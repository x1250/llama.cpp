# Inkling 2 Mini en strix-halo (2026-11-21)

Inkling-2-Mini (Thinking Machines) es un MoE de 26 B parámetros con 3.4 B activos y la arquitectura `inkling2_loop`:
- 128 expertos con 8 activos, en NVFP4 en el checkpoint.
- 24 capas físicas; las 6-17 se recorren tres veces (48 aplicaciones), con controladores de mensaje entre pasadas.
- Atención SWA de 1024 con 4 capas globales y un sesgo de posición relativa por cabeza (tabla de 1025 distancias).
- Convolución corta con estado recurrente, activación SiTU y residuales de atención (`attn_res`): cada capa mezcla los
  bloques anteriores del residual con una consulta aprendida.
- Visión (`inkling2v`), contexto nativo de 1M y razonamiento que no se apaga (`reasoning_effort`, 0.9 por defecto).

Se instaló como `inkling2mini` en strix-halo. Upstream lo soporta desde b12603, pero lo corría lento en esta máquina;
este documento registra lo que se cambió, cómo se midió y lo que queda abierto. Las cadenas, los logs y las salidas
están en `~/ProjectHub/llama.cpp-evidence/inkling2mini/`.

**Desinstalado el 2026-11-22** por decisión del Director, después de dos fallas del razonamiento en el uso con agente
(sección 8.4). Las dos se vieron con los pesos sin filtro; el modelo base no llegó a probarse en ese uso, y que las
fallas sean suyas no quedó demostrado.

**Reinstalado el mismo día:** el modelo base, con el GGUF propio en Q8_0, para que el Director lo pruebe en ese uso
(sección 9).

**Desde el 2026-11-24/25:** LoopCD en producción (sección 11), DFlash2 medido y no adoptado (sección 12) y las fusiones
del decode adoptadas: 4050 a 1974 despachos por token, una petición de 28.3 a 31.5 t/s (sección 13).

## 0. Resultado

**Instalado** como `inkling2mini` en `strix`. Configuración:
- El GGUF propio `Inkling-2-Mini-NVFP4-Q8_0.gguf` (sección 1) con el mmproj F16 de Unsloth en la GPU.
- 1048576 × 2 slots, KV q8_0, ubatch 512 (sección 4).
- Muestreo de Unsloth: temp 1.0, top_p 1.0, top_k 0, min_p 0.

**Velocidad** (llama-bench, contexto corto, `final-ab.sh`): el build instalado (master `6fb6b6ed7`) contra la base
medida al inicio y al final de la misma cadena. La base es b12603 con el arreglo de SCALE (punto 1 de la sección 2),
el mismo GGUF y el sesgo denso de upstream.

| | Base | Instalado |
|---|---|---|
| Prefill pp2048, ubatch 512 (producción) | 500 t/s | 931 t/s |
| Prefill pp2048, ubatch 2048 | 448-451 t/s | 1035 t/s |
| Decode tg128 | 30.7-31.4 t/s | 35.2-35.4 t/s |

Upstream tal cual, con el UD-Q8_K_XL de Unsloth y sin el arreglo de SCALE, decodificaba a 16.6 t/s. En el server, con
el muestreo de producción y 2 slots, el decode es de 32-33 t/s.

**A 131k de profundidad** (con ubatch 2048, no el 512 de producción): prefill de 298 t/s y decode de 17.0 t/s (18.0 con
KV q8_0). El decode a profundidad está en el límite de memoria: cada atención global lee 537 MB por token a
~210 GB/s.

**Memoria:** se carga con 53.9 GiB y deja 48 GiB de MemAvailable (sección 5).

**Calidad:** KLD de 0.0029 contra la base con el mismo GGUF, con ubatches de 2048 a 4096 de contexto. La referencia
sustituta, la base con `-fa off`, da 0.0031 (sección 3). Contra upstream puro y hasta 163k de contexto, la KLD media es
de 0.0012 a 0.0040, el tamaño de la diferencia entre dos corridas de un mismo build (sección 9.2).

**No adoptado:**
- DSpark (el drafter de bloques de PR #32218): con el muestreo de producción baja el decode (sección 6).
- Las matrices de control del loop en F16: +1.2 % de prefill a cambio de una KLD de 0.0010 (sección 6).

## 1. El archivo

- **Base:** el UD-Q8_K_XL de Unsloth en `Mainline_Support/` (el layout `inkling2_loop` que lee upstream; los GGUF de la
  raíz del repo usan el de Unsloth), rev `ea0c79b2`, con su sha256 verificado contra el LFS del Hub. `download-models.sh`
  lo enlaza como `.gguf` en la raíz (hard link).
- **El GGUF propio** (`quant.sh`): Unsloth sube a BF16 todos los pesos que no son NVFP4, porque ggml no tiene MXFP8, y
  el decode de Vulkan lee los BF16 a su límite o por debajo. `llama-quantize` pasó esos pesos a Q8_0, dejó los expertos
  NVFP4 del checkpoint intactos (copiados byte a byte, `tensor_same.py`) y el embedding en BF16.
  - Contra el UD-Q8_K_XL, 4 chunks de 4096: KLD 0.0033, 97.8 % de top igual, PPL 5.8250 frente a 5.8239.
  - Decode: de 22.1 a 30.3 t/s.

## 2. Los cambios

Cada uno es un commit en master, medido contra el build anterior con la base primero y última en la misma cadena:

1. **`vulkan: run SCALE on strided inputs`** (`8cf98d8bd`). SCALE sobre vistas no contiguas caía en la CPU. Decode de 16.6
   a 22.1 t/s.
2. **El sesgo relativo dentro de la atención** (`ggml_flash_attn_ext_rel`, `GGML_OP_FLASH_ATTN_EXT_REL`):
   - El problema: upstream construye el sesgo como un tensor denso n_kv × n_tokens × 24 cabezas y lo suma a la máscara.
     Con un ubatch de 2048 a 32k son 6 GiB por capa de atención, sobre el rango de 4 GiB de un binding de Vulkan pasado
     ~21k celdas, y su gather, copias y suma dominaban el prefill.
   - El op nuevo lee la tabla por distancia dentro del kernel (shaders escalar y coopmat1; CPU).
   - Prefill pp2048 de 445 a 622 t/s; un prompt de 57k de 217 a 472 t/s; decode a 57k de 17.8 a 22.1 t/s.
3. **SiTU como op GLU** (`GGML_GLU_OP_SITU`): la activación de los expertos era 9 pasadas sobre las activaciones de los
   expertos; ahora es una. Prefill de 622 a 687 t/s.
4. **El sink plegado en el producto de la tabla de sesgo** (un prompt batch): una fila más en el producto en vez de una
   pasada de resta sobre la tabla.
5. **`MUL_MAT_ID_MUL` en el camino matriz-matriz**, con su regla en el optimizador de grafo y las escalas de los expertos
   construidas antes del producto. La escala del down se pliega en los pesos del router:
   - Antes, cada capa NVFP4 hacía tres pasadas sobre las salidas de los expertos (escala de gate_up, escala de down y
     pesos del router), 136 por ubatch.
   - Ahora los shaders de matriz-matriz escalan cada fila al escribirla.
   - El optimizador de grafo separaba cada producto de su MUL: metía entre ellos la normalización del router, que es
     independiente. Ahora un MUL_MAT_ID espera a que la otra fuente de su MUL esté calculada.
   - Las 92 fusiones del ubatch se disparan. La salida es idéntica bit a bit con y sin fusión (KLD 0).
   - El ubatch de 2048 pasa de 2443 a 2260 ms de GPU; prefill de 454 a 917 t/s acumulado contra la base.
6. **Atención sobre la profundidad** (`ggml_depth_attn`, `GGML_OP_DEPTH_ATTN`):
   - Cada mezcla de los residuales de atención era un stack, normas, scores, softmax y una suma pesada.
   - En el decode eran más de 4 mil despachos por token, y en el prefill un producto matriz-vector por token a ~15 GB/s.
   - Ahora es un despacho por mezcla: un workgroup por token, con una reducción de orden fijo.
   - Prefill de 914 a 1038 t/s; decode de 32.3 a 33.2 t/s.
7. **El recorte de norma L2** (`ggml_l2_norm_clip`, `GGML_OP_L2_NORM_CLIP`):
   - Los recortes de q, k y la proyección relativa eran 8 despachos cada uno, unos 1150 por token.
   - Ahora es uno por recorte. Decode de 33.2 a 35.3 t/s.
8. **`ggml : the views after a tensor larger than max_size are initialized`**, un defecto de upstream:
   - `ggml_backend_buft_alloc_buffer_n_plan` abre un buffer nuevo cuando el tensor siguiente no cabe, y una vista cuenta
     0 bytes.
   - Detrás de un tensor más grande que `max_size` (1 GiB en Vulkan, el bloque de suballocación), la vista siguiente
     abría un ítem de tamaño 0. Si detrás solo venían vistas, ese último ítem no se agregaba al plan y las vistas
     quedaban sin buffer.
   - Con 1048576 × 2 slots, K y V de cada capa global pesan 2.2 GB: las vistas por stream de la última capa quedaban
     sin inicializar. La atención no lo notaba (arma sus vistas en cada grafo), pero guardar un slot en el prompt cache
     abortaba el server (la sonda de instalación, sección 5).
   - Ahora una vista se queda en el buffer en curso. `test-alloc` tiene el caso nuevo: falla sin el cambio y pasa con él.

Los ops nuevos (2, 3, 6 y 7) tienen kernel de CPU y de Vulkan, con sus casos en `test-backend-ops`: FLASH_ATTN_EXT_REL
54/54, SITU 99/99 con SWIGLU, DEPTH_ATTN 38/38, L2_NORM_CLIP 41/41 con L2_NORM y MUL_MAT_ID_FUSION 24/24 con y sin
matrix cores. Otros backends no los tienen: el scheduler los manda a la CPU.

## 3. Calidad

La compuerta de calidad del fork (CLAUDE.md) compara la KLD del build nuevo contra la base con la de una referencia de
puro redondeo, la base con `GGML_VK_DISABLE_FUSION=1`.

**En Inkling esa referencia da 0:** KLD 0 y 100 % de top igual. Las fusiones de la base son exactas bit a bit en este
modelo, así que la regla "a lo sumo el doble de la referencia" no puede medir nada.

**La referencia sustituta** es la misma base con `-fa off`:
- La misma matemática, con el softmax en kernels F32 propios.
- El sesgo en F32; con flash attention la base lo redondea a F16 con la máscara.
- Es justo la precisión que recupera el sesgo dentro del kernel, y perturba ese camino en los cuatro modos.

Build nuevo `f917064f6` contra la base, por modo (`qgate.sh`, `qref.sh`):

| Modo | KLD nuevo | KLD referencia | Top igual nuevo | Top igual referencia |
|---|---|---|---|---|
| p: ubatch 2048 a 4096 | 0.0030 | 0.0031 | 97.9 % | 98.0 % |
| d: ubatch 3 a 4096 | 0.0018 | 0.0018 | 98.1 % | 98.1 % |
| P: ubatch 2048, 2 × 16k | 0.0028 | 0.0033 | 98.1 % | 97.8 % |
| M: ubatch 3, 16k | 0.0033 | 0.0036 | 98.0 % | 97.9 % |

- **Dirección:** contra la corrida de sesgo F32 (`-fa off` como base), en el modo d el build nuevo queda a la misma
  distancia que la base (0.0018 y 0.0018). En el modo p queda 0.0003 más lejos (0.0034 frente a 0.0031), que es del
  orden de la referencia: SiTU y la mezcla reordenan sumas.
- **Commits siguientes:** cada uno se midió contra la misma base en los modos p y d y quedó en ese orden (0.0029 y
  0.0018 con el build final).
- **KV q8_0:** frente a f16, KLD 0.0032 (p) y 0.0021 (d), el tamaño de la referencia.
- **Determinismo:** `inkling_repeat.py` envía la misma petición cinco veces con `cache_prompt` false y otra petición
  distinta antes de cada una: 48 tokens greedy y sus log-probabilidades top-4. Se repitieron idénticos con los builds
  de la fusión, de la atención sobre la profundidad y el instalado.
  - La sonda de 4.7k del fork (`repeat_probe.py`) no sirve en Inkling: su primer token abre el razonamiento al 100 %.

## 4. Contexto y ubatch

Una atención global de un ubatch tarda en proporción a ubatch × celdas: 579 ms con 2048 a 131k y 305 ms con 1024
(`depth-bench.sh`). Cada una va sola en su submit, porque el reparto por flops de Vulkan las separa, así que el límite
es el timeout de 2 s del anillo de cómputo (`amdgpu.lockup_timeout`). Además, con ubatch 2048 la máscara de una capa
global a 1M ocupa 4 GiB, el tope de un binding de Vulkan.

| ubatch | Atención global a 1M | Prefill pp2048, contexto corto | A 131k |
|---|---|---|---|
| 2048 | ~4.6 s | 1031-1035 t/s | 298 t/s |
| 1024 | ~2.3 s | | 291 t/s |
| 512 | ~1.2 s | 926-930 t/s | |

Producción usa 512, el único que entra en el timeout al contexto nativo (-10 % de prefill a contexto corto).
`UBATCH=2048` sigue disponible hasta ~300k de contexto.

## 5. Memoria y sonda de instalación

`install-probe.sh`, con master y el arreglo del punto 8:
- **Memoria:** 1048576 × 2 slots, KV q8_0. MemAvailable de 102.0 a 47.7-48.1 GiB al cargar (−53.9 a −54.5 GiB). Con
  `srv_mem`, GTT 1.89 + VRAM 51.98 + anónima 0.26 GiB al cargar y 54.9 GiB después de todos los checks. `kv_gb` = 40
  en `config.ini` (pesos 17.2 + mmproj 0.6 + 40).
- **Checks** (`server_check.py`, `inkling_repeat.py`):
  - **Razonamiento:** aparece en `reasoning_content` y la respuesta es correcta (37 × 43 = 1591).
  - **Tool call:** `get_weather` con `{"city":"Santiago"}`.
  - **Prompt cache:** una petición, otra en el mismo slot y la primera otra vez dan la misma respuesta greedy; la tercera
    evalúa 4 de 19 tokens y lee el resto del caché. Esta es la petición que abortaba antes del punto 8.
  - **Dos a la vez:** 43.9 t/s en total con ubatch 512 y 46.4 con 2048 (`vision_check.py`).
  - **Lectura de imagen:** una imagen de 1024 × 768 con "CODE 4817 BLUE" escrito. La respuesta es "4817 BLUE" con
    ubatch 512 y con 2048 (`vision_check.py`).
  - **Velocidad del server**, muestreo de producción, 512 tokens con `reasoning_effort` none: 32.2 t/s en código y 32.3
    en prosa.
  - **Imagen de 2048 × 2048 en el slot 1:** 2734 tokens en 9.2 s (442 t/s). Con ubatch 512 no aborta: el encoder de
    Inkling no exige un ubatch del tamaño de la imagen.
  - **Historia:** la misma petición greedy en el slot 0, antes y después de esa imagen, da los mismos 48 tokens y
    log-probabilidades.
  - **Determinismo:** cinco peticiones idénticas, con otra distinta antes de cada una, dan lo mismo.

## 6. No adoptado

- **DSpark** (PR #32218, rama `inkling-dspark` y worktree `llama.cpp-dspark`). Se portó al checkpoint en loop, con el
  converter, la cabeza cortada en n-max y una variante de POOL_1D por subgrupo para el argmax de 201024 logits.
  - Un paso con n-max 2 cuesta 61-64 ms frente a 34 sin draft.
  - Con el muestreo de producción acepta 0.74 tokens por paso: 27.4 t/s frente a 29.4 sin draft. Con código sube a
    ~38 t/s y en prosa baja a ~26.
  - Como el razonamiento no se apaga y es prosa, el balance es negativo. No está en master.
- **Matrices de control del loop en F16:** `llama-quantize` deja los tensores `.loop_` como vienen (BF16), y esta GPU
  no tiene BF16, así que sus productos corren en el shader escalar.
  - La conversión exacta salvo 5062 subnormales de 28.3 M valores (`ctrl_f16.py`) dio +1.2 % de prefill con ubatch
    2048.
  - Pero la KLD contra el original fue 0.0010: el producto F16 redondea también las activaciones. Descartado.
- **HIP:** no se evaluó. Los ops nuevos son de CPU y Vulkan; HIP exigiría portar cuatro. El backend HIP de Strata para
  este modelo está leído en la sección 10.

## 7. Abierto al instalar (cerrado por la desinstalación)

Lo que quedaba al instalar. Las secciones 8.1 a 8.3 atacaron el prefill a profundidad, los expertos NVFP4 y la cola de
despachos; la desinstalación cerró el resto sin trabajarlo.

- **El prefill a profundidad:** la atención global corre a 5.7 TFLOPS en el coopmat1 y es dos tercios del ubatch a 131k.
- **Los expertos NVFP4:** el MMQ coopmat1 corre a 16-18 TFLOPS (Q8_0 denso, 24); es el 35 % del ubatch a contexto corto.
- **Ubatch 2048 a 1M:** dividir las atenciones globales en bloques de consultas acotados por trabajo, como hacía el
  fork en qwen4exp.
- **La cola de despachos chicos del decode:** unos 5600 por token. Los controladores de mensaje del loop son ~1150, y la
  convolución con su residual y su escala ~960.
- **DSpark con draft adaptativo** (apagado cuando acepta poco).

## 8. Después de la instalación y desinstalación (2026-11-21/22)

Las velocidades son de llama-bench con KV q8_0, con la base primera y última en la misma cadena.

### 8.1 Adoptado

- **El sesgo relativo en la atención coopmat1 volteada** (`b318a4692`, `flip-check.sh`). FLASH_ATTN_EXT_REL caía en el
  coopmat1 sin voltear. Con 24 cabezas de consulta sobre 8 de K/V, ese shader corre a un tercio del volteado.
  - Perf de la atención global a 32k con K/V f16: de 4.6 a 10.3 TFLOPS con ubatch 512 y de 3.7 a 10.0 con 2048. Un KV
    q8_0 llega por el dequant a f16.
  - Prefill pp2048 a 32k de profundidad: de 585 a 750 t/s con ubatch 512 y de 651 a 846 con 2048.
  - A 131k: 522 t/s con ubatch 1024 y con 2048, y 14 TFLOPS en la atención (`fmt-depth.sh`).
- **La convolución corta con su residual y su escala** (`f533b0cd8`, `conv-check.sh`). Es una fusión SSM_CONV + ADD +
  SCALE en el shader de SSM_CONV. El optimizador de grafo deja pasar el ADD por delante de una vista pendiente de un
  tensor ya calculado (su residuo): así se funden 156 de las 192 convoluciones de un paso. De las otras 36, 16 son las
  dinámicas de las capas globales (K y V en las 8 visitas globales: el perfil trae 16 de sus proyecciones por token).
  Las otras 20 coinciden en número con las visitas SWA sin mensajes; su causa no se investigó.
- **La cola de los controladores del loop** (`22cac0e76`, `GGML_OP_TANH_BLEND`): `a*(1 - tanh(f)) + b*tanh(g)` en una
  pasada, en vez de seis.
  - Con las dos fusiones, contra el build anterior: decode de 34.8 a 36.5 t/s, y prefill pp2048 de 968 a 1008 t/s con
    ubatch 512 y de 1070 a 1122 con 2048 (`fuse2-check.sh`).
  - Las dos son exactas: la KLD en los modos p y d es idéntica a la del build anterior.
  - Instaladas y verificadas en producción (`install-fusions.sh`): 33.6 t/s en el server.
- **La escala del down fuera de los pesos del router con un LoRA** (`34b6627a9`):
  - El pliegue de la escala por experto (punto 5 de la sección 2) también escalaba el delta de un LoRA sobre
    `ffn_down_exps`. Ahora el pliegue se omite cuando un adaptador toca la proyección.
  - Con el LoRA uncensored de AtomicChat sobre el GGUF propio, la KLD contra el modelo sin él es 0.0092 (0.0074 con
    el pliegue; su ficha da 0.0078). `lora-unc.sh`.

### 8.2 Medido y no adoptado

- **Kernels NVFP4** (rama `inkling-nvfp4-mmvq`, `nvfp4-ab.sh`). Tres cambios:
  - NVFP4 en el matriz-vector entero (activaciones q8_1).
  - La escala UE4M3 calculada en el matmul int8 coopmat, en vez de la tabla compartida.
  - La rama de dos escalas de ese matmul reestructurada.

  En `test-backend-ops perf`, a la forma de un experto con un token, el producto bajó un tercio: gate_up de 67 a 45 µs y
  down de 39 a 29 µs, cerca de MXFP4 (40 y 24 µs). En el modelo no rindió: decode de 34.8 a 35.0-35.1 t/s, prefill
  igual, y la KLD algo más alta (0.0019 frente a 0.0018 en el modo d, 0.0033 frente a 0.0031 en el p). La causa probable
  es que el perf repite los mismos pesos, que quedan en caché, y en el modelo cada experto se lee de la RAM; no se midió
  el ancho de banda del decode.
- **Otros formatos de 4 bits para los expertos** (`fmt-depth.sh`, a la forma de los expertos): con 512 y 2048 tokens,
  IQ4_NL, MXFP4 y Q4_0 tardan 18-22 % menos que NVFP4. Convertirlos sería recuantizar los pesos NVFP4 del checkpoint, con
  pérdida. No se probó en el modelo.

### 8.3 No terminado

- **DEPTH_ATTN con workgroups de 1024 hilos** cuando hay menos tokens que núcleos (el decode).
  - En el perfil del decode, cada llamada bajó de 16.9 a 10.9 µs: 1.3 ms frente a 2.0 por token.
  - La cadena se detuvo para devolver producción y no se completaron la KLD ni el A/B.
  - El cambio quedó en `depth-attn-wide.patch` en la carpeta de evidencia.

### 8.4 Los loops de razonamiento

El Director vio dos fallas distintas en el uso con agente (pi). Las dos fueron con pesos sin filtro: el LoRA uncensored
de AtomicChat sobre el GGUF propio, o su GGUF horneado. El modelo base corrió un solo turno en pi, abortado a mano: no
quedó probado. Las cifras son de una sesión larga de pi con texto en español (`pi_session_loops.py`,
`runs/pi-session-loops.out`).

**Loop de bloque** (tres turnos seguidos, de 42k a 60k de contexto, esfuerzo medium):
- El razonamiento repite líneas enteras: 102 de 185, 137 de 153 y 272 de 338 líneas largas. En los dos últimos la cola
  repite un bloque exacto, de 2749 y de 673 caracteres.
- Los dos primeros turnos terminaron solos (9448 y 7352 tokens). El tercero llegó al tope de 16384.
- pi tenía elegido un modelo de otra familia mientras el servidor era Inkling, así que envió el muestreo de ese modelo
  (temp 0.7, top_p 0.95, top_k 20) y no el de Inkling (temp 1.0 sin truncar).
- El contexto de cada turno incluye la salida completa del anterior, razonamiento incluido (42558 + 9448 → 52098). El
  segundo y el tercero partieron con el loop anterior en el contexto y repitieron casi desde el principio: solo el
  primero es un inicio independiente.
- Al elegir Inkling en pi, el razonamiento de esos turnos salió del contexto (pi descarta el de otro modelo; de 76k
  bajó a 30k). Los 19 turnos siguientes, de 30k a 123k, no repitieron ninguna línea; uno generó 20744 tokens.

**Colapso** (a más de 130k de contexto, esfuerzo xhigh):
- No es un loop. El razonamiento se deshace en la palabra "casi" traducida a unos 30 idiomas, sin período: el último
  turno generó 65536 tokens en 63 min, el tope de salida, con 35605 palabras y 7093 distintas.
- Empieza de a poco. Hay una palabra suelta en el turno que cruza 65536 tokens de contexto y después 11 turnos limpios,
  de 72k a 123k. Desde el turno que cruza 131072 aparecen en todos: 1, 3, 14, 16, 12 y 85 palabras entre 130k y 159k, y
  el colapso total a 162.6k.
- Las palabras quedaron en el texto y en los argumentos de las herramientas, o sea en el contexto de los turnos
  siguientes: la falla se alimenta sola.
- 131072 es la mayor profundidad que se midió en esta instalación, y solo en velocidad; la calidad se midió hasta 16k
  (sección 3). Dos coincidencias con una potencia de 2 no alcanzan para concluir. Después de la reinstalación se midió
  hasta 163k contra upstream puro (sección 9.2).
- **Otro build:** el llama.cpp de Unsloth (prebuilt b12588-mix-c0c3de8 Vulkan) con el GGUF horneado de AtomicChat
  (AD-Q4_K) retomó esa misma conversación a 162.6k y colapsó igual: 433 palabras en 5.6k caracteres de razonamiento.
  - Descarta los parches del fork para ese estado, y nada más. Los pesos sin filtro, el historial con las palabras, el
    muestreo y el KV q8_0 eran los mismos.
  - El prebuilt es otro build de llama.cpp, no otro motor: comparte la implementación de upstream.

**De quién es cada falla:**
- **Loop de bloque:** es lo que describe la ficha de Thinking Machines y lo que terceros reportan con los pesos
  originales. En este uso tuvo un solo inicio independiente, con el muestreo de otro modelo, y no volvió en 19 turnos.
- **Colapso:** no está en la ficha. El reporte más cercano es el del hilo #7, también a gran profundidad. Quedó sin
  separar de:
  - Los pesos sin filtro: el LoRA edita también `token_embd`.
  - El muestreo: el del servidor es temp 1.0 sin truncar, y el que envió pi con Inkling elegido no quedó registrado.
  - El KV q8_0, la implementación de llama.cpp a más de 131k y el esfuerzo xhigh.
- El modelo se desinstaló sin separar esas causas. No se probó el base en pi, ni el loop de bloque con el muestreo de
  Inkling, ni el colapso con los pesos originales o con otro motor.
- La sección 9.2 midió después dos de ellas, sobre texto corriente y hasta 163k: los parches del fork contra upstream
  puro y el KV q8_0 contra f16. Ninguna mueve la distribución más que el redondeo. No reemplaza la prueba con agente.

**Lo que hay publicado:**
- **Ficha de Thinking Machines:** el modelo "a veces se queda atascado en loops largos de razonamiento, o vuelve a
  revisar su trabajo sin avanzar", sobre todo depurando código. Recomiendan topes de salida y de turnos. No recomiendan
  un muestreo.
- **Reportes de terceros:**
  - Hilos del repo oficial: en el #7, con pi y SGLang a 256k, el razonamiento se degrada a repetir una palabra después
    de varias compactaciones; en el #7 y el #14, párrafos repetidos con los pesos originales en vLLM; en el #16,
    salidas de más de 65k tokens reescribiendo comandos de bash.
  - autotrust (`inkling-2-mini-concise`) midió el original con vLLM y un tope de 32k: no terminan el 30 % de las
    respuestas de LiveCodeBench ni el 11 % de las de GPQA.
- **Mitigaciones conocidas:**
  - Topes de salida, esfuerzo bajo y presupuesto de razonamiento.
  - El loop guard de Maya (`serve/loop_guard.py`): detecta un bloque que se repite exacto y escala de una penalización
    temporal a cerrar el razonamiento. Cubre el loop de bloque; el colapso no tiene período.
  - La edición de rango 1 de autotrust sobre `attn_output` de las capas 18-23: menos generaciones desbocadas, −4.7 en
    IFBench.

  Ninguna elimina el loop de bloque. No se aplicó ninguna.

Maya y el PR de Strata que agrega Inkling (Niko1221/Strata#6506) usan el mismo motor CUDA (`ik2`). No corre en esta GPU
ni lee los expertos Q4_K y Q5_0 del GGUF de AtomicChat.

### 8.5 Desinstalación

- Se borraron los GGUF (el propio, el UD-Q8_K_XL, los de AtomicChat y los drafters), la descarga parcial del
  checkpoint original y el prebuilt de Unsloth.
- Se borraron los worktrees de experimentos (`llama.cpp-fa`, `-relfa`, `-mix`, `-dspark`). Las ramas `inkling-*` siguen.
- strix-halo revirtió sus entradas en `d42c259`. pi volvió a su configuración anterior y producción volvió a
  `qwen38flash`.
- Los parches de las secciones 2 y 8.1 siguieron en master sin ningún modelo de llama-server que los usara, hasta la
  reinstalación (sección 9).

## 9. Reinstalación del modelo base (2026-11-22)

El Director pidió el modelo base de vuelta, con el GGUF propio, para probarlo más a fondo: las dos fallas de la sección
8.4 se vieron solo con los pesos sin filtro.
- **Cuantización:** pidió primero el Q6. Con la medición de la sección 9.1 eligió el Q8_0, para que una falla del
  razonamiento no deje la cuantización como una causa más. El Q6_K quedó en disco, medido.
- **Los parches del fork:** preguntó si las optimizaciones podían estar detrás de los loops. La sección 9.2 compara el
  fork con upstream puro a la profundidad de las fallas.
- **Las pruebas en uso con agente las hace el Director.** Esta sección registra lo que se midió antes de entregarle el
  servidor.

### 9.1 Los archivos: Q8_0 y Q6_K

`Inkling-2-Mini-NVFP4-Q6_K.gguf` sale de la receta de la sección 1 con el tipo como parámetro (`q6.sh`). El Q8_0 se
generó de nuevo de la misma fuente, como base de la comparación.

- **Qué cambia:** los 185 tensores que la sección 1 pasó de BF16 a Q8_0. 181 quedan en Q6_K (3.1 GB, el 18 % del
  archivo). Los otros 4 son `ffn_down_exps` de las capas 20-23: sus filas de 1152 no son múltiplo de 256 y
  `llama-quantize` los deja en Q8_0.
- **Qué no cambia:** los 40 tensores de expertos NVFP4 (11.5 GB, el 67 % del archivo), el embedding en BF16 y el
  resto, iguales byte a byte a la fuente (`tensor_same.py`).
- **Tamaño:** 17.08 GB frente a 18.01 GB.

Calidad contra el UD-Q8_K_XL, en tres modos de la sección 3:

| Modo | KLD Q8_0 | KLD Q6_K | Top igual Q8_0 | Top igual Q6_K | PPL UD / Q8_0 / Q6_K |
|---|---|---|---|---|---|
| p: ubatch 2048 a 4096 | 0.0032 | 0.0044 | 97.8 % | 97.1 % | 5.826 / 5.816 / 5.822 |
| d: ubatch 3 a 4096 | 0.0022 | 0.0029 | 97.8 % | 97.5 % | 7.082 / 7.068 / 7.069 |
| P: ubatch 2048, 2 × 16k | 0.0030 | 0.0039 | 98.0 % | 97.7 % | 4.997 / 5.002 / 4.997 |

Velocidad, con el Q8_0 medido antes y después del Q6_K. llama-bench va con KV q8_0 y ubatch 512; el server, con 2 slots
de 32768 y respuestas greedy de 256 tokens.

| | Q8_0 | Q6_K |
|---|---|---|
| Prefill pp2048 | 983-988 t/s | 953 t/s |
| Lotes chicos pp2 / pp4 / pp8 | 57 / 82-83 / 107-108 t/s | 61 / 87 / 108 t/s |
| Decode tg128 | 36.1-36.2 t/s | 38.9 t/s |
| Server, una respuesta | 35.4-35.9 t/s | 38.5-38.6 t/s |
| Server, dos a la vez (cada una) | 26.3-26.9 t/s | 28.2-28.8 t/s |

El Q6_K decodifica un 7 % más rápido y lee el prompt un 3 % más lento, con 0.9 GB menos y una KLD 0.001 mayor.

### 9.2 El fork contra upstream puro, hasta 163k

Las compuertas de calidad (sección 3) comparan dos builds del fork hasta 16k de contexto. Las fallas de la sección 8.4
aparecieron de 42k a 60k (el loop) y de 65k a 163k (el colapso). Además, sobre las 8192 posiciones el modelo escala Q y
la proyección relativa de sus capas globales por tau = 1 + 0.1 · ln(posición / 8192) (`src/models/inkling2.cpp`): tau
vale 1.07 a 16k y 1.30 a 163k, y ninguna compuerta había pasado de 1.07.

**La sonda** (`depth_equiv.py`) envía una misma secuencia de peticiones, sobre los mismos tokens, a cada servidor:
- **Texto:** el test de wikitext-2, tokenizado una vez por el servidor (`runs/depth-equiv-ids.json`). Entre los tokens
  40000 y 64000 lleva un bloque de 600 tokens repetido, la forma de un loop de razonamiento.
- **Zonas:** diez profundidades, de 2048 a 163000 tokens. Las de 45000 y 60000 caen dentro del bloque repetido.
- **En cada zona:** el prefijo de una vez, 128 pasos de un token sobre el texto, 64 tokens generados y la misma
  predicción leída otra vez. Son 129 predicciones sobre el texto por zona.
- **Qué se compara:** las 100 log-probabilidades más altas de cada predicción, antes de todo sampler. La KLD cuenta los
  tokens comunes de las dos listas y el resto como un solo evento. La generación es greedy.

**Los servidores** (`depth-equiv.sh` con el Q6_K, `install.sh` con el Q8_0), con un slot de 172032 y KV f16:
- **Upstream puro:** llama-server de master (`1fb8a2696`, build 2020, 25 commits después de la base del fork), en el
  worktree `llama.cpp-ref` con la configuración de build del fork (`ref-build.sh`). Va con ubatch 128: su sesgo relativo
  denso, fuera de la flash attention, ocupa 2 GiB por capa global a 172k celdas × 128 tokens × 24 cabezas.
- **Fork:** el build instalado (`ee9ec0746`, build 2464), con el ubatch 512 de producción.
- **Fork sin fusiones:** el mismo con `GGML_VK_DISABLE_FUSION=1`. Mide lo que mueve el redondeo solo, con el mismo
  reparto en lotes.

Cada sonda comprobó que el server mapeaba solo las bibliotecas de su build, y registró el build y el archivo que servía.

KLD media por zona del fork contra upstream puro, y de las fusiones contra el mismo fork:

| Zona | Q6_K: KLD | Q6_K: top igual | Q8_0: KLD | Q8_0: top igual | Sin fusiones contra el fork (Q6_K) |
|---|---|---|---|---|---|
| 2048 | 0.00158 | 98.4 % | 0.00124 | 97.7 % | 0.00012 |
| 8192 | 0.00250 | 98.4 % | 0.00339 | 99.2 % | 0.00032 |
| 16384 | 0.00322 | 99.2 % | 0.00267 | 98.4 % | 0.00033 |
| 30000 | 0.00283 | 97.7 % | 0.00198 | 96.1 % | 0.00034 |
| 45000, en el bloque | 0.00000 | 100 % | 0.00000 | 100 % | 0.00000 |
| 60000, en el bloque | 0.00003 | 100 % | 0.00003 | 100 % | 0.00000 |
| 70000 | 0.00250 | 96.1 % | 0.00293 | 96.1 % | 0.00027 |
| 100000 | 0.00241 | 96.1 % | 0.00240 | 97.7 % | 0.00021 |
| 130000 | 0.00374 | 99.2 % | 0.00400 | 99.2 % | 0.00049 |
| 163000 | 0.00260 | 98.4 % | 0.00243 | 99.2 % | 0.00030 |

Producción (1048576 × 2 slots, KV q8_0, ubatch 512, el Q8_0), con la misma secuencia dos veces en el mismo server:
«solo» es el slot 0 con el otro libre, y «a la vez», los dos slots con la secuencia al mismo tiempo. La última columna
compara el slot 0 de las dos corridas (`runs/depth-equiv-prod-runs.out`).

| Zona | Solo contra upstream | A la vez contra upstream: slots 0 / 1 | Solo contra fork KV f16 | A la vez contra solo |
|---|---|---|---|---|
| 2048 | 0.00138 | 0.00116 / 0.00105 | 0.00152 | 0.00125 |
| 8192 | 0.00318 | 0.00387 / 0.00393 | 0.00210 | 0.00266 |
| 16384 | 0.00256 | 0.00237 / 0.00261 | 0.00208 | 0.00249 |
| 30000 | 0.00248 | 0.00155 / 0.00250 | 0.00224 | 0.00218 |
| 45000, en el bloque | 0.00000 | 0.00001 / 0.00000 | 0.00001 | 0.00002 |
| 60000, en el bloque | 0.00003 | 0.00003 / 0.00003 | 0.00000 | 0.00001 |
| 70000 | 0.00292 | 0.00249 / 0.00294 | 0.00158 | 0.00148 |
| 100000 | 0.00260 | 0.00274 / 0.00307 | 0.00267 | 0.00277 |
| 130000 | 0.00406 | 0.00354 / 0.00304 | 0.00289 | 0.00356 |
| 163000 | 0.00223 | 0.00278 / 0.00216 | 0.00229 | 0.00245 |

**Lectura:**
- **No crece con la profundidad.** La zona de 163k (0.0022 a 0.0028) queda como las de 8k y 16k (0.0024 a 0.0039), que
  las compuertas ya cubrían. La mediana por token es de 0.0005 a 0.0016 en todas las zonas fuera del bloque.
- **En el bloque repetido los dos builds dan la misma distribución:** KLD de 0.00003 o menos, el mismo token más
  probable en las 129 predicciones y los mismos 64 tokens generados.
- **La distancia a upstream es la de dos corridas de un mismo build.** La última columna compara dos corridas del build
  instalado, en el mismo server y el mismo slot. Entre las dos cambia el otro slot, activo en una, y con él el reparto
  de los tokens en lotes. Da de 0.0012 a 0.0036, lo mismo que el fork contra upstream (0.0012 a 0.0040), que además leen
  con ubatches distintos.
- **Las fusiones**, con el mismo reparto en lotes, mueven diez veces menos: de 0.0001 a 0.0005.
- **El KV q8_0 y la forma de producción** no se distinguen de ese piso: 0.0015 a 0.0029 contra el fork con KV f16.
- **Tokens sueltos:** la KLD máxima de un token llega a 0.146 entre el fork y upstream, y a 0.054 entre las dos
  corridas de producción. Los 64 tokens generados coinciden enteros en 5 de las 10 zonas con el Q6_K y en 3 con el
  Q8_0, y en 3 entre las dos corridas de producción. En las demás se separan en algún token.

**Conclusión:** a la profundidad de las fallas, el fork da la distribución de upstream dentro de lo que cambia entre dos
corridas de un mismo build. Con esta medición los parches dejan de ser un candidato a causa de los loops. Coincide con
la sección 8.4, donde otro build de llama.cpp colapsó igual sobre la misma conversación.

**Lo que no cubre:** es texto corriente y no el razonamiento del modelo, una sola secuencia, y llega a 163k. Compara
dos builds de llama.cpp, no llama.cpp contra la implementación de referencia de Thinking Machines.

Velocidad de la misma secuencia con el Q8_0 (subproducto; el prefijo se lee con ubatches distintos):

| | Upstream puro, ubatch 128 | Fork, ubatch 512 |
|---|---|---|
| Leer el prefijo, a 2k y a 163k | 356 y 102 t/s | 846 y 418 t/s |
| Un paso de un token, a 2k y a 163k | 79 y 201 ms | 76 y 154 ms |
| Generar, a 2k y a 163k | 27.1 y 9.6 t/s | 31.6 y 17.7 t/s |
| La secuencia entera | 22.0 min | 7.8 min |

Al cargar, MemAvailable bajó 26.2-27.7 GiB con upstream y 23.0-23.4 con el fork. Ningún reset de anillo.

### 9.3 Una predicción leída tres veces

En cada zona, el último paso, el primer token generado y la relectura final son la misma predicción sobre el mismo
caché. Cambia el camino del último token:
- **Agregar:** el paso lo decodifica a continuación del caché.
- **Restaurar:** si el prompt ya está entero en el caché, el server vuelve a su último checkpoint y relee el final
  (`server_prompt::get_reuse`). Así llegan la generación y la relectura.

Con un solo slot activo (`reread_detail.py`, `runs/reread-detail-q6.out` y `-q8.out`; seis corridas, 60 zonas):
- **Restaurar dos veces da lo mismo, bit a bit,** en 56 de las 60 zonas. Las otras 4 son la zona 70000 de las cuatro
  corridas con ubatch 512 (el fork en Q6_K con y sin fusiones, el fork en Q8_0 y producción).
- **Agregar y restaurar difieren siempre,** también en upstream puro: KLD de 8e-7 o menos en 56 de las 60 zonas, con
  una brecha máxima de 0.001 a 0.008 nats fuera del bloque repetido.
  - Dentro del bloque la brecha máxima no mide nada: llega a 4.4 nats en tokens de la cola, con el token más probable
    sobre 0.999 y una KLD de 1e-5 o menos.
- **Peticiones idénticas dan resultados idénticos por el mismo camino:** `inkling_repeat.py` repitió igual en 5
  corridas, con otra petición antes de cada una (sección 9.5).

**La causa es el layout del caché SWA:**
- Las capas con ventana guardan K y V en un anillo de 1536 celdas con ubatch 512 y de 1280 con 128
  (`llama-kv-cache-iswa.cpp`: la ventana más un ubatch, en múltiplos de 256).
- `llama_kv_cache::find_slot` reparte las celdas recorriendo el anillo desde su cabeza. Al restaurar un checkpoint,
  `state_read_meta` limpia la secuencia y las pide de nuevo: la ventana queda en otras celdas, según dónde estaba la
  cabeza.
- La atención suma en orden de celda. Los mismos valores en otro orden redondean distinto.
- `swa_layout.py` porta a Python esas funciones y las mueve como el server. Predice el patrón completo
  (`runs/swa-layout-ub512.out`, `-ub128.out`): agregar y restaurar dejan layouts distintos en las 10 zonas; con ubatch
  128 las dos restauraciones dejan el mismo, y con 512 también, salvo en la zona 70000. Coincide con las 60 zonas
  medidas.
- Es código de upstream, no del fork: su server puro lo muestra igual.

**Sin explicar:** en 3 de las 60 zonas, fuera del bloque, agregar y restaurar difieren bastante más que el resto.

| Corrida | Zona | KLD | Brecha máxima |
|---|---|---|---|
| Upstream puro, Q6_K | 30000 | 7.4e-5 | 0.10 nats |
| Fork sin fusiones, Q6_K | 130000 | 2.9e-4 | 0.68 nats |
| Producción, slot 0 solo | 163000 | 1.9e-3 | 0.22 nats |

- Aparece en upstream puro y sin fusiones: no es de un parche ni de una fusión.
- En la de producción, el log del server confirma que el otro slot no corría nada y que cada lectura releyó un solo
  token (`runs/srvlog-prod.log`).
- **Hipótesis, sin verificar:** la diferencia de redondeo cambia una decisión discreta del modelo, por ejemplo uno de
  los 8 expertos que elige el router entre 128 cuando dos puntajes casi empatan. El mismo mecanismo explicaría que dos
  corridas enteras queden a una KLD de 0.001 a 0.004 cuando el prefijo se repartió distinto.
- Queda para la medición controlada de la sección 9.4.

### 9.4 Dos slots a la vez

Producción tiene dos slots y el caché no es unificado: cada slot es un stream. `install.sh` corrió la sonda en los dos
slots a la vez, como dos agentes.

**Cómo comparten un lote** (leído en el código):
- El server pone primero los tokens muestreados de los slots que generan y después los de prompt, en orden de slot.
  Con `--decode-share`, el token de un slot que genera viaja con un paso de prompt del otro.
- La memoria híbrida parte el lote con `split_equal` secuencial (`llama-memory-hybrid-iswa.cpp`): un token del slot 0 y
  uno del slot 1 comparten ubatch solo si el del slot 0 va primero.
- En ese ubatch, la atención recorre la mayor cantidad de celdas de los dos streams (`llama_kv_cache::get_n_kv`).

**Qué se midió** (`both_slots_timeline.py`, `runs/both-slots-timeline.out`): el log del server da el lanzamiento, los
tiempos y la liberación de cada tarea, y con eso cada una de las tres lecturas queda clasificada como sola o acompañada.
Cuando el otro slot leía un prompt de varios lotes, la clasificación se infiere de las reglas del lote y de los tiempos:
el log no registra cada ubatch.

| | Slot 0 | Slot 1 |
|---|---|---|
| Lecturas solas / acompañadas | 30 / 0 | 3 / 27 |
| Zonas en que restaurar dos veces da lo mismo, bit a bit | 10 de 10 | 0 de 10 |
| KLD entre las dos restauraciones | 0 | 2e-8 a 2e-3 |

- **Zona 2048 del slot 1:** las dos lecturas solas coinciden en el tamaño del layout (KLD 3.9e-8, brecha 0.002 nats).
  La acompañada difiere de las dos (KLD 2.4e-6 y 2.6e-6, brecha 0.11 nats).
- **Slot 0, agregar contra restaurar:** KLD de 1e-8 a 1e-6, como con un slot solo, salvo en la zona 100000 (3.2e-5,
  brecha de 0.13 nats). Es un caso más de los que la sección 9.3 deja sin explicar.
- **La distancia a upstream no cambia:** de 0.0010 a 0.0039 en los dos slots, como el slot 0 solo (sección 9.2).

**Lo que significa:** con dos agentes a la vez, los logits de una petición dependen, en el tamaño del redondeo, de si su
token comparte ubatch con el del otro slot y de con cuál. Las respuestas no son reproducibles bit a bit. No se midió
ningún efecto en la calidad, y producción muestrea a temperatura 1.0.

**Lo que no está identificado:**
- **El mecanismo en los kernels.** Descartado en el código: el producto de los expertos usa el mismo kernel con uno y
  con dos tokens (`ggml_vk_use_mul_mat_vec_id`, hasta 8 tokens). Candidatos sin verificar: la atención sobre la mayor
  cantidad de celdas de los dos streams, los kernels matriz-vector de dos columnas y el KV q8_0.
- **Otro caso de la misma clase sí está identificado, a nivel de op** (sección 9.7): cuando los dos slots leen prompt
  en un mismo ubatch (64 tokens o más de cada uno), la atención global con KV q8_0 corre en otro kernel que con un
  slot solo y da otros bytes para el mismo stream. No es el de esta medición, donde compartían ubatch tokens sueltos.
  La rama `vk-fa-flip-q8` lo quita; no está adoptada.
- **Por qué dos lecturas acompañadas difieren entre sí.**
- **Si upstream puro lo muestra:** su referencia corrió con un solo slot. El reparto del lote es código de upstream;
  `--decode-share` es del fork.

**Ninguna compuerta de determinismo lo ve:** todas usan un slot activo a la vez. El check "dos a la vez" de
`server_check.py` comprueba que las dos peticiones respondan y mide la velocidad, no compara logits.

**Medición pendiente,** con producción descargada: un server nuevo a contexto reducido y dos slots, con el mismo token
decodificado solo y acompañado de compañeros elegidos; con KV f16 y q8_0, con `--decode-share` y sin él, y con el caché
unificado (`-kvu`). Los dos registros están en `latent-defects.md`.

### 9.5 Instalación

`install.sh`, con el build instalado (`ee9ec0746`, build 2464):
- **Archivo:** `Inkling-2-Mini-NVFP4-Q8_0.gguf` (`quant = NVFP4-Q8_0` en `config.ini`). Con `QUANT=NVFP4-Q6_K` el
  launcher sirve el Q6_K.
- **Forma:** la de la sección 0. 1048576 × 2 slots, KV q8_0, ubatch 512, muestreo de Unsloth.
- **Build:** agrega `34b6627a9`, que solo actúa con un LoRA, al build 2461 de la instalación anterior.
- **Memoria** (`srv_mem.py`): 54.14 GiB al cargar (GTT 1.89 + VRAM 51.99 + anónima 0.26). Después de 163k tokens en
  cada slot, 61.78 GiB (anónima 7.51). MemFree no bajó de 35.1 GiB. Ningún reset de anillo.
- **Checks** (`server_check.py`, `inkling_repeat.py`):
  - Razonamiento, tool call, caché de prompt e historia: correctos.
  - Velocidad: 33.4-33.5 t/s una respuesta y 46.6 t/s en total con dos a la vez.
  - Imagen de 2048x2048 en el slot 1: 8.8 s.
  - Determinismo: 5 corridas idénticas.
- **pi** (`effort_check.py`): la entrada `inkling2mini` de `~/.pi/agent/models.json` lleva el contexto de 1048576, el
  tope de salida de 65536 y el muestreo de Inkling. Cada nivel de razonamiento de pi llega a la plantilla como el
  esfuerzo que corresponde: minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh y max 0.99. pi no ofrece apagarlo.
- **A profundidad** (la sonda en el slot 0): lee el prefijo a 873 t/s a 2k y a 413 t/s a 163k, y genera a 32.3 y a
  19.4 t/s. Con los dos slots a 163k, cada uno genera a 11.0-12.5 t/s.
- **Servidor entregado:** las sondas dejan 163k tokens en cada slot, así que la cadena lo cargó de nuevo. Quedó arriba
  un server sin historia (54.12 GiB).

**Sin medir:** más de 163k tokens en un slot.

**Queda en disco:** el Q6_K (17.08 GB), la fuente UD-Q8_K_XL (22.87 GB) y el worktree `llama.cpp-ref` (1.8 GB), que
hace falta para la medición pendiente.

### 9.6 En la prueba del Director: el marcador de medios en el historial

**Síntoma** (2026-11-22, una sesión de pi a 67.7k de contexto): toda petición falla con
`400 Failed to tokenize prompt`, también al reintentar.

- **Causa:** el agente consultó `/props` con una herramienta. La respuesta lleva `media_marker`, el texto con que el
  server marca dónde va cada imagen dentro del prompt (`get_media_marker`, `server-common.cpp`: `<__media_`, una cadena
  aleatoria y `__>`, uno por proceso). Desde ese turno el marcador está en el historial como texto.
- **Efecto:** `mtmd_tokenize` parte el prompt por el marcador y espera una imagen por cada uno. El log del server lo
  registra en cada intento (`runs/media-marker-400.log`):
  `number of media markers in text (1) exceeds number of bitmaps (0)`. La primera vez, 124 ms después del turno que
  lanzó la herramienta.
- **Alcance:** es código de upstream, igual en su master (`1fb8a2696`). Vale para todo modelo de llama-server con
  mmproj y para cualquier texto que contenga el marcador del proceso.
- **Salida de la sesión, sin el arreglo:** retomarla desde antes de esa llamada. Recargar el server también la
  destraba según el código, porque el marcador cambia en cada arranque y el del historial queda como texto corriente.
  No se probó.

**Arreglo** (`9a62394d4`, por orden del Director): el marcador es de cada petición.
- **Diseño:** cada petición de chat lleva sus archivos y un marcador aleatorio propio, que nunca se publica
  (`server_prompt_media`). El server lo escribe donde va cada imagen, parte el prompt ya renderizado por esa cadena
  exacta y exige un marcador por archivo (`process_mtmd_prompt`, sobre `mtmd_tokenize_from_parts`). El texto del
  cliente no lo puede contener, porque no lo conoce. El marcador publicado, dentro de un mensaje, es texto.
- **El divisor de la plantilla de Inkling** (`common/parsers/inkling.cpp`) tenía el mismo defecto por la forma: sacaba
  de su bloque de texto cualquier cadena `<__media_…__>`. Ahora parte por el marcador de la petición.
- **Rutas cubiertas:** `/v1/chat/completions`, `/v1/responses`, `/v1/messages`, sus contadores de tokens y
  `/v1/completions`, que no recibe archivos. También `/v1/audio/transcriptions` y el contenido con formato OAI de
  `/embeddings`, sin probar: este modelo no tiene codificador de audio ni se sirve para embeddings.
- **Lo que queda igual:** `/props` publica `media_marker`. En los caminos crudos (`/completion` con `prompt_string` y
  `multimodal_data`) es el mecanismo con que el cliente ubica sus archivos, y `/apply-template` devuelve el prompt
  con ese marcador. Ahí un texto que lo contenga sigue contando como un archivo (`latent-defects.md`).
- **Superficie:** `tools/server` y `common`, sin tocar mtmd ni libllama. Son archivos de upstream: el parche se
  reaplica en cada merge mientras upstream no cierre el defecto. Sus issues #25285, #27249 y #29118 están cerrados
  sin arreglo en master; el PR #29291, que quita del texto del usuario los marcadores sueltos, seguía abierto.

**Verificación contra el build anterior** (`marker-check.sh`, `marker_check.py`): 2 slots de 32768, cuatro cargas, el
build anterior (2464) primero y último. Cada server publica un marcador fijado con `LLAMA_MEDIA_MARKER`. Se compara el
estado HTTP, los tokens de prompt, 16 tokens greedy y sus 4 log-probabilidades más altas.

| Comparación | Casos | Resultado |
|---|---|---|
| Build anterior, dos cargas, sin marcador en el texto | 13 | iguales |
| Build nuevo contra el anterior, sin marcador en el texto | 16 | iguales |
| Build nuevo: el marcador publicado en el texto, contra la misma cadena sin publicar | 19 | iguales |
| Build anterior: el marcador publicado en el texto | 6 | 400 en los tres de chat y en `/v1/completions`, 500 en el contador, 200 en `/apply-template` |
| Build nuevo: el marcador publicado en el texto | 6 | 200 |

- **Sin marcador en el texto:** un chat de texto, un turno de agente con llamadas a herramientas, una imagen, una
  imagen entre dos textos, tres imágenes, una de 2048x2048, `/v1/completions`, los tres contadores, `/v1/responses`,
  `/v1/messages`, la transcripción (501 en los dos builds), `/apply-template`, `/completion` crudo con su imagen y un
  marcador sin su archivo (400 en los dos).
- **Con el marcador en el texto:** un mensaje, el turno de agente (el marcador en el resultado de una herramienta, en
  un texto del asistente y en los argumentos de otra llamada), un mensaje con imagen, `/v1/completions`, el contador
  y `/apply-template`.
- **El divisor por forma, en el build anterior:** la misma cadena sin publicar daba 65, 427 y 211 tokens de prompt,
  contra 62, 421 y 208 en el nuevo: quedaba fuera de su bloque de texto.

**Producción** (`marker-install.sh`, build 2466, la forma de la sección 9.5):
- **Memoria** (`srv_mem.py`): 54.13 GiB al cargar. MemFree no bajó de 33.2 GiB. Ningún reset de anillo.
- **Checks** (`server_check.py`, `inkling_repeat.py`, `effort_check.py`, `vision_check.py`):
  - Razonamiento, tool call, caché de prompt, historia y lectura de una imagen: correctos.
  - Velocidad: 33.3-33.4 t/s una respuesta y 46.1 t/s en total con dos a la vez.
  - Imagen de 2048x2048 en el slot 1: 8.7 s.
  - Determinismo: 5 corridas idénticas.
  - Los seis niveles de razonamiento de pi llegan a la plantilla.
- **El marcador de ese server en el texto del cliente:** 200 en los seis casos. Los 16 sin marcador dan el mismo
  estado y los mismos tokens de prompt que en la verificación.
- **Servidor entregado:** uno sin historia, cargado después de los checks. Con su marcador en un mensaje responde
  200, a 33.8 t/s.
- **La sesión del Director:** no se tocó ni se volvió a correr. El marcador que quedó en su historial es ahora
  texto. El caso equivalente medido es el turno de agente de la verificación.

**Registrado sin arreglar** (`latent-defects.md`): el marcador publicado en los caminos crudos, y el defecto vecino
que sigue.

**Defecto vecino, sin arreglar: las etiquetas de control en el texto del cliente.** El server tokeniza el prompt de
chat ya renderizado con `parse_special` en todo su texto, el de la plantilla y el del cliente por igual. Una etiqueta
de la plantilla citada en un mensaje, en un archivo leído o en el resultado de una herramienta llega al modelo como
el token de control. Apareció al revisar el marcador.
- **Medido en producción, sin usar slots** (`special_in_text.py`, `runs/special-in-text.out`):
  - Citadas en un mensaje, `<|end_message|>`, `<|message_model|>` y `<|content_thinking|>` quedan cada una en un
    token. El contador del chat da los 28 tokens de esa tokenización; como texto son 69.
  - Un resultado de herramienta con el volcado de `/props`, que trae la plantilla entera: las 26 apariciones de sus
    12 etiquetas quedan como 26 tokens de control (4036 tokens; como texto, 4345). Dentro del resultado el modelo
    recibe 7 encabezados de rol (`<|message_…|>`), 8 `<|end_message|>` y 2 `<|content_model_end_sampling|>`. El
    primero está a 2095 de sus 9753 caracteres.
- **Ocurrió en la sesión del Director** (`session_tags.py`, `runs/session-tags.out`; solo conteos): la llamada que
  trajo el marcador trajo también la plantilla, 26 apariciones de las 12 etiquetas en un resultado de herramienta.
  pi no compactó el historial después. Al contarlo (10:23), las 32 respuestas posteriores se habían generado con
  esos tokens de control en el prompt, de 72.6k a 97.8k tokens, y todas terminaron sin error. En las otras dos
  sesiones con respuestas de `inkling2mini` no hay ninguna etiqueta citada.
- **El efecto en las respuestas no se midió.** Lo que esa sesión muestre después de ese turno no separa al modelo de
  este defecto.
- **Upstream:** es su código, para todo modelo. Lo sigue el issue #28249 del mantenedor, abierto: que la plantilla
  marque qué texto viene del cliente. Sus dos primeros pasos están en master y en este árbol: el prompt renderizado
  sale en partes con la marca `is_input` (`common_chat_prompt`, `common/chat.h`), y el server las aplana en una
  cadena (`oaicompat_chat_params_parse`). El tercero, usarlas al tokenizar, es el PR #32033: un borrador del
  2026-11-10, en conflicto con master, que solo actúa con `--no-chat-input-special`. Su autor advierte que depende
  de la plantilla. Busca en el texto el marcador publicado: no reemplaza a `9a62394d4`.
- **Arreglo candidato:** tokenizar por partes, con `parse_special` solo en las de la plantilla, sobre el marcador de
  cada petición. No está diseñado ni medido. Cambia los tokens de todo prompt que cite una etiqueta, y falta medir
  que los demás prompts queden con los mismos tokens de hoy. Pide un build y su ventana de medición, con producción
  descargada: decisión del Director, pendiente.

### 9.7 KV q8_0 en el kernel volteado de la atención (rama `vk-fa-flip-q8`, sin adoptar)

**Origen** (2026-11-22): el Director señaló una publicación sobre el llama.cpp de Unsloth
(`x.com/bebuu_ai/status/2124495992062177632`) y pidió detectar el parche e implementarlo, sin descargar producción.
- **Lo que dice:** con Inkling 2 Mini, un caché KV de 8 bits ocupaba más VRAM que uno F16. El kernel de atención no
  aceptaba K/V cuantizados y llama.cpp caía en un camino más lento, con copias F16 de K y V. Unsloth lo arregló con
  K/V cuantizados en ese kernel y tres kernels fusionados. Sus cifras, en una B200: prefill +60 %, decode +39 % y el
  buffer de cómputo a 1M de 9.3 a 2.1 GB, con resultados idénticos bit a bit.
- **El parche:** el PR #399 de `unslothai/llama.cpp` (commit `7db4a7064`, K/V cuantizados en la atención en banda con
  sesgo relativo). Es solo de CUDA y no está en upstream.

**En Vulkan el defecto no existe en esa forma** (leído en el código):
- `supports_op` acepta K/V cuantizados, y un caché q8_0 ocupa menos que uno f16.
- **El análogo es la copia f16.** Con K/V q8_0, 64 consultas o más y un caché contiguo, cada atención de un ubatch
  de prompt deshace la cuantización de K y V en un buffer auxiliar (`use_dequant_kv`, `pipeline_dequant_transpose`)
  y el kernel volteado f16 lee esa copia. En las capas globales son 4096 B por celda: 128 MiB a 32k.
- **Lo que la copia no toma** iba al kernel sin voltear, que decodifica q8_0 dentro del kernel:
  - K/V con paso: los dos slots leyendo prompt en un mismo ubatch (dos streams de un caché sin llenar no son
    contiguos, `is_dense_kv_cache`), y las vistas.
  - Una copia que no cabe en un buffer: los 4 GiB de n_kv 1048576, las últimas 256 celdas del contexto nativo.
- **El decode no pasa por aquí:** con menos de 64 consultas no aplican ni la copia ni el kernel volteado.

**El parche** (`9bad70a63`, rama local `vk-fa-flip-q8`, worktree `~/ProjectHub/llama.cpp-kvq8`): el kernel volteado
deshace la cuantización q8_0 de K y V al cargar sus teselas en LDS, con el redondeo de `dequant_q8_0.comp`. Es el
mismo módulo, con el tipo de K/V como constante de especialización. Solo en UMA, donde se midió;
`GGML_VK_DISABLE_FA_FLIP_QUANT=1` vuelve al kernel sin voltear.

**Dos formas medidas.** La primera es el equivalente directo del parche de Unsloth: leer q8_0 en el lugar siempre que
el kernel volteado aplica, sin copia (`kvq8-in-place-everywhere.diff`, `kvq8-op-in-place-everywhere.out`). Da los
mismos bytes que la copia en cinco formas. En un caché contiguo de 32k celdas, con un slot:

| Consultas | La copia | En el lugar |
|---|---|---|
| 64 | 3.9-4.0 ms | 3.7 ms |
| 128 | 12.1-12.3 ms | 12.2-12.6 ms |
| 256 | 20.1-21.4 ms | 20.1-20.4 ms |
| 512 (producción) | 28.0-34.9 ms | 29.6-30.6 ms |
| 2048 | 92.8-94.3 ms | 114.1-114.6 ms |
| Memoria de GPU que agrega la op, con 512 y con 2048 | 138 y 156 MiB | 10 y 28 MiB |

- **Con 2048 consultas es 22 % más lenta.** Con 512 o menos no se separa del ruido de esta medición.
- **Leído en el código, sin medir el reparto:** la copia deshace la cuantización una vez por op y deja cada cabeza
  contigua. En el lugar, cada bloque de 64 consultas lee de nuevo los bloques de 34 bytes, con las cabezas
  intercaladas.
- **Forma elegida:** la copia conserva todos los casos que ya tomaba, y el kernel lee en el lugar solo los que la
  copia no toma. No ahorra la memoria de la copia, que es la cifra principal de la publicación: aquí cuesta 128 MiB
  por cada 32k celdas, y el objetivo en esta máquina es la velocidad.

**Verificación de la forma elegida** (`kvq8-op.sh`, `kvq8-op.out`, `fa_bits.cpp`): a nivel de op, con producción
cargada y decodificando durante las corridas, y cada paso por la compuerta. "Antes" es el mismo build con
`GGML_VK_DISABLE_FA_FLIP_QUANT=1`. Ningún reset de anillo; producción siguió arriba.
- **Tests** (`test-backend-ops` en Vulkan0 contra la CPU, con el parche y antes): FLASH_ATTN_EXT_REL 66 de 66, con
  casos nuevos en el layout del caché, de uno y de dos streams. FLASH_ATTN_EXT 5693 de 5693, con los mismos 82 casos
  sin soporte.
- **Bytes del resultado** de la atención global (24 cabezas de consulta sobre 8 de K/V, la tabla de 1025 distancias).
  K y V son, como en el server, las primeras celdas de cada stream de un caché más grande:

  | Comparación | Con el parche | Antes |
  |---|---|---|
  | Un stream, cinco formas (de 1024 celdas × 64 consultas a 32768 × 2048), contra el kernel f16 sobre los valores ya descuantizados | iguales | iguales (la copia) |
  | Dos streams (2048 × 512 y 32768 × 256), contra ese kernel f16 | iguales | distintos |
  | Dos streams: el stream 0 contra la corrida de ese stream solo | iguales | distintos |

  El control: el kernel sin voltear da otros bytes en las siete formas, y con dos streams da los de "antes".
- **Memoria:** sin cambio. Con un stream la copia sigue (de 230.2 a 368.1 MiB a 32768 × 512, con el parche y antes);
  con dos streams no la había (de 366.3 a 376.2 MiB).
- **Velocidad** a 32k celdas, en ms por op, con el camino de antes primero y último:

  | Caso | Antes | Parche | Antes | Parche | Antes |
  |---|---|---|---|---|---|
  | Dos slots, 256 consultas cada uno, caché q8_0 | 79.5 | 27.2 | 79.5 | 28.6 | 84.5 |
  | El mismo con caché f16 (control) | 25.3 | 23.4 | 25.1 | 24.9 | 25.5 |
  | Vistas con paso, 512 consultas, q8_0 | 75.2 | 21.9 | 57.7 | 30.4 | 59.1 |
  | Vistas con paso, 2048 consultas, q8_0 | 314.8 | 109.3 | 213.2 | 112.3 | 221.4 |
  | Un slot, caché q8_0, 512 consultas (la copia en las cinco) | 26.8 | 21.6 | 21.5 | 29.2 | 29.3 |
  | Un slot, caché q8_0, 2048 consultas (la copia en las cinco) | 93.9 | 87.6 | 99.8 | 91.2 | 90.1 |

  - **Dos slots:** de 2.8 a 3.1 veces más rápida, a 2-5 ms del caché f16. En la cadena de la primera forma el mismo
    caso midió 62.7-65.1 ms antes y 27.3-27.6 ms después (2.3 veces).
  - **Un slot:** las cinco corridas toman el mismo camino, así que su dispersión (de 21.5 a 29.3 ms con 512
    consultas) no es del parche. Su causa no se investigó: el slot 0 de producción decodificaba en todas.

**Lo que cambia en producción si se adopta:**
- **Un agente a la vez:** nada. El prefill toma el mismo camino, con los mismos bytes y la misma memoria, y el decode
  no pasa por este kernel.
- **Dos agentes leyendo prompt a la vez** (64 tokens o más de cada uno en un mismo ubatch): la atención global corre a
  6.8-9.4 TFLOPS en vez de 2.4-3.9. Cuánto pesa eso en el t/s del modelo no se midió.
- **El resultado de esa op deja de depender del otro slot.** Antes, con los dos slots en el ubatch, corría en otro
  kernel y daba otros bytes que el mismo stream solo. Es un mecanismo identificado de la dependencia de la sección
  9.4, pero no el que se midió allí: allí compartían ubatch tokens sueltos, que no pasan por este kernel.
- **Margen del anillo a profundidad** (extrapolación lineal desde 32k, sin medir): una atención de 512 consultas
  sobre 1M de celdas son 32 veces la de 32k. A 2.4-3.9 TFLOPS serían 1.7-2.7 s, alrededor del timeout de 2 s; a
  6.8-9.4, 0.7-1.0 s (la sección 4 estima ~1.2 s desde la medición a 131k). Vale para los dos slots leyendo prompt a
  la vez (con la tasa más baja medida, desde unas 775k celdas) y para las últimas 256 celdas del contexto, donde la
  copia no cabe en un buffer. Ninguna de las dos situaciones se ha dado: lo más profundo medido son 163k.

**Sin medir, pendiente con producción descargada** (decisión del Director):
- El A/B en el modelo con los dos slots leyendo prompt a la vez, con la base primera y última.
- Las compuertas de adopción (calidad, determinismo, historia, profundidad) y la memoria a 1M.
- La copia que no cabe en un buffer (n_kv 1048576): el camino está leído en el código. Una op de ese tamaño no se
  corrió junto a producción, por el timeout del anillo.

**Observación abierta, fuera del parche:** en el layout del caché a 32k celdas, un stream de 128 o de 256 consultas
cuesta casi el doble por consulta que uno de 2048 o que dos de 256. Pasa igual con f16 y con q8_0. Por ahí pasan el
último ubatch de cada prompt y los turnos cortos de un agente. Sin investigar. Las filas f16 no dependen del parche:
la tabla junta las diez corridas de las dos cadenas (`kvq8-op.out`, `kvq8-op-in-place-everywhere.out`).

| Consultas, caché f16 | ms por op | µs por consulta |
|---|---|---|
| 64 | 2.7-4.3 | 42-67 |
| 128 | 10.7-11.6 | 84-91 |
| 256 | 20.1-21.8 (una corrida, 13.8) | 79-85 (54) |
| 2 streams × 256 | 23.4-25.5 | 46-50 |
| 512 | 28.2-30.0 (una corrida, 21.6) | 55-59 (42) |
| 2048 | 88.7-94.3 | 43-46 |

**Queda en disco:** el worktree `llama.cpp-kvq8` (0.7 GB), con su build.

## 10. Las optimizaciones de Strata para este modelo (2026-11-22, estudio sin mediciones)

**Origen:** Strata v0.2.19 (2026-11-22, motor `strata-inkling` 0.1.65) corre Inkling 2 Mini con un backend HIP propio
y publica para esta máquina cifras sobre las del fork. El Director decidió no mover el modelo a Strata y pidió mirar
qué hizo.
- **Leído:** su código en el commit `c6f6427d9` de `Niko1221/Strata`, carpeta `engine/src/inkling2/`: `ink_serve.h`
  y `hip/ink_hip_backend.hip.cpp`, citados abajo como `serve` y `hip` con su línea. Y su ficha de Hugging Face y las
  notas de la versión (copias en `strata-study/`).
- **Contra qué:** los perfiles vigentes del fork, del build de la sección 8.1 (`runs/prof-tg-fuse2.log` y
  `runs/prof-pp-fuse2.log`). Todas sus ops están en `runs/strata-study-tg-ops.out` y `runs/strata-study-pp-ops.out`
  (`perf_top.py`).
- **Nada se midió:** producción siguió cargada, en las pruebas del Director, y el motor de Strata no se instaló.

**Resultado:**
- **Decode:** su cifra es con un draft que su propio server apaga cuando se muestrea, y producción muestrea. Sin
  draft no publica cifra para esta máquina.
- **Prefill:** su cifra no trae el largo del prompt. A igual prompt, la diferencia no está establecida.
- **Lo que hace mejor y aplica acá:** lanza de 13 a 27 kernels por visita de capa, unos 900 por token; el fork
  despacha 80 por visita, 3830 por token. Esa cola de despachos chicos es 11.6 de los 30.0 ms de GPU de un token
  (sección 10.3).

### 10.1 Lo publicado y su protocolo

| | Strata (ficha, Strix Halo 128 GB) | El fork (sección 8.1) |
|---|---|---|
| Prefill | 1466 t/s, "the highest measured on that machine", sin largo de prompt; 1654 con 64 peticiones a la vez | pp2048: 1008 t/s con ubatch 512 y 1122 con 2048 |
| Una petición | 47.3 t/s (SQ-20) y 48.0 (SQ-10), con el draft de árbol W=4 y sin razonamiento | 36.5 t/s, sin draft |

- **El draft exige decodificación greedy.** El server entra al paso especulativo solo con `greedy()`, que es
  `temperature <= 0` (`serve:66`, `serve:631-634`). Producción muestrea con temperatura 1.0 (sección 0), la misma que
  recomienda la ficha: con ese muestreo Strata decodifica sin draft.
- **Sin draft no hay cifra para esta máquina.** En las dos filas que traen las dos, el draft multiplica por 1.43
  (Olares One: 80.5 y "115+"; RX 7900 XT: "54.7+" y "about 78+"). Con esa razón, 47.3 serían ~33 t/s. Es una
  inferencia.
- **El prefill a 2K:** en las tres filas que traen además esa cifra, queda de 10 a 27 % bajo la principal (8722 de
  11306, 7645 de 10452, 3951 de 4367). Con esa razón, 1466 serían de 1070 a 1330 t/s. Otra inferencia.
- La ficha lleva un comentario HTML de su autor, sin resolver: añadir la versión y los tamaños de contexto de las
  cifras.

### 10.2 Lo que hace su backend HIP

| Técnica | En Strata | En el fork | ¿Sirve acá? |
|---|---|---|---|
| Pocos kernels por visita | Una visita SWA simple son 13: la mezcla de residuos con su norma (`hip:2498`); Q, K, V, R, G y el sumidero en un producto (`hip:2843`); el par de convoluciones de K y V (`hip:2892`); clips, norma de V, guardado de K/V y sesgo relativo (`hip:2911`); la atención en dos (`hip:2984-2990`); wo con su convolución (`hip:3033`); la mezcla del FFN con su norma (`hip:3062`); el router en dos, el producto y el top-k con sus pesos (`hip:3076-3080`); y los expertos en tres, el último con la convolución (`hip:3099-3109`). Una visita con mensajes suma 11: la mezcla de los mensajes (`hip:3600`), 3 kernels para los controladores de Q, K y V juntos (`hip:2770-2777`), 3 para el del FFN (`hip:2688-2738`), las normas aparte de las mezclas y K y V en productos propios | 80 despachos por visita (3830 en 48) | Sí: sección 10.3 |
| Replay del paso | Graba el paso de decode como grafo HIP y lo relanza (`hip:2523-2561`); el draft igual (`hip:1095-1245`) | Graba los command buffers en cada grafo | No. Descartado en qwen4exp: se midió que la GPU no espera al host, y reusar los command buffers recuperaría solo los 0.38 ms previos al primer submit (`decode-plan-2026-09-28.md`, sección 2). Sin repetir en este modelo |
| Tronco en BF16 | Las matrices 2-D grandes van en BF16 (`hip:2016`) y los expertos en su códec de 4 bits | Tronco Q8_0, expertos NVFP4 | No. El decode lee 3708 MB de pesos por token a 202 GB/s, el límite de la máquina; en BF16 el tronco serían 2577 MB en vez de 1369 |
| Kernel de expertos del prefill | En esta GPU, FP16 WMMA (`hip:3201-3207`); el int8 es solo de RDNA4 (`hip:507`) | MMQ coopmat1: 15.7-17.5 TFLOPS (Q8_0 denso, 24.2) | Sin evidencia. En este fork el coopmat f16 midió más lento que los cuantizados, 15-17 frente a 20-22 TFLOPS (`qwen4exp-cost-breakdown-2026-09-09.md`, sección 11) |
| Draft de árbol W=4 | Ramas verificadas en un batch, con un modelo de costo que decide cuántos nodos (`hip:1510-1543`). Solo greedy | DSpark lineal, sin adoptar (sección 6) | No con el muestreo de producción |

El conteo es propio, sobre el camino de un token de `forward_batch_chunk` con sus ramas fusionadas: 13 kernels en una
visita SWA simple y 24 en una con mensajes, 19 y 27 en las capas globales; 924 en las 48 visitas. No es una medición.
Cuenta un kernel por cada producto que lanza un helper (`launch_bf16_opt`, `launch_f32_linear`, `launch_trunk_linear`,
sin leer) y no cuenta las copias entre buffers.

### 10.3 Dónde está el costo del fork

**Decode** (30 grafos: 30.02 ms de GPU y 3830 despachos por token):
- **Productos con pesos Q8_0 y NVFP4:** 449 despachos, 18.39 ms. Leen 3708 MB a 202 GB/s (la máquina da ~210).
  Solos, serían 54 t/s.
- **La cola:** 3381 despachos, 11.62 ms. El perfil no nombra tensores: las ops compartidas se reparten por conteo,
  según el grafo (`src/models/inkling2.cpp` y `build_moe_ffn`).

| Grupo (veces por token) | Ops | Despachos | ms | Strata | Fundido | ms que quitaría |
|---|---|---|---|---|---|---|
| Controladores del loop (96) | RMS_NORM ×2, CONCAT, 3 productos BF16, TANH_BLEND | 672 | 2.11 | 6 kernels por visita con mensajes (el fork, 28 despachos) | 1 op por controlador: −576 | 1.0-1.3 |
| Router, pesos y escalas de los expertos (48) | producto f32, SIGMOID, ADD, ARGSORT, GET_ROWS ×3, SUM_ROWS, SCALE ×2, DIV, REPEAT ×2, MUL | 652 | 1.83 | 2 kernels | ~3 por visita: −500 | 0.9-1.1 |
| Convoluciones cortas (192) | CONCAT, CPY del estado, SSM_CONV | 576 | 1.42 | Dentro de otros kernels; el par K/V en uno | 1 op por convolución: −384 | 0.7-0.85 |
| Preparación de la atención (48) | L2_NORM_CLIP ×3, MUL ×2, norma de V, PAD, tabla del sesgo, SUB, FILL, SET_ROWS | 572 | 1.48 | 1 kernel | ~3 por visita: −430 | 0.8-0.95 |
| Proyecciones Q, K, V, R, G y sumidero (48) | 6 productos Q8_0 (están en los 449 de arriba) | 288 | 5.01 | 1 kernel en la visita SWA simple; 3 en las demás, con K y V aparte | Lo mismo: −184 | 0.3-0.4 |

- **Lo demás de la cola:** DEPTH_ATTN (120 despachos, 2.00 ms; sección 8.3), RMS_NORM_MUL (154, 0.81), la atención
  (48, 0.56), los taps de las convoluciones dinámicas (192, 0.46), la compuerta de la atención, SiTU y los
  residuales.
- **Constantes que se recalculan en cada token:** el PAD de la tabla del sesgo y el FILL de los sumideros. Son 96
  despachos y 0.25 ms del grupo de la atención. No es algo de Strata: salió al repartir el perfil.
- **Lo que vale un despacho quitado:** 1.8 µs de reloj (L2_NORM_CLIP: ~1000 despachos menos, de 33.2 a 35.3 t/s) y
  2.0-2.2 µs de GPU bajo el logger (`runs/strata-study-tg-totals.out`: 5633, 4394 y 3830 despachos en 33.90, 31.13 y
  30.02 ms). La última columna es despachos por 1.8-2.2 µs.
- **La suma:** 2074 despachos menos, de 3.7 a 4.6 ms sobre los 27.4 ms de reloj de un token: de 36.5 a 42-44 t/s. Es
  un techo aritmético. La máquina está limitada por potencia y una palanca puede recuperar menos de lo que quita
  (`qwen4exp-cost-breakdown-2026-09-09.md`, sección 28.4); el orden real lo da el A/B de cada una.

**Prefill** (un ubatch de 2048 tokens: 1806.8 ms de GPU, 3974 despachos, 1133 t/s):

| Grupo | Despachos | ms | % |
|---|---|---|---|
| Expertos NVFP4 (17.5 y 15.7 TFLOPS) | 88 | 682.1 | 37.7 |
| Productos Q8_0: tronco, 4 capas de expertos y salida (24.2 TFLOPS en Q y G) | 361 | 320.5 | 17.7 |
| FLASH_ATTN_EXT_REL | 48 | 153.1 | 8.5 |
| Productos BF16 de los controladores (shader escalar, 2.0-3.2 TFLOPS) | 288 | 94.5 | 5.2 |
| DEPTH_ATTN | 120 | 65.4 | 3.6 |
| Productos f32: la tabla del sesgo y el router | 96 | 60.0 | 3.3 |
| El resto: normas, convoluciones, copias y ops elemento a elemento | 2973 | 431.2 | 23.9 |

- **Los expertos NVFP4** siguen siendo la partida mayor (sección 7). A los 24 TFLOPS del Q8_0 denso serían ~200 ms
  menos, +13 %. El código de Strata no responde eso: su kernel en esta GPU es FP16.
- **El resto** son pasadas enteras sobre las activaciones del ubatch. Las fusiones de la tabla del decode quitan
  pasadas también acá; cuánto, no se estima por despacho: se mide.
- **Los productos BF16** en F16 ya se midieron: +1.2 % con una KLD de 0.0010, descartado (sección 6).

### 10.4 Candidatos que deja el estudio

Por costo medido en el fork, de mayor a menor. Ninguno está implementado ni medido.
1. **Los controladores del loop en un op.** Hoy son 7 despachos por controlador (TANH_BLEND ya fundió su cola).
2. **El router con su top-k, sus pesos normalizados y las escalas de los expertos.**
3. **La convolución corta con su historia y su estado**, sin el CONCAT ni la copia.
4. **La preparación de la atención.** Dentro de ella, las dos constantes por token son la parte exacta por
   construcción y la más barata.
5. **Q, R, G y el sumidero en un despacho**, y K y V con ellos donde comparten la entrada (las 20 visitas SWA sin
   mensajes). Ahorra solo despachos (los bytes leídos son los mismos) y pide un op de varias salidas: el más invasivo.

Cada uno es un op nuevo con kernels de CPU y Vulkan, como los que ya rindieron en el decode: L2_NORM_CLIP, 6 %
(sección 2), y TANH_BLEND con SSM_CONV_ADD_SCALE, 5 % entre las dos (sección 8.1). Entra con su A/B (la base primera y
última), la KLD y las compuertas de determinismo: eso exige producción descargada. DEPTH_ATTN con workgroups de 1024
hilos (sección 8.3) sigue abierto y vale 0.7 ms por token.

### 10.5 Sin leer y sin medir

- **Sin leer:** los cuerpos de los kernels fusionados de Strata (`k2_qrkv_kvstore_relpos_fused`,
  `k_moe_reduce_sconv_add`, `k4_moe_router_topk_f16`, los de los controladores, las atenciones `k3_flash_*`, el de
  expertos FP16 del prefill) y `forward_tree_verify`. De ellos se usaron el nombre, los argumentos y el lugar del
  lanzamiento.
- **Sin medir:** el decode sin draft y el prefill a 2K de Strata en esta máquina. Es lo que cerraría la comparación,
  y pide producción descargada y su pack (19.35 GB el SQ-20).
- **Hipótesis:** que un tronco BF16 no pueda ganarle al Q8_0 en decode supone que HIP tiene el mismo techo de ancho
  de banda que Vulkan en esta memoria.

## 11. LoopCD en producción (2026-11-24, `04ea3c68a`)

- **Qué es:** el contraste de la última pasada del loop contra la primera (arXiv 2610.02185): los logits finales son
  `z3 + 0.3·(z3 − z1)`, con z1 la primera pasada llevada por su propia copia de la coda (6 slots de caché más).
  También está la variante en los estados que entran a la coda (`--loop-contrast-type hidden`), sin costo extra.
- **Instalado** con `loop_contrast = 0.3` en logits (`config.ini`, `run-server.sh`), la fuerza del paper para
  generación. El Director lo prueba en uso real; no pasó por la compuerta de calidad, porque cambia la distribución a
  propósito y su efecto se juzga en las respuestas.
- **Verificado:** con la fuerza 0 los logits son los de master (KLD 0, 100 % de top igual); el contraste en logits es
  exacto contra `last + s·(last − una pasada)` en el modelo sintético (`test-llama-archs --loop-contrast`).
- **Costo, medido el 2026-11-25** en la forma de producción, con la base primero y al final de la cadena
  (`lcdhead.sh` en la carpeta de evidencia):
  - Memoria: 4.30 GiB más (58.45 contra 54.15 GiB del servidor recién cargado, `srv_mem.py`). Es la cuenta de
    `config.ini`: los 4.25 GiB de la capa global 23 a 1048576 x 2 en q8_0, más poco de las 5 capas de ventana. Los
    "11 GiB" que este punto decía antes comparaban MemAvailable de dos días y estados distintos de la máquina.
  - Decode: 5.6 ms por token, un 18 %. Una petición da 36.9-37.3 t/s sin LoopCD y 30.1-30.9 t/s con él; dos a la vez
    dan 53.5 y 44.9-45.2 t/s en total.
- **El head en un solo producto** (`c73afe5ca`): las dos predicciones pasan juntas por un producto de 2 columnas, y
  los pesos del head (2304 x 201024, 469 MiB en Q8_0) se leen una vez por token en vez de dos. El contraste sigue en
  los logits porque el softcap de 30 después del head no es lineal.
  - Una petición sube a 32.3-33.3 t/s (+7 %) y dos a la vez a 47.5 t/s (+5 %); instalado (`lcdhead-install.sh`), 33.3-33.8 t/s. Del costo de LoopCD quedan unos 3.3 ms
    por token: la copia de las 6 capas de la coda.
  - En Vulkan el producto de 2 columnas va por MMVQ, que cuantiza la entrada a Q8_1 (`ggml_vk_should_use_mmvq`:
    "MMVQ is generally good for batches"). Con una columna y pesos Q8_0 en RDNA no la cuantiza. Es la precisión que
    el head ya tenía con dos slots en un ubatch y en los modos de 3 tokens. Contra la base, la secuencia de 2k-163k da
    un KL medio de 1-2e-5 por zona, dos órdenes bajo las referencias de redondeo.
  - Compuertas: los modos p y d sin contraste son idénticos a la base, y los modos pL y dL con contraste dan KLD 0.
    En el modo d, `llama-perplexity` da 99.976 % de top igual también para la base contra sí misma (dos veces,
    `lcdhead-d.sh`). Es un empate en las log-probs cuantizadas que guarda el archivo base, no una diferencia del
    build; el FAIL de `lcdhead.sh` en ese modo es su criterio de 100 %. También pasan `repeat_probe` (5 corridas
    idénticas), la secuencia 2k-163k contra sí misma y `server_check`, `inkling_repeat` y `vision_check`.
  - No adoptado, sin medir: juntar también las dos pasadas por la coda. Las proyecciones de la atención se leerían
    una vez, pero los expertos solo se comparten cuando las dos pasadas eligen los mismos, y la atención sigue
    separada porque cada pasada tiene su caché. Lo estimo en 1 ms por token como mucho, y exige que cada capa sirva a
    dos cachés a la vez.

## 12. DFlash2 de incoai, no adoptado (2026-11-25)

- **El drafter:** `incoai/Inkling-2-Mini-DFlash2` (5 capas, bloques de 8), en el GGUF de `ufrik`, reescrito por
  `dflash2/fix_gguf.py`: el vocabulario del objetivo (200058, el GGUF traía el relleno de 201024), los índices de capa
  de la referencia de SGLang ([5, 23, 41, 48], los de DSpark; ufrik escribió [6, 24, 42, 48]) y el softcap del selector
  después de su corrección (`dflash.selector_softcap_after_correction`).
- **La rama** `inkling-dflash2` (worktree `llama.cpp-loopcd`): el soporte de drafters de upstream para Inkling
  (`33aeb17f8`, PR #32218), las features indexadas por visita y no por slot (con LoopCD hay 6 slots más que visitas) y
  los dos cambios del grafo de DFlash2. Sin merge.
- **Medido** (`dflash2-check.sh`, NP=1, 32k, LoopCD 0.3, verificación por muestreo; cortado por el Director antes de
  los controles):

| | Greedy | Muestreo de producción | ms por paso (muestreo) |
|---|---|---|---|
| Sin draft | 29.7 t/s | 28.3 t/s | 35 |
| n-max 7 | 29.3 t/s | 19.3 t/s | 150 |
| n-max 3 | 33.8 t/s | 21.1 t/s | 100 |

- **Por qué pierde:** verificar 8 tokens cuesta 100-117 ms de GPU contra 34 ms de un token (en el MoE cada token del
  lote lee sus propios expertos), y hay ~50 ms por paso fuera de la GPU, sin identificar. Acepta 1.9 tokens por paso
  con muestreo, la mitad de lo que reporta incoai en una H200; los controles que separaban LoopCD y la conversión no
  llegaron a correr.

## 13. Las fusiones del decode (2026-11-25, `571dd49fa`..`0392fef19`)

Los candidatos de la sección 10.4, más la atención en profundidad con workgroups anchos (sección 8.3), implementados
en ramas propias y en una de integración (`inkling-fuse`), cada uno con su arnés y su informe en
`llama.cpp-evidence/inkling2mini/fusions/<grupo>/REPORT.md`.

| Grupo | Commits | Mecanismo | Exactitud |
|---|---|---|---|
| Atención en profundidad | `571dd49fa` | Un pipeline de 768 hilos para lotes de hasta 40 tokens | Reagrupa sumas (nmse 1e-14) |
| Controladores del loop | `4a4a0b4fc`, `b8e02a7a3`, `879657923` | Ops `RMS_NORM_CONCAT_MUL_MAT` y `TANH_BLEND_MUL_MAT`, lotes de hasta 32 tokens; en CPU por productos BF16 como mul_mat | Vulkan: activaciones en F32 (nmse 1e-14 hasta 8 tokens, 5e-7 de 9 a 32) |
| Router | `ef24daaa6`, `f1b2cbb23`, `590bade97` | `topk_moe` de upstream con la suma de Inkling y las escalas de los expertos | Bit a bit, empates incluidos |
| Convoluciones cortas | `0377d8d99`, `2bd209726` | Fusión `SSM_CONV_STATE`: lee historia y entrada en su lugar y escribe el estado | Bit a bit, estado incluido |
| Preparación de la atención | `2a8017c88`..`0392fef19` (6) | Tabla y sumideros al cargar; `L2_NORM_CLIP` con escala y factor; normas con la rotación y el guardado de K/V; la tabla del sesgo dentro del clip de r en decode | Bit a bit salvo la tabla (suma en F32: más cerca de la CPU) |

**Medido** (`fuse3-all.sh`: la base primera y última; `fuse3-faoff.sh`; `fuse3-install.sh`):

| Etapa | Despachos por token | GPU por token | tg128 (ub 512) | pp2048 ub 512 / 2048 |
|---|---|---|---|---|
| Base (master `04ea3c68a`) | 4050 | 30.7-31.8 ms | 34.8-35.9 t/s | 979-985 / 1086-1105 |
| + atención en profundidad | 4050 | 30.4 | 37.0 | 993 / 1102 |
| + controladores | 3474 | 29.4 | 37.9 | 991 / 1099 |
| + router | 2918 | 27.7 | 38.6 | 995 / 1104 |
| + convoluciones | 2526 | 26.6 | 39.3 | 962 / 1128 |
| + preparación sin la tabla | 2070 | 26.8 | 40.3 | 978 / 1130 |
| + la tabla | 1974 | 25.9 | 40.8 | 989 / 1134 |

- **En producción** (1M × 2 slots, KV q8_0, LoopCD): una petición de 28.3 a 31.5 t/s (+11 %); dos a la vez de 38.1 a
  44.2 t/s. Instalado: 31.1-31.5 t/s, 41.5 t/s con dos.
- **Compuertas:** `test-backend-ops` 2494/2494 y `test-llama-archs` 158/158 en Vulkan; determinismo (`repeat_probe`,
  cinco lecturas idénticas, y `depth_equiv` de 2k a 163k contra la base: KLD media de 0.002 a 0.004, la de dos
  corridas de un build); la forma de producción, base y rama: razonamiento, tool call, caché del prompt, imagen de
  2048 × 2048, historia, peticiones repetidas idénticas y los dos slots a la vez.
- **Calidad:** con ubatch de 2048 (modos p, P y pL) la rama da la KLD de la base sin fusiones. Con ubatch de 3 (d, M y
  dL) la referencia de `GGML_VK_DISABLE_FUSION=1` mueve casi nada en este modelo (0.0005 en d) y la rama quedaba a
  3.5 veces; ya la primera etapa (una reagrupación de sumas de 1e-14) da 0.0018, y las siguientes quedan en esa meseta:
  con lotes chicos el modelo amplifica cualquier redondeo hasta ahí. Contra la referencia sustituta de la sección 3,
  la base con `-fa off`, pasa en los tres:

| Modo | KLD rama | KLD `-fa off` | Top igual rama / `-fa off` |
|---|---|---|---|
| d (ubatch 3, 4096) | 0.0019 | 0.0020 | 98.1 / 97.8 % |
| M (ubatch 3, 16k) | 0.0028 | 0.0033 | 98.0 / 97.9 % |
| dL (d con LoopCD) | 0.0032 | 0.0034 | 97.4 / 97.4 % |

- **Queda:** 4 despachos por visita en la preparación de la atención (3 pide juntar cadenas independientes); las 18
  convoluciones dinámicas por token con sus taps (un op nuevo, ~−288 despachos); en CPU, los controladores fundidos con
  lotes de 9 a 32 tokens quedan hasta un 30 % bajo el camino sin fundir (el mul_mat por lotes usa el GEMM de
  llamafile). Las notas sobre upstream que encontraron las ramas están en `latent-defects.md`.
