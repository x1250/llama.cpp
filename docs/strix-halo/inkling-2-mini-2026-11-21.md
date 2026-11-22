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
- **HIP:** no se evaluó. Los ops nuevos son de CPU y Vulkan; HIP exigiría portar cuatro.

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
  tensor ya calculado (su residuo): así se funden 156 de las 192 convoluciones de un paso. Las otras 36 son las
  dinámicas de las capas globales.
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
  sin arreglo en master; los PR #28249 y #29291 seguían abiertos.

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

**Registrado sin arreglar** (`latent-defects.md`): el marcador publicado en los caminos crudos, y un defecto vecino
que apareció al revisar este: una etiqueta de control de la plantilla citada en un mensaje llega al modelo como el
token de control (`special_in_text.py`, `runs/special-in-text.out`).
