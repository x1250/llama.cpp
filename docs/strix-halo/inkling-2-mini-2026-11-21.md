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

**Calidad:** KLD de 0.0029 contra upstream con el mismo GGUF, con ubatches de 2048. La referencia de redondeo da
0.0031 (sección 3).

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
  (sección 3). Dos coincidencias con una potencia de 2 no alcanzan para concluir.
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
- Los parches de las secciones 2 y 8.1 siguen en master, pero ningún modelo de llama-server los usa (CLAUDE.md).
