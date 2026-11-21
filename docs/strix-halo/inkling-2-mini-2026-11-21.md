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

## 7. Abierto

- **El prefill a profundidad:** la atención global corre a 5.7 TFLOPS en el coopmat1 y es dos tercios del ubatch a 131k.
- **Los expertos NVFP4:** el MMQ coopmat1 corre a 16-18 TFLOPS (Q8_0 denso, 24); es el 35 % del ubatch a contexto corto.
- **Ubatch 2048 a 1M:** dividir las atenciones globales en bloques de consultas acotados por trabajo, como hacía el
  fork en qwen4exp.
- **La cola de despachos chicos del decode:** unos 5600 por token. Los controladores de mensaje del loop son ~1150, y la
  convolución con su residual y su escala ~960.
- **DSpark con draft adaptativo** (apagado cuando acepta poco).
