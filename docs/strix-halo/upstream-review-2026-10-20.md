# Cruce con upstream: Strata y llama.cpp (2026-10-20)

Revisión de solo lectura, pedida por el Director con la GPU ocupada: no se compiló, cargó ni aplicó nada. Los merges se
simularon en memoria con `git merge-tree`, que no toca los árboles de trabajo. Cada afirmación cita el código en la
revisión indicada; lo que necesita la GPU está en la sección 3.

| | Nuestro | Upstream | Distancia |
|---|---|---|---|
| Strata (producción) | fork `c195c0c4` = v0.1.40.1 (`82f46a8c`) + 16 commits | `e88ee39d` (v0.2.0-beta, motor 0.1.45.3, 2026-10-19) | 622 commits first-parent en 11 releases; 14 archivos en conflicto |
| llama.cpp (respaldo e investigación) | master `8448355ce` | `dd002b91f` (2026-10-20) | 739 commits nuevos desde la revisión del 2026-09-17, ninguno en el fork; 31 archivos en conflicto (con la base real, sección 2.1) |

## 0. Resultado

1. **Un defecto de upstream llama.cpp afecta a producción hoy: el preprocesado de imágenes.** `strata-vision` enlaza el
   mtmd de nuestro fork (`strata/tools/vision/CMakeLists.txt:41-44`, `LLAMA_DIR` en `build-strata.sh:25,79`). Para
   `qwen3vl_merger`, ese mtmd ajusta la imagen dentro del tamaño de `smart_resize` y rellena el resto con negro
   (`PAD_CEIL`, `tools/mtmd/clip-model.h:67`; el caso de `clip.cpp:1651-1668` no lo cambia). Además redondea con
   `std::round` (`mtmd-image.cpp:130`). La referencia oficial (`preprocessor_config.json` del repo de Qwen:
   `Qwen2VLImageProcessorFast`) redimensiona directo, sin relleno, y con el `round()` de Python. Resultado: bandas
   negras en toda imagen cuyos lados no son múltiplos de 32 px, y una fila o columna de tokens de más cuando el
   cociente termina en .5.
   - El servidor de Strata no redimensiona: pasa los bytes (o un PNG convertido) a `strata-vision`, que devuelve
     también la cuenta de tokens (`serve/server.py:1778-1810`; `strata_vision.cpp:183-192`).
   - Calculado con el algoritmo de cada lado:

     | Imagen | Fork | Referencia |
     |---|---|---|
     | 1920×1080 | 1920×1080 dentro de 1920×1088, 8 px negros | estira a 1920×1088 |
     | 1366×768 | 10 px negros | estira |
     | 800×600 | 8 px negros | estira |
     | 1170×2532 (móvil) | 15 px negros | estira |
     | 2048×2048, 1280×800, 2560×1440 | igual que la referencia | — |

   - Upstream `f711c8f30` (#30736, de hoy) lo corrige. Su autor mide en Qwen3.5 el coseno del peor token contra
     transformers: 0.04-0.30 antes, 0.98-0.999 después; 640×488, con 5 px de banda por lado, daba 0.23. Son 5 archivos
     y aplica limpio salvo un test.
   - **Es lo único que recomiendo tomar ya.** Las compuertas de imagen esperan a la GPU. Por ser un commit de horas,
     conviene mirar si upstream le agregó un arreglo antes de tomarlo.
   - Otra diferencia, aparte y sin propuesta: nuestro tope es de 4096 tokens por imagen (`max_tokens` del codificador)
     y el de la referencia, de 16384 (`longest_edge` 16777216 px). Una foto de 3024×4032 da 4015 tokens aquí y 11844
     en la referencia.
2. **Strata upstream no arregla ningún fallo observado en producción, pero sí cuatro defectos latentes de nuestro
   HEAD** (sección 1.3). También trae mejoras de velocidad; dos se pueden probar sin merge (sección 1.4):
   - las lecturas cortas del agente, solo con variables de entorno;
   - el top-k del decode a profundidad, un cherry-pick de 6 líneas.
3. **No conviene un merge completo de Strata ahora.**
   - Upstream reescribió la lectura de prompts con dos slots y la cesión del servidor con otra política: cede solo a una
     petición de a lo sumo la mitad de largo, y en una GPU lee de a un chunk. Tres commits del fork se reescriben.
   - Upstream sigue con el defecto de imágenes con dos slots que arregla nuestro `bf7f443`.
   - Trae defaults que habría que fijar a mano: un watchdog que recarga el motor fuera de la compuerta de memoria, y el
     cierre del pensamiento al 75 % de `max_tokens`.
   - Lo que vale de upstream se puede traer por cherry-pick (sección 3).
4. **llama.cpp: un merge caro y con peligros silenciosos; no es urgente porque producción no corre llama.cpp.**
   - Lo de valor: el matmul int8 por coopmat1 (PR #27952, medido en Strix Halo por su autor) y cuatro arreglos de
     MTP/servidor que el fork también necesita.
   - Dos errores de la revisión del 09-17 quedan corregidos aquí: el bloque lazy-off vuelve **en silencio** con un
     `git merge` normal (no como conflicto), y el PR #28244 está cerrado sin merge desde el 2026-09-03.

## 1. Strata

### 1.1 Qué trajo upstream

- **0.2.0 es sobre todo producto:** app, terminal, instaladores y motores empaquetados. El paquete de Strix Halo (Linux,
  ROCm 7.14.1) es upstream sin nuestros 16 commits. Los mantenedores nunca lo corrieron en un Strix Halo y no trae
  tablas hipBLASLt (planeadas para 0.2.1). Nada de eso sirve para producción.
- **El tier de archivo de Linux** (prefetch de I/O, zero-copy, chequeo de RAM con cgroup, plan de chunk) es para GPUs
  que no guardan todos los expertos. Con los 24576 en VRAM solo actúa el prefetch, y solo durante el llenado de arranque
  (sección 1.5).
- Defaults de gfx12, Intel y NVIDIA: no aplican a gfx1151.
- Arreglos de motor y servidor: sección 1.3.

### 1.2 La configuración de producción frente a upstream

Todos los argumentos y variables de `strix-halo/strata/qwen38flash.json` siguen existiendo con el mismo parseo, y el
esquema de `render-config.py` se lee igual. Diferencias:

| Item | Upstream | Consecuencia |
|---|---|---|
| `STRATA_FILE_RELEASE` | sigue solo en Windows (`up:src/core/expert_source.cpp:3027-3050`) | conservar `310bfe4`; el merge simulado lo conserva |
| `STRATA_BATCH_DECODE_SHARE` | reparte entre chunks (`up:src/program/generate.cpp:13052-13059`) | nuestro `d810a71` reparte entre capas: conservarlo |
| `STRATA_HC_Q8=1` | con una GPU ya no guarda la copia BF16 de las proyecciones hc; el prompt las descuantiza de Q8_0 en cada chunk (`b60aab1b`; `up:src/core/native_dense.cpp:507-535`) | otra numérica del prompt: compuerta de calidad. Nuestro GGUF tiene las hc en Q8_0 (cabecera leída: `hc_*_{down,up}` Q8_0 × 48), así que la extensión IQ4_NL de `2485459` no se usa en producción. Un GGUF con hc en IQ4_NL y este código juntos leerían IQ4_NL como Q8_0 |
| `STRATA_HIP_WMMA=1` | si un lanzamiento falla, pasa al kernel FP32 sin salir (`up:src/kernels/cuda/qsa_prompt_attn.cu:2257-2262`) | las compuertas deben buscar esa línea en el log |
| `-DSTRATA_W_LB=8` | se respeta; con clang 23 en gfx1151 el default nuevo (`edf61f18`) es el mismo tope para nuestros formatos | sin cambio |
| `STRATA_STAGE_PIN` (0.1.41: el fijado por default corrompía IQ3_S tras un prompt largo) | opt-in | **producción no está expuesta:** la base usa búferes paginables (`82f46a8c:src/core/expert_source.cpp:831`) y solo en el llenado bloqueante de arranque |

### 1.3 Defectos de nuestro HEAD que upstream corrige

| Upstream | Defecto en HEAD (verificado en el código) | Peso aquí | Compuertas |
|---|---|---|---|
| `7a0e3023` + `e372d4ff` (#606, #1448) | tres productores de q8_1 escriben `make_half2(d, sum)` sin acotar `d` (`fused_gr.cu:202`, `verify_kernels.cu:187`, `iq_kernels.cu:2330`). Un bloque con amax > ~8.3M da una escala inf/NaN, la fila de logits termina en NaN y el muestreador devuelve el id 0, `!`. La tabla de gfx1151 enciende `STRATA_QFUSE` (`src/core/arch_defaults.cpp:36`), que usa dos de ellos | bajo en probabilidad, alto si ocurre: el texto degenera y queda en el caché. Bits iguales en bloques normales | determinismo, historia |
| `b97a36a4` + `7bab660b` (#1712, #1996) | el prefill sube los registros de pasos QSA y los ids de tokens con `cudaMemcpyAsync` desde un `std::vector` paginable (`prefill.cpp:1805,1875,2674-2675,2699`). En gfx1100, el primer prompt de texto tras uno con imagen leyó basura en 5 de 5 | sin medir en gfx1151 | texto → imagen → texto ×5 junto a un cliente Vulkan, `batch_test` |
| `a6830008` + `62185aab` + `0d613031` (#2026, #537) | un `<tool_call>`, `</tool_call>`, `<tool_response>` o `</think>` citado dentro de un mensaje se tokeniza como token especial (`serve/frontend.py:163`) | pi lee y escribe código de agentes: un archivo que contiene esas etiquetas corta la llamada o el pensamiento | pi en vivo, suite del servidor |
| `dd08a7d0` (#2518) | con dos chats en dos slots y dos turnos que llegan juntos, `pick_slot` puede mandar un turno al slot del otro (`serve/server.py:1368-1381`); ese turno relee el prompt entero | velocidad, no exactitud | `batch_test`, `parking_test` |
| `ef6c18a1` + `c5fa1b58` (#1059, #1994) | un `ERR` del motor retiene las líneas de control hasta 300 s (`serve/server.py:1316-1323`) | robustez | suite del servidor |
| `2bf1d07c` (#1615) | el texto del error de contexto excedido no coincide con los patrones de pi (`serve/server.py:2920-2923`), así que pi no compacta | bajo | pi en vivo |

Además hay menores con el mismo veredicto (tomar): #1620 (cancelar conserva la conversación), #1666, #1700 (borde
del contexto), `940011da` y `53f72571` (caché de conversaciones) y `f332a855` (guarda de logits no finitos).

### 1.4 Velocidad

| Qué | Cómo se obtiene | Medición de upstream (hardware) | Numérica |
|---|---|---|---|
| **Lecturas cortas.** El log del lanzador (`~/.strix-engine.log`, 2026-10-07..10) da 274 t/s en 143 lecturas de 128-1023 tokens, 822 t/s en 48 de 1024-4095 y 1274 t/s en 87 de ≥ 4096. Bajo 1024 tokens los expertos no van por el camino de streaming (`STRATA_PREFILL_STREAM_MIN`, default 1024, `prefill.cpp:112-116`); bajo `PF_SWITCH_MIN_T` (4096 en producción) no corren las conmutaciones rápidas | solo variables, ya existentes en nuestro build: `STRATA_PREFILL_STREAM_MIN=128`; `STRATA_PF_SWITCH_MIN_T=2048` | gfx1151, `--no-prefill-borrow`: lecturas de 441/855 tokens de 3.1/4.0 s a 0.9/1.2 s, un turno de agente de 3.4 a 2.1 s (#1391, KL 0.016); +15-20 % en un chunk de 4K (#2021, #2208) | cambia: compuerta de calidad |
| **Top-k del decode a profundidad.** El decode llama al top-k sin el conteo de bloques activos, así que su alcance es la capacidad: `max_cells/4+2` = 65538 bloques a 262144 (`src/core/verify.cpp:424`), más que los 33792 del kernel de registros; en HIP cae al radix de referencia de 4 pasadas (`qsa_select.cu:1244-1250`) | cherry-pick `82922f55` (6 líneas; más `6ef23b89` y `b6ccf873`, también exactos) | gfx1151: 2.24 → 0.77 ms por llamada a 131K bloques; decode a 512K de 19-22 a 24-26 t/s | exacto (los mismos ids) |
| `STRATA_PF_F16=1`: expertos del prompt con activaciones f16, nuestros formatos (IQ3_S/IQ4_XS gate/up, IQ4_NL down) | merge (#2020) | gfx1151, UD-IQ4_XS: 4K +13.5 %, 32K +9.9 % | cambia (KL del primer token 0.0104 contra 0.0076 del camino int8) |
| `STRATA_SPEC_GUMBEL=1`: aceptación del draft con muestreo | merge | un Strix Halo a temperatura 1.0: aceptación de 52.8 a 59.9 %, +9 % de salida | otra muestra de la misma distribución |
| `STRATA_GDN_CHUNKED=1` | merge | gfx1151: +2.1 % de prompt | cambia (KL 0.0039) |
| Commit por lotes (`6f3e3810`, default) | merge | R9700: 2.35 → 0.65 ms por ventana. En gfx1151 solo corre la mitad QSA: la tabla pone `GDN_SPLIT=1` (`up:src/kernels/cuda/verify_kernels.cu:1044-1054`); el PR abierto #3226 lo extiende | exacto |

- **Lo que pesa contra el top-k:** a poca profundidad no se ve. Producción a 262144 da 78.3 t/s en la corta (sección
  10.12 del documento de Strata), igual que a 65536 (77.7, sección 10.2).
- **Dónde estaría la ganancia:** a profundidad. Nuestros 46.2 t/s a 125K se midieron con ctx 131072, que todavía usa el
  kernel de registros.
- **Cómo medirlo sin tocar código:** decode a la misma profundidad con 131072 y con 262144 por slot.

### 1.5 Si se hace el merge

**Defaults nuevos que se encienden aquí, y cómo conservar lo de hoy:**

| Default | Efecto aquí | Recomendación |
|---|---|---|
| `answer_room` (`d0e990ba`) | pi siempre envía `max_tokens` 16384; upstream cierra el pensamiento a los 12288 para dejar un cuarto a la respuesta | **decisión del Director**; `"answer_room": false` lo apaga |
| `STRATA_ENGINE_STALL_S` = 90 s | psutil está instalado: un motor sin salida, CPU ni I/O durante 90 s se mata y se vuelve a cargar con la petición siguiente, dentro del servidor y sin pasar por la compuerta de memoria | `=0` |
| `STRATA_IO_PREFETCH` (on con el GGUF leído en su lugar, `up:include/strata/core/expert_source.hpp:998`; `up:src/program/generate.cpp:5520`) | 8 hilos de `pread` en el llenado de arranque. Nuestro `POSIX_FADV_DONTNEED` debería descartar esas páginas (mismo inode), pero no está medido | `=0` (`STRATA_COLD_OVERLAP=0` también; con todos los expertos en VRAM es inerte) |
| `fit_max_tokens` | un 400 en el borde del contexto pasa a ser una respuesta acortada | dejar el de upstream, o `false` |
| `apply_saved_tuning` | superpone presets de `~/.local/state/strata` | `STRATA_NO_PRESET=1` (una sola fuente de configuración) |
| Opciones de CMake `STRATA_MMQ_KQUANTS`, `STRATA_Q6K_EXPERTS`, `STRATA_MMQ_TQ_OVERLAY` pasan a ON (`up:CMakeLists.txt:79-84`); `STRATA_PORTABLE` del codificador pasa a ON (`up:tools/vision/CMakeLists.txt:26`) | un build nuevo compila más y el codificador queda en AVX2 sin OpenMP | fijar las tres primeras en OFF y `-DSTRATA_PORTABLE=OFF` en `build-strata.sh` |

**Lo que no cambia:**
- El plan de chunk por RAM no puede pisar `--prefill 8192`: no corre con un chunk explícito (`up:src/program/generate.cpp:7473`).
- La PLE sigue por `O_DIRECT`.
- La guardia de memoria de las conversaciones solo se vuelve más estricta.

**Los 16 commits del fork:**

| Veredicto | Commits |
|---|---|
| Sobran: upstream tiene el mismo cambio | `e3b3edb` (= `0a22c467`), `4eae8e5` (= `3b499f17`, luego `edf61f18`) |
| Se reaplican tal cual | `310bfe4`, `71d9e69` (la forma de las tools; upstream sigue sin ella, `up:serve/frontend.py:544`), `57c6375`, `a2bcaf8`, `5c812ba`, `d8a6ce9` |
| Se adaptan | `d3669c8` y `91ab622` (triviales); `bf7f443` (moderado: upstream manda imágenes a los slots con `BGENI`, `up:serve/server.py:1869`, y el motor fusionado lo rechaza); `ee7e7c5` (casi todo lo cubre upstream #1869; faltan las sumas de drafts y de lecturas de expertos) |
| Se reescriben | `d810a71`, `0f43590`, `c195c0c` sobre el `read_part` nuevo (437 líneas en conflicto), o se adopta la política de upstream: **decisión del Director** |
| `2485459` | solo si vuelve un GGUF con hc en IQ4_NL (ver 1.2) |

### 1.6 Los límites de la sección 10.10

Upstream no levanta ninguno por default. Nuestra cesión cede antes que la suya, y lo de imágenes con dos slots sigue
roto upstream (una tabla M-RoPE por dispositivo y `BGENI` aceptado). Para el rendimiento con dos slots solo hay
opt-ins medidos en layer split con varias GPU.

### 1.7 Lo publicado de Strix Halo

| Fuente | Configuración | Prompt t/s | Decode t/s |
|---|---|---|---|
| Mantenedores (`up:docs/STRIX_HALO.md:147-157`) | UD-IQ4_XS, prefill 16384, MTP, lookup 3 | 8K 1293, 128K 1320 | 53.8, 51.4 |
| Comunidad, EVO-X2, 0.1.41 | UD-IQ4_XS, prefill 16384 con préstamo | 32K 1165, 128K 1109 | 49.6, 47.5 |
| Comunidad, laptop de 70 W | `STREAM_MIN=128`, `--spec 5 --mtp-window 8192` | 8K 969, 128K 1031 | 59.2, 42.0 |
| Producción | UD-small NL3S, chunks de 8192, 262144 × 2 | 39.5K 1324; 249.5K 1247 | corto 78, agente ~67, 249.5K 43.1 |

En lectura de prompt largo estamos en el techo publicado; en decode, por encima. La comparación no es exacta: cambian
los prompts y los GGUF.

Un dato de la comunidad sin probar aquí: `--spec 5` dio 54.3-54.7 t/s contra 50.3-50.4 con `--spec 4` (greedy, a 32K)
y `--spec 6` fue peor.

## 2. llama.cpp

### 2.1 El merge: mecánica y peligros

- **La base que calcula git es falsa.** El fork lleva 129 copias reescritas de los commits de upstream
  `5eec3ad01..f3f1a8f27` (la reescritura del 2026-09-09; el árbol de `6d640e6a0` es idéntico al de `f3f1a8f27`), así que
  `git merge` toma `5eec3ad01` como base.
  - Con esa base salen 170 archivos en conflicto, casi todos espurios.
  - Lo grave: el bloque de upstream que resuelve lazy AUTO → OFF en iGPU (`#28160`, el freeze #7) **vuelve sin
    conflicto**. Árbol simulado `e68b9006c`, `src/llama-model.cpp:1604-1610`.
  - La revisión del 09-17 decía que volvería como conflicto: es falso.
- **Procedimiento:**
  1. `git merge -s ours f3f1a8f27`: registra la ascendencia sin cambiar el árbol, porque master ya contiene esos
     cambios.
  2. Después, el merge de upstream, con base `f3f1a8f27`: 31 archivos en conflicto, sin el bloque lazy.
  3. Tras cualquier merge, `grep -n '#28160' src/llama-model.cpp` tiene que salir vacío.
- **Choque silencioso en la FA sparse.**
  - El fork marca su pipeline sparse con el bit 16 del spec constant `Flags` (`FA_PIPELINE_FLAG_SPARSE 16u`,
    `ggml-vulkan.cpp:1556, 4331`).
  - Upstream usa el mismo bit para la suya (`USE_SPARSE = (Flags & 16)`, `flash_attn_base.glsl:28`). Su indexación
    entra sin marcadores de conflicto (árbol `4a86b3865`, `flash_attn_base.glsl:238, 251`).
  - Si se resuelven solo los marcadores, el pipeline sparse del fork aplica también la indexación de upstream sobre su
    propia lista: atención incorrecta sin error de compilación.
  - Resolución: tomar enteros `flash_attn_cm1.comp` y `flash_attn_base.glsl` del fork, y mover el bit de upstream en
    `flash_attn.comp`.
- **El scheduler.** El merge conserva lo que el fork necesita, con tres puntos a cuidar:
  - La regla split-on-inputs sigue ausente en upstream y el merge la conserva sin conflicto.
  - `ggml_backend_sched_reset` de upstream sigue reusando el layout, y el merge conserva el descarte del fork.
  - Nuevo: `972b5fcc1` registra cada reasignación del gallocr (`up:src/llama-context.cpp:2712-2729`). Con el re-layout
    deliberado del fork dispararía en cada grafo no reusado: hay que condicionarlo.
  - En `ggml-backend.cpp`, la sincronización por split del fork (`fb27b714f`) choca con `ggml_backend_sched_copy_input`.
    Si se toma upstream, se pierde en silencio, y solo cuesta velocidad.
- **Reestructuras previas.** `91f6a6cf3` (el tipo de A como spec constant) es ahora la base de todo el matmul de
  upstream (`70c4e1582`, `966baae76`, `7ec420bc0`, `bbd488c42`), así que deja de ser opcional. `f172be756` partió
  `ggml-vulkan.cpp` en seis archivos:
  - el conflicto de ese archivo son 53 hunks;
  - hay ~150 líneas del fork que reubicar;
  - la excepción de lectura < 64 KiB de `c668f5d57` se pierde en silencio si se toma el `buffers.cpp` de upstream.
- **`llama-memory-hybrid-idx`** no se resuelve con "ours": `glm5-next.cpp` de upstream usa su API k-pool. Hace falta la
  unión, con `indexer_kpool = 0` para qwen4exp, y ese es el lugar natural para `311d4211b`.
- **Peligros conocidos que siguen bien:**
  - `14a9d09f7`: ningún script ni el lanzador usa los flags eliminados (`run-server.sh:454` usa `-lm mmap`).
  - PR #28118: sigue abierto y ya no aplica limpio (8 sitios a mano).
  - **PR #28244: cerrado sin merge el 2026-09-03.** `CLAUDE.md` todavía lo lista como pendiente.
  - PR #27952: entró como `70c4e1582`.

### 2.2 Verdictos

| Commit | Qué | Veredicto | Valor aquí |
|---|---|---|---|
| `f711c8f30` | resize de Qwen-VL como transformers | **tomar ya**; recompilar `strata-vision` y llama-server | sección 0, punto 1; afecta a los dos motores |
| `e467e5b5e` | el KV plano respeta `PARTIAL_ONLY` | tomar (el hunk de `llama-kv-cache.cpp`) | cada checkpoint copia hoy el KV entero del draft MTP (`llama-kv-cache.cpp:2105`, los flags se ignoran): en `~/dbg/depth/srvlog-v33lv.log` crece 2072 B/token (186.5 MiB a 37.4k tokens, 190.6 a 39.5k; sin MTP, 112.6 fijos). A 262k serían cientos de MiB por checkpoint, hasta 8 checkpoints por slot (estimado) |
| `4453b535f` | filas NextN sin máscara en orden de lote | tomar | el fork las extrae en orden de ubatch y el hook MTP las lee por índice de lote (`common/speculative.cpp:1617, 1680`); con NP=2 y dos prompts a la vez el draft recibe estados equivocados. Afecta la aceptación, no la salida |
| `d280808f5` | no aceptar drafts después de un EOG | tomar, más el mismo corte en la variante `dists` del fork | `common/sampling.cpp:704` corta solo por desacuerdo |
| `23b0202a1`, `8e1642198` | slot ocupado pedido por `id_slot`; corte de `keep_first` dentro de un chunk de imagen | tomar | 5 líneas y 1 línea |
| `1a3011cc0`, `436f6f89e` | re-reserva al cambiar los flags NextN; un solo `get_rows` de estados | tomar con la sonda de carga a 135168 | el buffer de cómputo de 3.9 GiB puede cambiar |
| `806eee984`, `6c59c4007`, `526247161`, `5e4878e97`, `ff5888f99`, `411d54851`, `83dd71f86`, `47f5a64ae`, `05f2dcfdb`, `5b335f413` | arreglos de Vulkan y ggml | tomar | todos los defectos están en master; ninguno se alcanza con las formas de qwen4exp salvo `05f2dcfdb` (alineación en im2col, el patch embed del ViT) |
| `70c4e1582` (PR #27952) | matmul int8 por coopmat1 en RDNA3/4 | **ventana propia**, después de `91f6a6cf3` | su autor midió en Strix Halo MUL_MAT IQ4_NL 1.21×, Q8_0 1.27× y MUL_MAT_ID IQ4_NL 1.32×. El gate+up IQ3_S, el nodo más caro, no está cubierto. Estimado: −5 a −9 % del tiempo de prefill, decode igual. Cambia la numérica y falta portar el epílogo de escala (`cad680f62`) |
| `1fb7ef3e3` | draft probabilístico con rechazo | ventana de medición | producción muestrea a 0.7: upstream mide +11-12 % de aceptación en draft simple, sin cifras de MTP. Unificar con la variante `dists` del fork |
| `966baae76` | duplicados de `mul_mat_id` en el prepass | entra forzado con `91f6a6cf3` | se espera bit-idéntico; menos workgroups vacíos |
| `0bc845d35` | reuso de descriptor sets | tomar y medir el decode | solo host |
| `7f51953d7`, `781dbc5ac` | el draft MTP con imágenes | adaptar después de batch_ext | hoy la caché del draft tiene huecos en las posiciones de imagen |
| `c061df198` | MTP de qwen4exp en upstream | saltar el grafo; adaptar el loader | el draft denso del fork está medido (§41 del documento de costes). Upstream trae un mezclador hc propio del MTP (`nextn.hc_head_*`); con un GGUF convertido por upstream, el fork leería el del tronco sin avisar |
| `4d756bc72`, `5c53396b8`, `7490357f2`, `bbd488c42`, `b3daa077a`, `dac308739`, `94a0ae3e7` + `3c9e747f7` | — | saltar | el fork tiene versiones mejores o medidas (`c668f5d57`, `615d21ba6`, su FA sparse); `bbd488c42` es el `78dce736c` retirado por −2.1 %; los dos últimos se anulan |
| `733f57a99`, `5c200e0c8`, `0f701c882`, `35822afe5`, `03a667aa3`, `0878f7c30`, `7ec420bc0`, `e0bcaccbe`, `a8d9a7e1e`, `b1ff4ca23`, `32dd62ee6`, `b9f3599fb` | — | arrastrar en el merge | inertes en gfx1151 o en nuestro grafo |

### 2.3 Hallazgo lateral: la selección QSA del fork se aparta de la referencia

- **Referencia** (`modeling_qwen4_exp.py:734-763`): toma los `min(512, bloques)` mejores bloques más la cola de 0-3
  tokens.
- **Fork:** hace un top-k de ancho 2051 sobre celdas (`src/models/qwen4exp.cpp:1007, 1063`) con la cola sesgada a 1e9
  (`llama-memory-hybrid-idx.cpp:925-972`). Cuando la cola tiene menos de 3 celdas, entran además 1-3 celdas del bloque
  513. Eso pasa en 3 de cada 4 posiciones con contexto > ~2k.
- **Upstream `66e0c17ee`** implementa la regla de la referencia; coincide con la vía 19 del documento de costes.
- **Magnitud:** sin medir. Solo afecta a llama.cpp; Strata no se revisó en este punto.

## 3. Pendiente con la GPU, en orden

1. **`f711c8f30`** en el fork de llama.cpp, más la recompilación de `strata-vision` y de llama-server. Compuertas:
   - conteo de tokens contra `smart_resize` en 3-4 tamaños no alineados;
   - texto → imagen → texto;
   - `img_big.py` 2048 × 2048: debe quedar igual, porque 2048 es múltiplo de 32;
   - compuerta de historia de Strata.
2. **Ventana de medición de Strata, sin cambiar código:**
   - las dos variables de lectura corta, con la compuerta de calidad;
   - el decode a la misma profundidad con 131072 y con 262144 por slot (el top-k).
3. **Cherry-picks al fork de Strata:**
   - las seis correcciones de 1.3;
   - `82922f55` + `6ef23b89` + `b6ccf873`, si el punto 2 muestra la ganancia.
   - Compuertas: las de la sección 10 del documento de Strata, con el cliente Vulkan de ruido.
   - Simulados contra HEAD con `git merge-tree --merge-base=<c>^`:
     - **Limpios:** `e372d4ff`, `a6830008`, `2bf1d07c` y el top-k (`82922f55`, `6ef23b89`, `b6ccf873`). El kernel
       ancho ya está en HEAD (`qsa_select.cu:629`), solo apagado en HIP. Los helpers `q8_1_finite`/`q8_1_quant` de los
       arreglos de q8_1 también están.
     - **Con conflictos chicos, a mano:**
       - `7a0e3023`: 1 hunk de 5 líneas;
       - `b97a36a4` + `7bab660b`: 4 hunks, ~85 líneas en `prefill.cpp`;
       - `62185aab` y `0d613031`: 2-3 hunks, más los tests;
       - `dd08a7d0`, `ef6c18a1` y `c5fa1b58`: 1-3 hunks en `server.py`, el más grande de 89 líneas, más los tests.
       Los del servidor chocan con la cesión y el reparto de nuestros commits del 2026-10-07.
4. **Cherry-picks a llama.cpp:** los "tomar" de 2.2, con la suite de determinismo (imagen incluida) y la sonda a 135168.
5. **Los merges completos:** cuando haya una razón medida. Para Strata, `PF_F16` y `GUMBEL`; para llama.cpp, la ventana
   de `70c4e1582`. Cada uno con el procedimiento y los peligros de 1.5 y 2.1.

**Decisiones del Director:**
- si un merge de Strata adopta la política de cesión de upstream o reescribe la nuestra;
- `answer_room`;
- quitar #28244 de los pendientes de `CLAUDE.md`.
