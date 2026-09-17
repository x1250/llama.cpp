# qwen4exp en Strix Halo: dónde va el tiempo y vías de optimización (2026-09-09)

Estado del árbol: master `c0e776ec4` (filas compactas de la FA sparse para lotes pequeños,
prepass determinista, V del indexador estrechado, hoist de P en la FA coopmat1).
Producción: `strix load qwen38flash` (NP=2, `-c 524288`, ubatch 2048, KV q8_0, lazy mode auto,
draft MTP IQ4_NL n-max 2, mmproj, `--ctx-checkpoints 8`).

Fuentes de datos:

- Perfiles del Vulkan perf logger a 40k: `~/dbg/depth/strix-decode-prof-cd-vk.log` (prefill de
  39.5k + decode) y `strix-decode-prof-cdec-vk.log` (decode con filas compactas). El logger
  serializa cada op (barrera + timestamp por nodo), así que sus tiempos son cotas superiores:
  la suma por paso de decode es 72 ms frente a 66 ms medidos en producción.
- Rigs de profundidad `~/dbg/depth/depth-*.log` (turno 1 = prefill 39512 tokens, turnos 2-8 =
  re-prefill de ~2048 + 256 generados).
- `LLAMA_SPEC_TIMING` y `LLAMA_INPUT_TIMING` en `~/dbg/merge/srvlog-p40-*.log`.
- Lectura del grafo (`src/models/qwen4exp.cpp`), de los inputs de host
  (`src/llama-kv-cache.cpp`, `src/llama-memory-hybrid-idx.cpp`), del bucle especulativo
  (`tools/server/server-context.cpp`, `common/speculative.cpp`) y del backend Vulkan
  (ver `vulkan-backend-audit-2026-09-09.md`).

Dimensiones del modelo que fijan los bytes: n_embd 2560, hc 4 (residual ancho 10240),
48 capas (36 GDN + 12 QSA), 512 expertos de 640 usados 10 + 1 compartido, vocab 248320,
LM head Q6_K, wqkv de GDN Q5_K, router `ffn_gate_inp` F32, indexador BF16, resto IQ4_NL,
o_proj Q8_0. Pesos por token: densos ~3.1 GB, expertos 1.33 GB.

## 1. Decode a 40k con MTP: paso de 66 ms = 2.42 tokens (36 t/s)

| Bloque | ms/paso | Observación |
|---|---|---|
| Expertos del verify (3 tokens) | 18.2 | `ggml_vk_mul_mat_vec_id_q_f16` despacha un dispatch por token (ggml-vulkan.cpp:11517); los expertos compartidos (27 % de los pares) los relee la Infinity Cache, no la DRAM (sección 5): ~2.9 GB únicos a ~160 GB/s, frente a 210-227 de los densos grandes |
| LM head (target q6_K 2.3 + 2 del draft iq4_nl 1.6) | 5.6 | 227 GB/s, en el límite del bus |
| Densos grandes (wqkv q5_K 3.0, z 1.7, ssm_out 1.8, wq 1.1, wo 1.0, shexp 1.2) | ~11 | 190-215 GB/s |
| Router f32 (`m=512 n=3`, 49 dispatches de 54.8 µs) | 2.7 | 96 GB/s: único mat-vec con NUM_ROWS=1 (ggml-vulkan.cpp:5898) |
| Hyper-connection: 3 mat-vecs (`m=320` 1.72, `m=10240 k=320` 1.58, `m=4` 1.40) + norm, scale, silu, gated_mean, inject | 6.7 | 3.7 MB de pesos por paso, suelo 1.5 ms. `m=4` corre en un solo workgroup |
| FA sparse (12 capas × 0.35 ms) + FA densa del draft (2 × 0.5) + prepass | 5.7 | Modo compacto sin split_k: 6 workgroups en 40 CUs para 2051 celdas |
| Estado GDN: gather (36 get_rows) + 3 snapshots copiados (36 CPY) + kernel 0.8 | 3.1 | ~450 MB movidos por paso por el rollback de MTP (n_rs_seq = 2) |
| TOPK_QSA (12 × 85 µs) | 1.05 | Lineal con n_kv: ~6 ms a 262k |
| ~2500 dispatches pequeños (norms 285, sigmoid 139, glu 105, cont 176, cpy 156, set_rows 65, scale 199, add 153) | ~6 | 3600 dispatches por paso; barrera global entre nodos dependientes |
| Host: sampling 1.2, huecos ~3 | ~4 | |

Suelo de ancho de banda: ~7.8 GB de pesos por paso (densos 3.1 + expertos 3.2 con dedup +
draft 1.5) = 32 ms a 240 GB/s, el 49 % del paso. El resto es latencia de kernels pequeños,
kernels mal dimensionados y ocupación baja. La afirmación anterior de que "el 70 % del paso
son los pesos" era incorrecta.

Datos de apoyo:

- Draft (2 tokens): 8 ms por paso = LM head 1.6 + expertos ~1.2 + FA densa 0.5 (crece con
  n_kv) + eh_proj 0.1 + resto 0.6, por token.
- Aceptación 1.42 tokens por paso a temp 0 (0.71 por token del draft). n-max 3 midió peor.
- Graph reuse activo en decode (`graphs reused` en print_timing).

## 2. Prefill de 39.5k tokens: 108 s de pared, 20 ubatches de 2048

| Bloque | s | % |
|---|---|---|
| GPU total (suma del perf logger) | 80.7 | 75 |
| · Densos MUL_MAT | 22.5 | 21 |
| · · hyper-connection: `m=320 n=2048 k=10240` 3.0 (7.5 TFLOPS), `m=10240 k=320` 2.78 (7 TFLOPS), `m=4 k=10240` 2.65 (0.1 TFLOPS) | 8.4 | 7.8 |
| · · wqkv q5_K 3.09 (22 TFLOPS), ssm_out 2.36 (18), z 1.86 (22), wq 1.32 (23), shexp 1.15, wo q8_0 0.70, router f32 0.37 (12.7), `m=48` 0.27, `m=1` f32 0.25 | 11.4 | 10.5 |
| · Expertos MUL_MAT_ID (down `m=2560 k=640` 14.5 ms/dispatch a 4.6 TFLOPS; gate_up `m=640 k=2560` 13.3 ms a 10 TFLOPS) | 20.0 | 18.5 |
| · Flash attention (sparse del target ~12 s, densa del draft ~3 s) | 15.8 | 14.6 |
| · Elementwise sobre el residual ancho de 84 MB: MUL 3.0, RMS_NORM(+MUL) 2.95, HC_GATED_MEAN 1.9, MULTI_ADD 1.65, HC_INJECT 1.6, CONCAT 0.94, CONT 0.92, ADD 0.87, GLU 0.67, SSM_CONV 0.65 | ~13 | 12 |
| · `eh_proj` del draft como mat-vec por lote (`m=2560 n=4 k=5120 batch=2048`, 130 ms por ubatch) | 2.5 | 2.3 |
| · Selección QSA (TOPK_QSA 1.1, RELU 0.37, TOP_K 0.22, FILL/SET_ROWS 0.17) | ~2 | 1.9 |
| · GDN (36 × 2.1 ms por ubatch, kernel secuencial en tokens) | 1.43 | 1.3 |
| No GPU | 27 | 25 |

FA sparse por nodo de 2048 consultas según n_kv (perf logger, 12 nodos por ubatch):
2.3k 6.8 ms, 4.4k 19.6, 6.4k 32.5, 8.4k 43.7, 10.5k 53.9, 12.5k 64.3, 14.6k 71.6
(régimen denso con máscara), 18.7k ~83, de 20k a 40k satura en 67-80 ms. La FA densa del
draft a 37k cuesta 229 ms por nodo (7.2 TFLOPS). La sparse a 40k equivale a atender ~13k
celdas por tile de 16 consultas: 6.3× las 2051 seleccionadas por token.

Input fill medido con `LLAMA_INPUT_TIMING`: 140-250 ms por ubatch de 2048, plano con la
profundidad (~0.1 ms por token); 39 ms para 512 tokens; 0.3 ms para 2 tokens a 40k. El
componente plano con la profundidad apunta a `prefetch_ple_rows` (`src/models/qwen4exp.cpp`):
hasta 2 × 10240 llamadas síncronas a `posix_madvise` por ubatch (ubatch actual y siguiente).

Los ~20 s restantes no atribuidos: build+alloc del grafo (no se reutiliza en prefill: n_kv
cambia en cada ubatch), lado host de `graph_compute`, copia D2H de `h_nextn` (84 MB por
ubatch), las tres copias de 84 MB del catch-up del draft, huecos entre submisiones.

Medición previa del tamaño de ubatch (prompts de 4.7k, 2026-09-09): ub512 326 t/s,
ub2048 425, ub8192 348. El sobrecoste fijo por ubatch no es el término dominante.

## 3. Vías de optimización, ordenadas de mejor a peor ganancia estimada

Regla (2026-09-10, tras el punto retirado de la sección 5): ninguna vía se implementa sin la
comprobación previa de su columna, un experimento acotado que aísle el cuello de botella con
herramientas que ya existen (test-backend-ops perf, variables de entorno, perf logger). La
ganancia es un porcentaje de la fase a 40k: del prefill de 108 s o del paso de decode de 66 ms.
"Evidencia" dice de dónde sale la estimación; "riesgo" es la probabilidad de que la ganancia no
exista aunque el cambio sea correcto.

| # | Vía | Fase | Ganancia estimada | Evidencia | Riesgo | Comprobación previa |
|---|---|---|---|---|---|---|
| 1 | MUL_MAT_ID del prefill: tile por filas por experto (mediana 21, BN=128), hoisting de ids para 512 expertos, cargas de B fuera de `_ne1`, BK_STEP=1. **Hoisting hecho y medido (7fc9ae43d, 2026-09-16): prefill 109.6 → 94.7 s (−13.5 %), re-prefill 6.05 → 5.30 s, decode igual.** El tile por filas por experto (`GGML_VK_MMID_SMALLN=1`) resultó neutro (94.4 s) y queda como sonda apagada; ubatch 4096 pierde un 9 % y deja los checkpoints a 4096. Pendientes: cargas de B fuera de `_ne1`, BK_STEP | prefill | −13.5 % medido (hoisting); resto sin estimar | Medida directa: 4.6 y 10 TFLOPS frente a 22 en los densos; causas leídas en `mul_mmq.comp`; externa, misma GPU: halo-box selecciona el tile por filas esperadas por experto (`GGML_VK_MMID_SMALLN` con `M128` y `BM64`, sobre un prepass de listas de filas) y midió +16.3 % pp512 en Qwen3.6-35B-A3B (~32 filas por experto), MUL_MAT_ID q5_K de 2.33 a 4.43 TFLOPS; es la implementación de referencia (auditoría, sección 10). Ruta entera (MMQ q8_1) frente a coopmat, medido a nivel de modelo el 2026-09-16 con `GGML_VK_DISABLE_COOPMAT_MMQ=1`: para IQ4_NL la entera gana 2.4 % de prefill (104.0 frente a 106.5 s); para IQ3_S la coopmat gana 2.1 % (109.7 frente a 112.0 s), y en el microbench aislado la coopmat gana en ambos tipos: el +3.7 % de b9c196c1c se sostiene solo en el grafo real | Medio: otro límite (L2, ocupación) puede aparecer al llenar los tiles | test-backend-ops perf con 64 expertos (320 filas por experto): si alcanza 15-20 TFLOPS, el tile es la causa; ubatch 4096 en el rig con MTP (80 filas por experto; compute buffer 4.3 GiB a 2048, verificar el tamaño con `-lv 4` antes de cargar) |
| 2 | FA sparse token-major en prefill: tiles de 12 cabezas × 1 token (la ruta GQA del decode) en vez de 16 tokens × 1 cabeza, unión 6.3× la lista. **Hecha y medida (2026-09-16, sección 9): prefill 94.7 → 88.8 s (−6.3 %), re-prefill de 2k 5.3 → 4.75 s (−10 %), decode igual, determinismo verificado.** | prefill | −7.5 % a 40k (−8 s), más a mayor profundidad | Medida directa: 67.5 ms sparse vs 229 densa por nodo; uniones del 33 % medidas por el backend a 16k | Medio: asume FA limitada por cómputo; el gather compacto no ganó, lo que apunta a cómputo pero no lo prueba | `LLAMA_QSA_QUERY_BLOCK=8` fuerza bloques de 8 tokens por la ruta GQA existente; el perf logger da el coste de la FA por token sin escribir shader |
| 3 | Forma del workgroup del mat-vec de expertos en decode: NUM_ROWS mayor o reparto de k en `down` (k=640, una iteración y media por hilo), gate+up como tensor fusionado (m=1280) | decode | hasta −6.5 % (18.2 → ~14 ms) | Tasas medidas: 160 GB/s en expertos frente a 210-227 en densos grandes de la misma GPU; halo-box midió sobre este modelo en Vulkan el troceado por columnas del mat-vec por lotes: gana en q8_0 y q6_K (MTP n-max 4: 9.6 → 15 t/s) y pierde en q4_K por releer los pesos por trozo; nuestro mat-vec-id ya despacha por token | Medio-alto: estimación por tasa, del mismo tipo que la vía retirada | test-backend-ops perf con las formas reales (640×2560, 2560×640, 10 de 512, n=3) y variantes de NUM_ROWS antes de tocar el grafo |
| 4 | Mezcladores hyper-connection en prefill: tiles para m=320 y k=320 (48 workgroups, B en f32), formulación transpuesta o fusión del inject m=4 **Medida (2026-09-16, sección 11): `hc_*_inject` fusionado en `hc_*_down` (archivo y loader) −5.2 % prefill y decode +15 %; tile medio alineado para k=320 −1.3 %; split_k para pocos tiles negativo.** | prefill | −5.5 % (−6 s) | Medida directa: 7.5 TFLOPS y 0.1 TFLOPS en formas concretas | Bajo-medio | test-backend-ops perf en esas tres formas con la heurística de split_k relajada (una línea) |
| 5 | GEMM densa con sombra f16 por nodo en prefill: desempaquetar el peso cuantizado a f16 una vez por nodo y ubatch (scratch transitorio) y correr el pipeline coopmat f16, en vez de dequantizar cada tile de A en cada workgroup que lo usa (16 veces por ubatch de 2048 con BN=128) **Medida (2026-09-16, sección 11): negativa, +4 % a +7 %; el kernel f16 es más lento que los cuantizados en todas las formas (15-17 frente a 20-22 TFLOPS). Cerrada.** | prefill | central −3 %, hasta −5.5 % (los densos grandes: 11.4 s a 18-23 TFLOPS; la sombra cuesta leer el quant y escribir f16 una vez por nodo) | Externa: la ablación de pwilkin en HIP vale 1.42× para "bf16 WMMA dequant GEMM" con ubatch 24576 (sección 6); halo-box atribuye la ganancia de wave32 al conteo de instrucciones de dequant inline (q6_K 3907 → 3433) | Medio-alto: otro backend y otro ubatch; en Vulkan la ruta coopmat f16 debe rendir ~2× la cuantizada para que la sombra se pague | test-backend-ops perf MUL_MAT f16×f32 frente a q5_K, q8_0 e iq4_nl×f32 en las formas densas de la sección 2 (wqkv, ssm_out, z, wq, wo, shexp); si f16 no dobla la tasa, la vía no existe |
| 6 | **Medidas el 2026-09-16, neutras en este modelo (109.8, 110.1 y 110.9 s frente a 109.6 de la base; corrección 80/80 y 103/103):** compuertas ya compiladas y apagadas del backend: `GGML_VK_DENSE_WAVE32=1` (retile a wave32 de los pipelines coopmat cuantizados densos; `=2` también los f16), `GGML_VK_MMID_WAVE32=1` y `GGML_VK_MMID_WG256=1` (mismo retile y workgroup de 256 hilos para MUL_MAT_ID) | prefill | 0 a −5 % | Externa, misma GPU (halo-box, RADV gfx1151): GEMM densa q6_K +5.2..+10.8 %, q8_0 +5.4..+8.4 %, q4_K +0.7..+9.1 %, q4_0 −1.5..+1.8 %; a nivel de modelo pp2048 +3.9..+7.2 %; activadas allí por defecto desde 2026-08-30. IQ4_NL sin medir en ningún sitio | Medio: el grueso de nuestros pesos es IQ4_NL, del que no hay dato; q5_K y q8_0 (wqkv, o_proj) sí lo tienen | Tres variables de entorno, sin código: test-backend-ops perf en las formas densas y de expertos, luego el rig de 40k con la combinación que gane (auditoría, sección 10) |
| 7 | split_k en el modo compacto de la FA sparse de decode (hoy 6 workgroups por capa, 350 µs) | decode | hasta −4.5 % (−3 ms), no probado | Inferencia: 6 workgroups en 40 CUs | Alto: el modo índice con split_k=13 fue más lento que el compacto sin split_k | test-backend-ops perf con la FA sparse de 3 tokens y split_k forzado |
| 8 | Ponderación y suma de expertos en el epílogo del MUL_MAT_ID de prefill (MUL 3.0 s, MULTI_ADD 1.65 s). **Ponderación hecha y medida (2026-09-16, sección 10): prefill 89.0 → 86.8 s (−2.4 %), salida bit-idéntica; la suma (MULTI_ADD, 1.6 s) queda, no hay forma determinista de meterla en el kernel.** | prefill | −4.3 % (el MUL solo, −2.8 %) | Pasos de memoria medidos, 1 ms por tensor de 84-210 MB; implementación de referencia en halo-box: `GGML_VK_MMID_SCALE_EPILOGUE` aplica la escala por (experto, token) al escribir el MUL_MAT_ID de prefill (no en coopmat2) | Bajo para el MUL, medio para la suma | Ninguna para el MUL; la suma requiere diseño (orden de acumulación) |
| 9 | Cabeza del draft MTP recortada a un subconjunto de vocabulario (248k → ~47k filas, tabla índice → token, muestreo del draft sobre K logits; converter + `common/speculative.cpp`). Aplazada por decisión del Director (2026-09-10) | decode | +4 % (la cabeza del draft mide 1.62 ms por token, 3.24 ms por paso; con 47k filas ~0.6 ms) | Medida directa del mat-vec de la cabeza (221 GB/s) | Medio: la aceptación cae con los tokens fuera del subconjunto; el subconjunto debe salir de nuestro tráfico (español), no de uno de código | Contar en las respuestas de los rigs qué fracción de tokens cae en los 47k más frecuentes de nuestro tráfico |
| 10 | Mezclador hyper-connection en decode: reparto de k del mat-vec m=4 (14 µs en un solo workgroup), scale plegado, silu y gate fusionados | decode | hasta −3.8 % (−2.5 ms), rebajado desde −4.4 | Cadena medida: 61 µs por mezclador × 96; suelo por dispatch 2.5 µs | Alto si se fusiona todo en un kernel serial por token; medio como reparto de k | test-backend-ops perf del mat-vec m=4, k=10240 con reparto de k |
| 11 | IQ4_NL en las listas f16-B del backend (B viaja en f32; `matmul_iq4_nl_f16` ya se genera) **Medida (2026-09-16, sección 11): neutra (86.7 frente a 86.7-87.1 s). No se adopta.** | prefill | −1 a −3 % | Medición del fork en otros tipos (+4.5 %) | Medio: los densos grandes van a cómputo, no a bytes de B | Dos líneas; se mide directamente |
| 12 | **Línea de tiempo medida el 2026-09-16 (`LLAMA_INPUT_TIMING=1`, 18 ubatches de 2048): set_inputs 177 ms, alloc 40 ms, build 2 ms, host de graph_compute 1579 ms y espera de GPU 3499 ms por ubatch; fuera de llama_decode 204 ms por ciclo. El prefill está limitado por la GPU: el host serial vale 3 a 4 % (set_inputs + alloc) más 3.7 % fuera de decode; los "27 s no GPU" eran un subconteo del perf logger.** Input fill del prefill: `prefetch_ple_rows` emite 2 × 10240 `posix_madvise` síncronos por ubatch (150-250 ms medidos); el resto del 25 % no GPU sin atribuir. Hipótesis añadida el 2026-09-15: la máscara KQ densa. La CPU la rellena celda a celda en un solo hilo (`set_input_kq_mask_impl`, `src/llama-kv-cache.cpp`): 2048 × n_kv por ubatch, 84M celdas a 40k y 260M a 126k, creciente con la profundidad; y cada capa QSA hace fill, set_rows y add sobre n_kv × 2048 (`build_attn_qsa`), unos 5 × n_kv × 2 B por token y capa: ~1 s por prefill de 40k, ~10 s a 126k (pwilkin: "guard KQ mask write" +2 % y "maskless KQ" 1.48× en HIP con ubatch 24576) | prefill | −4 a −7 % (medido el techo el 2026-09-16) | Medido con `LLAMA_INPUT_TIMING`; el resto es una resta pared − GPU | Bajo para la parte medida; el resto no está cuantificado | Línea de tiempo por ubatch (set_inputs, build/alloc, compute host, D2H, catch-up del draft); set_inputs separa máscara y filas PLE, y el perf logger da FILL/SET_ROWS/ADD de la máscara por capa |
| 13 | Router f32: mat-vec con NUM_ROWS=1, 96 GB/s frente a 165 del bf16 de igual forma | decode | −2.7 % (−1.8 ms); −3.5 % con router q8_0 en el GGUF, con PPL/KLD | Medida directa | Bajo | Constante de especialización; se mide |
| 14 | Estado GDN in place: 36 CPY de snapshots y 36 gathers por paso | decode | −2.3 % (−1.5 ms) | Familias CPY y GET_ROWS medidas; la porción GDN es estimada | Medio | Contar en el perfil los dispatches de estado por capa |
| 15 | `eh_proj` del draft como una sola matmul (`ggml_reshape_2d` a [5120, 4·T]) en vez de mat-vec por lote de 2048 (130 ms por ubatch). Implementada (bbded2ebe) y medida el 2026-09-16: prefill de 39.5k 104.0 s frente a 107.4 s de la referencia anterior (−3.2 %), decode igual, repeat idéntico | prefill | −3.2 % medido | Medida en el rig | Cerrada | — |
| 16 | QSA en el bloque del draft reutilizando la selección de bloques del target (IndexShare, como SGLang en su soporte de día 0: el draft no ejecuta el indexador; toma la lista de la última fila aceptada más N+1 columnas para las posiciones drafteadas). La FA densa del draft cuesta 229 ms por ubatch a 37k, lineal con la profundidad, y en decode 0.5 → 3.3 ms por paso de 40k a 262k | prefill y decode | −2.2 % a 40k, crece con la profundidad; decode −6 ms por paso a 262k | Medida directa del coste; diseño de referencia publicado (SGLang) | Medio: la aceptación del draft con una selección prestada se mide al implementar; el target no cambia | Ninguna barata: la ganancia es el coste medido de la FA densa; la aceptación se comprueba en el rig (draft acc) |
| 17 | Menos dispatches en decode: cont+cpy de los slots de rollback (108), scale plegado en `hc_down` (96), sigmoid×mul de la norma GDN, máscara fill/set_rows/add si la FA consume la lista | decode | −1 a −3 % | Suelo por dispatch medido (2.4-2.6 µs) | Bajo | Aritmética directa |
| 18 | gate y up de los expertos en un solo tensor `ffn_gate_up_exps` (el loader ya lo soporta y el converter tiene `--fuse-gate-up-exps`; nuestro GGUF los trae separados). Transformación exacta del archivo: un MUL_MAT_ID por capa en vez de dos, un prólogo de ids menos por capa en prefill, −49 dispatches por paso en decode | prefill y decode | −1 a −2 % en prefill (estimado), −0.3 % en decode | Bit-idéntico según oMLX (`qwen35_moe_gate_up.py`); dispatches medidos | Bajo | Construir el GGUF fusionado con gguf-py (concatenar filas por experto, sin recuantizar) y medir en una ventana; cambio de archivo, decisión del Director |
| 19 | Selección QSA a granularidad de bloque y FA que consuma la lista top-k sin construir la máscara | prefill y decode | −1.4 % a 40k, lineal con n_kv | Pasos de memoria medidos | Medio | Diseño; cuantificar a 128k con el perfil |
| 20 | concat + transpose del `ssm_conv` | prefill | −0.8 % | Medida directa | Bajo | Ninguna |
| 21 | Mat-muls diminutos en prefill (m=48, m=1 f32, router f32) | prefill | −0.6 % | Medida directa | Bajo | Ninguna |
| 22 | MMVQ IQ4_NL fuera del allowlist del mat-vec | decode | < 1 % | El commit b9c196c1c midió tg64 sin cambio | Alto | test-backend-ops perf |
| 23 | GDN chunked en prefill (hoy kernel secuencial, 2.1 ms por nodo de 2048 tokens = 1.03 µs por token y capa) | prefill | ≤ −1.4 %, probablemente nula | oMLX: su kernel secuencial optimizado rinde 0.93 µs por token y capa a 16k (paridad con el nuestro) y su variante chunked resultó más lenta de extremo a extremo | Alto | Ninguna barata; queda al final por la evidencia de oMLX |
| M | Caché del indexador como una clave agrupada por bloque de 4 celdas más un anillo de 4 slots con las claves crudas del bloque incompleto (SGLang, oMLX): hoy guardamos la clave cruda de cada celda, 12 × n_ctx × 128 × 2 B = 1.6 GB a 524k | memoria | −1.2 GB del sobre de producción; rendimiento neutro | Diseño de referencia publicado | Medio: toca `llama-memory-hybrid-idx` y el camino de recomputación del prefill | Ninguna; es memoria, se mide al cargar |
| — | Mat-vec-id agrupado por experto (retirado, sección 5) | decode | medido: −5 % de decode, más lento | Inferencia de bytes sin prueba de cuello de botella | Fallido | — |

Suma de las vías 1, 2, 4, 8, 11, 12 (parte medida) y 15 en prefill: −30 a −35 s de 108 con
confianza media; la parte no atribuida del host es adicional y desconocida. Las vías 5 y 6
(2026-09-15) añadirían entre 0 y −9 s con evidencia externa: no entran en la suma hasta su
comprobación. En decode, las vías 3, 10, 13, 14 y 17 suman −9 a −13 ms de 66 con confianza
media-baja: cada una exige su comprobación previa.

Referencias externas revisadas el 2026-09-11: oMLX (jundot/omlx, Apple Silicon/MLX) y el blog de SGLang del
soporte de día 0 de Qwen3.8-Flash-Next. Coinciden con las vías 2 y 19 (atención sparse GQA de índice
directo, sin máscara, por token) y con nuestro modo compacto de decode; aportan IndexShare (vía 16), el
anillo del caché del indexador (fila M) y gate+up fusionado (vía 18), y sitúan el GDN chunked al final.
El mecanismo Causal Encoder-Decoder de DeepSeek-V4.1-Flash (KV de las capas altas proyectado desde el
encoder) es una aproximación entrenada, con pérdida por diseño y sin lugar para el estado recurrente de las
capas GDN: fuera de esta lista. Revisión del 2026-09-15 (pwilkin Strix Halo Lab y
halo-box/strix-llama.cpp): sección 6.

## 4. Mediciones previas pendientes

- Línea de tiempo host por ubatch de prefill, producción descargada: set_inputs, build/alloc,
  compute host, copias D2H, catch-up del draft. Decide cuánto vale la vía 12 y si existe una
  vía mayor.
- Coste de la FA sparse por token en prefill con `LLAMA_QSA_QUERY_BLOCK=8` y el perf logger.
  Confirma o descarta la vía 2 antes de escribir el shader. **Hecha (sección 9): el brazo de
  bloques de 8 no mide la geometría sino la ocupación (16 workgroups por dispatch, 3× más lento);
  el brazo base fijó la FA sparse en 0.77-0.98 s por 2048 consultas (12 capas) entre 16k y 37k,
  y la vía se midió directamente con el kernel.**
- test-backend-ops perf de las formas reales de expertos (vías 1 y 3) y de los mezcladores
  (vías 4 y 10).
- El solape de expertos entre los tokens del verify ya está medido (sección 5): 27 % a n=3, y
  no es una vía.
- Compuertas compiladas (vía 6): test-backend-ops perf con `GGML_VK_DENSE_WAVE32=1`,
  `GGML_VK_MMID_WAVE32=1` y `GGML_VK_MMID_WG256=1`, por separado, en las formas densas y de
  expertos; después el rig de 40k con la combinación que gane.
- Sombra f16 de los densos (vía 5): perf de MUL_MAT f16×f32 frente a q5_K, q8_0 e iq4_nl en las
  formas densas.
- Ubatch 4096 en el rig con MTP (vía 1): tamaño del compute buffer con `-lv 4` y prefill a 40k;
  hoy solo hay medición a 4.7k (ub2048 425 t/s, ub8192 348).
- Robustez, sin rendimiento: acotar cada submisión por bytes movidos (halo-box
  `GGML_VK_MAX_MB_PER_SUBMIT`, 8 GiB por defecto) como guarda genérica del timeout de 2 s del
  kernel 7.0. Hoy solo la FA está acotada (`qsa_query_block`); los nodos sin estimación de flops
  (copias, set_rows, fill de máscaras) crecen con el contexto y `GGML_VK_MAX_NODES_PER_SUBMIT`
  cuenta nodos, no bytes.
- Merge de upstream: tras nuestra base `6d9c82ea2` hay cuatro commits Vulkan relevantes:
  `50182a53f` fusión topk_moe en prefill (#28422), `6788edb4f` matrices M pequeñas para qwen
  (#28457), `28ff09582` escrituras CPU en `cpy_tensor_async` con el contexto inactivo (#28618) y
  `481c65f09` carrera y OOB en argsort grande (#28705, corrección). Entran con el próximo merge y
  sus comprobaciones (lazy mode, split-on-inputs, sonda a n_ctx 135168).
- Memoria: draft MTP con `token_embd` y `output` compartidos con el target (pwilkin
  `--mtp-shared-embd`, draft Q8_0 de 2.8 GB sin ambos). Nuestro
  `Qwen3.8-Flash-Next-MTP-IQ4_NL.gguf` lleva los dos (2 × 636 M elementos IQ4_NL, ~0.7 GB):
  ahorro del mismo orden en el sobre; requiere cargador y archivo.

Reglas de toda ventana de medición: producción descargada, gate de 40 GiB (37 para los rigs
de profundidad), baseline repetido en la misma cadena, MTP encendido, y las comprobaciones de
determinismo (`repeat_probe.py`, `graph_diff4`, `depth_repeat.py`) tras cualquier cambio de
backend.

## 5. Medición del solape de expertos (decode, punto 1)

Herramienta: `LLAMA_MOE_IDS_LOG=<archivo>` (common/common.cpp), volcado de los expertos que
elige cada capa por ubatch; estadísticas con `~/dbg/merge/moe_ids_stats.py`. Corrida: rig de
profundidad a 40k con MTP n-max 2 (`~/dbg/depth/moe-ids-40k.txt`, 45305 líneas, 2026-09-09).
El callback parte el grafo en cada nodo observado, así que los tiempos de esa corrida no son
comparables (prefill 273 t/s, decode 26-32 t/s); las selecciones sí.

Verify (troncal, 48 capas):

| Tokens del verify | Grafos-capa | Expertos únicos / pares (token, slot) | Bytes de expertos ahorrados con dedup |
|---|---|---|---|
| 3 | 238 | 0.791 | 21 % |
| 4 | 40608 | 0.730 | 27 % |
| 5 | 432 | 0.533 | 47 % |

Por capa a n = 3: entre 18.0 y 27.4 expertos únicos de 30; las capas altas (39-47) solapan más.
El bloque MTP (capa 48) a n = 3: 0.846 (15 %).

Proyección: los expertos del verify cuestan 18.2 ms por paso leyendo 3.98 GB a 219 GB/s (cada
token relee sus expertos); con una lectura por experto único quedan ~2.9 GB → −4.9 ms por paso
de 66 (−7.4 %, +8 % de decode). Además la fusión `MUL_MAT_ID + MUL` hoy exige un solo token
(ggml-vulkan.cpp, `mmid_mul_ok`: `scale->ne[2] == 1`), así que a n = 3 la ponderación es un
MUL aparte (48 dispatches por paso) que la variante agrupada puede absorber.

Prefill (ubatch de 2048, troncal): tokens consecutivos comparten el 36.6 % de sus expertos
(t, t+1) y el 28.3 % (t, t+2); expertos activos por capa 398 de 512; filas por experto activo
media 52, p50 18-21, p90 103-116, máximo ~1960 (un experto casi universal por capa). Con el tile
BN = 128 del MUL_MAT_ID (sección 3, vía 1) la mediana de 21 filas ocupa el 16 % del
tile: confirma el diagnóstico de utilización.

Diseño implementado y medido (2026-09-10, retirado): un solo dispatch por nodo con eje y =
pares (token, slot) en vez de un dispatch por token; el workgroup del par p toma su experto e,
sale si un par anterior ya lo tiene, y si no calcula las columnas de todos los tokens que
eligieron e (NUM_COLS especializado por n_tokens) con offsets de B y D por columna y el índice
de escala `t·n_used + s`, lo que extiende la fusión `MUL_MAT_ID + MUL` a varios tokens.

Resultado: correcto y determinista, pero más lento. test-backend-ops MUL_MAT_ID 927/927,
MUL_MAT_ID_FUSION 17/17, MUL_MAT 1176/1176; graph_diff4 seq_rm 2048,631 sobre MUL_MAT_ID: 0 de
432 nodos difieren; logits a 40k bit a bit idénticos a la build anterior (q1: El −0.3867,
We −1.286, The −3.2772, Okay −6.0566 en ambas; sonda de producción idéntica). A/B en el rig de
40k con MTP, mismas condiciones, una carga tras otra:

| Build | Turnos 2-5, decode t/s | Turno 1 | Runs del repeat (small batch) |
|---|---|---|---|
| Anterior (dispatch por token) | 38.50, 36.69, 36.56, 36.00 | 33.34 | 34.9-35.4 |
| Agrupada, corrida 1 | 36.93, 35.25, 34.71, 34.21 | 31.61 | 33.7-34.2 |
| Agrupada, corrida 2 | 36.04, 35.06, 33.96, 33.71 | 31.26 | 33.2-33.7 |

Lectura: −5 % de decode. Las relecturas del 27 % ya las servía la Infinity Cache de 32 MB (los
tres dispatches de un nodo leen los mismos expertos con microsegundos de diferencia), así que la
agrupación no quitó tráfico de DRAM, y añadió a cada workgroup el barrido de los ids y hasta
n_tokens columnas de B y de FMAs aunque su experto lo use un solo token. La ineficiencia real del
mat-vec de expertos (160 GB/s frente a 210-227 de los densos grandes) está en la forma del
workgroup, no en los bytes: `down` tiene k = 640 (una iteración y media por hilo, 640
workgroups por par que pagan la reducción entera) y `gate`/`up` m = 640 (160 workgroups por
par); son dos nodos separados por capa (gate y up no se fusionan al cargar). Próxima vía para
este bloque: NUM_ROWS mayor o reparto de k para `down`, y gate+up en un tensor fusionado
(m = 1280, la mitad de dispatches); ambas requieren medición propia. El commit del kernel
agrupado se retiró; quedan los tests de forma real a 2-8 tokens con expertos compartidos
(`tests: MUL_MAT_ID mat-vec cases with tokens sharing experts`).

## 6. Referencias externas (2026-09-15): pwilkin Strix Halo Lab y halo-box/strix-llama.cpp

Fuentes: https://pwilkin.github.io/strix-halo/ (y su `journey.html`), el instalador `install.sh`
del repo pwilkin/strix-halo (rama `strix-halo` de pwilkin/llama.cpp, commit `d67d58836`, ROCm
10.0, `-DGGML_HIP=ON -DGPU_TARGETS=gfx1151`) y https://github.com/halo-box/strix-llama.cpp
(Vulkan, misma GPU). Los commits del fork comunitario que ya están en el nuestro y los que
faltan están tabulados en `vulkan-backend-audit-2026-09-09.md`, sección 10.

Sus números son llama-bench en HIP (`-b 24576 -ub 24576 -p 16384 -n 128 -d 0,40000 -r 3`, y
16384 en el instalador, que mide igual: 1204.31 ± 2.31 frente a 1199.63 ± 7.12), sin draft, un
slot, ctx 65536, sin mmproj, pesos en memoria anónima (`--load-mode none`) y PLE servida por
`pread` desde un pool de hilos (`--lazy-mode on-direct`). Su lanzador de producción añade MTP
n-max 3 con un draft Q8_0 de 2.8 GB que comparte `token_embd` y `output` con el target. Los
nuestros son el server con el feature set de producción (Vulkan, ubatch 2048, KV q8_0, MTP
n-max 2, mmproj, checkpoints): rigs `~/dbg/depth/depth-p40-{mtp,nomtp}.log`,
`depth-merge-rev-125k.log` y `depth-chunk125.log`.

| Medida | pwilkin, HIP, ubatch 16384 | Nuestro, Vulkan, ubatch 2048 |
|---|---|---|
| Prefill, profundidad 0 | 1204 ± 2 t/s | ~425 t/s (prompts de 4.7k, MTP cargado) |
| Prefill, 40k | 1086 ± 1 t/s | 368 con MTP, 401 sin MTP |
| Prefill, contexto largo | 947 t/s (150k) | 299 con MTP (126.5k) |
| Decode, 40k | 16.6 ± 0.1 t/s | 25.9 sin MTP, 36.0 con MTP |
| Decode, contexto largo | 6.9 t/s (150k) | 21.8 sin MTP (build anterior), 31.4 con MTP (126.5k) |

Prefill: 2.7 a 3.2× a favor de HIP. Decode: 1.5 a 3× a favor nuestro sin draft, hasta 4.5× con
MTP. Su punto de partida con HIP upstream fue 191 t/s con ubatch 24576, la mitad de nuestro
Vulkan: el salto (6.07×) es su rama. Sobre upstream lleva 26 commits del 12 al 14 de septiembre
(15 de kernels CUDA/HIP, 5 de modelo y grafo sobre `qwen4exp.cpp` y `llama-graph`, el resto
pruebas, defaults y guardas), más el anillo de entradas del scheduler de agosto y los TOP_K
wave32 de ROCm. Su trabajo de decode empezó el 13 de septiembre (`hip: enable sparse QSA decode
and incremental indexer state`). Su capa de runtime (listas PM4 retenidas para HIP graphs) no
interviene: por su propia nota los HIP graphs nunca se activan en el prefill de este modelo, y
Vulkan no tiene esa re-codificación.

Ablación de pwilkin (desactivando cada pieza en su binario final) mapeada a nuestro desglose de
108 s:

- GDN por tiles, 2.37× en su serie: no aplica. Nuestro GDN cuesta 1.43 s (1.3 %), ~1.03 µs por
  token y capa; su baseline era el kernel genérico de HIP.
- GEMM densa bf16 WMMA con dequant una vez por grafo, 1.42×: vía 5.
- Kernel de atención sparse, 1.71×, y "sparse apagada del todo" 0.98-1.01×: su FA densa es tan
  rápida que el indexador (12 capas × 80 franjas por prefill) anula lo que ahorra la sparsity con
  ubatch 16384. Confirma la vía 2: la palanca es el kernel, no la sparsity; una FA densa rápida
  de head 256 serviría también a la atención densa del draft (229 ms por nodo a 37k).
- Camino sin máscara KQ, 1.48× con ubatch 24576 (máscara de 24576 × n_kv escrita por el host en
  memoria managed): en nuestro grafo la máscara densa existe y es la hipótesis añadida a la vía 12.
- HC gate GEMM + mezcla fusionados, 1.19×; streams HC en bf16, 1.08×: vías 4 y 10. Los streams
  en bf16 cambian la numérica y exigirían una compuerta de calidad (PPL).
- Lector PLE directo, 2.75× (233 → 396 t/s aislado): patología de la memoria managed de HIP,
  que hacía fault por página dentro del driver. Nosotros leemos filas de la page cache con
  `posix_madvise` (vía 12); solo el diseño (pool de hilos con `pread`) es candidato si la línea
  de tiempo confirma el input fill.
- Indexador WMMA, compactación MMQ, tuning de FA hd256 y concat traspuesta: neutros en su serie.
  Nuestra concat ya es la traspuesta por tiles (1.1 s por corrida de 40k, 32 µs por dispatch).
- Empates en TOP_K (−1.9 %, corrección): la misma clase de defecto que corregimos en el radix
  top-k el 2026-09-09.

Veredicto (2026-09-15): el 3× requiere ubatch 16384 (8× menos lanzamientos de grafo, costes
fijos por ubatch amortizados 8×, 320 filas por experto en vez de 40) y kernels a eficiencia
WMMA nativa (FA densa hd256, GEMM densas, indexador). Ubatch 16384 está fuera del sobre de
memoria de producción (compute buffer 4.3 GiB a 2048, margen ~22 GiB con producción cargada);
ubatch 4096 es el máximo razonable de probar. En Vulkan la ruta hardware es la misma (coopmat es
WMMA en RDNA3.5), pero alcanzarla es reescribir kernels, no ajustar los que hay, y nadie ha
publicado ese resultado en Vulkan con este modelo: halo-box lo corre en Vulkan sin publicar
prefill y con decode MTP de 15 t/s a profundidad no indicada. El techo de nuestra tabla queda en
~2×: hoy 108 s; central 67 s (~590 t/s) con la tabla anterior y ~64 s (~620 t/s) con la vía 5;
optimista 55-58 s (~700 t/s); frente a 1086.

Migración a HIP: esta máquina no tiene ROCm (sin `/opt/rocm` ni paquetes; build solo Vulkan).
ROCm 10.0 es un cambio de sistema que decide el Director. La pila de producción es Vulkan de
punta a punta: las tres correcciones de determinismo viven en shaders Vulkan, y los caminos QSA,
checkpoints, visión y el presupuesto del timeout de 2 s se revalidarían completos sobre kernels
HIP; sus commits de modelo tocan `qwen4exp.cpp` y chocan con nuestro trabajo de QSA y PLE.
Semanas, con decode hoy 1.5× peor que el nuestro. Decisión (2026-09-15): no se abre. Si se
abriera, el primer paso sería instalar ROCm y una ventana de medición con sus argumentos
exactos, producción descargada.

## 7. Quant nl3s-oproj8: gate y up de expertos en IQ3_S sin recuantizar (2026-09-16)

Origen: el registro del requant de producción (`~/dbg/imatrix/requant.log`, 2026-09-01) muestra que
nl-oproj8 se construyó desde unsloth UD-IQ4_XS, donde gate y up de los expertos vienen en IQ3_S (47
capas) e IQ4_XS (blk.2), y los infló a IQ4_NL con `--allow-requantize`: 42.2 GiB para guardar 32.4 GiB
de información, más un paso de cuantización. La receta nueva (`~/dbg/imatrix/pipeline-nl3s.sh`) es la
misma con `--tensor-type 'blk\.([013-9]|[1-4][0-9])\.ffn_(gate|up)_exps\.=iq3_s'`: llama-quantize copia
sin tocar los tensores que ya tienen el tipo destino, blk.2 sigue a IQ4_NL, el resto es idéntico.
Archivo `NL3S/qwen4exp-nl3s-oproj8.gguf`, 83.5 GiB (93.3 antes); residente con lazy 56.5 GiB (66.5).

Medido en la misma cadena, mismo binario, producción descargada:

| | nl-oproj8 (actual) | nl3s-oproj8 (nuevo) |
|---|---|---|
| PPL holdout, 12 chunks c=4096 | 1.5939 ± 0.0158 | 1.5941 ± 0.0159 |
| MemAvailable / MemFree con carga de producción (NP=2, 262144, MTP, mmproj) | ~24 / ~1 GiB | 38 / 28 GiB |
| Prefill 39.5k con MTP | 104.0 s (380 t/s) | 109.2 s (362 t/s), −4.9 % |
| Re-prefill de 2k a 40k | 5.95 s | 6.05 s |
| Decode turnos 2 / 5 | 38.6 / 36.0 t/s | 38.9 / 33.8 t/s (corridas previas 37.1 / 35.8 y 36.8 / 35.7) |
| Determinismo | — | repeat idéntico a 40k (small batch y re-prefill por lotes), sonda ×5 idéntica |

Coste de IQ3_S por nodo, aislado (test-backend-ops perf, gate/up 640×2560, 512 de 10 expertos):
20 a 30 % más que IQ4_NL a n = 2048 y a n = 3, por cualquier ruta; es la búsqueda en la rejilla de
512 entradas y los signos, no la ruta. Un cargador MMQ de enteros para IQ3_S (78dce736c, retirado,
tests conservados en 00b7dec7d) resultó correcto pero 2.1 % más lento que la ruta coopmat a nivel de
modelo (112.0 frente a 109.7 s); la configuración final es la de antes de ese commit: ruta entera
para los down (IQ4_NL) y coopmat para gate y up (IQ3_S). El coste de prefill se recupera con la vía 1,
que reduce el peso del cargador para cualquier formato.

La medición costó dos incidentes: el freeze #10 (load de la GPU sobre el archivo recién escrito, con
sus páginas sucias sin volcar; reglas en CLAUDE.md) y tres timeouts del anillo de la GPU a las 13:16,
13:19 y 13:21 que coincidieron, con 4 a 9 s de diferencia, con aperturas del dispositivo por otro
proceso del Director (ComfyUI en la misma GPU): los rigs bajo contención se descartaron y el vigilante
de MemFree descargó el rig a 2 GiB libres. Toda medición exige la GPU sola.

Adopción en producción: `quant = nl3s-oproj8` en `strix-halo/config.ini` (decisión del Director), y la
compuerta final con producción cargada: `repeat_probe.py <tag> 5 0 0` y `graph_diff4` con
`GD_MODES=seq_rm GD_SPLIT=2048,631`.

## 8. Ventana de palancas del 2026-09-16 (quant nl3s-oproj8, build 375/376, rigs de 40k con MTP)

| Palanca | Prefill 39.5k | Veredicto |
|---|---|---|
| Base (build 375) | 109.6 / 109.6 s | referencia de la cadena |
| `GGML_VK_DENSE_WAVE32=1` | 109.8 s | neutra (los densos son IQ4_NL) |
| `GGML_VK_MMID_WAVE32=1` | 110.1 s | neutra |
| `GGML_VK_MMID_WG256=1` | 110.9 s | neutra |
| Ubatch 4096 (con MTP) | 119.4 s | −9 %, y los checkpoints pasan a 4096: cada turno reprocesa 4100 tokens |
| Línea de tiempo del host | 109.7 s | vía 12 acotada a 4-7 %; el prefill es GPU |
| Hoisting de ids a 1024 expertos (vía 1, build 376) | 94.7 / 95.0 s | **−13.5 %**, decode igual, determinismo verificado, en producción |
| + `GGML_VK_MMID_SMALLN=1` | 94.4 s | neutro; sonda apagada |

Microbench del nodo gate/up (640×2560, 512 de 10, n=2048): iq4_nl 10.2-11.4 → 8.6 ms, iq3_s 12.4-13.6 →
9.4 ms con el hoisting; los casos de 64 expertos (ya hoisteados) no se mueven. Estado de producción al
cierre: quant nl3s-oproj8, build 376 (7fc9ae43d), prefill 94.7 s a 40k (417 t/s), decode 36-38 t/s,
MemAvailable 36-38 GiB con producción cargada.

## 9. Vía 2 medida: FA sparse token-major en el prefill (2026-09-16, build 378)

Cambio (backend Vulkan, `ggml_vk_flash_attn`, `flash_attn_base.glsl`, `flash_attn_sparse_idx.comp`):
en el modo índice de la FA sparse, un workgroup del prefill toma las 12 cabezas de un token (la
disposición GQA que ya usaba el decode con N ≤ 8) y atiende la lista del propio token, en vez de
un tile de 16 tokens de una cabeza con la unión de sus listas. Las listas pasan a ser por fila de
máscara (`list_rows` = 1; el prepass es el mismo shader con Br = 1) y el shader recibe la altura
de la lista como push constant en lugar de asumir Br. El modo compacto (decode, lotes ≤ 64
filas) no cambia. Compuerta de apagado: `GGML_VK_DISABLE_SPARSE_FA_TOKEN_MAJOR=1`.

Geometría a 40k por capa QSA y ubatch de 2048: antes 24 cabezas × 128 tiles × ~13k celdas
(624k pasos de tile de 64 celdas); ahora 2 cabezas KV × 2048 tokens × 2051 celdas (135k pasos,
4.6× menos) con 12 de las 16 filas coopmat en uso.

Pre-check (perf logger, ubatch 512, modo compacto apagado, sección 4): la FA sparse costaba
0.77-0.98 s por 2048 consultas y 12 capas entre 16k y 37k de profundidad (creciente con la
unión). El brazo `LLAMA_QSA_QUERY_BLOCK=8` dio 2.3-3.6 s: no mide la geometría sino la
ocupación (cada nodo de 8 tokens despacha 16 workgroups con split_k 5 y un prepass por trozos
de dos pasadas), así que no sirve como proxy y la vía se midió con el kernel.

Rigs de 40k con MTP (`mtp_depth.sh`, base con la compuerta de apagado, primero y último):

| Config | Prefill 39.5k | Re-prefill 2k | Decode | Determinismo |
|---|---|---|---|---|
| Base (token-major apagado) | 94.7 / 94.7 s (417 t/s) | 5.31 / 5.32 s | 33-38 t/s | — |
| Token-major | **88.8 s (445 t/s, −6.3 %)** | **4.75 s (−10 %)** | 32-39 t/s (igual) | depth_repeat: runs 2 = 3 y 1 = 5 idénticos en tokens y logprobs |

Correctness: test-backend-ops FLASH_ATTN_EXT 5197/5197 en tres pases (por defecto, modo
compacto apagado y token-major apagado); los casos sparse con GQA 12 y lotes de 128-512 filas a
32k y 131k toman la ruta nueva. Gate (`chain_via2.sh`): graph_diff4 seq_rm 2048,631 con los
mismos 72 nodos SET_ROWS de vistas de caché sobre 12300 que la referencia del 2026-09-09 (benignos)
y repeat_probe sobre producción con una sola distribución en 5 prefills. Estado de producción al
cierre: quant nl3s-oproj8, build 378, prefill 88.8 s a 40k (445 t/s), decode 32-39 t/s,
MemAvailable 36-37 GiB con producción cargada.

La ganancia (5.9 s sobre 12 ubatches en régimen sparse, ~0.5 s por ubatch) es el 60 % de la FA
sparse medida en el pre-check: el resto es el prepass por token (2048 workgroups que recorren la
fila de máscara completa, con una barrera por bloque de 128 columnas, 16× más iteraciones de
barrera que el prepass por tile) y las 4 filas coopmat vacías. Siguiente palanca en esta vía:
construir la lista por token desde los índices del selector QSA en vez de barrer la máscara.

## 10. Vía 8 medida: la ponderación por los pesos del router en el epílogo del MUL_MAT_ID (2026-09-16, build 380)

Perfil fresco del build 378 (perf logger, ubatch 2048, MTP, `~/dbg/merge/prof_nodes.py`): GPU 4.0 s por
ubatch, 80.3 s por prefill de 39.5k sobre 88.8 s de pared. Por familia: MUL_MAT_ID 28.4 s (gate/up
IQ3_S 19.3, down IQ4_NL 8.4), MUL_MAT 22.4, FA 9.7, MUL 3.05, RMS_NORM_MUL 2.5, HC_GATED_MEAN 1.9,
HC_INJECT 1.6, MULTI_ADD 1.6. El MUL de 3.05 s es `ffn_moe_weighted`: `experts` [2560, 10, 2048] ×
`weights` [1, 10, 2048] (420 MB de lectura+escritura por capa, kernel `mul` con broadcast a 137 GB/s).
La fusión MUL_MAT_ID+MUL del backend existía solo en el kernel mat-vec (n ≤ 8) y exigía un solo token
(`scale->ne[2] == 1`), así que el prefill y el lote de verificación del MTP corrían sin fusión.

Cambio: los kernels de tiles (`mul_mm.comp` coopmat y escalar, `mul_mmq.comp`) multiplican cada fila
(slot de experto, token) por `scale[token * nei0 + slot]` al escribir (binding 6, flag en el push
constant); el host pasa el tensor del MUL como salida del MUL_MAT_ID y su `src[1]` como escala; la
condición de fusión acepta la ruta de tiles (no en coopmat2, que no tiene el epílogo) y escalas
[1, n_used, n_tokens]; el mat-vec indexa la escala por token (`expert_i1 * nei0 + slot`), con lo que
el lote de verificación (3 tokens) también fusiona. El producto se hace en f32 sobre el mismo
acumulador que antes se guardaba y volvía a leer: **salida bit-idéntica** (misma aceptación del draft
139/230 en base y fusionado). Compuerta de apagado: `GGML_VK_DISABLE_MMID_SCALE_EPILOGUE=1`.

Rigs de 40k con MTP (base con la compuerta, primero y último):

| Config | Prefill 39.5k | Re-prefill 2k | Decode | Determinismo |
|---|---|---|---|---|
| Base (epílogo apagado) | 89.0 / 89.1 s | 4.81 / 4.81 s | 33-38 t/s | — |
| Epílogo | **86.8 s (455 t/s, −2.4 %)** | **4.68 s** | 33-38 t/s | depth_repeat idéntico (2 = 3, 1 = 5) |

Correctness: test-backend-ops MUL_MAT_ID_FUSION 23/23 con y sin epílogo (casos nuevos: down IQ4_NL
n = 2048 y 100, gate/up IQ3_S, f16), MUL_MAT_ID 935/935. La ganancia (2.2 s) es menor que los 3.05 s
del MUL porque parte del MUL solapaba con otros nodos por la concurrencia del optimizador de grafo.
La suma sobre los 10 expertos (MULTI_ADD, 1.6 s) queda: hacerla en el kernel exige atómicos entre
workgroups (orden por planificación, rompe la identidad entre peticiones) o un tiling por token que
destruye el esquema expert-major de `row_ids`; descartada.

Gate: graph_diff4 seq_rm 2048,631 con los mismos 72 nodos SET_ROWS que la referencia; repeat_probe
sobre producción con una sola distribución en 5 prefills. Incidente de la ventana: el harness de la
sesión mató las cadenas en segundo plano durante la recarga de producción ("memoria baja", con
MemAvailable en 40 GiB, lo normal de toda carga) y la carga murió con ellas; desde entonces las
recargas de producción se lanzan con `setsid` fuera del árbol de procesos de la cadena. Estado de
producción al cierre: build 380, prefill 86.8 s a 40k (455 t/s).

## 11. Vías 4, 5 y 11 medidas en una ventana (2026-09-16, build 382)

Reconocimiento (mezcladores hyper-connection, `build_hc_mix`): (a) `hc_*_down` [10240, 320] corre el
kernel grande alineado con 48 workgroups y sin split_k (los umbrales de `ggml_vk_guess_split_k` están
escritos para GPUs con ≥ 72 CUs); (b) `hc_*_up` [320, 10240] corre el kernel grande **sin alineación**
porque k = 320 no es múltiplo de 128 (bounds checks en cada carga); (c) `hc_*_inject` [10240, 4] corre un
tile de 32×32 con 4 filas válidas en 64 workgroups: latencia, no cómputo (1.4 ms por nodo, 0.11 TFLOPS;
en f16 tarda lo mismo). Los tres son IQ4_NL en el quant (el inject era f32 en el checkpoint).

Pre-check (test-backend-ops perf, formas reales, n = 2048): el kernel f16 es más lento que los
cuantizados en todas las formas grandes (q5_K 5.23 → f16 6.84 ms, iq4_nl m=2560 k=6144 2.92 → 4.32,
q8_0 3.11 → 4.40): leer 2.9-3.6× más bytes de A cuesta más que el desquantizado en línea. Con la sombra
(desquantizar A a f16 y correr el kernel f16) los nodos cuantizados suben 20-110 %. IQ4_NL con B en f16:
los mezcladores ganan (a −10 %, b −6 %, c −37 %) y los nodos grandes pierden 2-3 %.

Rigs de 40k con MTP (base primero y último, 86.7-87.1 s):

| Palanca | Prefill 39.5k | Re-prefill 2k | Decode | Veredicto |
|---|---|---|---|---|
| Base (build 382 = vía 8) | 87.1 / 86.7 s | 4.70 / 4.70 s | 33-38 t/s | referencia |
| split_k 2 para < 2 rondas de tiles (`GGML_VK_MM_SPLITK_SMALL`) | 91.0 s | 4.89 s | igual | negativo (reducción y parciales), eliminado |
| split_k 3 | 88.6 s | 4.78 s | igual | negativo, eliminado |
| **Tile medio alineado cuando k es múltiplo de 64 y no de 128** | **85.8 s (−1.3 %)** | 4.65 s | igual, misma salida | **adoptado por defecto** |
| Sombra f16 hasta 32 MiB (`GGML_VK_DENSE_SHADOW_F16`) | 90.7 s | 4.92 s | igual | negativo, eliminado |
| Sombra f16 hasta 64 MiB | 93.4 s | 5.03 s | igual | negativo, eliminado |
| IQ4_NL en la lista f16-B (vía 11) | 86.7 s | 4.72 s | igual | neutro, eliminado |
| **`hc_*_inject` fusionado en `hc_*_down` (archivo `nl3s-hcdi-oproj8`)** | **82.4 s (480 t/s, −5.2 %)** | 4.71 s | **38-41 t/s (+15 %)** | **adoptado**; depth_repeat idéntico |

La fusión de (c): el archivo lleva `blk.N.hc_{attn,ffn}_down_inject.weight` [10240, 324] IQ4_NL, las
filas de `down` seguidas de las de `inject` (concatenación de bytes, sin recuantizar; script
`scripts/qwen4exp-merge-hc-inject.py`, que copia los datos dentro del kernel y suelta la caché cada
2 GiB: 6.5 min, MemFree estable). El loader acepta ambos formatos (`create_tensor_hc_down_inject`) y el
grafo hace un matmul de 324 filas y dos vistas (`lo` contiguo para scale+silu, `inject` con stride para
HC_INJECT). Desaparecen 96 matmuls de 4 filas por ubatch y, en decode, 96 mat-vec por token: de ahí el
+15 % de decode. El valor de `inject` cambia de redondeo (otro kernel) y la salida greedy diverge en
algún punto respecto del archivo anterior (aceptación del draft 153/202 frente a 139/230), pero es
idéntica entre peticiones (depth_repeat, graph_diff4 y repeat_probe en la cadena de adopción).

Intento de fusionar (c) sin cambiar el archivo: no hay nada barato; la forma [4, n] no tiene ruta
eficiente en el backend (mat-vec necesita n ≤ 8 columnas, aquí n = 2048) y transponer requería el
inject en f32.

Estado de producción al cierre (build 382 con el tile alineado y el archivo `qwen4exp-nl3s-hcdi-oproj8.gguf`,
`quant = nl3s-hcdi-oproj8` en config.ini): prefill 80.8 s a 40k (489 t/s; 109.6 s y 361 t/s al empezar el
día), re-prefill de 2k 4.68 s, decode 38-41 t/s, MemAvailable 36 GiB con producción cargada.
