# qwen4exp en Strix Halo: dónde va el tiempo y vías de optimización (2026-09-09, actualizado 2026-09-16)

## Estado actual (2026-09-23)

- Árbol: master con el hint del server, la conv sin concat, el guard de columnas y BK_STEP 2 del MMQ (sección 20, build 410), las listas de la FA sparse desde la selección QSA (sección 21, build 413), la carga IQ3_S de 16 valores por hilo (sección 22, build 415), los tiles de expertos que saltan las columnas vacías (sección 23, build 416) y los quick wins de la sección 26: la marca del caché QSA agrupado por rango y el orden por counting sort (decode tras una imagen +12 %), y `--decode-share 0.4` con `--no-cache-idle-slots` para varios agentes a la vez.
  Producción: `strix load qwen38flash`
  (NP=2, `-c 524288` = 262144 × 2 slots, batch = ubatch 2048, KV q8_0, lazy mode auto, draft MTP IQ4_NL
  n-max 2, mmproj con el encoder de visión en la GPU desde el 2026-09-22 (sección 24), `--ctx-checkpoints 8`,
  `--decode-share 0.4`, `--no-cache-idle-slots`),
  archivo `NL3S/qwen4exp-nl3s-hcdi-gu-oproj8.gguf` (`quant = nl3s-hcdi-gu-oproj8`).
- Rendimiento a 40k de profundidad (rig `mtp_depth.sh`, MTP on): prefill de 39.5k **70.4 s (561 t/s)**,
  re-prefill de 2k 4.11-4.20 s, decode 35-41 t/s (64 ms por paso de 2.46 tokens, sección 24). Turno con una
  imagen de 448×448 en producción 4.1 s (antes 30.9). Residentes 56.5 GiB; MemAvailable ~35 GiB con producción
  cargada. Determinismo verificado (repeat_probe, graph_diff4, depth_repeat) tras cada cambio.
- Lista de trabajo vigente: sección 13 (prefill) y sección 25 (encoder, imágenes, ejecución paralela, con la
  ventana v21 del 2026-09-23). Desglose por nodo vigente: sección 2. Las secciones 5-12 son el
  registro de cada ventana de medición, en orden cronológico.

Cronología del prefill de 39.5k a 40k (misma máquina, mismo rig):

| Paso | Prefill | t/s | Cambio |
|---|---|---|---|
| 2026-09-09 | 108 s | 366 | base de este documento |
| 2026-09-16 mañana | 109.6 s | 361 | build 375 (referencia de la ventana de palancas) |
| eh_proj como matmul (bbded2ebe) | 104.0 s | 380 | −3.2 % |
| quant nl3s-oproj8 (gate/up IQ3_S, −10 GiB) | 109.2 s | 362 | −4.9 % de prefill a cambio de la memoria |
| hoisting a 1024 expertos (7fc9ae43d) | 94.7 s | 417 | −13.5 % |
| FA sparse token-major (4ac37b4a4) | 88.8 s | 445 | −6.3 % |
| epílogo de escala del MUL_MAT_ID (cad680f62) | 86.8 s | 455 | −2.4 % |
| tile medio alineado (6459c5bb9) | 85.8 s | 461 | −1.3 % |
| hc inject fusionado en down (c12f6777c, archivo hcdi) | 80.8 s | 489 | −5.8 %, decode +15 % |
| 2026-09-17: cherry-picks de upstream (build 396) | 81.5 s | 485 | correcciones, sin efecto |
| gate y up fusionados (archivo hcdi-gu, 2026-09-18) | **80.9 s** | **488** | −0.8 %, re-prefill −2.8 % |
| vía 12 medida (2026-09-18, sección 16) | 80.9 s | 488 | cerrada sin cambio: el input PLE es E/S |
| lote de riesgo bajo (2026-09-22, sección 18) | 80.4-80.5 s | 491 | sin cambio |
| draft MTP recortado a las filas de salida (2026-09-22, sección 19) | **76.8 s** | **515** | −4.6 %, re-prefill −6.8 % |
| hint del server para el PLE + conv sin concat + guard y BK_STEP 2 del MMQ (2026-09-22, sección 20) | **73.3 s** | **539** | −5.0 %: el hint −3.3 s, las otras ~−0.6 s |
| listas de la FA sparse desde la selección QSA, vía 2 bis (2026-09-22, sección 21) | 73.2 s | 539 | −0.2 s (−0.3 %); −9 % del nodo FA a 131k |
| carga IQ3_S de 16 valores por hilo en el coopmat (2026-09-22, sección 22) | **72.9 s** | **542** | −0.6 s (−0.9 %), bit-idéntica |
| tiles de expertos: sub-tiles de columnas vacías saltados (2026-09-22, sección 23) | **70.4 s** | **561** | −2.0 s (−2.8 %), bit-idéntica |

Lo que sigue es el documento original del 2026-09-09 con sus secciones anotadas; el desglose de
la sección 2 está reemplazado por el perfil del build actual.

Fuentes de datos (2026-09-09):

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

## 1. Decode a 40k con MTP (perfil del 2026-09-16 noche, build 382+, archivo hcdi)

**Corrección (2026-09-22, sección 24.1):** la tabla de esta sección promedia el grafo de verificación del target
con el grafo del hook del draft, que también corre sobre 3 tokens. La verificación cuesta 45-50 ms, no 25.7, y el
paso completo 64 ms, no 39.3; con 2.46 tokens por paso eso da los 38 t/s medidos. Las proporciones entre nodos
de la tabla siguen siendo útiles; los milisegundos absolutos, no.

Perf logger a 40k (`~/dbg/depth/strix-decode-prof-prof383.log`, `GGML_VK_PERF_LOGGER_FREQUENCY=1`,
que serializa cada nodo: sus sumas son cotas superiores). Un paso = un grafo de verificación del
target (3 tokens) más los grafos del draft (2 tokens): 39.3 ms de GPU por paso, ~1.5 tokens
aceptados por paso → 38-41 t/s medidos en producción.

| Bloque (grafo de verificación, 3 tokens) | ms/paso | Observación |
|---|---|---|
| Mat-vecs densos | 9.6 | wqkv q5_K 1.52, router f32 1.32 (NUM_ROWS=1, 96 GB/s), LM head q6_K 1.16, ssm_out 0.88, hc down+inject 0.86, z 0.83, hc up 0.77, wq 0.56, wo 0.49 |
| Expertos gate/up IQ3_S (`batch=3`) | 5.4 | 48 dispatches; el nodo más caro también en decode |
| Expertos down IQ4_NL con la escala fusionada | 2.8 | epílogo de escala activo en el lote de verificación |
| FA sparse (12 capas, modo compacto) | 2.3 | 0.5-0.56 ms por capa a 40k |
| GET_ROWS + CPY (estado GDN, rollback MTP) | 1.2 | |
| TOPK_QSA | 0.5 | lineal con n_kv |
| RMS_NORM_MUL, GDN, CONT, SCALE, ADD, SIGMOID, HC_INJECT, ARGSORT | ~2.0 | ~2500 dispatches pequeños |
| **Total verificación** | **25.7** | |
| Grafos del draft (2 tokens) | 13.6 | expertos 4.8 (mat-vec-id), densos 4.4, FA densa 1.65 (crece con n_kv), resto 2.7 |
| **Total GPU por paso** | **39.3** | |

Cambios respecto del 2026-09-09 (66 ms por paso, 36 t/s): el mat-vec `m=4` del inject desapareció
(archivo hcdi), el MUL de la ponderación se fusionó y el quant nl3s bajó los bytes de expertos.
Observación nueva: los grafos del draft cuestan 13.6 ms por paso, el 35 % del total; su expertos
(4.8 ms para un bloque) y su FA densa a 40k son las palancas de decode que quedan sin explorar
(vía 16 y la cabeza recortada, vía 9).

## 2. Prefill de 39.5k tokens: 80.8 s de pared, 20 ubatches de 2048 (perfil del 2026-09-16 noche)

Mismo perfil. GPU 3.63 s por ubatch = 72.6 s por prefill; los ~8 s restantes (10 %) son host y
huecos entre submisiones (línea de tiempo de la sección 8: set_inputs 177 ms, alloc 40, fuera de
`llama_decode` 204 ms por ciclo). Por familia y por nodo, en segundos por prefill de 39.5k:

| Bloque | s | % de la GPU |
|---|---|---|
| Expertos gate/up IQ3_S (`MUL_MAT_ID`, 94 dispatches por ubatch, 9.5 ms cada uno, 6.4 TFLOPS) | 17.8 | 24.5 |
| Expertos down IQ4_NL con la escala fusionada (`MUL_MAT_ID_MUL`, 47 × 9.1 ms, 7.3 TFLOPS) | 8.5 | 11.7 |
| Densos `MUL_MAT` | 18.3 | 25.2 |
| · wqkv q5_K `m=10240 k=2560` 3.31 (23 TFLOPS), hc down+inject `m=324 k=10240` 3.16 (8 TFLOPS), ssm_out `m=2560 k=6144` 2.58 (18), z `m=6144` 2.01 (23), hc up `m=10240 k=320` 1.53 (tile medio alineado), wq `m=12288` 1.43, shexp 0.84, wo q8_0 0.75, router f32 0.4 | | |
| Flash attention (sparse del target ~7, densa del draft ~2.7) | 9.8 | 13.4 |
| Elementwise sobre el residual ancho: RMS_NORM_MUL 2.52, HC_GATED_MEAN 1.88, HC_INJECT 1.61, MULTI_ADD 1.44, MUL 0.96, CONCAT 0.93, CONT 0.85, ADD 0.79, GLU 0.66, SSM_CONV 0.64 | ~12.3 | 17 |
| GDN (36 × 2 ms) | 1.45 | 2.0 |
| Selección QSA (TOPK_QSA 1.32 + RELU, TOP_K, FILL) | ~1.7 | 2.3 |
| eh_proj del draft (matmul `m=2560 n=8192 k=5120`) y resto del draft | ~1.5 | 2 |
| **GPU** | **72.6** | 100 |
| Host y huecos | ~8 | |

Lectura: los expertos son el 37 % de la GPU y gate/up en IQ3_S el nodo más caro (sección 12: en
IQ4_NL sería un 12 % más rápido, descartado por memoria; Q3_K un 17 % más lento). Los densos van
a 18-23 TFLOPS salvo el mezclador `m=324` (8 TFLOPS, 48 workgroups, sin split_k rentable). La FA
sparse ya corre token-major; su prepass barre la máscara entera por token (vía 2 bis). El
elementwise sobre el residual de 84 MB suma 12 s: fusiones pendientes (fila 6 de la sección 13).

Perfil anterior (2026-09-09, 108 s): GPU 80.7 s, densos 22.5, expertos 20.0, FA 15.8, elementwise
13, `eh_proj` 2.5, no GPU 27; input fill 140-250 ms por ubatch. La medición previa del tamaño de
ubatch (4.7k, 2026-09-09): ub512 326 t/s, ub2048 425, ub8192 348; con MTP a 40k ub4096 dio −9 %.

## 3. Vías de optimización, ordenadas de mejor a peor ganancia estimada (tabla original, estados al 2026-09-16)

Cada fila lleva en negrita su estado cuando se midió; la lista de pendientes ordenada y vigente es
la sección 13. Regla (2026-09-10, tras el punto retirado de la sección 5): ninguna vía se implementa sin la
comprobación previa de su columna, un experimento acotado que aísle el cuello de botella con
herramientas que ya existen (test-backend-ops perf, variables de entorno, perf logger). La
ganancia es un porcentaje de la fase a 40k: del prefill de 108 s o del paso de decode de 66 ms.
"Evidencia" dice de dónde sale la estimación; "riesgo" es la probabilidad de que la ganancia no
exista aunque el cambio sea correcto.

| # | Vía | Fase | Ganancia estimada | Evidencia | Riesgo | Comprobación previa |
|---|---|---|---|---|---|---|
| 1 | MUL_MAT_ID del prefill: tile por filas por experto (mediana 21, BN=128), hoisting de ids para 512 expertos, cargas de B fuera de `_ne1`, BK_STEP=1. **Hoisting hecho y medido (7fc9ae43d, 2026-09-16): prefill 109.6 → 94.7 s (−13.5 %), re-prefill 6.05 → 5.30 s, decode igual. Columnas vacías del tile saltadas (2026-09-22, sección 23): −2.0 s.** El tile por filas por experto (`GGML_VK_MMID_SMALLN=1`) resultó neutro (94.4 s) y queda como sonda apagada; ubatch 4096 pierde un 9 % y deja los checkpoints a 4096. Pendientes: cargas de B fuera de `_ne1`, BK_STEP | prefill | −13.5 % medido (hoisting); resto sin estimar | Medida directa: 4.6 y 10 TFLOPS frente a 22 en los densos; causas leídas en `mul_mmq.comp`; externa, misma GPU: halo-box selecciona el tile por filas esperadas por experto (`GGML_VK_MMID_SMALLN` con `M128` y `BM64`, sobre un prepass de listas de filas) y midió +16.3 % pp512 en Qwen3.6-35B-A3B (~32 filas por experto), MUL_MAT_ID q5_K de 2.33 a 4.43 TFLOPS; es la implementación de referencia (auditoría, sección 10). Ruta entera (MMQ q8_1) frente a coopmat, medido a nivel de modelo el 2026-09-16 con `GGML_VK_DISABLE_COOPMAT_MMQ=1`: para IQ4_NL la entera gana 2.4 % de prefill (104.0 frente a 106.5 s); para IQ3_S la coopmat gana 2.1 % (109.7 frente a 112.0 s), y en el microbench aislado la coopmat gana en ambos tipos: el +3.7 % de b9c196c1c se sostiene solo en el grafo real | Medio: otro límite (L2, ocupación) puede aparecer al llenar los tiles | test-backend-ops perf con 64 expertos (320 filas por experto): si alcanza 15-20 TFLOPS, el tile es la causa; ubatch 4096 en el rig con MTP (80 filas por experto; compute buffer 4.3 GiB a 2048, verificar el tamaño con `-lv 4` antes de cargar) |
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

## 4. Mediciones previas pendientes (estado al 2026-09-16)

Hechas: la línea de tiempo del host (sección 8: el prefill es GPU, host 4-7 %), la FA sparse por
token (sección 9), el perf de las formas de expertos (secciones 8 y 12) y de los mezcladores
(sección 11), las compuertas compiladas (sección 8, neutras), la sombra f16 (sección 11,
negativa) y ubatch 4096 (sección 8, −9 %). Quedan:

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

## 12. Los expertos gate/up en IQ3_S: alternativas de tipo medidas (2026-09-16, microbench)

El nodo gate/up (512 expertos, 10 activos, m=640 k=2560, n=2048) es el más caro del prefill (19.3 s de
80.8). Volverlo a IQ4_NL costaría los 10 GiB que ahorra IQ3_S y queda descartado por orden del Director.
test-backend-ops perf, tres pases (el primero descartado: compila pipelines dentro de la medición):

| Tipo | Ruta | ms por nodo |
|---|---|---|
| IQ4_NL | integer-dot | 8.0-8.6 |
| IQ3_S (producción) | coopmat | 9.0-9.2 |
| Q3_K (mismo tamaño, 3.44 bpw) | integer-dot | 10.5-10.8 |
| Q3_K | coopmat | 10.5 |

Q3_K descartado: 17 % más lento que IQ3_S y habría exigido recuantizar desde el UD-IQ4_XS (no hay BF16
local). Quedan, sin tocar la memoria: gate y up en un solo tensor (vía 18, −1 a −2 s) y el desquantizado
IQ3_S del kernel coopmat (−2 a −3 s, incierto).

## 13. Lista de trabajo del prefill (vigente, 2026-09-16 noche)

Prefill de 39.5k a 40k: 80.8 s. Pendientes ordenados por ganancia esperada por unidad de riesgo;
la ganancia es sobre esos 80.8 s. Ninguna se implementa sin su comprobación previa y sin la
cadena de medición completa (base primero y último, MTP on, determinismo).

| # | Palanca | Ganancia estimada | Riesgo / coste | Comprobación previa |
|---|---|---|---|---|
| 1 | Vía 18: gate y up de expertos en un solo tensor `ffn_gate_up_exps`. **Hecha (2026-09-18, sección 15): −0.6 s de prefill, −2.8 % de re-prefill, PPL idéntica; archivo `nl3s-hcdi-gu-oproj8` en producción** | −1 a −2 s | bajo | ninguno |
| 2 | Vía 2 bis: lista por token de la FA sparse construida desde los índices del selector QSA (hoy el prepass barre la fila de máscara entera, 40k columnas por token) y las 4 filas coopmat vacías. **Hecha la lista desde los índices (2026-09-22, sección 21): −0.4 s a 40k, −0.2 s a 55k, −9 % del nodo a 131k; el barrido era ~1 % del nodo, no el 40 % que suponía la sección 9: lo que queda de la FA sparse son los gathers de K/V por token. Las 4 filas vacías no cambian ese tráfico** | — | — | — |
| 3 | Desquantizado IQ3_S en el kernel coopmat de MUL_MAT_ID (recuento de instrucciones, cargas de 16 bits, tabla en shmem). **Hecha la anchura de carga (2026-09-22, sección 22): 16 valores por hilo, −4 a −7 % del nodo, −0.6 s en el modelo, bit-idéntica. Lo que queda del nodo es el MMA desperdiciado del tile (128 columnas para 40 filas por experto), no el desquantizado** | ≤ −0.5 s más | medio | microbench |
| 4 | Vía 12: host (`prefetch_ple_rows`, 20k llamadas síncronas por ubatch; huecos entre submisiones, 204 ms por ciclo). **Cerrada (2026-09-18, sección 16): el input PLE son 130-190 ms de E/S aleatoria por ubatch; agrupar o repartir el encolado no cambia el total, batch 8192 pierde decode. El read-ahead entre decodes por hint del server está hecho (2026-09-22, sección 20): −3.3 s, set_inputs 157 → 27 ms por ubatch** | — | — | — |
| 5 | Vía 16: el draft reutiliza la selección QSA del target (sin indexador en el draft). **Re-valorada (2026-09-23, sección 25.4): la fila 9 dejó la atención densa del draft solo en las filas de salida (draft 232 → 64 ms por ubatch), así que en el prefill ya no queda casi nada; sigue como palanca de decode (FA densa del draft, ~1.3 ms por paso a 40k)** | ~0 en prefill | medio-alto: flujo del MTP | diseño |
| 6 | Elementwise sobre el residual ancho (RMS_NORM_MUL 2.5 s, HC_GATED_MEAN 1.9, CONT 0.85, ADD 0.8): fusiones adicionales. **Rebajada (2026-09-22, sección 20): quitar el CONCAT entero (0.93 s en el perfil serializado) dio −0.13 s en el modelo; el backend solapa estos nodos con las matmuls, así que el perfil serializado sobreestima su costo real varias veces** | ≤ −0.5 s | medio | perfil por nodo (sección 2) |
| 7 | Vía 1 bis: expertos `down` (k=640, ~9 ms por dispatch, 7.3 TFLOPS): cargas de B fuera del bucle de filas, BK_STEP. **Hecha en parte (2026-09-22, sección 20): ceros en las columnas sin fila del tile, −3.8 % en el nodo MMQ (−0.3 s); BK_STEP=2 medido en el microbench (sección 20)** | — | — | — |
| 8 | Vías menores 19-23: QSA por bloques, ssm_conv (= fila 11), mat-muls diminutos (techo 0.5 s repartido en 2-3 arreglos, sección 18), MMVQ IQ4_NL, GDN chunked | −0.5 a −1 s cada una | bajo-medio | varias |
| 9 | Draft MTP en prefill recortado a n_outputs (sección 17.1). **Hecha (2026-09-22, sección 19): −3.7 s de prefill (−4.6 %), re-prefill −6.8 %, exacta; crece con la profundidad** | — | — | — |
| 10 | Tile medio para la matmul alineada de pocos workgroups (sección 17.2). **Cerrada (2026-09-22, sección 18): igual en el nodo aislado, −1 % en el modelo y no bit-idéntica; el nodo no está limitado por ocupación** | — | — | — |
| 11 | `ggml_ssm_conv` con un src de historial y x con sus strides (sección 17.3; es la vía 20 de la lista original): hoy `ggml_concat(state, transpose(x))` materializa 84 MB por capa GDN por ubatch solo para el layout. **Hecha (2026-09-22, sección 20): `ggml_ssm_conv_state`, −0.13 s en el modelo (el concat solapaba con las matmuls), 108 nodos y 117 GB de tráfico menos por prefill, exacta** | — | — | — |
| 12 | RMS norm agrupada como un op (sección 17.4). **Cerrada (2026-09-22, sección 18): el nodo ya corre a 210 GB/s, la premisa de 123 era un error de reparto; un kernel por subgrupo probado y neutro** | — | — | — |
| 13 | Host entre batches del prompt (sección 17.5). **Hecha (2026-09-22, sección 18): −7 ms por batch (0.2 %); las 2048 sincronizaciones costaban 7 ms, no 60** | — | — | — |
| 14 | Camino split de la QSA: concats encadenados. **Cerrada por aritmética (2026-09-22, sección 18): 0.1 % incluso a 262k** | — | — | — |
| 15 | Indexador QSA: relu, suma de las 4 cabezas y bias en un solo kernel (hoy RELU + CONT + 3 ADD + ADD de bias sobre [n_blocks, 4, 2048] f32 por capa QSA y ubatch) (sección 24.3) | −0.7 s serializado a 40k, cuadrático con la profundidad; real una fracción | bajo-medio: un kernel o un patrón de fusión, exacto si suma en el mismo orden | perfil por nodo con formas |
| 16 | Máscara QSA sin construir: con `kv_idx` la FA puede leer la máscara base en las celdas candidatas (FILL + SET_ROWS + ADD sobre [n_kv, 2048] por capa) (sección 24.3) | −0.4 s serializado a 40k, cuadrático | medio: cambia la semántica del op; la CPU y el modo denso de Vulkan deben honrar `kv_idx` | diseño |
| 17 | Fusiones de pesos de igual tipo y entrada: z + alfa + beta (iq4_nl, m=6240), gate y up del shexp (iq4_nl, m=1280), router + gate del shexp (f32, m=513) (sección 24.3) | −0.5 s en prefill, −0.5 ms por paso de decode | medio: dos exigen transformar el GGUF (decisión del Director, procedimiento del freeze #10); m=513 no está alineado y puede empeorar el prefill del router | ninguna |

Suma realista de 2 y 3: ~4-6 s (73.7 → ~68-70 s, ~570 t/s). Techo con todo: ~62-64 s. El lote de riesgo bajo
(filas 10, 12, 13, 14 y vía 21) se midió el 2026-09-22 y no mueve el prefill (sección 18); la fila 9 está hecha
(sección 19); el hint del server (fila 4), la fila 11 y parte de la 7 están hechas (sección 20).

Cerradas (no volver sin un dato nuevo):

- IQ4_NL para gate/up: +10 GiB residentes; descartado por orden del Director (2026-09-16).
- Q3_K para gate/up: 17 % más lento que IQ3_S (sección 12).
- Sombra f16 por nodo (vía 5): +4 a +7 % (sección 11).
- split_k con menos de 80 tiles: +2 a +4.5 % (sección 11).
- IQ4_NL en la lista f16-B (vía 11): neutro (sección 11).
- Loader integer-dot para IQ3_S: correcto, −2.1 % (sección 7).
- Tile por filas por experto (`GGML_VK_MMID_SMALLN`), `DENSE_WAVE32`, `MMID_WAVE32`, `MMID_WG256`: neutros (sección 8).
- Ubatch 4096: −9 % y checkpoints de 4096 (sección 8).
- `LLAMA_QSA_QUERY_BLOCK=8` como proxy de la FA por token: mide ocupación, no geometría (sección 9).
- Suma sobre expertos dentro del MUL_MAT_ID: atómicos u orden por planificación, rompe el determinismo (sección 10).
- Vía 12, encolado del prefetch PLE: `process_madvise` por lotes, reparto en 4/8 hilos, batch 8192 con
  read-ahead síncrono o en hilo: el total por ubatch no se mueve, es E/S (sección 16).

Decode (38-41 t/s a 40k; 39.3 ms de GPU por paso, sección 1), pendientes por interés:

| # | Palanca | Ganancia estimada | Nota |
|---|---|---|---|
| D1 | Grafos del draft: 13.6 ms por paso (35 %), con 4.8 ms de expertos por mat-vec-id para un solo bloque y una FA densa que crece con n_kv; vías 9 (cabeza recortada) y 16 (selección QSA compartida) | −3 a −5 ms por paso | hallazgo nuevo del perfil del 2026-09-16; primero un perfil del draft por nodo |
| D2 | Vía 3: forma del workgroup del mat-vec de expertos (gate/up IQ3_S 5.4 ms + down 2.8 por paso) | −2 a −3 ms | microbench de `MUL_MAT_ID` a n=3 por variante |
| D3 | Vía 13: router f32 con NUM_ROWS=1 (1.32 ms por paso a 96 GB/s) | −0.7 ms | una línea en la tabla de NUM_ROWS |
| D4 | Vía 14: estado GDN in place (GET_ROWS + CPY 1.2 ms por paso) | −0.8 ms | |
| D5 | Vía 7: split_k en el modo compacto de la FA sparse (2.3 ms por paso, 6 workgroups por capa) | −1 ms | |
| D6 | Vía 17: menos dispatches pequeños (~2500 por paso, ~2 ms) | −0.5 a −1 ms | fusiones |
| D7 | Agrupación QSA en CPU incremental: `set_input_qsa` es O(n_kv) por ubatch y en decode corre con la GPU parada (sección 24.3) | −1 ms por paso a 40k, crece con n_kv | el set_inputs del grafo de verificación mide 1.8 ms |
| D8 | Hueco de host de ~2 ms entre el hook del draft y el primer paso del draft (sección 24.3) | hasta −2 ms | sin atribuir; medir con `LLAMA_SPEC_TIMING` |

Re-valoración de D1-D6 con el perfil corregido (sección 24.3): expertos IQ3_S 10.8 ms por paso a ~172 GB/s (D2),
mezcladores hc 3.2 ms a 104-124 GB/s (vía 10), router f32 2.5 ms a 99 GB/s (D3), estado GDN 2.5 ms (D4), FA del
target 3.8 ms (D5), cabeza del draft 3.2 ms (vía 9, aplazada).

## 14. Los cuatro frentes: estado y orden propuesto (2026-09-17)

| Frente | Estado | Medido | Ganancia realista | Primer paso |
|---|---|---|---|---|
| Prefill | 80.8 s a 40k (489 t/s) | sí (secciones 2, 13) | −8 a −10 s con las filas 1-4 de la sección 13 (~550 t/s); techo ~68 s | vía 18 (gate/up en un tensor) |
| Decode | 38-41 t/s a 40k; 39.3 ms de GPU por paso | sí (sección 1) | −6 a −9 ms por paso (46-48 t/s): los grafos del draft son el 35 % del paso y sus expertos cuestan 4.8 ms para un bloque frente a 8.5 ms de los 48 del target | perfil por nodo del draft (`GGML_VK_PERF_LOGGER` sobre los grafos pequeños) |
| Imágenes | **resuelto (2026-09-22, sección 24.2): el encoder corría en CPU por un flag del launcher. En la GPU: 448×448 30.9 → 3.4 s, captura 1920×1080 103 → 7.7 s, 2048×2048 (tope de 4096 tokens) 381 → 14.6 s, misma memoria, mismas respuestas** | sí | — | queda: los tokens de imagen van al ritmo del texto; el encoder (27-47 % del prompt) y el decode posterior a una imagen (+13 ms por paso a 40k por el caché QSA agrupado, ~+21 % con su arreglo) tienen palancas medidas en la sección 25 |
| Ejecución paralela (NP=2) | **medida (2026-09-23, sección 25.3): dos agentes generando a 40k dan 19.6-20.0 t/s cada uno, 38.6 en total (uno solo: 33.7); el MTP sigue ganando con dos streams; un agente que genera cae a 0.6 t/s mientras el otro procesa un prompt** | sí | el reparto del paso entre prompt y decode (C1) | decisión del Director sobre el reparto |

Prefill de 76.5k tokens en producción (log del Director, 2026-09-16, NP=2, con cortes de imagen):

| Tramo | Tiempo por ubatch de 2048 | t/s marginal |
|---|---|---|
| 0-8k | 4.0-4.3 s | ~500 |
| 20-40k | 4.2-4.5 s | ~470 |
| 60-76k | 4.9-5.1 s | ~410 |
| Acumulado a 40k | 83.8 s | 480 (rig NP=1: 80.8 s, 489) |
| Acumulado a 76.5k | 172 s | 444 |

La caída con la profundidad (~20 % de 0 a 76k) es el término lineal con n_kv: la atención densa del
draft (un bloque sin QSA sobre todo el contexto), el prepass de la FA sparse (barre la fila de máscara
completa por token) y el top-k del selector QSA. A 40k pesan poco; a 128k dominan (2026-09-03: 126k a
241 t/s; el ritmo de hoy extrapola a ~410-420). Son las filas 2 y 5 de la sección 13 y D1. Los
ubatches cortos de los cortes de imagen pagan el coste fijo por ubatch sin amortizarlo.

Orden propuesto con la GPU libre: dos ventanas cortas de solo lectura (imágenes, paralelo) para saber
si hay problema, y en paralelo la vía 18 y el perfil del draft.

## 15. Vía 18 medida: gate y up de los expertos en un solo tensor (2026-09-18, build 396)

`scripts/qwen4exp-merge-gate-up-exps.py` escribe `blk.N.ffn_gate_up_exps.weight` [2560, 1280, 512] con, por
experto, las filas de gate seguidas de las de up (concatenación de bytes por experto, sin recuantizar;
la disposición que `build_moe_ffn` parte en las dos vistas). El loader ya aceptaba ambos formatos
(`create_tensor_gate_up_exps`). Archivo `NL3S/qwen4exp-nl3s-hcdi-gu-oproj8.gguf` (1080 tensores), 7 min
de copia en el kernel.

Pre-check (test-backend-ops perf, 512 expertos de 10, n = 2048): IQ3_S dos nodos de m=640 18.3 ms frente
a uno de m=1280 16.8 ms (−8 %); IQ4_NL 16.6 → 15.8 (−5 %).

Rigs de 40k con MTP: base 81.5 s (485 t/s), fusionado **80.9 s (488 t/s, −0.8 %)**, re-prefill 4.70 →
4.57 s (−2.8 %), decode igual, misma aceptación del draft, depth_repeat idéntico; la base 2 dio 89.5 s con
presión de E/S (psi_io 9-11 % durante ese prefill, 44 GiB disponibles frente a 47-48 el día anterior) y no
cuenta. Menos que el microbench (1.4 s por prefill) porque el nodo fusionado ahorra sobre todo el prólogo
por nodo, que en el modelo solapa con otros nodos.

Gate: graph_diff4 seq_rm 2048,631 con 12156 nodos (144 menos), los mismos 72 SET_ROWS y la misma
distribución; conversación con imagen ×2 idéntica entre sí y con el archivo anterior; repeat_probe 5/5
idéntica. La sonda cambió de distribución respecto del archivo anterior ('The' 98.2 % frente a 99.7 %):
el nodo fusionado no es bit-idéntico al par (otro reparto de tiles y de fusiones del grafo). Perplejidad
en la misma ventana: fusionado 1.5928 ± 0.0158, anterior 1.5928 ± 0.0158; sin efecto en la calidad.
Adoptado: `quant = nl3s-hcdi-gu-oproj8`.

## 16. Vía 12 medida: el input PLE está acotado por la E/S (2026-09-18, build 399)

La palanca host del prefill. Instrumentación nueva, bajo `LLAMA_INPUT_TIMING=1`: `llm_graph_result::set_inputs`
lista por clase de input los que cuestan 1 ms o más en cada ubatch de 512+ tokens (`input timing detail`), y el
input PLE de qwen4exp desglosa sus fases (`ple input detail`: hash, prefetch, set, next). Rigs de 40k con MTP,
forma de producción (batch = ubatch 2048) salvo donde se indica.

Qué es el input PLE: `per_layer_token_embd.weight` [160, 320 001 536] IQ4_NL, 28.8 GB, mmap lazy en disco;
90 bytes por fila. Cada token trae 3 n-gramas × 8 cabezas = 24 filas por hash, 49k filas por ubatch de 2048,
que tras ordenar y fusionar son ~30k rangos de una página de 4 KiB cada uno, aleatorios sobre 28.8 GB:
120-190 MB de E/S de páginas por ubatch para 4.4 MB de datos útiles. El hash cuesta 0.1 ms.

Medido (por ubatch de 2048, medias de 18 ubatches; primeros ubatches hasta 190 ms):

| Variante | set_inputs | compute host | gpu_wait | total | prefill |
|---|---|---|---|---|---|
| `posix_madvise` por rango (producción) | 150 | 1130 | 2510 | 3818 | 489.5 t/s |
| `process_madvise` por lotes de 1024 rangos | 168 frente a 158 | | | | sin cambio |
| lotes repartidos en 4 hilos | 117 | 1167 | 2505 | 3819 | 489.0 t/s |
| lotes repartidos en 8 hilos | 78 | 1201 | 2496 | 3803 | 491.4 t/s |

Lo que se ahorra en `set_inputs` reaparece íntegro en `compute` (la suma de ambos es constante, 1280 ms): el
`get_rows` del PLE en CPU espera esas mismas páginas. El costo es el dispositivo sirviendo ~30k lecturas
aleatorias de 4 KiB por ubatch (130-150 ms), no el encolado en el hilo principal. Decode sin efecto en
todas (el reparto en hilos solo se activaba con 2048+ rangos; un ubatch de decode trae ≤ 192).

El read-ahead del siguiente ubatch (`get_next_ubatch`, que existía) nunca corre en producción: con
batch = ubatch cada `llama_decode` lleva un solo ubatch (`next = 0.0 ms` en todos). Con `BATCH=8192`
(knob nuevo del launcher, strix-halo 1d44594; ubatch 2048) sí corre, y se midió con el encolado del siguiente en el hilo
principal y en un hilo aparte:

| Config | prefill | re-prefill 2k | decode | PLE por ubatch |
|---|---|---|---|---|
| base 2048 | 494.5 t/s | 448.7 t/s | 36.3-39.9 t/s | prefetch 134-137 |
| batch 8192, siguiente en el hilo principal | 486.9 (−1.5 %) | 455.0 | 35.1-35.6 | prefetch 45 (páginas ya pedidas) + next 87 |
| batch 8192, siguiente en un hilo | 498.2 (+0.7 %) | 460.6 (+2.7 %) | 34.6-34.8 | prefetch 46 + next 0.1 |

Descartado: el decode cae 5-13 % con el batch grande (los buffers `embd_nextn` y `embd_layer_inp` del
draft escalan con n_batch), los checkpoints del server se crean por batch (4x más espaciados: el re-prefill
tras una edición de contexto crece hasta 8192 + nuevos), y el grafo se reconstruye en cada ubatch de 2048
(`reused 0`), así que el estado "este ubatch ya fue leído" que vive en el objeto de input muere con él y
el prefetch síncrono se repite (46 ms) aunque las páginas estén pedidas. La generación a temperatura 0 es
idéntica entre 2048 y 8192 en el turno 1 y difiere desde el turno 2 (otro punto de checkpoint, otro corte
de ubatches en el re-prefill); dentro de una misma config es idéntica entre corridas.

Conclusión: el input PLE cuesta 130-190 ms por ubatch de 2048 (3.6-5 % del ciclo de 3.8 s) y es E/S. La
única forma de quitarlo es leer las filas del batch siguiente mientras computa el actual, y con batch =
ubatch eso exige que el server pase los tokens del próximo batch al contexto (hint) y que el estado viva
en el contexto, no en el grafo: cambio transversal server → contexto → input del modelo, techo 3.9 % de
prefill más lo que hoy espere el `get_rows` dentro de `compute` (sin atribuir). No se implementa sin
decisión del Director. Código: la instrumentación queda (inerte sin la variable); los helpers medidos sin
ganancia se retiraron.

Hallazgos laterales de la ventana: (1) el gate de descargas del launcher (`pgrep -x curl`) atrapaba su propio
`curl` de health y mató una recarga de producción en menos de 1 s (00:46, producción caída 9 min); ahora
ignora los procesos que apuntan a 127.0.0.1 y el log de la carga anterior se conserva como
`.strix.log.prev` (strix-halo 9fae19f, 4ac4e01). (2) `~/.local/bin/llama-server --version` reporta el
commit del 2026-09-09: el ejecutable es un stub que no se relinkea; las librerías que mapea producción son
las de `build/bin` y llevan el build vigente. (3) El "compute host" de 1130 ms por ubatch (30 % del ciclo)
sigue sin atribuir entre la grabación de comandos Vulkan y el gather PLE con sus faults.

## 17. Auditoría de código de palancas de prefill no inventariadas (2026-09-18, solo lectura)

Tres lecturas independientes (backend Vulkan: alimentación de las matmuls densas; grafo qwen4exp sobre el
residual ancho; camino host por ubatch: contexto, scheduler, server, speculative), sin ninguna ejecución
(GPU ocupada). Las cifras son las del perfil de la sección 2 (72.6 s de GPU por prefill de 39.5k) y de la
línea de tiempo de la sección 16; los cambios propuestos no se han medido.

Correcciones al perfil de la sección 2, verificadas por aritmética sobre las formas:

- El elementwise sobre el residual ancho es 6.0 s (RMS_NORM_MUL 2.52 + HC_GATED_MEAN 1.88 + HC_INJECT
  1.61; 13 pasadas de 84 MB por capa = 1010 GB por prefill a 168 GB/s efectivos), no 12.3: CONCAT 0.93
  es el estado conv de la GDN (17.3), MULTI_ADD 1.44 es la suma de los 10 expertos sobre vectores de 2560
  (214 GB, ya en su óptimo de banda), CONT 0.85 y MUL 0.96 son el gate de atención y las compuertas.
- El perfil por nodo serializa el grafo (barrera tras cada nodo): son cotas superiores sin solape entre
  nodos. Su suma (3.63 s por ubatch) coincide con compute + gpu_wait de la línea de tiempo: el solape del
  optimizador no aporta en prefill.
- El residual ancho es F32 por los asserts de `ggml_hc_gated_mean` y `ggml_hc_inject` (ggml.c); el repo no
  documenta el dtype de la referencia. Pasarlo a f16 ahorraría como mucho 3 s (la mitad de 6.0) a cambio
  de f16 en todos los ops hc (shaders + ops) y una compuerta de PPL; no se propone.
- El tiempo host entre decodes es bimodal: 52-82 ms por batch dentro del prefill (19 intervalos, media 60)
  y 530-590 ms en los 7 bordes de turno (256 pasos de decode con ~2.2 ms de server cada uno, 3.5-4 % del
  decode). La media de 193 ms de la sección 16 mezclaba ambos.
- La grabación Vulkan de los ~12k nodos (`compute` 1130 ms por ubatch, 93 µs por nodo) corre por delante
  de la GPU y queda oculta (compute + gpu_wait = GPU); sin syncs ni readbacks dentro del grafo (verificado:
  un solo `waitForFences` por grafo, submits cada 100 nodos o 200 GFLOP). Se vuelve suelo cuando la GPU baje
  de ~1.3 s por ubatch.

### 17.1 Draft MTP en prefill: atención y FFN a n_tokens con un solo head (−3.6 s, crece con la profundidad)

`graph_mtp` (qwen4exp.cpp:472-581) corre el bloque completo a n_tokens y recorta a n_outputs solo antes
de la cabeza; `res->t_h_nextn = flat` se fija antes del recorte para un segundo head encadenado. El draft
de producción tiene `qwen4exp.nextn_predict_layers = 1` (`n_mtp_layers = 1`, `chain_heads = false`,
speculative.cpp:1425-1504): nadie consume `t_h_nextn` del draft. Lo que el draft necesita a n_tokens en
un batch de prompt es K, V y su store al cache (y `eh_proj`, `hc_mix` de entrada, que los alimentan); Q,
el gate, la FA densa (2.7 s por prefill, lineal en n_kv), `wo`, `hc_combine`, `hc_mix` de FFN y la FFN
(~0.9 s) solo hacen falta para las filas de salida. El recorte sigue la regla del tronco (`!embeddings_nextn
|| embeddings_nextn_masked`, qwen4exp.cpp:648-659; el contexto draft corre masked). Gate: aceptación del
draft y depth_repeat idénticos. Defecto latente anotado: con masked el host lee n_outputs filas desde el
inicio de `t_h_nextn` (llama-context.cpp:2005-2016) mientras el nodo no está recortado: filas equivocadas
si algún día se encadenan heads.

### 17.2 Tile medio para la matmul alineada de 48 workgroups (−1.5 a −2 s)

`ggml_vk_guess_matmul_pipeline` (ggml-vulkan.cpp:9569-9578, rama coopmat1) elige el tile L en cuanto
m > 64 y n > 64; la regla k-alineada del fork (10026-10031) baja a `a_m` solo con `!aligned`. Para
`hc_*_down_inject` (m=324, k=10240, n=2048, IQ4_NL, f32-B) k es múltiplo de 128 → aligned → L:
3 × 16 = 48 workgroups de 256 hilos en 40 CU, 60 de 384 filas de padding, 8 TFLOPS (3.16 s por prefill).
Con `a_m` (64×64): 6 × 32 = 192 workgroups. Cambio: extender la regla con una condición de ocupación
(`aligned && tiles_l < 2 × shader_core_count`). Distinto del split_k cerrado (sección 11): cambia el tile,
no parte k. Dato adicional: MMQ denso (B a q8_1) no existe en coopmat (`create_mmq_pipelines(dense=false)`),
los IQ4_NL densos corren f32-B sin pasada de conversión, y los Q5_K/Q8_0/Q6_K f16-B con una conversión de
31.5 MB por nodo (k=2560), compartida por wq/wk/wv: sin palanca ahí. Un `ggml_cast` a f16 delante de una
matmul IQ4_NL hoy desreferencia un pipeline vacío (275-282, 8511-8514): no usar.

### 17.3 El estado conv de la GDN: 84 MB por capa por ubatch solo por layout (−0.9 s)

`ggml_concat(state, ggml_transpose(x))` (qwen4exp.cpp:1709) materializa [n_tokens+3, 10240] f32 desde
una lectura transpuesta, para que `ggml_ssm_conv` vea tiempo-canal contiguo con las 3 columnas de
historial: 36 capas × 19.3 ubatches × 168 MB = 117 GB, lo que explica CONCAT 0.93 s y parte de SSM_CONV
0.64. Cambio: `ggml_ssm_conv` con un segundo src de historial y x leído con sus strides. Op compartido
por todas las arquitecturas SSM (referencia CPU + shader Vulkan): cambio de ggml, con test-backend-ops.

### 17.4 RMS norm agrupada (−0.7 s)

El par `ggml_rms_norm` + `ggml_mul` sobre [2560, 4, n_tokens] (qwen4exp.cpp:375-376) se fusiona en el
backend, pero corre a 123 GB/s frente a 186 (HC_GATED_MEAN) y 217 (HC_INJECT) sobre los mismos 84 MB.
Un op de norma agrupada con gamma [n_embd, hc] leería y escribiría una vez a la banda de los otros.

### 17.5 Host entre batches del prompt: 2048 sincronizaciones y dos copias de 84 MB (−1 s)

Tras cada `llama_decode` del prompt, `common_speculative_process` (speculative.cpp:1605-1690) copia los
83.9 MB de `embd_nextn` del target (n_embd_out = 10240 × 2048) a `batch.embd` (desplazados una fila),
decodifica el draft, y luego copia las 2048 filas a `verify_h` con 2048 llamadas a
`llama_get_embeddings_nextn_ith`, cada una con `ctx->synchronize()` (llama-context.cpp:3974-3978) →
`ggml_backend_vk_synchronize` → `ggml_vk_graph_cleanup` incondicional (reset del command pool, vectores,
descriptores; 17529-17565). `accept()` solo lee las filas `min(n_accepted, n_rows-1)` con n_accepted ≤
n_max (2) y `pending_h` la última (1849-1856): las filas 3..2046 son inalcanzables. Cambio: una sola
sincronización (`llama_get_embeddings_nextn` ya la hace) y `verify_h` con las filas 0..n_max y la última.
Techo: los 60 ms por batch medidos (1.6 % del ciclo); el reparto exacto entre las 2048 sincronizaciones y
las copias no es determinable sin medir. El viaje D2H → host → H2D de los 84 MB es inherente a los dos
contextos (el fork prohíbe dos contextos en vuelo: "syncobj interlock", speculative.cpp:1659-1663).

### 17.6 Verificado y descartado o menor

- Selector QSA en régimen denso: se construye siempre (condición estática, qwen4exp.cpp:1154) y no es
  trabajo perdido: la máscara restringe la atención también bajo el umbral (que vive en el backend,
  ggml-vulkan.cpp:11779). El único salto exacto (`width == n_kv`, :990) solo aplica a contextos cortos.
- Materialización de la máscara QSA en 3 pasadas de [n_kv × n_tokens] f16 (fill, set_rows, add): ~0.45 s
  (0.6 %); pasar `top_k` directo a la FA es medio-alto en riesgo (vía 19, ya listada).
- Reuso del grafo entre ubatches de 2048 (build + alloc 28 ms, 0.7 %): lo bloquean n_kv en la máscara y
  en el input QSA (padeables) y `kv_head` horneado en offsets de vistas de la GDN (delta-net-base.cpp:493-600):
  refactor, no cambio barato; el beneficio GPU sería inferido.
- Sin ops del modelo en CPU salvo el `get_rows` del PLE (buft CPU de la tabla lazy; `offload_op` lo deja
  en host, ggml-vulkan.cpp:20714-20731): un split de CPU al inicio, 6.55 MB H2D por ubatch. Es la espera
  de E/S de la sección 16, no un costo nuevo.
- `apply_ubatch` del KV corre dos veces por ubatch (prepare en seco + apply, llama-kv-cache.cpp:761-820 y
  2725-2735): sub-ms. `seq_pos` como `std::set` por token: sub-ms.
- El `ggml_cont_2d` del gate de atención (qwen4exp.cpp:1205): 50 MB por capa de atención, ~0.15 s (0.2 %).
- La caché de B convertida del backend tiene una sola entrada (ggml-vulkan.cpp:2620-2623): con las formas
  de este modelo no hay reconversiones que valgan (31.5 MB por capa).

## 18. Lote de riesgo bajo de la sección 17 medido (2026-09-22, build 402)

Orden acordado: filas 10, 13, 14, vía 21 y fila 12; las de riesgo medio quedan para después. Resultado: el lote
no mueve el prefill. Rigs de 40k con MTP, timing en todos (`LLAMA_INPUT_TIMING=1`), footprint del rig medido
68 GiB y gate 40 GiB (el escritorio con 10-12 GiB de navegadores dejó MemAvailable en 103-108 y la ventana
abrió a 113 tras cerrar apps).

| Rig | Prefill 39.5k | Re-prefill 2k | Decode | Ubatch de 2048 | Host entre batches |
|---|---|---|---|---|---|
| base 1 (tile y kernel RMS apagados) | 80.28 s (492.2 t/s) | 444.1 t/s | 36.1-39.6 | 3795 ms | 50-57 ms (mediana 53) |
| fila 10: tile medio para m=324 | 81.10 s (487.2) | 439.3 | 36.5-41.4 | 3825 ms | |
| fila 12: RMS norm por subgrupo | 80.64 s (490.0) | 451.5 | 36.5-38.1 | 3802 ms | |
| base 2 | 81.09 s (487.2) | 447.0 | 35.5-39.4 | 3825 ms | |

- Fila 13 (host entre batches): una sola lectura de `embd_nextn` con su única sincronización y un `memcpy` por
  secuencia en vez de 2048 llamadas que sincronizaban y limpiaban el contexto Vulkan. Medido en el "outside" de la
  línea de tiempo: 52-82 ms por batch (mediana 60, sección 17) → 50-57 (mediana 53). **−7 ms por batch, 0.2 % del
  ciclo**: las 2048 sincronizaciones costaban ~7 ms, el resto del batch es otro trabajo host. Se queda por exacta y
  más simple, no por ganancia.
- Fila 10 (tile medio para `hc_down_inject` m=324): en el nodo aislado (test-backend-ops perf) tile L 1360.7 µs y
  tile medio 1358.0 (10 TFLOPS ambos); en el modelo −1 % (487.2 frente a 492.2 t/s, +30 ms por ubatch) y no
  bit-idéntico (otra aceptación del draft). El nodo no está limitado por ocupación. **Retirada**.
- Fila 12 (RMS norm por subgrupo: un subgrupo por fila, la fila en registros como vec4, `subgroupAdd`, 4 filas por
  workgroup): correcta (test-backend-ops RMS_NORM 71/71, par fusionado RMS_NORM_MUL 20/20 con casos nuevos de 256 y
  2560 columnas, con vista y con gamma [n, 4]); en el nodo aislado 821 µs frente a 809 del kernel actual (192 frente
  a 194.5 GB/s, ambos en el límite de banda de la iGPU); en el modelo neutra (490.0 frente a 492.2 t/s) y no
  bit-idéntica; en los grafos de decode del perfil por nodo un 23 % más lenta (994 frente a 808 µs: pocas filas,
  4 por workgroup). **Retirada**; los tests del par fusionado quedan (no existían). La premisa de la sección 17.4
  (123 GB/s) era un error de reparto: la familia RMS_NORM_MUL (2.52 s) incluye los QK norms (128, 48, 2048) 0.76 s
  y otros; el nodo del residual (2560, 4, 2048) son 1.53 s = 0.80 ms por nodo = **210 GB/s**, igual que aislado.
- Fila 14 (concats del camino split): descartada por aritmética. La salida por bloque de la FA son ~4 MB (256
  queries × 24 cabezas × 256 × f32); a 262k son 8 bloques por stream y 28 copias por capa: ~1 GB por ubatch, 5 ms
  sobre 3.8 s (0.1 %).
- Vía 21 (mat-muls diminutas), con el perfil por nodo fresco (18.8 ubatches, 146 claves): router f32 m=512
  0.37 s pero a 13.8 TFLOPS (ya eficiente), `m=48` IQ4_NL 0.28 s (1.2 ms por nodo, 0.4 TFLOPS: 16-32 workgroups),
  `m=1` f32 0.27 s (un producto punto por token en el pipeline de matmul), bf16 m=512 0.23 y m=128 0.10 del
  indexador, f32 m=256 n=49152 0.12 y m=64 n=196608 0.09 de los resúmenes QSA. Techo realista **0.5 s (0.6 %)**
  repartido en dos o tres arreglos distintos (m=1 como MUL + SUM_ROWS, tile pequeño para m=48). Queda en las menores
  con esa cifra; no se abre ventana.

Perfil por nodo fresco del prefill (2026-09-22, kernel RMS apagado; `prof_nodes2.py`, clave del down con la escala
fusionada): GPU 3594 ms por ubatch = 71.9 s por 20. MUL_MAT 18.35 s (25.5 %), MUL_MAT_ID gate/up IQ3_S 17.01 s
(23.7 %), FLASH_ATTN_EXT 9.81 s (13.6 %, la densa del draft crece de 30 a 49 ms por ubatch entre 8k y 15k de n_kv),
MUL_MAT_ID_MUL down 8.57 s (11.9 %), RMS_NORM_MUL 2.52, HC_GATED_MEAN 1.87, HC_INJECT 1.61, GATED_DELTA_NET 1.45,
MULTI_ADD 1.44, TOPK_QSA 1.32, MUL 0.96, CONCAT 0.93, CONT 0.84, ADD 0.78, SSM_CONV_SILU 0.64. Densos: wqkv q5_K
3.33, hc_down_inject 3.15, ssm_out 2.59, z 2.02, hc up 1.54, wq 1.44, shexp 0.83, o_proj q8_0 0.75. Sin cambios
respecto de la sección 2 salvo el orden de los dos primeros.

Rigs y herramientas: `mtp_depth.sh` pasa la estimación del footprint con MTP de 74 a 68 GiB (medido en cinco rigs
del 18/09: 110.6-111.2 disponibles al inicio, 43-44 en el mínimo) con el gate de 40 GiB de CLAUDE.md;
`prof_nodes2.py` es `prof_nodes.py` con la clave del down actual.

## 19. Fila 9 medida y adoptada: el draft MTP recortado a las filas de salida (2026-09-22, build 405)

`graph_mtp` corría el bloque completo del draft para los 2048 tokens de cada ubatch de prompt y recortaba a las
filas de salida solo antes de la cabeza. Con un único bloque nextn (`qwen4exp.nextn_predict_layers = 1` en el
GGUF del draft, `chain_heads = false`) nadie lee su hidden por token: K y V de todos los tokens siguen yendo al
cache, y Q, sus filas de máscara y el gate se recogen para las filas de salida (`build_layer_attn` con `out_rows`),
con lo que la atención densa, `wo`, los mezcladores hc y la FFN corren sobre esas filas. El tensor de hand-over se
expande al grafo para que el host siga leyendo sus filas de salida (el primer intento asertó en
`llama-context.cpp:2010` por no hacerlo). La condición es la del tronco (`!embeddings_nextn ||
embeddings_nextn_masked`, el contexto draft corre masked) más un solo stream en la máscara; `LLAMA_MTP_NO_TRIM=1`
restaura el bloque completo. Sin recorte en los batches de verificación (todas las filas son salida) ni en el draft
de un token; los batches de imagen no pasan por el draft.

Rigs de 40k con MTP, misma cadena (base = `LLAMA_MTP_NO_TRIM=1`):

| Rig | Prefill 39.5k | Re-prefill 2k | Decode | Ubatch del draft | Aceptación |
|---|---|---|---|---|---|
| base 1 | 80.51 s (490.7 t/s) | 4.59 s (447.6) | 36.8-40.4 | 232 ms | 153/202, 161/187 |
| **recorte** | **76.77 s (514.7)** | **4.27 s (480.2)** | 37.4-41.3 | **64 ms** (gpu_wait 170 → 27) | 153/202, 161/187 |
| base 3 | 80.40 s (491.4) | 4.57 s (449.1) | 37.4-40.3 | 221 ms | 153/202, 161/187 |

**−3.7 s (−4.6 %) de prefill y −6.8 % de re-prefill**, decode igual. Exactitud: los turnos a temperatura 0 (1-5)
idénticos entre base y recorte (6-8 son muestreados a 0.7 y difieren igual que entre dos bases); `depth_repeat`
idéntico; probe 5/5 con la distribución de siempre (98.2 %); conversación texto → imagen → texto ×2 idéntica entre
sí y con la de la ventana gu (sección 15). La ganancia crece con la profundidad: la FA densa del draft es lineal en
n_kv (30 ms por ubatch a 8k, 49 a 15k en el perfil de la sección 18), así que a 126k el ahorro por ubatch es ~3
veces mayor. Adoptado; producción recargada con el recorte activo.

Anotación lateral: la conversación con imagen b corrió a 17-19 t/s de decode y 47.7 s de imagen (a: 36-40 t/s y
30.5 s), con las mismas respuestas: otra carga en la máquina durante ese minuto, no el cambio.

## 20. Las tres palancas simples medidas: hint del server para el PLE, conv sin concat, guard del MMQ (2026-09-22, build 407)

Tres cambios en una ventana (chain_v16, base primero y último, MTP on, rig `mtp_depth.sh` de 40k) más una
segunda de calidad (chain_v16b: perplejidad y el experimento BK_STEP). Base: build 405 con el recorte del
draft, 77.09 / 77.21 s (512.6 / 511.8 t/s), re-prefill 4.25 s, decode 38.4 t/s; los textos greedy de las
dos bases son idénticos entre sí (los rigs son reproducibles entre cargas).

| Config | Prefill 39.5k | t/s | Re-prefill 2k | Δ prefill |
|---|---|---|---|---|
| base (405) | 77.09 / 77.21 s | 512.6 / 511.8 | 4.25 s | — |
| **las tres (407)** | **73.65 s** | **536.5** | 4.11-4.20 s | **−3.5 s (−4.5 %)** |
| sin hint (`LLAMA_NO_NEXT_TOKENS_HINT=1`) | 76.77 s | 514.7 | 4.20 s | −0.4 s |
| sin conv nueva (`LLAMA_GDN_CONV_CONCAT=1`) | 73.78 s | 535.5 | 4.20 s | −3.4 s |

Reparto: el hint −3.3 s; la conv sin concat −0.13 s; el guard del MMQ ~−0.3 s (por diferencia y por
microbench). Decode sin cambio en todas (35-41 t/s, mismo rango que la base).

### 20.1 `llama_hint_next_tokens`: las filas PLE del batch siguiente se leen mientras computa el actual

El mecanismo de read-ahead existía (`get_next_ubatch`, sección 16) pero nunca corre con batch = ubatch. Ahora el
server anuncia en `pre_decode` el siguiente trozo de texto de cada prompt (hasta `n_batch` tokens, hasta el
próximo chunk de imagen) con `llama_hint_next_tokens(ctx, seq, p0, tokens, n)`; el contexto guarda un run por
secuencia y, al final de `llama_decode`, con el grafo ya enviado y la GPU computando 2.5 s, llama al hook
`llama_model::prefetch_tokens` (qwen4exp: predecesores desde las celdas con `get_prev_tokens(seq, p0, tokens)`,
hash de n-gramas, `madvise WILLNEED` de las filas). Es solo un prefetch: los resultados no dependen de él y un
decode que no siga el hint solo cuesta esa lectura. Los 87 ms de encolar 30k madvise (sección 16) quedan fuera
del camino crítico.

Diagnóstico con `LLAMA_INPUT_TIMING=1` (fuerza la espera de la GPU por ubatch, no es producción), por ubatch de 2048:

| | set_inputs | compute | gpu_wait | total | `ple input detail` prefetch |
|---|---|---|---|---|---|
| sin hint | 157.2 ms | 1097 | 2449 | 3740 | 141.5 ms |
| con hint | 27.4 ms | 1079 | 2464 | 3607 | 12.7 ms |

−133 ms por ubatch: la E/S desaparece del `set_inputs` y no reaparece en `compute` (el `get_rows` en CPU ya
encuentra las páginas), y el prefetch síncrono del ubatch actual baja a 12.7 ms sobre páginas residentes, así que
no hace falta lógica para saltarlo. El re-prefill de 2k no cambia (una sola batch, el hint cubre 4 tokens).

Incidente: la primera petición con imagen de la verificación mató producción (14:31-14:33): el server copiaba el
trozo siguiente con `server_tokens::get_tokens()`, que asserta con medios en el prompt. Corregido copiando por
índice (`operator[]`, se detiene en el primer `LLAMA_TOKEN_NULL`); la conversación texto → imagen → texto es la
compuerta que lo atrapó y tras el arreglo pasa (5/5 turnos, respuestas idénticas a las de la ventana gu y a las
de la build 405).

### 20.2 `ggml_ssm_conv_state`: la conv de la GDN lee el historial y `qkv_mixed` sin el concat

Mismo op `GGML_OP_SSM_CONV` con `src[2]` opcional: `state` [d_conv−1, d_inner, n_s] y `x` [d_inner, n_t, n_s]
(canales contiguos, como sale de la proyección); CPU y Vulkan lo implementan (mismo orden de acumulación, el
`dot` de 4 taps sobre los mismos valores), los otros nueve backends lo rechazan en `supports_op`. `build_conv_state_at`
devuelve el historial y escribe las colas de rollback directamente desde `x` (transposición de 3 columnas);
el concat solo se construye cuando una cola alcanza el historial (ubatches de 1-3 tokens en decode, 245 KB) o
para la conv dilatada del PLE (una capa). `LLAMA_GDN_CONV_CONCAT=1` restaura el operando concatenado.

Microbench (test-backend-ops perf, 10240 canales, 2048 tokens, silu fusionado): 733 µs el operando concatenado,
729 µs la forma nueva, 215 GB/s ambas: el kernel ya estaba al ancho de banda. En el modelo −0.13 s por prefill
frente a los 0.93 s del CONCAT en el perfil serializado: el backend Vulkan solapa los nodos elementwise sin
dependencias con las matmuls, y quitarlos apenas mueve el total. Eso rebaja la fila 6 (fusiones del residual).
Quedan 108 nodos menos por grafo (12156 → 12048) y 117 GB menos de tráfico por prefill de 40k. Tests: SSM_CONV
90/90 y SSM_CONV_BIAS_SILU 180/180 en ambas formas.

### 20.3 MMQ de expertos: ceros en las columnas del tile sin fila

Con el hoisting de ids, `row_ids[]` queda sin inicializar para las columnas del tile por encima de `_ne1` (88
de 128 con 40 filas por experto) y `mul_mmq.comp` hacía gathers de B con esos índices (dentro o fuera del
buffer) para resultados que después descarta. Ahora `block_b_to_shmem` recibe `col_in_bounds` y escribe ceros.
Microbench de la ruta entera (IQ4_NL, 512 de 10, n = 2048): m=640 8.81 → 8.47 ms (−3.8 %), m=1280 17.09 → 16.27
(−4.8 %), 64 expertos (columnas casi llenas) 5.57 → 5.42; el nodo down (m=2560, k=640, caso perf nuevo) 8.26 ms
(8.12 TFLOPS). La ruta coopmat, sin tocar, oscila ±3 % entre corridas (IQ3_S 9.26 → 9.58), que es la escala del
ruido del microbench. MUL_MAT_ID 935/935, MUL_MAT_ID_FUSION 23/23.

BK_STEP=2 para MUL_MAT_ID (dos bloques de k por barrera, la mitad de barreras; `build-bk2` aparte con
`BUILD_DIR`, solo test-backend-ops): MUL_MAT_ID 935/935, MUL_MAT_ID_FUSION 23/23; microbench frente al guard
solo, misma sesión: m=640 k=2560 8.47 → 7.70 ms (−9.0 %), m=1280 16.27 → 15.19 (−6.6 %), 64 expertos 5.42 → 5.13
(−5.4 %), down m=2560 k=640 8.26 → 8.09 (−2.1 %: con k=640 son 20 bloques de k, hay menos barreras que ahorrar).
En producción solo el down va por la ruta entera (gate/up son IQ3_S coopmat), así que vale ~−0.2 s por prefill;
se adopta por ser consistente en todas las formas MoE de la ruta entera. Rig de confirmación: sección 20.6.
Nota de proceso: el rebuild de emergencia de las 14:31 (arreglo del hint) tomó esta línea del árbol antes de
medirla, así que producción corrió BK_STEP=2 desde las 14:33 con el probe y las imágenes en verde pero sin rig;
la sección 20.6 lo cierra.

### 20.4 Exactitud

- Determinismo: depth_repeat 5 corridas idénticas (tokens y logprobs); repeat_probe 5/5 con la misma
  distribución que la build 405 (`'The':98.2%`); textos greedy iguales entre corridas de la misma configuración.
- No bit-idéntica a la 405: los textos greedy a 40k difieren desde el carácter 478 del turno 1 y el primer
  logprob de depth_repeat pasa de −0.2109 a −0.2154. Las tres palancas son exactas por construcción (mismos
  valores, mismo orden de acumulación, o solo prefetch); el cambio de bits viene de otro orden de nodos y otro
  reparto de fusiones del grafo (108 nodos menos), como en la fusión gate+up (sección 15). Perplejidad en la
  misma ventana (holdout, c=4096, 12 chunks): **1.5928 ± 0.01584 con la build 407 y 1.5928 ± 0.01584 con la 405**.
- graph_diff4 (seq_rm, split 2048,631): 72 nodos de 12048 difieren entre corridas, todos `SET_ROWS` de los cachés
  QSA (`cache_idx_k`, `cache_k`, `cache_v` de las 12 capas, dos ubatches); la build 405 da exactamente lo mismo
  (72 de 12156). Es un artefacto del tool desde el merge del camino SET_ROWS: compara la vista entera del caché
  en el momento del nodo, y las celdas del segundo ubatch, aún sin escribir, valen cero en la corrida 1 y guardan
  el dato en la 2. Los otros 11976 nodos y las probabilidades finales son idénticos. Pendiente: que el tool
  compare solo las filas escritas.

### 20.5 Compuertas y herramientas

Compuertas de medición: `LLAMA_NO_NEXT_TOKENS_HINT=1`, `LLAMA_GDN_CONV_CONCAT=1`. `build-llama.sh` acepta
`BUILD_DIR` para un directorio de experimento con las mismas compuertas de memoria. Casos perf nuevos: la conv de
la GDN en ambas formas y el nodo down IQ4_NL (m=2560, k=640).

### 20.6 BK_STEP=2 confirmado en el modelo (build 410)

Rig de 40k con MTP, mismo día y misma base: 73.65 s (las tres palancas, BK_STEP 1) → **73.26 s (539.3 t/s)** con
BK_STEP 2, re-prefill 4.16 s, decode igual. Textos greedy de los turnos 1-5 idénticos a los de la corrida sin
BK_STEP 2 y depth_repeat con los mismos logprobs (`'El' −0.2154`): el cambio no mueve ningún bit (misma
acumulación por columna, solo cambia cuántos bloques de k entran por barrera). Adoptado.

## 21. Vía 2 bis medida: las listas de la FA sparse desde la selección QSA (2026-09-22, build 412)

`ggml_flash_attn_ext_set_kv_idx(a, idx)`: el nodo FA recibe en `src[5]` las columnas candidatas de cada fila de
la máscara (I32 [n_idx, filas], distintas por fila; `n_idx` acota `n_kv_max`). qwen4exp pasa el `top_k` del
selector, que ya tiene el layout de las filas de la máscara (`build_attn_mha` gana el parámetro `kv_idx`; los otros
callers pasan `nullptr`). Los backends sin FA sparse lo ignoran. En Vulkan, el prepass del modo sparse construye
la lista de un tile desde las candidatas en vez de barrer las `nem0` columnas de sus filas: un workgroup por
(tile, batch) marca en un bitmap compartido de la fila (`WORDS` = potencia de dos ≥ nem0/32, hasta 64 KB =
524288 celdas) las candidatas cuya entrada de máscara es finita y compacta los bits en orden de columna
(exclusive scan por subgrupo): la misma lista ascendente que el barrido, así que la atención acumula las mismas
celdas en el mismo orden; los repetidos colapsan y las candidatas enmascaradas se saltan. Vale para el modo
índice (una lista por token, prefill) y para el compacto (una lista por tile de `Br` filas, decode), y elimina
el barrido en dos pasadas por chunks del decode. `GGML_VK_DISABLE_SPARSE_FA_KV_IDX=1` mantiene el barrido.

Primera versión (bitonic de 4096 claves en memoria compartida, 78 barreras): correcta pero cara: microbench
(512 filas × 12 cabezas, q8_0, 2051 celdas) 6.95 → 7.07 ms a 32k y 9.86 → 9.31 a 131k; rigs 73.85 (barrido) →
73.33 s (bitonic) a 40k y 100.88 → 100.98 a 55k. La versión bitmap (3 barreras):

| | barrido | bitmap |
|---|---|---|
| microbench 32k, 512 filas | 6.90 ms | 6.78 ms (−1.8 %) |
| microbench 131k, 512 filas | 9.86 ms | 8.98 ms (−8.9 %) |
| rig 40k (base 410: 73.69 / 73.40 s) | — | **73.16 s (540.1 t/s, −0.4 s)** |
| rig 55k (base 100.88 s) | — | **100.66 s (−0.2 s)** |
| re-prefill 2k a 40k | 4.26 s | 4.16 s |

Lo que enseña: el barrido por token costaba ~1 % del nodo FA sparse a 40k (la sección 9 le atribuía, con las 4
filas vacías, el 40 %). El resto del nodo son los gathers de K/V por token (571 MB por 512 filas a 84 GB/s
efectivos: latencia de gather, no MMA ni prepass), y crecen con la profundidad por la localidad del caché,
no por el barrido. Las 4 filas coopmat vacías no cambian ese tráfico. Ganancia real: ~0.5 % del prefill a
cualquier profundidad medible, más el decode sin barrido a profundidad (sección 21.1).

Exactitud: el rig con la misma build y el barrido (`GGML_VK_DISABLE_SPARSE_FA_KV_IDX=1`) da textos greedy
idénticos a los del bitmap y a los del bitonic (turnos 1-5), es decir, las listas son las mismas; frente a la
build 410 los textos difieren por el src extra del nodo FA (otro orden de grafo, sección 20.4), con
depth_repeat 5/5 idéntico y probe igual a la 405. Tests: FLASH_ATTN_EXT 5221/5221 con 24 casos nuevos con
`kv_idx` (12 formas × f16/q8_0: las candidatas en orden barajado, con enmascaradas después de las finitas).
graph_diff4: los 72 SET_ROWS de siempre. Conversaciones con imagen: respuestas idénticas a las referencias.

### 21.1 Modo compacto (decode) con el mismo prepass (build 413)

El mismo shader con `list_rows` filas por lista: en el modo compacto (lotes ≤ 64 filas, decode y verificación
MTP) el workgroup del tile marca en el bitmap las candidatas finitas de sus `Br` filas y compacta la unión, en
lugar del barrido en dos pasadas por chunks de 2048 columnas (una pasada de conteo y otra de colocación por
chunk, sección 6). Rigs: 40k base 73.20 / 73.44 s, nuevo 73.32 s; 55k 100.78 s (base 100.88); decode 36-41 t/s
en ambos, sin diferencia atribuible (a 40k el barrido por chunks cuesta microsegundos por token; el ahorro crece
con la profundidad, a 262k son 128 chunks × 2 pasadas por token y capa). Textos greedy del decode idénticos a
los de la build con el modo índice solo (v17b) a 40k y 55k: la unión por tile reproduce la lista del barrido.
FLASH_ATTN_EXT 5221/5221 (los casos de 1-64 filas con `kv_idx` toman ahora este camino), depth_repeat 5/5,
graph_diff4 con los 72 SET_ROWS, probe igual a la 405, conversaciones con imagen idénticas a las referencias.

Balance de la vía 2 bis sobre las tres ventanas: bases 73.69 / 73.40 / 73.20 / 73.44 s (media 73.43), nuevo
73.16 / 73.32 s (media 73.24): **−0.2 s a 40k (−0.3 %)**, −0.1 a −0.2 s a 55k, −9 % del nodo FA sparse a 131k
en el microbench. Adoptada: exacta, contenida (un op opcional, un shader, un parámetro) y elimina de prefill y
decode el único trabajo que crecía linealmente con la profundidad del caché por token; la estimación de
−2 a −3 s de la sección 13 estaba equivocada en un orden de magnitud porque atribuía al prepass lo que son
los gathers de K/V.

## 22. IQ3_S en el coopmat de expertos: la anchura de carga (2026-09-22, build 415)

El tile coopmat de MUL_MAT_ID es 128×128×16 con 128 hilos: por bloque de k cada hilo desquantiza 16 valores de A.
Con IQ3_S los cargaba de 4 en 4: cinco cargas de bytes (d, qs, qh, signs, scales) y una búsqueda en la rejilla
por cada 4 valores. Ahora 16 por hilo (un hilo por fila del tile): dos lecturas empaquetadas de qs, un byte de
qh, dos de signos y un nibble de escala por 16 valores, cuatro búsquedas; la ruta de 8 queda disponible y la de
4 es el código anterior (`load_vec_quant` de `iq3_s` en el generador). Los valores escritos a memoria compartida
son los mismos, así que el resultado es bit-idéntico.

Microbench (test-backend-ops perf, 512 expertos de 10, n = 2048; dos sesiones, la segunda entre paréntesis):

| Forma | 4 (antes) | 8 | 16 |
|---|---|---|---|
| m=640 k=2560 | 9.57 ms | 9.28 | **8.91 (9.22)** |
| m=1280 k=2560 (gate+up de producción) | 16.85 | 16.36 | **16.21 (16.64)** |
| 64 expertos (320 filas por experto) | 5.60 | 5.05 | **5.01** |

Correctness por variante: MUL_MAT_ID 935/935, MUL_MAT_ID_FUSION 23/23, MUL_MAT 1176/1176. Rigs de 40k con MTP
(chain_v18b, base = build 414): base 73.79 / 73.38 s, **lv 16 72.94 s (541.7 t/s, −0.6 s)**, lv 16 + sonda de
tile pequeño (`GGML_VK_MMID_SMALLN=1`) 72.76 s (−0.2 s más, ruido: la sonda sigue apagada). Textos greedy
idénticos a la base en los 5 turnos, depth_repeat 5/5, probe igual a la 405, imágenes idénticas, graph_diff4
con los 72 SET_ROWS.

Lo que enseña el microbench sobre el nodo: a 512 expertos con 40 filas por experto, cada tile de 128 columnas
calcula 3.2× las multiplicaciones útiles, y el nodo corre a ~24 TFLOPS brutos (7.5 útiles), cerca del techo
coopmat de los densos. El desquantizado de A es secundario (por eso 16 valores dan 4-7 % y no 20-30 %), y la
sonda de tile pequeño no ayuda porque el tile chico carga A proporcionalmente más veces por columna útil. Lo
que queda en este nodo es el MMA desperdiciado, y la única vía contra eso es agrupar filas de varios expertos
en un tile (otra arquitectura de kernel), no el desquantizado.

## 23. Los tiles de expertos saltan las columnas vacías (2026-09-22, build 416)

Con 512 expertos y 10 activos, cada experto ve ~40 filas de un ubatch de 2048 y el tile de MUL_MAT_ID es de 128
columnas: las 88 restantes se cargaban con ceros (sección 20.3) y aun así entraban en todas las multiplicaciones,
y el almacenamiento las descartaba. Ahora cada warp calcula solo los sub-tiles de columnas que alcanzan una fila
del experto (`_ne1`): en el coopmat (gate/up IQ3_S) los sub-tiles de 16 columnas, en la ruta entera (down IQ4_NL)
las iteraciones de columna de 16. La cuenta es uniforme por warp y la selección se hace una vez por tile con un
`switch` de copias completamente desenrolladas del bloque de multiplicación: una rama dentro del bucle
desenrollado (`break` o `if`) desindexa estáticamente los acumuladores coopmat y el nodo empeora 3-11 % incluso
sin saltar nada (medido en dos intentos antes de llegar al `switch`). Las filas válidas se acumulan en el mismo
orden, así que el resultado es bit-idéntico.

Microbench (test-backend-ops perf, 512 de 10, n = 2048), lv 16 como base:

| Nodo | antes | ahora |
|---|---|---|
| gate+up IQ3_S coopmat m=1280 k=2560 | 16.2-16.6 ms | 16.07 |
| gate/up IQ3_S coopmat m=640 | 8.9-9.2 | 9.02 |
| down IQ4_NL entera m=2560 k=640 | 7.8 | 7.43 (−5 %) |
| IQ4_NL entera m=640 / m=1280 | 7.73 / 15.1 | 7.37 / 14.35 (−5 %) |

Rigs de 40k con MTP (chain_v19, base = build 415): base 72.48 / 72.31 s, **nuevo 70.38 s (561.4 t/s, −2.0 s,
−2.8 %)**, re-prefill 4.14 → 4.06 s, decode igual; con la sonda de tile pequeño encima 70.29 s (ruido, sigue
apagada). Textos greedy idénticos a la base en los 5 turnos, depth_repeat 5/5, probe igual a la 405, imágenes
idénticas, graph_diff4 con los 72 SET_ROWS. La ganancia en el modelo (2.8 %) supera la del microbench (1-5 % por
nodo): el microbench mide cada nodo aislado y en el grafo real los 47 nodos por ubatch corren solapados con lo
demás, donde el trabajo que se quita libera los CUs para los nodos vecinos.

Lo que enseña: el nodo coopmat apenas gana con el salto (1-3 %), así que no está limitado por el MMA
desperdiciado como suponía la sección 22; el techo coopmat f16 de la GPU está muy por encima de los 24 TFLOPS
brutos. Queda por identificar qué limita ese nodo (latencia de las cargas de A por hilo, ocupación).

## 24. Auditoría de solo lectura (2026-09-22 noche) y el encoder de visión en la GPU (build 416)

Auditoría del código y de los logs de las ventanas del día, sin cargar la GPU, seguida de una ventana para la
palanca mayor que salió de ella (la visión). Fuentes: `srvlog-v16-all-t.log` (`LLAMA_INPUT_TIMING`, rig de 40k,
build 407), `strix-decode-prof-v14prof-on.log` (perf logger, build 404, fases corta y de 40k) y
`strix-decode-prof-prof383.log` (perf logger, build 382), los rigs de 40k y 55k de las cadenas v17, y el código de
server, speculative, mtmd, qwen4exp y `llama-memory-hybrid-idx`.

### 24.1 Decode medido: el paso real

Con MTP n-max 2 a 40k, un paso da 2.46 tokens (150 aceptados de 208 propuestos en 104 pasos) y dura 64.3 ms:
38.3 t/s, lo mismo que miden los rigs. Línea de tiempo del rig con timing (cada decode sincroniza para muestrear,
así que el timing no cambia la forma del paso):

| Tramo | ms por paso |
|---|---|
| Verificación del target, 3 tokens: 13 ms de grabación en host solapados con la GPU + 37 ms de espera | 52 |
| Hook del draft sobre los 3 tokens | 1.7 |
| Dos pasos del draft, 1 token cada uno | 3.1-4.6 cada uno |
| Huecos de host: 0.44 tras la verificación, 2.07 tras el hook, 0.22 y 0.34 entre pasos | 3.1 |
| **Total** | **64.3** |

La verificación está limitada por la GPU: su suma de nodos en el perf logger es 49.5 ms (build 404, 3233
dispatches), igual a su pared. El perfil de la build 382 da 45.3 ms para el mismo grafo; la diferencia está en los
nodos que dependen de la profundidad (FA +1.75, TOPK_QSA +0.48, GET_ROWS +0.25), no en otro lado: no hay
regresión, y la tabla de la sección 1 promediaba este grafo con el del hook (1.5 ms).

Verificación por familia (perf logger serializado, cotas superiores; build 404, 40k):

| Familia | ms | Detalle |
|---|---|---|
| Mat-vec densos | 18.6 | wqkv q5_K 3.0 (216 GB/s), router f32 2.55 (99 GB/s; el bf16 de igual forma del mismo grafo va a 165), cabeza q6_K 2.3 (227), ssm_out 1.7 (184), hc down m=324 1.7 (104), z 1.6 (194), hc up k=320 1.45 (124), wq 1.0, wo 0.96 |
| Expertos gate/up IQ3_S (`batch=3`, camino mat-vec-id) | 10.8 | ~26-28 expertos distintos por capa para 3 tokens: ~172 GB/s |
| Expertos down IQ4_NL | 5.5 | ~224 GB/s |
| FA sparse (12 capas) | 3.8 | modo compacto |
| GET_ROWS + CPY (estado GDN) | 2.5 | 145 + 151 dispatches |
| TOPK_QSA, RMS_NORM_MUL, GDN y el resto | ~8 | ~2500 dispatches pequeños |

Los expertos del lote de verificación van por el mat-vec-id: el dispatch lo usa hasta 8 filas
(`ggml_vk_use_mul_mat_vec_id`), y el perf logger añade `_VEC` al nombre solo con 1 fila, así que sus nodos de 3
filas aparecen como `MUL_MAT_ID ... batch=3` sin serlo. (Corregido el 2026-09-23: esta tabla decía "camino de
matmul".)

Grafos del draft: el paso de 1 token suma 2.97 ms, de los que 1.58 son la cabeza iq4_nl de 248k filas y 0.53 la FA
densa a 40k; el hook suma 1.5 ms, con 0.75 de FA densa.

Profundidad: de ~1k a 40k (build 404, antes de las listas de la sección 21) el paso crece +7.9 ms: FA del target
+3.4, TOPK_QSA +0.9, GET_ROWS +0.5, FA del draft +1.5 y la agrupación QSA en CPU ~+1. De 40k a 55k, con la build
actual y prompts distintos, el paso cambia entre −0.9 y +3.1 ms: dentro del ruido entre prompts. No hay datos para
extrapolar a 262k.

### 24.2 Visión: el encoder corría en la CPU

El launcher pasaba `--no-mmproj-offload` desde su primer commit (strix-halo c8a5bdc, 2026-05-23), sin motivo
registrado. Sin offload, `tools/mtmd/clip.cpp` pone el ViT entero en el backend CPU. El ViT de Qwen3-VL tiene 27
capas, 1152 de ancho, FFN 4304 y 449M parámetros; con `--image-min-tokens 1024` procesa al menos 4096 parches
por imagen, y el proyector acepta hasta 4096 tokens (16384 parches). Todos sus ops tienen soporte en Vulkan
(conv de parches, resize bilineal de posiciones, rope multi, layer norm, flash attention con cabeza 72).

Cambio (strix-halo 55bb358): `MMPROJ_OFFLOAD=1` por defecto pone el encoder en la GPU, `MMPROJ_OFFLOAD=0` lo
devuelve a la CPU, y el gate de memoria del launcher suma el mmproj. Ventana chain_v20: rigs NP=1, ctx 57344 con
MTP, CPU / GPU / CPU, cada uno con la conversación de referencia dos veces y, en el primer par, una captura de
1920×1080 con texto chico y una imagen de 2048×2048 al tope de tokens:

| Turno con imagen | Tokens | CPU | GPU |
|---|---|---|---|
| 448×448 (conversación de referencia) | 1057 | 30.9 / 29.9 s | 3.4-3.6 s |
| Captura 1920×1080 | 2078 | 102.8 s | 7.7 s |
| 2048×2048, tope del proyector | 4134 | 381.2 s | 14.6 s |

- **Memoria:** MemAvailable tras cargar 47.0 GiB con el encoder en CPU y 47.1 GiB en GPU, 44.7 y 44.9 tras las
  imágenes: los pesos se mudan de la RAM del proceso a la GPU y el buffer de cómputo no pesa más que el de CPU.
- **Respuestas:** la conversación de referencia da las mismas respuestas greedy en CPU y en GPU, y entre
  repeticiones de cada una. Las dos imágenes grandes se leen bien en ambos casos (la barra "RAM 43.7 GB ... UPTIME
  3d 04h", "ORDER #7731 SHIPPED", las figuras y sus colores); cambia solo el formato de la respuesta.
- **Anillo de la GPU:** cero timeouts en el log del kernel durante la ventana; la imagen más grande que acepta el
  proyector completa en 14.6 s con el server sano.
- **Producción** (recargada con el encoder en GPU solo porque el rig pasó): turno con la imagen de 448×448 en
  4.1 s (antes 30.9), las mismas respuestas que el rig, probe de texto idéntico a las builds anteriores,
  MemAvailable 35 GiB tras cargar y 34 tras la imagen.

Solo se midió qwen38flash; el default cambia para todas las familias con mmproj del launcher, y el gate de
memoria del launcher suma ahora el mmproj en todos los modelos. `--image-min-tokens 1024` es un requisito de
grounding de Qwen-VL, no una perilla de rendimiento.

Reparto del turno con el encoder en la GPU, de las marcas de tiempo del log del rig (el lanzamiento de la tarea, el
envío de cada ubatch de imagen, que `find_slot` marca por sus posiciones no consecutivas, y el fin del prompt;
`llama_decode` es asíncrono, así que cada marca es un envío y el tiempo de GPU de un ubatch se lee hasta la marca
siguiente):

| Imagen | Tokens de imagen | ViT + preproceso + texto previo | Prefill de los tokens de imagen | Cola de texto + últimos 4 |
|---|---|---|---|---|
| 448×448 | 1024 | 0.78 s | 1.71 s (~600 t/s) | ≤ 0.35 s |
| 1920×1080 | 2040 | 2.43 s | ~3.0 s (~680 t/s) | ~0.44 s |
| 2048×2048 | 4096 | 5.99 s | ~5.6-5.9 s en 2 ubatches | ~0.35 s |

Los tokens de imagen se procesan al ritmo del texto; el encoder es el 27-47 % del prompt y crece más rápido que los
parches. (Corregido el 2026-09-23: esta sección decía que el resto del turno era un prefill de imagen más lento que
el de texto, y el reparto del 2026-09-22 atribuía 0.8 s a la cola de texto, que incluían 447 ms de generación.) El
inventario de palancas del encoder, de las imágenes y de la ejecución paralela está en la sección 25.

El log del kernel es legible para este usuario (122 líneas de amdgpu en el arranque), así que el cero de timeouts
de la ventana es una medida y no una ausencia de acceso. El mismo log muestra tres resets de la cola de cómputo del
día a las 15:46, 16:20 y 16:53, todos de `test-backend-ops perf -o FLASH_ATTN_EXT` (cadenas v17, v17b y v17c) en el
caso MLA heredado de upstream (hsk=576, hsv=512, kv=32768), que corrió después de los casos de qwen4exp: sus
números de la sección 21 valen, y esa suite ya no se corre completa (regla en CLAUDE.md).

### 24.3 Palancas nuevas y re-valoradas

Decode, milisegundos por paso a 40k:

| Palanca | Estado | Costo hoy | Ahorro estimado | Procedencia |
|---|---|---|---|---|
| Agrupación QSA en CPU incremental (D7) | nueva | ~1 ms, GPU parada | ~1 ms, crece con n_kv | set_inputs medido 1.8 ms; el ~1 ms sale del comentario del código a 33k (`llama-memory-hybrid-idx.cpp`) |
| Hueco tras el hook del draft (D8) | nueva | 2.07 ms | hasta 2 ms | medido; causa sin atribuir. El muestreo en GPU del server (`--backend-sampling`) se apaga con gramática o presupuesto de razonamiento, así que no sirve para tráfico de agentes |
| Fusiones z+alfa+beta, gate/up del shexp, router+gate del shexp (fila 17) | nueva | ~220 dispatches pequeños | ~0.5 ms | dispatches × ~2.5 µs |
| Expertos IQ3_S en el lote de verificación (D2) | re-valorada | 10.8 ms a ~172 GB/s | −2 ms si llegan a 220 GB/s | perf logger |
| Mezcladores hc (vía 10) | re-valorada | 3.2 ms a 104-124 GB/s | ~−1.5 ms | perf logger |
| Router f32 (vía 13, D3) | re-valorada | 2.55 ms a 99 GB/s | ~−1 ms | perf logger; el bf16 de igual forma del mismo grafo va a 165 GB/s |
| Estado GDN (vía 14, D4) | re-valorada | 2.5 ms | parcial | perf logger |
| Cabeza del draft (vía 9, aplazada por el Director) | re-valorada | 3.2 ms | ~−2 ms | perf logger |
| FA del target (vía 7, D5), TOPK_QSA (vía 19), FA densa del draft (vía 16) | re-valoradas | 3.8, 1 y 1.5 ms | crecen con n_kv | perf logger, build 404 |

Las tasas del perf logger son cotas superiores (serializa cada nodo). La comparación del router vale porque f32 y
bf16 de igual forma están en el mismo grafo; la de los grafos del draft no, porque sus pesos quedan en la MALL
entre pasos.

Prefill: poco nuevo a 40k. Filas 15 (indexador QSA fusionado, −0.7 s serializado), 16 (máscara QSA sin construir,
−0.4 s serializado) y 17 (fusiones de pesos, −0.5 s) de la sección 13. Las dos primeras salen de contar bytes
sobre el perfil, no de formas de nodo, y por lo aprendido con el concat (sección 20.2) el ahorro real será una
fracción: −0.5 a −1.2 s a 40k entre las tres; las dos primeras crecen de forma cuadrática con la profundidad del
prefill.

No cubierto: ejecución paralela con dos slots (NP=2); medida en la sección 25.3.

## 25. Inventario de palancas del encoder, las imágenes, la ejecución paralela y el prefill (2026-09-23, ventana v21)

Tres auditorías de solo lectura del código (encoder de visión, lado LLM de los prompts con imagen, servidor con dos
slots), cada afirmación verificada contra el código, y una ventana de medición para lo que el código no resuelve
(`chain_v21.sh`, build 416 más los casos de perf de la ViT en `test-backend-ops`): microbench de la flash attention
de la ViT, perfil de la ViT con el perf logger, decode a 40k con y sin una imagen al inicio (`LLAMA_INPUT_TIMING`)
y rigs NP=2 a 40k por slot con n-max 2, 1 y sin MTP. Cero timeouts de anillo en toda la ventana; el mínimo de
MemAvailable fue 41 GiB.

### 25.1 Encoder (ViT de Qwen3-VL, en la GPU)

La ViT corre con flash attention por el camino coopmat1 (`-fa on` → `CLIP_FLASH_ATTN_TYPE_ENABLED`,
`GGML_PREC_F32`, sin máscara): con `GGML_VK_DISABLE_COOPMAT=1` el mismo caso tarda 13.2 ms en lugar de 10.2. Un
nodo que supera el tope de FLOPs por envío (200 GFLOP como máximo, `ggml-vulkan.cpp:19083`) va en un envío propio
(`:19134-19143`), así que el tope de 2 s por envío no está en juego: 168 ms por capa al tope.

Reparto por familia (perf logger serializado; la FA coincide con el microbench):

| Imagen | Parches | ViT | FA | Matmul | ADD (bias) | Resto |
|---|---|---|---|---|---|---|
| 448×448 | 4096 | 0.61 s | 0.27 (44 %) | 0.22 | 0.05 | 0.07 |
| 1920×1080 | 8160 | 2.17 s | 1.48 (68 %) | 0.43 | 0.13 | 0.13 |
| 2048×2048 | 16384 | 5.95 s | 4.52 (76 %) | 0.89 | 0.27 | 0.27 |

El preproceso en CPU (base64, PNG, bicubic, normalización) y el texto previo suman ~0.1 s; la primera imagen tras
cargar paga ~0.15 s de compilación de pipelines. Ninguno es palanca.

Microbench de la FA de la ViT (16 cabezas, sin máscara, f32 acc; ms por capa):

| Cabeza | Parches | ms | TFLOPS |
|---|---|---|---|
| 72 | 4096 | 10.05 | 7.7 |
| 72 | 8160 (no múltiplo de 64: variante `Clamp`) | 54.96 | 5.6 |
| 72 | 8192 | 41.25 | 7.5 |
| 72 | 16384 | 168.6 | 7.3 |
| 64 | 16384 | 113.4 | 9.7 |
| 80 | 16384 | 141.9 | 9.7 |

La cabeza 80 hace un 11 % más de trabajo que la 72 y tarda un 16 % menos. Con 72 el último trozo de 16 columnas de K
(64-79) no está completo y va por `stage_k` (`flash_attn_cm1.comp:329`): dos barreras y una copia a memoria
compartida por cada bloque de 64 claves; con 80 se carga directo. La variante `Clamp` (N no múltiplo de 64, casi toda
imagen no cuadrada) activa los chequeos de límite en todos los bloques y cuesta un 33 %.

| # | Palanca | Ganancia | Base | Riesgo |
|---|---|---|---|---|
| E1 | **Probada la variante en el kernel (sección 26): más lenta, revertida; queda el relleno en el grafo de clip, decisión del Director.** Cabeza de la ViT rellenada a 80 con ceros (Q, K y V; la salida se recorta a 72). Los MMA son los mismos que hoy (72 ya se rellena a 80 dentro del kernel), así que se espera bit-idéntico | −0.72 s al tope, −0.16 s a 8192, −0.03 s en 448×448 | medido: 141.9 frente a 168.6 ms por capa × 27 | bajo si se hace en el grafo de clip; el costo del relleno (~1 ms por capa) está por medir |
| E2 | FA sin `Clamp` en los bloques completos: solo el último bloque con chequeo de límite | −0.37 s en 1920×1080 (55.0 → 41.2 ms por capa) | medido | medio: shader compartido con el LLM, compuertas de determinismo |
| E3 | Caché de embeddings por id de imagen (sha256 + tamaño, LRU acotado): hoy no existe (`mtmd.cpp:2150-2167`, el lote se resetea por tarea en `server-context.cpp:432`) | −0.7 / −2.4 / −6.0 s por imagen que se reprocesa (volver a un checkpoint anterior a la imagen, cambio de slot, fallo del prompt cache); 0 en un multi-turno normal | código | bajo; 10-40 MiB por imagen |
| E4 | `ffn_down` con K=4304, que no es múltiplo de 32 (la alineación mínima de los tiles, `ggml-vulkan.cpp:4712-4714`), va por el kernel no alineado: 10.2 TFLOPS, frente a 20.1 de `ffn_up` (mismos FLOPs) y 19.3 de la proyección de salida de la atención (misma m=1152, K=1152 alineado) | −0.21 s al tope, −0.11 s en 1080p, −0.06 s en 448 | medido (429 frente a 218 ms al tope, serializado) | bajo: relleno con ceros a 4352 de las filas de `ffn_up` y su bias y de las columnas de `ffn_down` al cargar el mmproj (GELU(0) = 0) |
| E5 | Bias de las matmuls fusionado en el epílogo (hoy un ADD aparte) | ≤ −0.27 s al tope | perf logger serializado | medio: epílogo del matmul compartido |
| E6 | mmproj en Q8_0 | ≤ −0.2 s al tope | inferido | calidad de lectura y grounding: decide el Director |

Suma realista de E1, E2, E4 y E5: al tope 5.95 → ~4.9 s (−18 %), en 1080p 2.17 → ~1.5 s (−30 %), en 448×448 0.61 →
~0.5 s. Techo por encima de eso: la FA de la ViT corre a 7-10 TFLOPS contra 18-23 de las matmuls densas; una FA con
tiles de más filas para N grande (hoy Br = 16: K y V se releen N/16 veces por cabeza) es un proyecto de kernel, sin
estimación medida.

### 25.2 Imágenes, lado del LLM

El reparto del turno está en la sección 24.2: los tokens de imagen se procesan al ritmo del texto. Lo que aparece
en la ventana es un costo del decode que queda para el resto de la conversación.

Decode a ~40k de profundidad, mismo texto y 256 tokens greedy, con y sin una imagen de 448×448 al inicio (rig NP=1,
`LLAMA_INPUT_TIMING`, medianas del grafo de verificación):

| Turno | t/s | Paso de verificación | build | alloc | set_inputs | compute + espera de GPU |
|---|---|---|---|---|---|---|
| Texto | 33.70 | 53.1 ms | 0 | 0 | 1.86 | 51.1 |
| Imagen + texto | **26.10** | **66.3 ms** | **2.17** | **11.1** | 3.45 | 51.2 |
| Texto (caché) | 34.26 | 52.4 ms | 0 | 0 | 1.35 | 51.0 |

Con la imagen el grafo de verificación se reconstruye y se reasigna en cada paso (+13.3 ms) y el set_inputs sube
1.6 ms. La caída de 33.7 a 26.1 t/s mezcla eso con la aceptación del draft, que es otra porque la respuesta es otra
(1.97 tokens por paso frente a 2.25; 75.4 frente a 66.6 ms por paso). Causa de los 13.3 ms (código): la marca de agua del caché de claves agrupadas del QSA (`pooled_valid`)
cuenta bloques por rango cuando la secuencia tiene una imagen (las celdas de imagen comparten posición y el
agrupamiento pasa a rango, `llama-memory-hybrid-idx.cpp:583-613`), pero `pooled_rm` la retrocede con la posición
del primer token borrado (`p0/ratio`, `llama-memory-hybrid-idx.cpp:383-406`). Tras cada verificación con rechazo el
server borra desde `pos_next()` (`server-context.cpp:4059`), la marca cae ~250 bloques (las 1024 celdas de la imagen
ocupan pocas posiciones), el ubatch siguiente vuelve a agrupar esos bloques y el tamaño de las tablas de bloques
sucios cambia de paso a paso, así que `llm_graph_input_qsa::can_reuse` falla (`qwen4exp.cpp:797`). Cada imagen
adicional suma su diferencia entre celdas y posiciones.

| # | Palanca | Ganancia | Base | Riesgo |
|---|---|---|---|---|
| I1 | `pooled_rm` con el rango de la primera celda borrada cuando la secuencia está ordenada por rango (el mismo orden que usa `set_input_qsa`) | −13.3 ms por paso de verificación a 40k con una imagen. **Medido (sección 26): 26.2 → 29.1 t/s (+11 %); la estimación de +21 % suponía otra reducción del paso de pared** | medido el síntoma, causa en código | bajo: exacto (los bloques anteriores no cambian); gates de determinismo y la conversación texto → imagen → texto |
| I2 | Orden por rango incremental por secuencia (hoy un `std::sort` de todas las celdas por ubatch con la GPU parada) | ≤ −1.6 ms por paso a 40k con imagen, crece con n_kv. **Hecho como counting sort (sección 26): set_inputs 3.50 → 2.83 ms, +1 %** | medido (set_inputs 3.45 frente a 1.86) | bajo, exacto |
| I3 | = E3 (caché de embeddings) | | | |
| I4 | Texto y tokens de imagen en un mismo ubatch (hoy texto previo, cada trozo de imagen y la cola van en `llama_decode` separados) | −0.2 a −0.5 s por turno con imagen | código; la cola cuesta ≤ 0.35 s | alto: el grafo elige token o embd por ubatch, el PLE y el grafo MTP exigen tokens |
| I5 | Presupuesto de tokens de imagen (hoy mínimo 1024, tope 4096) | el mínimo cuesta ~1.5 s por imagen chica; un tope de 2048 ahorra ~3 s de LLM y ~4 s de ViT al tope | código | calidad: decide el Director |
| I6 | Checkpoint después de un chunk de imagen | solo si se edita entre el fin de la imagen y el final del mismo prompt: 2.4-12 s en ese caso | código | raro en agentes |

No son palancas: `n_cache_reuse` con imagen (el estado GDN no permite quitar tramos intermedios), el PLE de los
tokens de imagen (un solo n-grama sustituto), el draft MTP durante la imagen (no corre) y su posición tras la imagen
(`b0dcb8192` ya está en el fork como 9361fffb2).

### 25.3 Ejecución paralela (NP=2)

Cómo arma el server un paso con dos slots (código): los tokens de los slots que generan entran primero al batch y
los prompts llenan el resto hasta n_batch = 2048, sin tope ni reparto (`server-context.cpp:3126-3145, 3548`); todo
va en un solo `llama_decode` y el hook del draft corre sobre el batch completo antes de muestrear
(`:3726, :3794`). El n-max del draft no mira a los otros slots (`:508-527`), y n_kv del grafo es el máximo entre
los streams (`llama-kv-cache.cpp:1260-1274`).

Rigs NP=2 (2 × 49152, ~39k tokens de prefijo por slot, 384 tokens greedy por stream):

| Configuración | Un stream (A / B) | Dos streams, por stream | Dos streams, total |
|---|---|---|---|
| MTP n-max 2 (base, primera carga) | 33.7 / 33.7 t/s | 19.6-19.9 | 38.6 |
| MTP n-max 1 | 32.9 / 31.9 | 19.7-20.3 | 38.8-39.2 |
| Sin MTP | 26.7 / 26.8 | 18.8-19.0 | 37.0-37.4 |
| MTP n-max 2 (base, última carga) | 34.0 / 34.0 | 19.6-20.0 | 38.7 |

El segundo agente suma un 15 % al total y cada uno va al 58 % de lo que iría solo. Con dos generando, el MTP todavía
gana un 4 % frente a apagarlo y n-max 1 queda dentro del ruido de n-max 2: la medición del 2026-09-02 del launcher
(−9 % con MTP y dos streams, otra quant, otro draft y sin los kernels de hoy) ya no vale.

Un agente que genera mientras el otro manda un prompt nuevo (15k tokens, n-max 2): el que genera pasa de 32.8 t/s
a **0.59 t/s** durante los 29 s del prefill ajeno (17 tokens, huecos de hasta 4.4 s) y vuelve a 33.3 después. Es
lo que predice el código: cada paso suyo espera un trozo de 2048 tokens del otro más el hook del draft.

| # | Palanca | Ganancia | Base | Riesgo |
|---|---|---|---|---|
| C1 | Reparto del paso cuando un slot genera y otro procesa un prompt: k pasos de decode por trozo de prompt, o un tope de tokens de prompt por paso. **Hecho (sección 26): `--decode-share 0.4`, 0.56 → 13.7 t/s** | el que genera: de 0.6 a ~7 t/s con el 20 % de la GPU, ~17 con el 50 % (solo va a 33.7); el prefill del otro se alarga en la misma proporción | medido el problema, reparto inferido | medio: planificador del server; el reparto es una decisión del Director |
| C2 | n-max según los slots que generan. **Cerrada por la medición: con dos generando, n-max 1 da +1 % (ruido) y sin MTP −4 %** | — | medido | — |
| C3 | `--no-cache-idle-slots` (**hecho, sección 26: ~0.1 s por turno**): hoy cada tarea nueva copia el estado entero del slot idle al prompt cache (`:2455-2468`), aunque al reasignar el slot se guarda igual (`:1654-1668`) | decenas a cientos de ms por turno según la profundidad | código, costo inferido | bajo |
| C4 | n_kv por stream: hoy el slot corto paga la profundidad del largo | hasta ~8 ms por paso a 40k de diferencia | inferido | alto: estructural |
| C5 | `-ub 1024`: la mitad de espera por paso ajeno y de re-prefill por checkpoint | a medir: ub512 era −23 % de prefill a 4.7k | inferido | bajo, medir a profundidad |
| C6 | Recorte del draft cuando dos prompts comparten ubatch (`qwen4exp.cpp:553-555` exige un stream) | ~170 ms por ubatch cuando ocurre | inferido | bajo |

Notas: con más de dos agentes, las entradas del prompt cache de más de ~160k tokens superan los 4 GiB y se
descartan (el agente desalojado paga un prefill completo); NP=3 daría 9 filas de verificación y sacaría a los
expertos del mat-vec-id; `-kvu` desactiva el caché agrupado del QSA.

### 25.4 Prefill de texto (70.4 s a 40k)

Sin palancas nuevas; lo vigente de la sección 13:

| # | Palanca | Ganancia | Nota |
|---|---|---|---|
| T1 | Nodo coopmat gate/up de expertos (~24 % de la GPU): limitante sin identificar (sección 23) | −2 a −4 s si el nodo gana 20-30 % | investigación, sin mecanismo |
| T2 | Fila 15: indexador QSA fusionado | −0.3 a −0.7 s | cuadrático con la profundidad |
| T3 | Fila 16: máscara QSA sin construir | −0.2 a −0.4 s | cuadrático |
| T4 | Fila 17: fusiones de pesos | −0.5 s y −0.5 ms por paso de decode | dos exigen transformar el GGUF (Director) |
| T5 | Filas 3, 6 y 8 (restos) | ≤ −0.5 s cada una | confianza baja |

La fila 5 (vía 16) ya no es palanca de prefill: la fila 9 dejó el draft en 64 ms por ubatch (era 232). Sigue siendo
palanca de decode (FA densa del draft, ~1.3 ms por paso a 40k).

### 25.5 Orden propuesto

1. I1 + I2: exactos y locales al caché QSA; quitan la reconstrucción del grafo en cada paso de toda conversación
   con imágenes (+21 % en el turno medido a 40k con una imagen, más con más imágenes).
2. C1: el reparto entre un agente que genera y otro que manda un prompt (hoy el que genera cae a 0.6 t/s);
   necesita la decisión del Director sobre el reparto.
3. E1, E4: la ViT al tope −0.9 s con cambios en el grafo de clip y en la carga del mmproj.
4. E3: caché de embeddings, para los re-procesos de imagen.
5. E2, E5: kernels compartidos con el LLM, con las compuertas completas.

## 26. Quick wins de la sección 25 medidos (2026-09-23, cadenas v22, v23 y v23b)

Cada cambio se midió contra la build de producción en la misma cadena (base primero y último, binario base copiado y
cargado con `LD_LIBRARY_PATH`), con las compuertas de exactitud: conversación texto → imagen → texto, turnos de 40k
con y sin imagen idénticos a la base en texto, logprob del primer token y aceptación del draft, el turno con imagen
repetido sobre el prefijo en caché idéntico, `graph_diff4` con solo los 72 SET_ROWS, `repeat_probe` y cero timeouts de
anillo.

| Palanca | Resultado | Estado |
|---|---|---|
| I1: marca del caché QSA agrupado por rango (68fc7b3bc) | paso de verificación a 40k con imagen 64.3 → 54.3 ms (build + alloc 12.1 ms → 0); decode 26.2 → 29.1 t/s (+11 %), exacto | en producción |
| I2: orden por rango con counting sort por posición | set_inputs del paso con imagen 3.50 → 2.83 ms; decode 29.2 → 29.5 t/s (+1 %), exacto | en producción |
| C1: `--decode-share` al 40 % (decisión del Director) | agente que genera durante un prefill ajeno de 15k: 0.56 → 13.7 t/s; el prefill ajeno 28.6 → 48.0 s (×1.65); dos agentes generando sin cambio | en producción, launcher `DECODE_SHARE=0.4` |
| C3: `--no-cache-idle-slots` | hueco entre lanzamientos de dos tareas: 0.1 / 135 ms sin la copia del slot idle, 138 / 244 ms con ella (~0.1 s por turno) | en producción, launcher |
| E1: el trozo parcial de K de la cabeza 72 preparado por subgrupo | todas las variantes de la FA 4-16 % más lentas, también las que no toman la rama (la 80 139 → 161 ms): la rama nueva en el bucle desenrollado cambia la compilación del kernel entero, como en la sección 23 | revertido |

Lecturas:

- **I1 no llega al +21 % estimado en la sección 25.2**: el paso con imagen ya cuesta lo mismo que con texto (67.6
  frente a 66.0 ms por paso de pared); lo que queda de diferencia en t/s es la aceptación del draft, propia de la
  respuesta (1.97 frente a 2.25 tokens por paso). Con varias imágenes el ahorro por paso crece: cada imagen sumaba su
  diferencia entre celdas y posiciones a los bloques que se volvían a agrupar.
- **I2** deja ~1 ms por paso sobre el texto: la segunda pasada de agrupación por rango.
- **C1** reparte el tiempo por déficit: cada paso con generación y prompts pendientes a la vez es de prompt (un trozo
  de 2048 o una imagen; los que generan van incluidos) o de decode (solo los que generan); un paso toma un trozo de
  prompt solo cuando los pasos de decode ya tuvieron el 40 % del tiempo desde que empezó la contención, y la cuenta
  se reinicia cuando uno de los dos trabajos se acaba. Sin contención el server se comporta como upstream. El agente
  que genera recibe su 40 % en ráfagas: huecos de hasta ~4.4 s (un trozo de 2048 a 40k) seguidos de ~2.3 s de decode.
- **C3**: la copia del slot idle explicaba ~0.1 s por turno, no los 0.25 s que se le atribuían; los otros ~135 ms del
  hueco entre lanzamientos están en el camino HTTP (tokenización o plantilla de un prompt de 39k) y no se tocaron.
- **E1** queda solo como relleno de la cabeza a 80 en el grafo de clip (medido: 141.9 frente a 168.6 ms por capa al
  tope, −0.7 s por imagen de 2048×2048); es un rodeo en el grafo para un límite del kernel y lo decide el Director.
- `graph_diff4` se recompiló contra los headers actuales (el binario del 2026-09-09 se caía con rc 139 al cambiar
  `common_params`); una herramienta fuera del árbol que usa `common.h` se recompila con cada cambio de esa estructura.
  Compilación (en `~/dbg/merge`, con `L` = el fork y producción descargada): `g++ -O2 -std=c++17 -I $L/include
  -I $L/common -I $L/ggml/include graph_diff4.cpp -L $L/build/bin -lllama-common -lllama -lggml -lggml-base
  -Wl,-rpath,$L/build/bin -o graph_diff4`.

Pendientes de la sección 25 fuera de esta ronda: E2 y E5 (kernels compartidos con el LLM), E3 (caché de embeddings,
un componente nuevo), E4 (exige transformar el mmproj, decisión del Director), C4 a C6 y las palancas de prefill de
texto (25.4).
