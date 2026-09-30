# qwen4exp en Strix Halo: dónde va el tiempo y vías de optimización (2026-09-09, actualizado 2026-09-16)

## Estado actual (2026-09-28)

- Árbol: master con el hint del server, la conv sin concat, el guard de columnas y BK_STEP 2 del MMQ (sección 20, build 410), las listas de la FA sparse desde la selección QSA (sección 21, build 413), la carga IQ3_S de 16 valores por hilo (sección 22, build 415), los tiles de expertos que saltan las columnas vacías (sección 23, build 416) y los quick wins de la sección 26: la marca del caché QSA agrupado por rango y el orden por counting sort (decode tras una imagen +12 %), y `--decode-share 0.4` con `--no-cache-idle-slots` para varios agentes a la vez; desde el 2026-09-28, los grafos de
  los ubatches chicos en slots con su propio scheduler (SP1 del plan de decode, sección 30), las entradas del paso
  sin recorridos caros (SP2, sección 33) y los huecos de host del paso (SP3, sección 35). Abierto: con el SP1 la salida
  depende de la historia del servidor (sección 35.1).
  Producción: `strix load qwen38flash`
  (NP=2, `-c 524288` = 262144 × 2 slots, batch = ubatch 2048, KV q8_0, lazy mode auto, draft MTP IQ4_NL
  n-max 2, mmproj con el encoder de visión en la GPU desde el 2026-09-22 (sección 24), `--ctx-checkpoints 8`,
  `--decode-share 0.4`, `--no-cache-idle-slots`),
  archivo `NL3S/qwen4exp-nl3s-hcdi-gu-rq8-oproj8.gguf` (`quant = nl3s-hcdi-gu-rq8-oproj8`, router en Q8_0,
  sección 27).
- Rendimiento a 40k de profundidad (rig `mtp_depth.sh`, MTP on): prefill de 39.5k **70.4 s (561 t/s)**,
  re-prefill de 2k 4.11-4.20 s, decode 35-41 t/s (61.7 ms por paso sin instrumentar tras el cambio del scheduler de
  la sección 28.7, antes 63.9; los tokens por paso dependen del texto). Desde el 2026-09-28 el paquete ya no corre
  limitado a 85 W (117-130 W, sección 30): con el mismo código el paso bajó a 55.4 ms y el prefill de producción a
  590 t/s, y el SP1 lo llevó a 54.2 ms en greedy. Desde el 2026-09-28 el lookup en la ronda MTP (sección 31) acelera
  los turnos que copian o editan texto un 29-45 %. A 125k (sección 32): prefill de 124.9k en 244 s (511 t/s) y
  decode 33.3-34.0 t/s con instrumentación (37.8-39.9 a 40k en la misma ventana); con el SP2, 34.4-35.1 t/s a 125k
  (+3.2-3.9 %) y +0.9 % a 40k. Turno con una
  imagen de 448×448 en producción 4.1 s (antes 30.9). Residentes 56.5 GiB; MemAvailable ~35 GiB con producción
  cargada. Determinismo verificado (repeat_probe, graph_diff4, depth_repeat) tras cada cambio.
- Lista de trabajo vigente: sección 13 (prefill), sección 25 (encoder, imágenes, ejecución paralela, con la
  ventana v21 del 2026-09-23), sección 28 (presupuesto del decode) y sección 29 (techo físico del decode, camino al
  2× del prefill y referencias externas del 2026-09-26); plan del camino de decode por subproyectos en
  `decode-plan-2026-09-28.md`. Desglose por nodo vigente: sección 2. Las
  secciones 5-12 son el registro de cada ventana de medición, en orden cronológico.

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
| 7 | split_k en el modo compacto de la FA sparse de decode (hoy 6 workgroups por capa, 350 µs) | decode | hasta −4.5 % (−3 ms); **hecha el 2026-09-30 (7c05f19ab, sección 38): −2.1 ms por paso a 40k** | Inferencia: 6 workgroups en 40 CUs | Alto: el modo índice con split_k=13 fue más lento que el compacto sin split_k | test-backend-ops perf con la FA sparse de 3 tokens y split_k forzado |
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
| 22 | MMVQ IQ4_NL fuera del allowlist del mat-vec | decode | < 1 % | El commit b9c196c1c midió tg64 sin cambio; medida el 2026-09-30 en el banco del SP6 (sección 37.5): sin ganancia, descartada | Alto | test-backend-ops perf |
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
- Ubatch 4096: −9 % y checkpoints de 4096 (sección 8). **Reabierta (2026-09-26, sección 29.6)**: se midió en la build
  375, antes del hoisting de ids y del salto de columnas vacías en el kernel de expertos, que son el dato nuevo.
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
| D5 | Vía 7: split_k en el modo compacto de la FA sparse (2.3 ms por paso, 6 workgroups por capa) | −1 ms | hecha el 2026-09-30 (sección 38): −2.1 ms |
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
  Una imagen es un solo paso de prompt (el ViT y todos sus ubatches van dentro de `process_mtmd_chunk`): con una
  imagen al tope, el agente que genera espera ~12 s antes de que el reparto le devuelva su parte. Con dos agentes
  generando, la salida de cada uno ya variaba entre corridas sin el reparto (aceptación 204/358 y 210/343 en la
  v22), así que la compuerta de ese escenario es el agregado dentro del 5 %, no la identidad. El launcher aplica
  el 40 % a todo modelo con NP >= 2; solo se midió qwen38flash. El ejecutable `~/.local/bin/llama-server` solo
  llama `llama_server(argc, argv)` de `libllama-server-impl`, así que un cambio de `common_params` no lo rompe.
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


### 26.1 Barrido de los mat-vec del lote de verificación (cadena v24): sin quick wins

Las vías 3, 10 y 13 atribuían parte del decode a la forma de los mat-vec (filas por workgroup, tamaño del
workgroup). Se midieron con perillas temporales (no quedan en el árbol) sobre los casos de perf del lote de 3 tokens
que ahora trae `test-backend-ops` (router f32, mezcladores hc, alfa/beta, proyecciones, cabeza, gate+up fusionado y
down de expertos), base primero y último:

| Variante | Router f32 m=512 | gate+up iq3_s m=1280 | down iq4_nl m=2560 | hc down m=324 | Resto |
|---|---|---|---|---|---|
| Base (repetida al final) | 9.1 µs | 160 µs | 92-98 µs | 12.6 µs | — |
| f32 con 2 / 4 filas (vía 13) | 9.6 / 9.9 | igual | igual | igual | igual |
| IQ con 8 filas (vía 3) | igual | 159 | 89 | 44.0 | alfa/beta 4.1 → 6.7 |
| IQ con 2 filas | igual | 180 | 105 | 12.2 | proyecciones +5-10 % |
| Workgroup grande en AMD, k ≥ 1024 / ≥ 256 | 11.6 | 212 / 206 | 94 / 163 | 11.6 | proyecciones +40-110 % |

Los valores por defecto de RDNA (4 filas en IQ, workgroup de un subgrupo) son los mejores para estas formas; la
primera medida de la base del router (15.4 µs) fue un artefacto de la primera corrida. Los tensores de menos de ~32
MB quedan en la MALL entre corridas del microbench (el router lee a >500 GB/s aislado, 99 GB/s en el modelo), así que
el microbench no reproduce el costo de DRAM de esos nodos: lo que cuestan en el modelo es ancho de banda y contexto
del grafo, no la forma del workgroup. Las vías 3, 10 y 13 quedan cerradas como afinación; lo que sigue en decode
exige menos bytes (router q8_0 en el GGUF, vía 13 alternativa; cabeza del draft recortada, vía 9) o menos trabajo
(vía 16), no otra forma del kernel.

## 27. Router en Q8_0 adoptado; la cabeza recortada del draft descartada (2026-09-23, cadenas v25 y v26)

De la lista de la sección 26.1, el Director autorizó las dos palancas de decode que exigían transformar archivos:
el router en Q8_0 (vía 13) y la cabeza del draft recortada a un subconjunto de vocabulario (vía 9).

### 27.1 Router en Q8_0 (vía 13): adoptado

`scripts/qwen4exp-requant-router.py` cuantiza a Q8_0 los 48 routers (`ffn_gate_inp`, F32 [2560, 512]) y copia todo lo
demás sin cambios, sobre `scripts/gguf_stream_copy.py` (copia dentro del kernel y page cache soltada cada 2 GiB, reglas
del freeze #10). Archivo `NL3S/qwen4exp-nl3s-hcdi-gu-rq8-oproj8.gguf`; el nombre no contiene el quant anterior, porque
el launcher elige el archivo por subcadena. `build_moe_ffn` acepta cualquier tipo de peso, y la fusión TOPK_MOE de
Vulkan empieza en el SOFT_MAX posterior al router, así que el tipo del router no afecta ninguna fusión.

| Medida (cadena v25, mismo binario) | Router F32 | Router Q8_0 |
|---|---|---|
| PPL holdout, 12 chunks c=4096 | 1.5928 ± 0.0158 | 1.5919 ± 0.0158 |
| Paso de verificación a 40k (mediana, `LLAMA_INPUT_TIMING`) | 53.36 / 52.74 ms (base primero y último) | 51.83 ms |
| Prefill de 39.5k | 74.0 s | 73.8 s |
| Prompts de agente (10, código y tool calls), t/s medio | 47.7 | 48.4 |
| `graph_diff4` | — | 72 nodos (los SET_ROWS conocidos) |

−1.2 ms por paso de verificación (−2.3 %). El t/s de un turno aislado se mueve además con la aceptación del draft,
porque el texto generado cambia con los logits (el turno de 40k aceptó 135/239 frente a 140/228). Adoptado en
`strix-halo/config.ini` (`quant = nl3s-hcdi-gu-rq8-oproj8`); la distribución de referencia del probe pasa a ser
`probe-v26.out`, porque el router cambia los logits por diseño.

### 27.2 Cabeza del draft recortada (vía 9): descartada por el Director

La cabeza del draft MTP (iq4_nl, 2560 × 248320, 341 MiB) cuesta 1.58 ms por paso de draft. La vía la recortaba a un
subconjunto de K tokens: en el archivo del draft, las filas de esos tokens (`output_sub`) y sus ids, y en el grafo
del draft los K logits esparcidos con -inf sobre el vocabulario. La salida del target no cambia, porque el decode
especulativo verifica cada token; lo que cae es la aceptación cuando el token correcto queda fuera del subconjunto.
El subconjunto se construyó con las salidas del propio modelo (26 respuestas de trabajo de agente en español), 12.5
millones de tokens de código y documentos locales, y el relleno por orden de merges del BPE: 99.81 % de cobertura
con K = 98304 y 99.44 % con K = 65536 sobre las respuestas reservadas, sin medir el inglés.

La primera carga falló por el nombre de la tabla de ids (el loader pedía `output_sub_ids.weight`), y el Director
descartó la vía antes de medirla: no quiere recortar la eficacia de la cabeza del draft. El código se retiró del
árbol sin commit y los archivos de draft se borraron. Queda documentado aquí qué implicaba: un cambio de loader y
grafo del draft, una tabla de ids por archivo y un subconjunto que tendría que cubrir al menos español e inglés, con
un corpus de construcción y de evaluación bastante más grande que 26 respuestas.

## 28. Presupuesto del paso de decode a 40k (2026-09-23, cadena v27)

Ventana sin cambios de código (binario 7aefa3489, archivo rq8, draft IQ4_NL). Cinco cargas NP=1 (ctx 57344, MTP
n-max 2), una a la vez: sin instrumentar, timing de host (`LLAMA_INPUT_TIMING` + `LLAMA_SPEC_TIMING`), perf logger
concurrente, perf logger serial y otra vez sin instrumentar. Cada carga hace el prefill de 39.5k una vez y dos turnos
de 1024 tokens sobre ese prefijo (384 con el logger): greedy y el muestreo de producción (temp 0.7, top_k 20, top_p
0.95), con `ignore_eos` para que cada carga decodifique los mismos tokens. Relojes, actividad y potencia de la GPU
muestreados cada segundo con `amdgpu_top -gm`. Cero timeouts de anillo; producción recargada con el probe idéntico al
de v26. Herramientas en `~/dbg/merge`: `decode_budget.py` (rig), `budget_parse.py` (línea de tiempo de host),
`vk_budget_parse.py` (bloques del perf logger por ubatch), `vk_family.py` (familias), `power_sampler.py`.

### 28.1 El paso

| Carga | Greedy, ms por paso | Producción, ms por paso |
|---|---|---|
| Sin instrumentar (primera) | 64.29 | 63.56 |
| Timing de host | 66.06 | 65.75 |
| Perf logger concurrente | 79.38 | 79.32 |
| Perf logger serial | 79.80 | 79.23 |
| Sin instrumentar (última) | 63.76 | 63.78 |

Un paso de verificación sin instrumentar dura **63.9 ms** y, en este texto, entrega 2.19 tokens (1024 tokens en 467
pasos, 34 t/s). El muestreo de producción cuesta lo mismo que greedy: el pedido greedy también pasa por el top_k 20
del server en la CPU. La carga con timing es 2.1 ms (3 %) más lenta, así que las partes de host de 28.2 vienen de un
régimen algo perturbado; las cifras de GPU no dependen de eso.

### 28.2 Reparto del paso (carga con timing, turno greedy, 465 pasos)

| Tramo | Pared, ms | GPU ocupada, ms | Detalle |
|---|---|---|---|
| Verificación del target, 3 tokens | 53.49 | ≤ 47.6 | set_inputs 1.80 con la GPU parada; grabación en host 15.6 solapada con la GPU; espera 35.9 |
| Hook del draft, 3 tokens | 1.78 | 0.38 | build 0.14 + alloc 0.59 en cada paso: el grafo nunca se reutiliza (0 de 482) |
| Dos pasos del draft, 1 token | 7.71 | 2 × 2.96 | cabeza del draft 1.61 por paso; el primer paso tras el hook se reconstruye (build 0.09 + alloc 0.53) |
| Huecos de host | 3.08 | 0 | muestreo del target en CPU 1.27 (3 posiciones), clon del sampler 0.21, post 0.22, resto 1.4 |
| **Total** | **66.06** | **≤ 54.0** | |

La GPU trabaja ≤ 54 ms de 66 (gfx_activity medio 84-86 %). Unos 12 ms del paso son host en serie con la GPU:
set_inputs de la verificación 1.8, host del scheduler antes del primer submit 3.8 (medido en la sección 28.6),
hook 1.4, pasos del draft 1.7 y huecos 3.1. Dónde se va el tiempo de
host del hook y del draft: `llama_context` guarda un solo grafo previo (`gf_res_prev`), y el hook (3 tokens) y el
paso del draft (1 token) se alternan en el mismo contexto, así que cada paso reconstruye y reasigna dos grafos.

### 28.3 La GPU de la verificación por familia

Método: el perf logger concurrente pone un timestamp en cada sincronización que el grafo ya tiene y mide cada grupo de
nodos entre dos barreras con su concurrencia real. El último grupo (la cabeza del target) queda sin timestamp y se
toma del serial; dentro de un grupo, el tiempo se reparte en proporción a los tiempos del serial. Son cotas
superiores, porque el logger agrega una barrera y un timestamp por sincronización. Las familias salen de los tipos y
formas del header del GGUF.

El grafo de verificación tiene **2327 sincronizaciones para 3269 nodos**, y el serial (48.0 ms) es casi igual al
concurrente más la cabeza (47.6): casi no hay concurrencia, así que el número de barreras es un blanco medido.

| Familia | ms | Bytes por paso | GB/s |
|---|---|---|---|
| Expertos gate/up (IQ3_S, 47 capas) | 11.18 | ~1.79 GB con ~27 expertos por capa | ~160 |
| Expertos down (IQ4_NL) | 5.61 | ~1.19 GB con ~27 | ~213 |
| FA sparse (12 capas QSA) | 3.78 | crece con la profundidad | — |
| Mezcladores hc (96 + 96 mat-vec: 324 × k 10240 y 10240 × k 320) | 3.37 | 360 MB | **107** |
| Copias y filas (CPY, CONT, GET/SET_ROWS, CONCAT) | 3.20 | — | — |
| GDN attn_qkv (q5_K) | 3.04 | 649 MB | 213 |
| Elementwise y otros ops chicos | 2.82 | — | — |
| Cabeza del target (q6_K) | 2.30 | 521 MB | 226 |
| GDN ssm_out / attn_gate (IQ4_NL) | 1.75 / 1.60 | 318 / 318 MB | 182 / 199 |
| Normas | 1.11 | — | — |
| QSA attn_q/k/v | 1.09 | 232 MB | 212 |
| Experto compartido | 1.05 | 133 MB | 127 |
| QSA attn_output (q8_0) / QSA top-k | 0.98 / 0.98 | 200 MB / — | 204 / — |
| Router (q8_0) / recurrencia y conv GDN | 0.74 / 0.74 | 67 MB / — | 91 / — |
| hc ops, ruteo MoE, indexer bf16, alfa/beta, otros mat-vec | 2.28 | — | — |
| **Total** | **47.6** | | |

Agrupado: pesos (mat-vec y expertos) 33.9 ms para ~5.9 GB (~174 GB/s de media), atención (FA y top-k QSA) 4.8 ms y
ops chicos 8.9 ms (copias, elementwise, normas, recurrencia, hc ops, ruteo), casi uno por barrera. El número de
expertos distintos por capa con 3 tokens no se midió en esta ventana (~27 según la sección 24.1), así que los GB/s de
los expertos son estimados; los de los pesos densos salen de bytes exactos. Grafos del draft: el paso de 1 token
suma 2.96 ms (cabeza 1.61, FA densa a 40k 0.50) y el hook 0.38 ms (22 nodos tras el recorte de la sección 19).

### 28.4 La máquina está limitada por potencia

Muestras de `amdgpu_top -gm` durante el decode de las cargas sin instrumentar (turno de producción, ~30 s):

| Medida | Decode | Prefill |
|---|---|---|
| Potencia del paquete | 85.0 W, plana | ~85 W (picos de 94-107 W al arrancar) |
| GPU / núcleos de CPU | 35-36 W / 1-1.5 W | 27-32 W / 1-5 W |
| Resto (SoC, memoria, fabric) | ~48 W | ~50 W |
| gfxclk | ~2.48 GHz (máximo 2.9) | 2.1-2.4 GHz |
| fclk / uclk | ~1.67 GHz (máximo 2.0) / 1.0 GHz (máximo) | ~1.65-1.75 / 1.0 |
| gfx_activity | 84-86 % | 94-99 % |

El contador `throttle_residency_sppt` avanza sin pausa mientras la GPU trabaja (unas 30000 unidades por turno de
30 s) y queda quieto en reposo; `spl` y `prochot` no se mueven, `fppt` y `thm_core` poco. Placa AXB35-03, BIOS 1.08,
`powerprofilesctl` en performance. No se sabe la unidad del contador ni los modos de potencia que expone esta BIOS.
El límite afecta al prefill además del decode: el prefill corre a 99 % de actividad con la GPU a 2.1-2.4 GHz.

Consecuencia para las palancas: con el paquete en su tope, quitar tiempo ocioso de la GPU sube la potencia media y
el SMU responde bajando relojes, así que las palancas de host recuperan menos que el tiempo que quitan; las que leen
menos bytes ahorran tiempo y potencia de memoria (~48 W no son ni GPU ni CPU). Las palancas se ordenan solo por ms
por paso medidos de punta a punta, base primero y último, nunca por la aritmética de este presupuesto.

### 28.5 Palancas que salen del presupuesto

| Palanca | Costo medido hoy | Ahorro | Estado |
|---|---|---|---|
| Modo de potencia del paquete (BIOS o límites PPT) | GPU a ~2.48 GHz y fclk ~1.67 GHz bajo SPPT, en decode y prefill | desconocido sin cambiar el modo | decisión del Director; la re-medida es esta misma cadena más el prefill, con los contadores térmicos |
| GPU parada antes del primer submit de la verificación | 5.6 ms (set_inputs 1.8 + scheduler 3.4 + primer lote 0.4) | **−2.5 ms medidos** | **hecho (sección 28.7, fb27b714f)**: 2 sincronizaciones por split en vez de una por entrada; quedan set_inputs 1.8 y el scheduler 0.8 |
| Grafos del draft reconstruidos (hook y primer paso) | ~1.3 ms (build + alloc de dos grafos) | ~1.3 ms | reutilizar dos grafos en el contexto del draft; hoy `gf_res_prev` guarda uno |
| Mezcladores hc a 107 GB/s | 3.37 ms | ~1.7 ms a 210 GB/s | 324 filas × k 10240 dan 81 workgroups; candidato split-k. El microbench no sirve (tensores en la MALL, sección 26.1) |
| Expertos gate/up a ~160 GB/s | 11.2 ms | ~2.5 ms a 210 GB/s | D2; sin ganancia por forma de kernel (sección 26.1) |
| Muestreo del target en CPU | 1.27 ms + clon 0.21 | < 1 ms | microbench de CPU en v28 (`sampler_bench.cpp`) |
| set_inputs de la verificación | 1.8 ms | ~1 ms | D7, agrupación QSA incremental |
| Ops chicos y barreras | 8.9 ms en ~2300 dispatches | fusiones de la fila 17 (~220 dispatches) | medido por familia; cada fusión se mide de punta a punta |

Ventana siguiente (v28), producción descargada, base sin instrumentar primero y último: una carga con
`LLAMA_SPEC_TIMING` solo, una traza de submits (`GGML_VK_TRACE_SYNC`), `GGML_VK_MAX_NODES_PER_SUBMIT` en 25, 50 y 200 y
el microbench del muestreo antes de la primera carga. Resultado en 28.6.

### 28.6 Dónde está parada la GPU en la verificación (cadena v28)

Cargas hechas: sin instrumentar (63.46 / 63.87 ms por paso, greedy / producción), `LLAMA_SPEC_TIMING` solo (65.12 /
64.48: la instrumentación sola ya agrega ~1.1 ms) y la traza (`LLAMA_INPUT_TIMING` + `GGML_VK_TRACE_SYNC`, 384 tokens por
turno). La traza marca, en el reloj del host, la entrada al `graph_compute` de Vulkan, cada submit y la espera del
fence; su desfase con `ggml_time_us` sale de la mediana de (fin del ubatch − fin de la espera). Mediana de 177 y 183
verificaciones:

| Tramo de la verificación | ms | GPU |
|---|---|---|
| set_inputs (agrupación QSA, hash y prefetch PLE) | 1.81 | parada |
| Scheduler antes del `graph_compute` de Vulkan (split de CPU y copias de sus salidas) | 3.44 / 3.58 | parada |
| Grabación hasta el primer submit | 0.38 | parada |
| Primer submit → fence (33 submits; la grabación termina a los 11.8 ms) | 46.5 | ocupada |
| **compute + espera de `LLAMA_INPUT_TIMING`** | **50.4** | |

Desde el primer submit la GPU no espera al host: 46.5 ms coinciden con el trabajo que mide el perf logger (≤ 47.6,
cota superior) y la grabación termina 35 ms antes que la GPU. El barrido de `GGML_VK_MAX_NODES_PER_SUBMIT` no se
corrió (el launcher rechazó las cargas por una descarga activa, regla del freeze #5) y ya no hace falta: el tamaño del
lote de submit no es el problema. La GPU parada son 5.6 ms por paso, todos antes del primer submit.

El grafo del target tiene 2 splits con batch chico (`graph splits = 6 (with bs=2048), 2 (with bs=1)`): el primero es de
CPU (los `get_rows` de `token_embd` y de la tabla PLE lazy) y el segundo el de Vulkan. La primera hipótesis, fallos de
página en la tabla PLE, quedó refutada en la cadena v29: el server tiene 0.01-0.05 fallos mayores por paso durante el
decode (`power_sampler.py` cuenta `majflt`), y la tabla está residente.

El mecanismo, en la traza cruda de la cadena v29: antes de cada `graph_compute` de la verificación, el contexto Vulkan
del target recibe **29 `synchronize`, 15 de ellos con trabajo grabado** (submit y espera del fence, 50-190 µs cada uno) y
los demás vacíos (~30 µs). Los emite `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`): las entradas
del grafo se asignan al backend de CPU (regla `1.inp` del scheduler), cada una es una entrada del split de Vulkan, y
sin eventos (sin pipeline parallel) el scheduler llama a `ggml_backend_synchronize(split_backend)` antes de copiar
**cada** entrada. En Vulkan cada una envía la copia asíncrona anterior y espera su fence, con la GPU parada. Upstream
tiene el mismo bucle. Con una sola copia de las entradas basta una sincronización antes de la primera copia (nada de
lo que se envía entre copias lee las copias de entrada) y otra después de la última copia asíncrona, para que ninguna
quede pendiente cuando el split de CPU del ubatch siguiente reescriba su origen; el código actual deja pendiente la
última. Ese cambio se mide en la cadena v30.

Muestreo del target: el microbench de CPU (`sampler_bench.cpp`, cadena v29) da 0.14 ms por posición para el llenado de
248320 candidatos y la cadena de samplers del launcher (0.077 + 0.063 ms), frente a 0.42 ms por posición que mide
`LLAMA_SPEC_TIMING` en el server (1.27 ms por 3 posiciones). Los ~0.85 ms restantes por paso están fuera de la cadena
del microbench (samplers del server que el microbench no incluye, como el logit bias de tokens suprimidos o el
presupuesto de razonamiento, y el clon y el accept); sin atribuir.

### 28.7 Una sincronización por split en las copias de entrada: adoptado (cadena v30, fb27b714f)

Cambio en `ggml_backend_sched_compute_splits` (`ggml/src/ggml-backend.cpp`), solo para el caso sin eventos: una
`ggml_backend_synchronize(split_backend)` antes de la primera copia de entrada de un split y otra después de la
última copia asíncrona, en lugar de una antes de cada entrada. Las copias de expertos (pesos inmutables en host) no
cuentan para la segunda. El camino con eventos (pipeline parallel) no cambia. Ventana NP=1 a 40k, base (copia de
`build/bin` cargada por `LD_LIBRARY_PATH`, verificada en `/proc/<pid>/maps`) primero y último:

| Carga | Greedy, ms por paso | Producción, ms por paso | t/s greedy / producción |
|---|---|---|---|
| Base (primera) | 64.77 | 64.16 | 33.82 / 33.08 |
| **Nueva** | **61.82** | **61.61** | **35.43 / 34.45** |
| Base (última) | 63.77 | 72.58 (*) | 34.35 / 29.24 |

(*) Turno anómalo: gfxclk bajó a 1.83 GHz en ese turno y el paso subió 8.8 ms; causa no identificada, se descarta. Las
bases sin instrumentar de las cadenas v27-v29 dieron 63.3-64.4 ms (greedy) y 63.6-64.5 (producción): la ganancia es
**−2.5 ms por paso (−4 %)**, +4 % de t/s.

Traza (`GGML_VK_TRACE_SYNC`, mediana de 177 y 183 verificaciones): sincronizaciones del target antes del
`graph_compute` 29 → 2 (15 → 1 con trabajo), host del scheduler 3.4-6.0 → 0.8 ms, compute + espera de la verificación
50.4-52.4 → 48.4-48.5 ms; la GPU sigue ocupada 46.8-46.9 ms desde el primer submit.

Exactitud: respuestas de la build nueva idénticas a las de la base en greedy y en muestreo de producción (1024 tokens
cada una, `ignore_eos`), base contra base idénticas, `graph_diff4` 72 nodos (las vistas SET_ROWS conocidas), probe de
producción idéntico a `probe-v26.out`, conversación texto → imagen → texto 5/5 y la imagen de 2048×2048 en 14.1 s
(el encoder de visión usa su propio scheduler, así que también pasa por el cambio), cero timeouts de anillo. NP=2 en
producción: un slot solo 33.8-34.1 t/s (v26: 32.2), dos slots a la vez 38.3-38.9 t/s agregados (v26: 36.9-38.5).

Queda de la GPU parada antes de la verificación: set_inputs 1.8 ms (D7) y 0.8 ms del scheduler (split de CPU, copias y
las 2 sincronizaciones).

### 28.8 Perfil de CPU del host durante el decode (cadena v31)

Con `kernel.perf_event_paranoid=2` (puesto por el Director, no persistente), una carga NP=1 con la build de producción
(fb27b714f), el prefill de 39.5k y dos turnos de 1024 tokens bajo `perf record -e cycles:u`: plano a 4999 Hz (turno
greedy, 62.00 ms por paso) y con pilas DWARF a 999 Hz (turno de producción, 62.12 ms por paso; `perf` no cambia el
paso). Clasificación por pila con `~/dbg/merge/perf_fold.py`, 402 pasos. El hilo principal del server está ocupado
34.1 ms por paso, de los que ~16 son la espera activa del fence mientras la GPU trabaja; el resto es trabajo real:

| Actividad | ms por paso | Camino crítico |
|---|---|---|
| Grabación de comandos Vulkan (todos los grafos) | 9.4 | no en la verificación (la GPU no espera desde el primer submit); sí en el hook y el draft |
| Chequeos de fusión dentro de la grabación (`ggml_can_fuse*`, `ggml_vk_can_fuse`) | 2.5 | ídem |
| Contadores atómicos de `shared_ptr` dentro de la grabación | ~2.2 | ídem |
| Reconstrucción de los dos grafos del draft: `ggml_vk_graph_optimize` | 0.76 | sí |
| Reconstrucción: resto de alloc y split del scheduler | 0.47 | sí |
| Reconstrucción: build del grafo del modelo | 0.18 | sí |
| `set_input_qsa` (agrupación QSA, D7) | 1.20 | sí |
| `set_input_kq_mask` (atención densa del draft a 40k, tres grafos por paso) | 0.51 | sí |
| Muestreo del target: llenado de 248320 candidatos × 3 | 0.67 | sí |
| Muestreo del target: top-k (`partial_sort`) | 0.38 | sí |

`ggml_vk_graph_optimize` corre en cada grafo reconstruido (0.38 ms por grafo): su ventana es de 20 nodos, pero por
cada candidato vuelve a probar ~13 patrones de fusión y, para el patrón QSA, todos sus desplazamientos. El llenado de
candidatos cuesta 0.18-0.22 ms por posición en el server frente a 0.077 en el microbench con logits en memoria normal;
los logits del server viven en el buffer de salida pinned de Vulkan (tipo de memoria sin verificar).

### 28.9 `match_pattern` del optimizador de grafos con salida temprana: adoptado (cadena v32, cda30216f)

`ggml_vk_graph_optimize` comparaba cada op de un patrón de fusión aunque el primero ya no coincidiera; ahora retorna en
la primera diferencia, con el mismo orden de salida. Perfil DWARF de la carga nueva (turno de producción, 415 pasos):
el optimizador baja de 0.77 a 0.60 ms de host por paso; el resto del reparto de la sección 28.8 no cambia. Paso sin
instrumentar 61.98 / 61.36 ms (greedy / producción) frente a 61.82 / 61.61 de la build de v30: la ganancia queda bajo el
ruido del paso. Respuestas idénticas a las de la build de v30 en los dos turnos, `graph_diff4` 72 nodos, probe idéntico
a `probe-v26.out`, conversación con imagen 5/5 y la imagen de 2048×2048 en 14.2 s, cero timeouts de anillo.

Lo que queda del optimizador (0.60 ms por paso) es `is_src_of` y la ventana de 20 nodos por candidato, en cada uno de
los dos grafos del draft que se reconstruyen por paso; la palanca completa es no reconstruirlos (sección 28.5).

## 29. Techo físico del decode, camino al 2× del prefill y referencias externas (2026-09-26, sin carga)

Revisión sin GPU ni carga de modelo: dos motores externos que corren este mismo modelo (TensorFold en MLX y Strata en
CUDA), los cuants GSQ-RCO de ISTA-DASLab y el drafter DFlash publicado para Flash Next, contrastados con el
presupuesto de la sección 28 y con el código del fork. Fuentes consultadas el 2026-09-26. Las cifras marcadas como
estimación salen de aritmética sobre datos medidos: la sección 28.4 impide ordenar palancas con esa aritmética, y aquí
solo responde si un 2× es físicamente posible.

### 29.1 Censo de bytes por paso de verificación y los cuants GSQ-RCO

`~/dbg/merge/quant_census.py` lee solo el header del GGUF de producción (formas y tipos) y aplica a esas formas los
tipos por tensor que ISTA-DASLab publica en `tensor-allocation/*.rco-allocation.txt`
(https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF; copias en `~/dbg/merge/gsq-rco/`). Gate + up y
hc down + inject, fusionados en nuestro archivo, se reparten por número de elementos. Los expertos cuentan ~27
distintos por capa de 512 para 3 tokens, la estimación de la sección 24.1 que usa también la 28.3, así que en los
expertos el censo no es una verificación independiente; las familias densas coinciden con los bytes exactos de la
28.3 (hc 0.36 GB, cabeza 0.52 GB). GB por paso de verificación de 3 tokens:

| Archivo | Expertos | hc | Cabeza | Resto denso | Total | Promedio de tareas (BF16: 93.12) |
|---|---|---|---|---|---|---|
| Producción (`nl3s-hcdi-gu-rq8-oproj8`) | 3.03 | 0.36 | 0.52 | 1.97 | **5.88** | — (PPL holdout 1.5919) |
| GSQ-RCO IQ3_S (83.6 GB) | 2.65 | 1.28 | 0.52 | 2.31 | **6.76** (+15 %) | 93.26 |
| GSQ-RCO IQ3_XXS (75.8 GB) | 2.26 | 1.28 | 0.44 | 2.06 | **6.04** (+3 %) | 92.57 |
| GSQ-RCO Q2_0 (66.4 GB) | 1.79 | 1.28 | 0.44 | 1.62 | **5.14** (−13 %) | 89.07 |

La calidad es la de su tarjeta (AIME25, GPQA-Diamond, LiveCodeBench v6), no comparable con nuestra PPL. Sus 290
tensores de hyper-connections van en BF16 (los nuestros en IQ4_NL) y anulan el ahorro en expertos: IQ3_S e IQ3_XXS
leen más o lo mismo que producción. Q2_0 lee menos, pero pierde ~4 puntos (LiveCodeBench 87.43 → 81.14). Ninguno
trae las transformaciones del fork (hc inject fusionado, gate_up, router Q8_0). Idea sin medir: injertar solo los
expertos IQ3_XXS en nuestro archivo, 5.11 GB por paso (−13 %). Incógnitas: la calidad de la mezcla (su asignación RCO
se optimizó para el archivo completo) y la tasa real en Vulkan de sus tipos por capa (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
IQ3_S, Q2_0, IQ4_NL); nuestros expertos IQ3_S corren a ~160 GB/s frente a ~213 de IQ4_NL (sección 28.3). El backend
Vulkan del fork tiene mat-vec y mat-vec-id para Q2_0 (`ggml-vulkan.cpp:5923`, `:6009`).

### 29.2 Drafts n-gram junto al MTP ("copy windows"): soportado en el código, nunca ejecutado

TensorFold (https://github.com/ashhart/TensorFold, receta `docs/recipes/qwen3.8-flash-next.md`; M3 Ultra, MLX, 4
bits, T 1.0 / top-k 20 / top-p 0.95, pensamiento activo) sirve el mismo modelo con 1-3 drafts MTP y "copy windows":
cuando el contexto contiene el texto que se escribe, una ronda verifica hasta 7 tokens copiados; la entrada exige 8
tokens coincidentes, porque con coincidencias cortas fallaron 56 de 70 tokens copiados y el total perdió un 5 %. Una
edición de archivo decodifica a 190 t/s, frente a 105-107 de una respuesta corta y 103-115 de su prompt de agente de
23k: la ganancia es solo de los turnos que copian texto. sushi midió la misma idea en el mismo modelo, con reglas de
entrada más finas, tarea por tarea (29.8).

Lectura del código del fork, sin ejecutar:

- `--spec-type` acepta una lista separada por comas (`common/arg.cpp:4292-4298`).
- Las implementaciones se crean con los tipos n-gram antes que los drafts por modelo
  (`common/speculative.cpp:2730-2737`) y en cada paso gana la primera que produce un draft (`:2923-2960`).
- El hook del MTP procesa el lote verificado completo, sea quien sea el que lo propuso, con un batch del tamaño del
  `n_batch` del contexto del draft (`:1469-1470`, `:1595-1690`), y su `accept` ignora `is_other` (`:1846`).
- Los defaults no sirven para este uso: ngram-mod con `n_match` 24 y drafts de 48-64 tokens, ngram-map con `size_n` 12
  y `size_m` 48 (`common/common.h:355-366`). Habría que llevarlos a una entrada de ~8 tokens y una ventana de ~7.

Costo de las filas extra en este MoE: 8 tokens consecutivos eligen ~40 expertos distintos por capa en sus 80 plazas
(medido por TensorFold en el mismo modelo), así que una verificación de 8 filas leería ~7.3 GB frente a los 5.88 de la
de 3 (estimación con el censo de 29.1). Su política de profundidad por umbrales de aceptación (1 draft bajo 80 %, 2
bajo 90 %, 3 en otro caso) dio 91.1 t/s frente a 86.8 con un tope fijo de 1; es otro controlador que nuestro
`--spec-draft-adaptive` (EMA calibrado para DFlash), que perdió con esta cabeza (`strix-halo/run-server.sh`,
comentario del MTP de qwen38flash).

### 29.3 Drafters publicados para Flash Next

https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash: drafter DFlash de 498M parámetros en BF16, bloque 7,
entrenado sobre el target NVFP4. τ 2.88 con bloque 5; frente al MTP nativo con k=4, +3.9 % agregado (matemática +28 %,
código +0.4 %, chat −5.9 %), en greedy, sin pensamiento, contexto de 8k, vLLM TP2 sobre dos DGX Spark; los autores
advierten que la varianza entre arranques es del tamaño del efecto. No es comparable con producción (pensamiento y
muestreo), pero no muestra un salto de aceptación sobre el MTP: hoy no hay un drafter que llene ventanas anchas en
este modelo. Como referencia de aceptación, Strata (greedy, hasta 4 tokens por ventana, drafts mientras p ≥ 0.5)
mide 2.42-3.63 tokens por ventana sobre un prompt de revisión de código; producción, 2.19-2.46 con muestreo y n-max 2.

### 29.4 Techo físico del decode a 40k

Paso de verificación de 3 tokens con MTP n-max 2 y los bytes del censo:

| Componente | ms |
|---|---|
| Pesos, 5.88 GB a 227 GB/s (mejor tasa medida, sección 3, fila 3) / 256 GB/s (teórica) | 25.9 / 23.0 |
| FA sparse + top-k QSA (medido, sección 28.3) | 4.8 |
| Draft: dos pasos de 1 token + hook (GPU, sección 28.2) | 6.3 |
| **Paso mínimo, sin host ni ops chicos** | **37 / 34** |
| Paso medido hoy (secciones 28.7 y 28.9) | 61.7 |

Con ~2.3 tokens por paso, el techo es 62-67 t/s frente a ~37 hoy: 1.7-1.8×. El 2× solo aparece en el límite teórico
y reduciendo además la atención y el draft. Entre el paso de hoy y el mínimo hay ~9 ms de host en serie con la GPU
(sección 28.2 menos la 28.7) y 8.9 ms de ops chicos (sección 28.3). **Corrección (2026-09-28)**: la primera versión
contaba el replay de command buffers y estimaba 43 / 47 / 55 t/s; el replay no recupera tiempo, porque desde el primer
submit la GPU no espera al host y la grabación termina 35 ms antes (sección 28.6). El plan por subproyectos
(`decode-plan-2026-09-28.md`) estima 42-46 t/s (+14-23 %) con el host del camino crítico, las fusiones y el draft
dentro de la verificación, y 44-49 t/s sumando la tasa de los pesos; por la sección 28.4 recupera menos de lo que
quita. Pasar el techo exige más tokens por lectura de pesos (29.2, 29.3) o menos bytes: la parte densa son 2.85 de los
5.88 GB (wqkv q5_K, cabeza q6_K, oproj q8_0), y bajarla es una decisión de calidad.

### 29.5 El reparto CPU/GPU de Strata no aplica aquí

Strata (https://github.com/Niko1221/Strata, paper en `docs/paper/Strata-Paper.pdf`): RTX 5070 de 12 GB (672 GB/s) y
64 GB de DDR5-5200 (41-52 GB/s). Deja en VRAM la parte densa (~3.5 GB por token en su Q2_0) y los ~4,500 expertos más
usados (acierto 0.71-0.78), y la CPU calcula el resto en paralelo. Sin MTP da 47-57 t/s a 4K, frente a los ~15 del
llama.cpp con expertos en la CPU que cita su autor: casi toda la ganancia es ese reparto, que no existe en una memoria
unificada. Su v1 atiende una petición a la vez, solo en greedy y sin caché de prompts. Lo transferible es el censo de
sus cuants (29.1).

### 29.6 Camino al 2× del prefill

Existencia en esta máquina: pwilkin, HIP, 1086 t/s (llama-bench, bloque de 16384 a profundidad 40000, ubatch 16384,
sin MTP ni mmproj; sección 6) frente a 561 t/s nuestros (server, 39.5k desde 0, con MTP y visión). Las condiciones
difieren; la brecha de ~1.9× tiene una causa identificada: con ubatch 2048 cada experto recibe ~40 filas y los
MUL_MAT_ID corren a 6.4-7.3 TFLOPS frente a 18-23 de los densos (sección 2), unos 24 de los 70.4 s (estimación: los
26.3 s del perfil de la sección 2 menos 2.6 s de las secciones 22 y 23). Con ubatch 16384 recibirían ~320. Que el lote
grande los lleve a la tasa de los densos en Vulkan es una hipótesis: el tile por filas por experto salió neutro
(sección 8), y el −9 % de ubatch 4096 se midió antes del hoisting y del salto de columnas vacías.

Estimación del camino: expertos a la tasa de los densos, ~24 → ~9 s (−15 s); atención sin máscara y FA más eficiente
(pwilkin midió 1.48× en su camino sin máscara), ~−5 s; fusiones del residual ancho y host, ~−5 s. Total ~45 s (~880
t/s, 1.55×); el techo de ~62-64 s de la sección 13 suponía ubatch 2048, otro régimen. El re-prefill de 2k de cada
turno de agente no gana con el lote grande (un bloque de 2048 ya es un ubatch), solo con la atención y las fusiones.

Obstáculos:

1. Memoria: compute buffer de 4.3 GiB con ubatch 2048 (sección 6; ~3.9 GiB a n_ctx 135168 NP=1 con la regla de
   split-on-inputs del scheduler) y ~22 GiB libres con producción cargada. Cuánto crece con el ubatch y qué tensores
   lo dominan no está medido: `-lv 4` en dos tamaños de ubatch antes de nombrar qué lo hace caber.
2. Los checkpoints están atados al ubatch y cada turno re-procesa desde el último: un lote grande exige guardar el
   estado GDN cada 2048 tokens dentro del lote.
3. Kernels de expertos a eficiencia WMMA (coopmat) en Vulkan con este modelo: no hay referencia publicada. Cada dispatch
   queda lejos del tope de 2 s del anillo (gate/up: 9.5 ms a 2048, ~76 ms a 16384 a la tasa de hoy).

### 29.7 Ventanas propuestas, en orden

1. Prefill: ubatch 2048, 4096 y 8192 en la build actual, NP=1, producción descargada, con el compute buffer (`-lv 4`)
   y los TFLOPS de expertos por nodo. Decide si el camino de 29.6 existe en Vulkan y cuánta memoria cuesta.
2. Modo de potencia: todos los GB/s y relojes de 29.4 y 29.6 son bajo SPPT (sección 28.4) y es la única palanca que
   mueve los dos techos. Cambio del Director en el BIOS; re-medida de decode y prefill con los contadores térmicos.
   **Observado el 2026-09-28 (sección 30)**: el paquete corre a 117-130 W; decode −10 % y prefill −7.2 % sin cambio de
   código.
3. Host del camino crítico y fusiones del decode: subproyectos SP1-SP6 de `decode-plan-2026-09-28.md` (el replay
   de command buffers queda descartado, 29.4).
4. N-gram + MTP con entrada ~8 y ventana ~7 y las reglas de sushi (29.8: regla de línea, a lo sumo 8 drafts, ronda
   elegida contra el costo medido), sobre un turno de edición y uno de prosa, base primero y último, con
   `repeat_probe` y `depth_repeat`. **Hecho y adoptado el 2026-09-28 (sección 31.3)** con clave y ventana de 7, sin la
   regla de línea ni la compuerta de costo, que quedan pendientes.
5. Draft MTP muestreado con la regla p/q (29.8): exacto; tokens por paso con el muestreo de producción, greedy sin
   cambios, base primero y último.
6. Aceptación "typical" (29.8): solo si el Director acepta que cambie la distribución de salida, con una compuerta de
   calidad que no sea solo el NLL del texto emitido.

### 29.8 sushi (MLX): la ventaja es el hardware; lo transferible está en la aceptación (2026-09-28)

https://github.com/beamivalice/sushi (motor en Zig sobre MLX, fork de mlx-serve; `docs/perf-baselines.md`,
`docs/engine-mtp.md`, `docs/engine-exl3-experts.md`, `docs/quality-kld.md`, `src/mtp_acceptance.zig`, consultados el
2026-09-28). Sirve este modelo con los expertos en formato EXL3 (trellis) y el resto en afín.

Dónde está su ventaja: en el M5 Max (~540 GB/s entregados y unidades de matriz NAX, ~55 TFLOPS emitidos) decodifica a
93-98 t/s con MTP y hace prefill a 1670-1960 t/s. El mismo motor en un M2 Max de 64 GB (400 GB/s, sin NAX; celda
reportada por un usuario, greedy, pensamiento activo) da 38.6 t/s de decode a 2k, 37.8 a 16k y 399-413 t/s de
prefill. Eficiencia de una pasada completa: su M2 Max lee 5.58 GB por token en 33-36 ms (155-167 GB/s, ~40 % de 400);
nuestra verificación, 5.88 GB en 47.6 ms de GPU (124 GB/s, ~48 % de 256). Es la misma clase. Su decode también está
limitado por los huecos entre dispatches (~860 kernels por token en el M5 Max, ~7 µs por frontera, 9.8 ms de kernels
en ~18 ms), el mismo diagnóstico de la sección 28; nuestra verificación tiene 3269 nodos y 2327 sincronizaciones.

Lo que transfiere, en orden:

1. Draft muestreado con la regla p/q (exacto; sin medir aquí). Su modo `exact` muestrea el draft de su distribución q
   y lo acepta con probabilidad min(1, p/q), remuestreando el residuo al rechazar. El fork toma el argmax del draft
   (`common/speculative.cpp:1776`) y el server acepta cuando la muestra del target coincide
   (`common/sampling.cpp:692`): equivale a la regla p/q con q concentrada en un token, y la aceptación es p(argmax),
   donde un draft muestreado obtendría Σ min(p, q), mayor en las posiciones de alta entropía. El fork ya tiene el
   camino del residuo (`common/sampling.cpp:722`) y el campo `dists` en los parámetros del draft
   (`common/speculative.h:79`), pero nada lo llena y el server no lo usa (`tools/server/server-context.cpp:4033`; el
   otro camino de ese punto, `server_sample_and_accept_synth`, es la aceptación sintética para benchmarks). No cambia
   la distribución de salida y greedy no se ve afectado.
2. Aceptación "typical" (con pérdida; decisión del Director). Acepta el draft si p(draft) > min(ε, δ·exp(−H(p))), con
   δ 0.2, ε 1 y H la entropía de la distribución filtrada del target, sin moneda aleatoria. Medido en su Sushi-4bpw a
   T 1.0 / top-k 20 / top-p 0.95, 16 prompts × 512, 2 semillas: 69.2 / 63.3 → 80.0 / 76.6 t/s (2.1-2.2 → 2.4-2.6
   tokens por ronda), NLL del texto emitido 0.7819 frente a 0.7805, dentro del ruido de semilla. Con la "cola greedy"
   (drafts de profundidad ≥ 1 por argmax), +24-29 % sobre `exact` en su 2.6bpw, pero el NLL cae de 0.680 a 0.603
   porque el texto se acerca al argmax: su proxy de calidad no ve la pérdida de diversidad. Cambia la distribución de
   salida y se midió a T 1.0; producción muestrea a 0.7, donde el target está más concentrado y la ganancia sería
   menor.
3. Lookup dentro de la ronda MTP (29.2), medido en el mismo modelo (Sushi-3bpw, M5 Max, greedy / muestreo 0.6, 0.95,
   20): copiar el archivo +21 % / +19 %, renombrar en él +21 % / +17 %, corregir un bug +16 % / +17 %, tool call
   write_file +9 % / +11 %; renombrar detrás de ~32k / ~64k de contexto +24 % / +18 %; diff unificado, código nuevo y
   prosa dentro del ruido. Reglas: los últimos 3 tokens confirmados más el siguiente coinciden antes en el prompt o la
   salida y concuerdan hacia atrás 8+ tokens; una coincidencia corta (sufijo < 32) debe concordar pasado el inicio de
   una línea (sin esa regla el diff perdía 4.8 % en greedy y 11.5 % en muestreo); a lo sumo 8 drafts; la ronda se
   elige contra el costo medido de la ronda MTP. Los tipos n-gram del fork no tienen la regla de línea ni la compuerta
   de costo.

Lo que no transfiere:

- EXL3 en los expertos: mejor calidad por bit (Sushi-4bpw, 63.68 GiB, KLD 0.0632 frente a 0.0818 del afín 4/8 de
  70.13 GiB; Sushi-3bpw, 49.33 GiB, 0.1047 frente a 0.1444 del afín q3 de 54.94 GiB), pero en su propio hardware el
  GEMV de decode está limitado por ALU (~313 GB/s frente a ~600 del afín), así que en esta GPU su efecto en velocidad
  puede ser nulo o negativo. Es una palanca de calidad y memoria, y exige un tipo ggml nuevo, shaders Vulkan de decode
  y prefill y un conversor.
- Lectura de las hyper-connections agrupada por filas (+3 % en su verificación): el mat-vec de Vulkan ya comparte la
  lectura de pesos entre columnas (`NUM_COLS`, `vulkan-shaders/mul_mat_vec_base.glsl`).
- Shortlist del draft sobre una copia gruesa de la cabeza con re-puntuación exacta: baja la eficacia del draft, la
  misma familia que la vía 9 descartada por el Director (sección 27.2).
- GDN en un dispatch (−0.33 a −0.45 ms por pasada) y claves QSA agrupadas en un kernel (~1 %): el orden de las fusiones
  de la fila 17 de la sección 3.

## 30. SP1 del plan de decode: grafos de ubatches chicos en slots (2026-09-28, cadena v33)

Cambio (e579358ba, `src/llama-context.{h,cpp}`): el contexto guardaba un solo grafo previo (`gf_res_prev`), y el
contexto del draft, que alterna el hook sobre las filas verificadas (3 por slot que genera) y el paso de 1 fila,
reconstruía y reasignaba los dos grafos en cada paso (0 de 482 reutilizados, sección 28.2). Los ubatches de hasta 32
tokens van ahora a un LRU de 4 slots, cada uno con su `llm_graph_result` y su `ggml_backend_sched`; los mayores siguen
en el scheduler principal con la reserva del peor caso. El scheduler que computó un ubatch ubica sus salidas, construye
el callback de su grafo y corre su compute; los slots se resetean donde se resetea el grafo previo (reserva del
scheduler y actualización de la memoria). Alternativas descartadas en el diseño: unificar las topologías del hook y del
paso (3 filas sin salidas frente a 1 fila con muestreo en el backend) y compartir la memoria entre dos contextos del
draft (`ctx_other` solo existe para Gemma4). El LRU cubre las formas de NP=2 sin enumerarlas.

Ventana bajo `gate.py` (clase pesada, huella 66 GiB, la caída medida de esta configuración en v32): NP=1, ctx 57344, MTP
n-max 2, dos turnos de 1024 tokens con `ignore_eos` sobre el prefijo de 39.5k, base (cda30216f, `bin-v33base`)
primero y último:

| Carga | Greedy, ms por paso | Producción, ms por paso | t/s greedy / producción |
|---|---|---|---|
| Base (primera) | 55.35 | 54.85 | 39.58 / 38.69 |
| **Nueva** | **54.18** | **53.95** | **40.43 / 39.34** |
| Nueva (seguida de `depth_repeat`) | 54.19 | 53.95 | 40.43 / 39.34 |
| Base (última) | 55.38 | 58.98 (*) | 39.56 / 35.98 |

(*) Turno perturbado: misma aceptación, reloj medio de 2868 MHz y 4.1 ms más que la primera base; como en la sección
28.7, se descarta. Ganancia: **−1.17 ms por paso en greedy (−2.1 %, +2.2 % de t/s) y −0.90 ms con el muestreo de
producción (−1.6 %)**, la estimación del plan (−1.2 ms).

**El límite de potencia del paquete cambió entre el 2026-09-23 y el 2026-09-28** (qué lo cambió no consta aquí). Los
contadores que `decode_budget.py` muestrea con `amdgpu_top`, mediana de cada turno:

| Medida | v27, v30, v32 (2026-09-23) | v33 (2026-09-28) |
|---|---|---|
| Potencia del paquete | 85.1-85.3 W, plana | 117-130 W (máximo 142 W) |
| Potencia de la GPU | 30-36 W | 50.7-51.2 W |
| Residencia SPPT por turno | 29000-95000 | 0-6600 |
| fclk | 1.65-1.69 GHz | 1.78-1.89 GHz |
| gfxclk | 2.38-2.51 GHz | 2.82-2.89 GHz |

Con el mismo código (la base de esta cadena es la build de la v32), el paso de decode a 40k bajó de 61.7-62.0 a 55.4 ms
(−10 %) y el prefill de producción de ~39.5k con NP=2 (`np2_rig.py`, calentamiento del prefijo A) de 72.2 s (547 t/s,
v30) a 67.0 s (590 t/s, −7.2 %). Es la ventana 2 de la sección 29.7, observada sin cambio de código: la máquina ya no
corre limitada a 85 W (sección 28.4), y todas las cifras anteriores a esta sección son del régimen de 85 W.

Cargas con `LLAMA_INPUT_TIMING` (384 tokens por turno; greedy / producción):

| Tramo, ms por paso | Base | Nueva |
|---|---|---|
| Hook del draft | 0.98 / 0.95 | 0.59 / 0.58 |
| Paso del draft (media de los dos) | 3.19 / 3.04 | 3.03 / 2.81 |
| Verificación del target | 46.89 / 46.67 | 46.76 / 46.63 |
| Paso | 55.9 / 55.2 | 55.0 / 54.3 |

Grafos del contexto del draft reutilizados: hooks 0 → 356 de 360, pasos 360 → 716 de 721; build y alloc del draft en
la carga, 194 → 3.5 ms. La verificación del target ya se reutilizaba (356 y 358 de 360).

Exactitud: respuestas idénticas en greedy y en muestreo entre las dos bases, entre las dos cargas nuevas, entre la
nueva y la base, y entre las cargas con timing; la misma aceptación del draft en las seis cargas (556/931 y 541/961;
205/354 y 200/366); `depth_repeat` a 40k idéntico en los dos caminos; `graph_diff4` con la partición del server 72 de
12048 nodos, y con colas chicas que corren en los slots (`GD_SPLIT=2048,2048,512,3,3,1,1`) 569 de 32296 nodos en la
nueva y en la base, con la misma lista (las vistas SET_ROWS del caché) y la misma distribución; cero timeouts de
anillo.

Producción (NP=2, contexto nativo) con la nueva: probe idéntico a `probe-v26.out`, conversación texto → imagen → texto
5/5, imagen de 2048×2048 en 12.7 s. Rig NP=2 (`np2_rig.py` con s2), base → nueva:

| Medida | Base | Nueva |
|---|---|---|
| Un slot solo, A / B, t/s | 38.5 / 38.9 | 40.3 / 39.6 |
| Dos slots generando, agregado, t/s | 44.9 / 45.2 | 44.1 / 45.6 |
| Slot que genera mientras llega un prompt de 15k, chunks/s antes / durante / después | 41.2 / 15.9 / 40.7 | 41.9 / 16.5 / 40.2 |

La salida con NP=2 varía entre cargas (A aceptó 205/354 en la base y 213/340 en la nueva), así que estas diferencias
quedan dentro del ruido: sin regresión y sin ganancia medible con dos slots. Pendiente: la reutilización con NP=2 no se
midió (solo las cargas NP=1 corrieron con `LLAMA_INPUT_TIMING`), y en el log de `-lv 4` cada contexto asignó slots para
más formas chicas que sus 4 slots (1, 2, 3, 4, 6 y 8 filas): la hipótesis es que con dos slots el LRU desaloja formas
estables. El paso siguiente es una carga NP=2 con timing que cuente la reutilización antes de tocar el número de slots.

Memoria: producción recién cargada, 35 / 24 GiB (MemAvailable / MemFree) con la base y con la nueva. Buffers de los
slots medidos con `-lv 4` en producción después del rig NP=2 (prefijos de ~39k por slot): hasta ~31 MiB por slot del
target y 3-5 MiB por slot del draft, ~145 MiB en total. Crecen con n_kv (máscaras y listas de la atención); con los
dos slots a 262k se estiman bajo ~0.5 GiB (estimación, no medida). La reserva del peor caso del scheduler principal no
cambia. El aviso "compute buffer size ... does not match expectation" al cerrar el server ya aparecía en ventanas
anteriores (`srvlog-p40-mtp.log`, `srvlog-gu-base1.log`); no viene de este cambio.

Adoptado en producción (build del árbol e579358ba). Fuentes: `~/dbg/merge/chain_v33.{sh,out}`, `budget-v33*.txt`,
`gd-v33*.log`, `np2-v33p{b,n}.out`, `~/dbg/depth/srvlog-v33*.log`, `repeat-v33n2.log`.

## 31. Presupuesto con la potencia nueva, NP=2 y lookup en la ronda MTP adoptado (2026-09-28, cadenas v34-v38)

### 31.1 El paso de decode a 40k con el paquete a 117-130 W (cadena v34)

Build del SP1, NP=1, ctx 57344, MTP n-max 2, dos turnos de 1024 tokens: 54.72 / 54.48 ms por paso (greedy /
producción) con `LLAMA_INPUT_TIMING` y `LLAMA_SPEC_TIMING`; perf logger concurrente y serial en cargas aparte
(`vk_family.py v34c v34s v34a`, `~/dbg/merge/family-v34.txt`).

| Tramo | ms por paso |
|---|---|
| Verificación del target (pared) | 46.6 |
| · expertos gate/up / down | 11.0 / 5.5 |
| · FA sparse | 3.6 |
| · mezcladores hc | 3.3 |
| · copias y filas (CPY, CONT, GET/SET_ROWS) | 3.2 |
| · GDN attn_qkv / elementwise / cabeza | 3.1 / 2.7 / 2.3 |
| Draft: dos pasos de 1 token + hook (GPU) | 5.9 (cabeza del draft 1.6 por paso) |
| Host en serie: set_inputs 0.93, muestreo 0.56, huecos 0.98, post 0.12 | ~2.6 |

Con la potencia nueva el host dejó de pesar (set_inputs bajó de 1.81 a 0.93 ms: la CPU también corre más rápido); lo
que queda del paso es GPU: ~47 ms de la verificación y ~6 del draft. Las tasas por familia del perf logger quedan
cerca de las de la sección 28.3, que son cotas superiores (el logger serializa).

### 31.2 Reutilización de grafos con NP=2 (cadena v34)

NP=2, ctx 114688, `LLAMA_INPUT_TIMING`, el rig NP=2 con los dos slots generando y un prompt de 15k que llega mientras
el otro genera. Grafos de hasta 8 filas por contexto:

| Contexto | Forma | Reutilizados |
|---|---|---|
| Target | verificación de 3 / 6 filas | 806 de 820 / 351 de 363 |
| Draft | paso de 1 / 2 filas | 1639 de 1657 / 709 de 719 |
| Draft | hook de 3 / 6 filas | 800 de 815 / 356 de 364 |

Solo se reconstruye en las transiciones: el LRU de 4 slots alcanza y la hipótesis de desalojo de la sección 30 queda
refutada.

### 31.3 Lookup de prompt en la ronda MTP

`ngram-simple` del fork junto a `draft-mtp` (`--spec-type ngram-simple`): cuando los últimos k tokens ya aparecieron en
el contexto, la ronda propone los tokens que les siguieron (tiene prioridad sobre la cabeza MTP; el hook del MTP
procesa las filas verificadas sea cual sea el draft). Rig `~/dbg/merge/lookup_rig.py`: una función de ~700 tokens
(`common_ngram_simple_draft`) en el prompt y las tareas de sushi (copiarla, renombrar una variable, corregir un bug
plantado, tool call `write_file`, código nuevo, prosa, y el renombrado detrás del texto de ~39k), sin pensamiento, en
greedy y con el muestreo de producción; A = solo MTP, B = MTP + lookup, A B B A.

- **Ventana de 8 (verificación de 9 filas, cadena v34): pierde 11-16 %.** Con 9 filas Vulkan sale de los mat-vec
  (`mul_mat_vec_max_cols = 8` en los densos, `ggml_vk_use_mul_mat_vec_id` hasta 8 tokens en los expertos) y el paso
  sube de 47.8 a 114-143 ms.
- **Ventana de 7 (8 filas) con 2 slots de rollback (cadena v35):** copiar +35-38 %, corregir +34-35 %, renombrar
  detrás de 40k +31-32 %, renombrar +25-28 %, `write_file` +27 %; código nuevo −5 %, prosa −5 % / −1 %.
- **Los slots de rollback del estado recurrente** seguían el n_max de la cabeza MTP (2, `need_n_rs_seq`): cada ronda
  de lookup guardaba un checkpoint del estado y cada rollback de más de 2 tokens lo restauraba y volvía a decodificar
  las filas aceptadas (`server-context.cpp:3161`, `:4046`). Desde 353c25248 los slots cubren el draft más largo de
  cualquier tipo configurado (`common_speculative_n_max`); con solo MTP no cambia nada. Costo: el estado recurrente
  pasa de 0.68 GiB (medido con `-lv 4`, 108 MiB por copia) a ~1.75 GiB (calculado) con NP=2.
- **Ventana de 7 con 7 slots (cadena v37, la medida de adopción):** el render de Blender que bajó los relojes en la
  cadena v36 (solo CPU, pero comparte la potencia del paquete: GPU de ~2.84 a ~2.3 GHz; v36 descartada) terminó al
  inicio de la carga A1; CPU ocupada 3-4 % y GPU a 2.82-2.89 GHz en las cuatro cargas.

| Tarea | Greedy, A → B (t/s) | Producción, A → B (t/s) | Cambio |
|---|---|---|---|
| Copiar | 62.7 → 91.2 | 63.2 → 91.9 | **+45.5 %** |
| Renombrar detrás de 40k | 54.3 → 76.8 | 55.0 → 78.8 | **+41 a +43 %** |
| Corregir el bug | 63.0 → 87.1 | 62.7 → 86.0 | **+37 a +38 %** |
| Renombrar | 63.0 → 85.5 | 63.3 → 85.5 | **+35 a +36 %** |
| `write_file` | 62.2 → 81.3 | 61.9 → 79.6 | **+29 a +31 %** |
| Código nuevo | 58.8 → 57.0 | 57.9 → 57.7 | −3.0 / −0.2 % |
| Prosa | 50.5 → 50.2 | 47.7 → 46.1 | −0.6 / −3.3 % |

Exactitud: respuestas de B idénticas byte a byte a las de A en copiar, corregir, código nuevo, el renombrado detrás
de 40k y el renombrado en greedy (el lookup cambia el número de filas de la verificación y los mat-vec de hasta 8
columnas dieron los mismos bits en estos textos; en prosa y en el renombrado con muestreo el texto diverge);
`depth_repro` + `depth_repeat` con el lookup activo idénticos en los dos caminos. El `depth_repeat` fallido de la
cadena v35 corrió sin el prefijo en caché (la corrida 1 hizo el prefill completo), un error de orden de la prueba.

Producción (cadena v38, NP=2, contexto nativo, lookup encendido por el launcher): probe idéntico a `probe-v26.out`,
conversación texto → imagen → texto 5/5, imagen de 2048×2048 en 12.8 s, cero timeouts de anillo. Rig NP=2: un slot
solo 39.1 / 39.3 t/s; dos slots generando prosa 43.3 / 43.9 t/s agregados (44.1-45.6 sin lookup en la cadena v33, una
baja de ~2-4 % dentro de la variación de NP=2); el slot que genera mientras llega un prompt de 15k, 16.45 chunks/s
durante (16.49 antes). Memoria recién cargada 33 / 23 GiB (MemAvailable / MemFree), 1 GiB menos que sin lookup (34 /
24). Adoptado: `strix-halo/run-server.sh` agrega `--spec-type ngram-simple --spec-ngram-simple-size-n 7
--spec-ngram-simple-size-m 7` a qwen38flash con MTP (`LOOKUP=0` lo apaga) y `config.ini` sube `kv_gb` de 6 a 7.

Pendiente del lookup: la regla de línea y la compuerta de costo de sushi (sección 29.8) contra las entradas falsas en
código nuevo y prosa (−3 a 0 % hoy), y una clave más larga que la ventana, que `ngram-simple` no admite
(`copy_max < n_draft_min`).

### 31.4 Orden de trabajo con el presupuesto nuevo

1. Fusiones en la verificación (SP4): copias y filas 3.2 ms, elementwise 2.7, estado GDN, normas; con la potencia
   nueva lo que se quita de GPU se convierte en tiempo casi completo.
2. Tasa de los pesos (SP6): mezcladores hc a ~107 GB/s (3.3 ms, split-k), expertos gate/up a ~160 GB/s (11 ms).
3. Draft dentro de la verificación (SP5): el draft es 5.9 ms de GPU por paso (11 %).
4. SP2 y SP3 al final: el host en serie es ~2.6 ms por paso.

Fuentes: `~/dbg/merge/chain_v3{4,5,7,8}.{sh,out}`, `lookup-v3{4,5,7}*.jsonl`, `family-v34.txt`, `np2-v38pl.out`,
`~/dbg/depth/srvlog-v3{4,5,7}*.log`, `repeat-v37B1.log`, `power-v3{5,6,7}.log`, `cpu-v37.log`.

## 32. Entradas del paso a 125k: dimensionamiento del SP2 (2026-09-29, cadena v39)

Ventana sin cambios de código (build de producción 1ea4ce5d8 en `build/bin`, archivo rq8, draft IQ4_NL), paquete a
117-130 W. Tres cargas NP=1 con `LLAMA_INPUT_TIMING` y `LLAMA_SPEC_TIMING`, MTP n-max 2, el lookup en la ronda MTP
(default del launcher desde la sección 31.3) y el mmproj cargado: 40k (ctx 57344, prompt de 39.5k), 125k (ctx 131072,
prompt de 124.9k: los primeros 530944 caracteres de wikitext-2 test, el corpus del prompt de 40k,
`~/dbg/merge/depth_prompt125.txt`) y otra vez 40k. Cada carga hace el prefill una vez y dos turnos de 1024 tokens con
`ignore_eos` (greedy y muestreo de producción) sobre el prefijo en caché (`decode_budget.py`, que toma el prompt de
`DEPTH_PROMPT`). Entradas por contexto y tamaño de ubatch con `input_depth_parse.py` (medianas sobre todos los ubatches
del turno). Huella de la carga de 125k: 68 GiB calculados, 69 medidos (MemAvailable 110 → 41). Mínimos del guard
(MemAvailable / MemFree): 43 / 30, 40 / 29 y 45 / 37 GiB; cero timeouts de anillo; CPU ocupada 5 % (mediana), sin render;
gfxclk 2.76-2.88 GHz en las tres cargas. Las cifras llevan la instrumentación (~1-2 ms por paso más que sin ella,
sección 28.1); las comparaciones valen solo dentro de esta ventana (la sección 31.1 corrió sin lookup).

### 32.1 El paso a 40k y a 125k

| Carga | Greedy: t/s, ms por paso | Producción: t/s, ms por paso |
|---|---|---|
| 40k (primera) | 39.92, 55.83 | 37.78, 55.60 |
| **125k** | **33.95, 64.11** | **33.30, 63.47** |
| 40k (última) | 39.51, 56.42 | 38.15, 55.06 |

Los tokens por paso casi no cambian (2.23 / 2.10 a 40k, 2.18 / 2.12 a 125k): la baja de 12-15 % en t/s es el paso,
8.0-8.3 ms más largo. Reparto (`LLAMA_SPEC_TIMING` en todos los intervalos; set_inputs de `input_depth_parse.py`):

| Tramo, ms por paso | 40k | 125k | Δ |
|---|---|---|---|
| Verificación del target (pared) | 47.97 | 53.75 | +5.78 |
| · set_inputs (GPU parada) | 0.88 | 2.20 | +1.32 |
| · cómputo del grafo y espera de la GPU | 46.70 | 50.75 (*) | +4.05 |
| Draft: hook y dos pasos | 6.03 | 8.09 | +2.06 |
| · set_inputs de los tres grafos | 0.19 | 0.53 | +0.34 |
| Muestreo, post y huecos | 1.66 | 2.12 | +0.47 |
| **Paso** | **55.67** | **64.01** | **+8.34** |

(*) De la línea de tiempo de `budget_parse.py`, que a 125k cubre solo los pasos posteriores a la última verificación
de más de 3 filas (131 y 324 pasos): las verificaciones del lookup (4 y 8 filas) cortan su secuencia. Los pasos del
draft pasan de 2.9 a 3.9 ms y el hook de 0.63 a 0.79 ms en ese tramo: la atención densa del draft lee todo el KV.

### 32.2 Entradas por contexto

Medianas por ubatch, en ms:

| Entrada | 40k (primera / última carga) | 125k | Pendiente por 1k celdas |
|---|---|---|---|
| Verificación del target, 3 filas | 0.865-0.922 / 0.845-0.901 | 2.168-2.230 | 0.0154 |
| Hook del draft, 3 filas | 0.077-0.078 / 0.076-0.084 | 0.207 | 0.0015 |
| Paso del draft, 1 fila | 0.054-0.056 / 0.053-0.060 | 0.161-0.166 | 0.0013 |

Con una recta por los dos puntos, las entradas del target tienen ~0.27 ms fijos y el resto crece con las celdas: 0.61
ms a 40k y 1.93 a 125k. Las del draft son casi todas proporcionales: 0.16 ms por paso a 40k y 0.50 a 125k. En total, la
parte que crece con la profundidad es **0.77 ms por paso a 40k (1.4 %) y 2.43 ms a 125k (3.8 %)**. Las entradas
de los tres grafos del draft (su máscara KQ y el embedding del target) suman 0.19 ms por paso a 40k; la sección 28.8
midió 0.51 ms para la máscara sola con `perf`, a 85 W (método distinto, diferencia no atribuida).

Qué contiene cada una: la verificación llena la agrupación QSA (`set_input_qsa`, `src/llama-memory-hybrid-idx.cpp:442`:
cada celda a su bloque, sobre n_kv), la memoria híbrida (índices de K y V, máscara KQ de 3 filas × n_kv y copias del
estado recurrente, `src/llama-graph.cpp:1091`) y las filas PLE; el draft llena el embedding del target y la máscara KQ
de su atención densa. El reparto del decode entre QSA y máscaras no está medido: `LLAMA_INPUT_TIMING` detalla por
entrada solo los ubatches de 512 tokens o más, y `perf` no está disponible (`kernel.perf_event_paranoid=4`). En el
prefill sí (ms por ubatch de 2048 tokens):

| Entrada | 40k (ubatches 17-19) | 125k (ubatches 58-60) |
|---|---|---|
| QSA del target | 15.6-20.0 | 49.6-65.9 |
| Memoria híbrida del target (máscara KQ, índices) | 10.5-13.7 | 25.6-34.2 |
| PLE del target | 12.3-15.4 | 13.5-17.4 |
| Máscara KQ densa del draft | 11.4-15.3 | 26.2-36.0 |

En el prefill las entradas suman 1.1-1.4 s de 69 s a 40k y 5.5 s de 244 s a 125k (2.2 %). El prefill de 124.9k dura
**244.4 s (511 t/s)**, frente a 69.0-69.4 s (569-572 t/s) a 40k.

### 32.3 Techo del SP2 y orden

Si el SP2 quitara toda la parte de las entradas que crece con la profundidad (una cota superior: una agrupación
incremental sigue costando algo en cada paso, y el reparto entre QSA y máscaras en el decode no está medido):

| Profundidad | ms por paso | Greedy, t/s | Producción, t/s |
|---|---|---|---|
| 40k | −0.77 (−1.4 %) | 39.7 → 40.3 | 38.0 → 38.5 |
| 125k | −2.43 (−3.8 %) | 33.95 → 35.3 | 33.30 → 34.6 |

La estimación del plan (−1.2 a −1.7 ms a 40k, ~3× a 126k) venía del régimen de 85 W; con la potencia nueva la cota es
la mitad a 40k. De los 8.3 ms que suma el paso entre 40k y 125k, las entradas son 1.7 ms (20 %); el resto es sobre
todo GPU: +4.05 ms en la verificación (sin desglose por familia a 125k) y ~+1.7 ms en los grafos del draft, cuya
atención densa lee todo el KV.

Orden (sección 6 del plan): el SP2 pasa delante del SP5. Su ganancia quedó medida y acotada (+1.4 % a 40k, hasta +4 %
a 125k, creciente con el contexto de los agentes), cuesta 2-3 días y no necesita decisión del Director; el SP5 sigue
siendo un spike de riesgo alto con estimaciones del régimen de 85 W. SP4 y SP6 siguen primero: aplican a cualquier
profundidad y valen más por día.

Fuentes: `~/dbg/merge/chain_v39.{sh,out}`, `budget-v39{a,d,a2}.txt`, `inputs-v39{a,d,a2}.txt`, `input_depth_parse.py`,
`depth_prompt125.txt`; `~/dbg/depth/budget-v39{a,d,a2}.jsonl`, `srvlog-v39{a,d,a2}.log`, `guard-v39*.log`,
`power-v39.log`, `cpu-v39.log`.

## 33. SP2 del plan de decode: entradas del paso sin recorridos caros (2026-09-29, cadena v40)

Dos cambios de host, mismos valores que antes, sin tocar el grafo:

- **Agrupación QSA** (`set_input_qsa`, `src/llama-memory-hybrid-idx.cpp`, 100f88835). Con una secuencia por stream
  (todo ubatch de decode del server) un bloque tiene un solo grupo, que toma el número del bloque: sin cadenas ni
  inserciones por bloque. El bloque y el slot de un índice salen por desplazamiento cuando el ratio es potencia de
  dos (4 en qwen4exp), los slots del bloque quedan en un registro mientras las celdas consecutivas lo comparten, el
  camino pooled guarda solo las celdas de los bloques sucios (antes llenaba una tabla del tamaño del caché) y, si todas
  las celdas usadas llevan la secuencia de la consulta (`llama_kv_cells::seq_in_all_cells`), el sesgo por bloque se
  llena por rangos en vez de probar la pertenencia bloque a bloque. Los demás layouts siguen por las cadenas.
- **Primera fila de la máscara KQ de cada secuencia** (`src/llama-kv-cache.cpp`, b3e7e2b3f), en el target y en el
  draft. Con todas las celdas usadas en la secuencia del token y sin ventana deslizante, la fila son las celdas usadas
  con posición <= la del token: una pasada vectorizada sobre las posiciones (`fill_used_pos_le`). Las celdas que el
  bucle registraba para los tokens siguientes salen del índice de posiciones de la secuencia (`seq_cells_from`) y solo
  a ellas se les aplica la prueba completa, que conserva la regla M-RoPE en la posición de la consulta.

La atribución previa se hizo en la CPU, sin GPU, con las celdas reales (`llama_kv_cells`) y copias exactas del código
anterior (`~/dbg/merge/sp2/`): a 125k celdas la agrupación QSA costaba 1.03-1.22 ms por ubatch de decode y la máscara
0.09-0.1 ms por fila; la agrupación era el 75-80 % de las entradas de la verificación. Equivalencia bit a bit con el
código anterior: 21 layouts × 20 semillas para la agrupación (celdas de imagen en orden de rango, huecos, celdas
desordenadas, dos secuencias en un stream, una celda usada sin secuencia, ratios 3, 4, 5 y 8, caminos pooled y
recompute) y 9 × 20 para la máscara. En el arnés: agrupación 1.03 → 0.41 ms a 125k celdas (2.5×) y 0.33 → 0.13 a 40k;
máscara 5-7× por fila de decode. La máscara también se verificó sin causalidad, en F32 y con dos streams (640 casos,
`sp2/mask_equiv2.cpp`): el cambio vive en la función compartida por todos los modelos.

Cadena v40 (NP=1, MTP n-max 2, lookup y visión cargados; base = build de producción 1ea4ce5d8 por
`LD_LIBRARY_PATH`, verificada en `/proc/<pid>/maps`). A 40k sin instrumentar, base, nueva, nueva, base:

| Carga | Greedy: ms por paso (t/s) | Producción: ms por paso (t/s) |
|---|---|---|
| Base (primera) | 55.37 (40.25) | 54.89 (38.27) |
| Nueva | 55.05 (40.49) | 54.46 (38.57) |
| Nueva | 54.93 (40.58) | 54.44 (38.59) |
| Base (última) | 55.53 (40.14) | 54.95 (38.23) |

**40k: −0.46 ms por paso (−0.8 %), 40.20 → 40.54 t/s en greedy y 38.25 → 38.58 con muestreo (+0.9 %).** A 125k,
cargas con `LLAMA_INPUT_TIMING` y `LLAMA_SPEC_TIMING` (el prompt de 124.9k de la sección 32):

| Carga | Greedy: ms por paso (t/s) | Producción: ms por paso (t/s) |
|---|---|---|
| Base | 64.47 (33.76) | 63.82 (33.12) |
| **Nueva** | **62.05 (35.08)** | **61.48 (34.38)** |

**125k: −2.0 a −2.4 ms por paso, +3.2 a +3.9 % de t/s**: contra la base de esta cadena (64.47 / 63.82) y contra la
de la cadena v39, misma build y configuración (64.11 / 63.47); a 125k no hubo base de cierre. A 40k la ganancia (0.46
ms) quedó por debajo de su cota (0.77, sección 32.3). Entradas, medianas por ubatch en ms:

| Entrada | 40k base → nueva | 125k base → nueva |
|---|---|---|
| Verificación del target, 3 filas | 0.97 / 0.96 → 0.53 / 0.52 | 2.25 / 2.23 → 0.96 / 0.94 |
| Hook del draft, 3 filas | 0.072 / 0.069 → 0.023 / 0.024 | 0.209 / 0.210 → 0.053 / 0.054 |
| Paso del draft, 1 fila | 0.056 / 0.055 → 0.010 / 0.010 | 0.172 / 0.173 → 0.023 / 0.023 |
| **Por paso (verificación, hook y dos pasos)** | **1.15 → 0.57** | **2.80 → 1.05** |

A 125k las entradas bajan 1.75 ms y el paso 2.0-2.4: contra la base de esta cadena la verificación de pared cae 1.8
ms y sus entradas 1.29, el draft 0.4 y los huecos 0.2; el resto queda sin atribuir. Lo que queda de las entradas de la
verificación a 125k (0.95 ms) es ~0.27 fijo (PLE, índices) y la agrupación.

Exactitud: respuestas de la build nueva idénticas byte a byte a las de la base en greedy y con muestreo a 40k (1024
tokens, `ignore_eos`) y a 125k, base contra base y nueva contra nueva también; misma aceptación del draft;
`depth_repro` + `depth_repeat` idénticos; `graph_diff4` con el split del server, 72 nodos (los conocidos). Producción
(NP=2, contexto nativo): probe idéntico a `probe-v26.out`, conversación texto → imagen → texto 5/5, imagen de
2048×2048 en 12.8 s, cero timeouts de anillo. Rig NP=2 base → nueva: un slot solo 38.7 → 38.7-38.9 t/s, dos slots
44.1-44.6 → 43.4-44.1 t/s agregados, el slot que genera mientras llega un prompt de 15k 16.05 → 16.44 chunks/s: sin
cambio medible (el rig corre a poca profundidad). Adoptado: la build queda en `build/bin`; producción estaba descargada
al abrir la ventana y quedó descargada.

Fuentes: `~/dbg/merge/chain_v40.{sh,out}`, `budget-v40*.txt`, `inputs-v40{nx,bx,db,dn}.txt`, `np2-v40p{b,n}.out`,
`probe-v40.out`, `img-v40-prod.out`, `imgbig-v40-prod.out`, `gd-v40n.log`, `sp2/` (arnés de CPU); `~/dbg/depth/`
`budget-v40*.jsonl`, `srvlog-v40*.log`, `repeat-v40n2.log`, `power-v40.log`, `cpu-v40.log`.

## 34. Día 1 del SP3: qué corre el host con la GPU parada (2026-09-29, cadena v41)

Sin cambios de código (build de producción con el SP2). `kernel.perf_event_paranoid=2` puesto por el Director. Una
carga NP=1 a 40k (ctx 57344) con `LLAMA_INPUT_TIMING`, que fuerza la espera de la GPU al final de cada ubatch, así que
entre ubatches la GPU está parada, y `LLAMA_SPEC_TIMING`. Tras el prefill, un turno greedy y uno con muestreo de
producción de 1024 tokens bajo `perf record -e cycles:u --call-graph dwarf,16384 -F 999 -k monotonic`: el reloj de
`ggml_time_us`. `gap_fold.py` ubica cada muestra del hilo principal en su ubatch o en su hueco de la línea de tiempo
del paso: 950 pasos, 24430 muestras. El paso con la instrumentación, 55.5 / 54.8 ms (greedy / muestreo).

| Hueco (GPU parada) | ms por paso | Qué corre |
|---|---|---|
| Tras el hook | 0.98 | muestreo del target: llenado de los 248320 candidatos 0.23 y top-k (`partial_sort`, `__heap_select`) 0.18; copia de los candidatos en `common_sampler_clone` 0.14 (el clon del server antes de aceptar, `server-context.cpp:4026`, copia el vector de 3 MB); limpieza del command pool de Vulkan 0.13; `seq_rm` de lo rechazado 0.105; lookup n-gram 0.03; resto de la ronda especulativa ~0.1 |
| Tras la verificación | 0.33 | espera de la lectura de logits 0.09, Vulkan 0.06, la ronda especulativa 0.05, preparación de `llama_decode` 0.04 |
| Tras el último draft | 0.20 | espera de lectura 0.05, server 0.05, `llama_decode` 0.03, `seq_rm` y `find_slot` 0.04 |
| Entre los pasos del draft | 0.13 | espera de la lectura de la salida del draft 0.07 |
| **Total** | **1.64** | 3 % del paso |

`seq_rm` recorre todas las celdas del stream (`llama_kv_cache::seq_rm`, `src/llama-kv-cache.cpp:405`) para las
posiciones del draft rechazado, en la atención y en el indexador del target y en el caché del draft, aunque no haya
nada que borrar: su costo sigue al tamaño del KV, no a la profundidad. Las cargas de prueba usan ctx 57344; producción
usa 262144 celdas por slot, así que ahí cuesta ~4.6 veces más (~0.4 ms por paso, estimado a escala; una pasada aislada
en la CPU cuesta 0.016 ms a 57k celdas y 0.074 a 262k, `~/dbg/merge/sp2/seqrm_bench.cpp`, y en el server la caché de CPU
llega fría).

Candidatos del SP3, todos con la misma salida:

| Cambio | Estimación a ctx 57k | En producción (262k por slot) |
|---|---|---|
| El clon del sampler sin el vector de candidatos (se reescribe en cada muestreo) | −0.14 ms | −0.14 ms |
| `seq_rm` por el índice de posiciones de la secuencia en vez de recorrer todas las celdas | −0.09 ms | ~−0.4 ms |
| Llenado y top-k del muestreo en una pasada, con el mismo orden de empates que `partial_sort` | −0.2 a −0.3 ms | igual |
| La limpieza del command pool de Vulkan fuera del camino crítico | −0.10 a −0.13 ms | igual |
| **Total** | **~0.55-0.65 ms (+1.0-1.2 %)** | **~0.85-1.0 ms (+1.5-1.8 %)** |

Las esperas de lectura dentro de los huecos (~0.2 ms) son trabajo de la GPU (copias a host), no del host. La
restricción del muestreo sigue: mismos candidatos y mismo orden de empates, o el probe cambia.

Fuentes: `~/dbg/merge/chain_v41.{sh,out}`, `gap_fold.py`, `gap-v41p.txt`, `perf-v41-fold.txt`, `budget-v41p.txt`;
`~/dbg/depth/perf-v41.data`, `perf-v41.script`, `srvlog-v41p.log`, `budget-v41{w,p}.jsonl`.

## 35. SP3 del plan de decode: huecos de host del paso (2026-09-29, cadenas v42-v46)

Tres cambios de la atribución de la sección 34, mismos valores que antes:

- **Clon del sampler sin candidatos** (`common/sampling.cpp`, f5149b386). El server clona el sampler del target antes de
  aceptar un draft en cada paso, y el clon copiaba el vector de candidatos del último muestreo (248320 entradas, 3 MB).
  Cada muestreo lo rellena (`set_logits`), así que el clon y la copia llevan solo el estado (samplers, gramática,
  presupuesto de razonamiento, historia, rng). Nadie lee los candidatos de un clon ni de un sampler restaurado antes de
  muestrear con él (el restore del checkpoint del server, los ejemplos de speculative).
- **`seq_rm` por el índice de posiciones de la secuencia** (`src/llama-kv-cache.cpp`, 2ed7aac92). Recorría todas las
  celdas del stream (262144 por slot en producción) para borrar los tokens rechazados del draft, en la atención y el
  indexador del target y en el caché del draft. Ahora toma las celdas de la secuencia en [p0, p1) de su índice
  (`llama_kv_cells::seq_cells_in`, el accessor de la primera fila de la máscara del SP2 generalizado a un rango). Mismas
  celdas, mismos conjuntos de secuencias y mismo head que el recorrido en 1500 borrados al azar
  (`~/dbg/merge/sp2/seqrm_equiv.cpp`).
- **Pool de comandos de Vulkan reseteado durante la espera siguiente** (`ggml-vulkan.cpp`, 7b0e8cf98). Cada contexto de
  backend tiene dos pools usados por turno: la limpieza tras la espera retira el actual y el grafo siguiente graba en el
  otro; el retirado se resetea al comienzo de la próxima espera del fence, con la GPU trabajando.

Descartado: el llenado de candidatos y el top-k en una pasada. Solo es exacto si los samplers anteriores al top-k no
cambian nada (logit bias, penalizaciones, gramática, presupuesto de razonamiento), una condición que depende de cada
pedido: frágil, fuera del SP3.

Cadena v42, NP=1 a 40k (ctx 57344), base (build del SP2) por `LD_LIBRARY_PATH`:

| Carga | Greedy: ms por paso (t/s) | Producción: ms por paso (t/s) |
|---|---|---|
| Base (primera) | 54.79 (40.68) | 54.53 (38.52) |
| Nueva | 54.12 (41.18) | 53.73 (39.09) |
| Nueva | 54.09 (41.20) | 53.70 (39.12) |
| Base (última) | 55.29 (40.31) | 54.83 (38.31) |

**NP=1 a 40k: −0.93 ms por paso, 40.50 → 41.19 t/s en greedy y 38.42 → 39.11 con muestreo (+1.7-1.8 %).** Huecos de host
del paso (carga con timing, turno con muestreo): 1.71 → 1.20 ms; tras el hook 0.98 → 0.60, tras la verificación
0.35 → 0.31, tras el último draft 0.24 → 0.16.

**Producción (NP=2, 262144 celdas por slot) a 40k, servidor recién cargado: 56.04 / 55.72 → 54.20 / 53.81 ms por paso,
39.77 → 41.12 t/s en greedy y 37.70 → 39.04 con muestreo (+3.4-3.6 %).** Con la misma historia previa en el servidor
(probe, conversación con imagen, imagen de 2048×2048), 55.78 / 54.99 → 54.36 / 54.25 ms (+2.6 / +1.4 %). En producción
rinde más que en NP=1 porque el `seq_rm` recorría un KV 4.6 veces mayor. Rig NP=2: un slot solo 39.7 / 38.7 → 39.8 /
39.7 t/s, dos slots 44.5-44.6 → 44.5-45.2 t/s agregados, el slot que genera mientras llega un prompt de 15k 15.85 → 16.62
chunks/s.

Exactitud: respuestas idénticas a la base en NP=1 (greedy y muestreo, también en las cargas con timing), base contra base
y nueva contra nueva; `depth_repro` + `depth_repeat` idénticos; `graph_diff4` 72 nodos; probe idéntico a `probe-v26`,
conversación texto → imagen → texto 5/5, imagen de 2048×2048 en 12.9 s, cero timeouts de anillo. En producción, con la
misma historia la nueva y la base dan lo mismo (cadena v43: A == B), y recién cargadas también (C == v42pb). Adoptado:
build en `build/bin`, producción descargada como estaba.

### 35.1 Defecto de determinismo que introdujo el SP1: la salida depende de la historia del servidor (resuelto en 35.2)

Apareció en la compuerta de producción del SP3 y se aisló en las cadenas v43-v51 (producción NP=2, contexto nativo,
pedidos fijados a un slot con `id_slot`). La misma petición (el decode greedy de 1024 tokens sobre el prompt de 39.5k)
da respuestas distintas según lo que el servidor procesó antes:

- Con el servidor recién cargado la respuesta es la misma en el slot 0 y en el slot 1, y la misma en los builds de
  antes del SP1 (cda30216f), con SP1 (1ea4ce5d8), con SP2 y con SP3 (cadena v45, lookup apagado).
- Tras una imagen de 2048×2048 (4096 tokens) en el otro slot, el decode en el slot 0 diverge entre los caracteres 1100 y
  2200, con MTP y lookup y también sin ellos (cadena v46; con MTP apagado, en el carácter 1147). Una conversación con
  una imagen de 448×448 no lo produce, y borrar el slot 1 antes del decode no lo evita. La divergencia se repite byte a
  byte con la misma historia, y el probe (la distribución de la primera posición) sigue idéntico.
- El build anterior al SP1 no tiene el defecto (cadena v48); el build con SP1 y sin SP2 sí, con las mismas salidas que
  el SP3 (cadena v49). SP2 y SP3 no cambian ninguna salida: con la misma historia, base y nueva coinciden (v43).

Hipótesis descartadas: (1) la disposición de memoria que el asignador de un slot reutiliza del grafo anterior cuando la
topología coincide (`ggml_gallocr_needs_realloc`): con cada grafo de slot planificado para sus propias formas, el
defecto sigue (v50); (2) restos en los buffers de cómputo del slot: con un scheduler y buffers nuevos para cada grafo de
slot, el defecto sigue (v51). En ambos casos la salida con el servidor recién cargado no cambió y el probe siguió
idéntico a `probe-v26`. El canal es otro estado que el SP1 modifica (qué grafos corren en el scheduler principal, qué
grafos se reutilizan, el orden de las reconstrucciones). Pendiente: comparar nodo a nodo el decode del servidor recién
cargado con el del que procesó la imagen para localizar el primer nodo que difiere, y corregir la causa. El build
anterior al SP1 no lo tiene con MTP y lookup apagados (v48; con la configuración de producción no se probó); revertir el
SP1 costaría ~1.2 ms por paso y no se revirtió. Las cargas de diagnóstico v43-v51 usaron el tamaño de producción (NP=2,
262144 celdas por slot) bajo la compuerta registrada con huella 66 GiB: debieron correr con contexto reducido (NP=2, ctx
114688, huella 68) como pide la regla de pruebas.

Fuentes: `~/dbg/merge/chain_v4{2,3,4,5,6,7,8,9}.{sh,out}`, `chain_v50.out`, `chain_v50b.out`, `chain_v51.out`,
`sp2/seqrm_equiv.cpp`; `~/dbg/depth/budget-v4{2,3,4,5,6,7,8,9}*.jsonl`, `budget-v5{0,1}*.jsonl`, `srvlog-v4*.log`;
el parche descartado de la hipótesis 1 en el scratchpad de la sesión (`discard_layout.patch`).

### 35.2 Causa y arreglo del defecto de la sección 35.1 (2026-09-29, cadenas v61-v71)

**Causa.** Las fusiones del backend Vulkan se deciden, en parte, por dónde están los tensores: el reuso in place del
asignador y los chequeos de solapamiento que desactivan una fusión cuando la salida pisa una entrada. El asignador de
grafos (`ggml_gallocr`) reutilizaba la disposición del grafo anterior de su scheduler cada vez que el grafo nuevo tenía
la misma cantidad de nodos y hojas y sus tensores cabían (`ggml_gallocr_needs_realloc`), aunque fuera otro grafo. Antes
del SP1 los ubatches chicos pasaban por el scheduler principal, así que el grafo anterior a un prefill era el mismo en
cualquier historia. Con el SP1 esos ubatches van a los slots de grafo y el grafo anterior del scheduler principal pasa a
depender de la historia del servidor: tras la imagen es el de la imagen. El primer ubatch de 2048 del prompt quedaba
con otra disposición, otras fusiones y otro redondeo desde la capa 40.

Cómo se localizó, todo a contexto reducido (NP=2, 57344 celdas por slot) con MTP y lookup apagados salvo donde se dice:

- v61: se reproduce (difiere en el carácter 2910) y desaparece con `LLAMA_GRAPH_REUSE_DISABLE=1`, que también apaga los
  slots del SP1.
- v62: en 1000 reutilizaciones, cada grafo reutilizado es estructuralmente idéntico a uno construido de nuevo (op,
  formas, strides, parámetros, offsets de vistas): no es un `can_reuse` incompleto.
- v63 y v64: llenar con ceros o NaN los intermedios, y después las entradas, de cada grafo recién asignado no cambia
  nada: no hay lectura de memoria sin inicializar.
- v65: los log-probs difieren desde el primer token, que calcula el último ubatch del prefill.
- v66 a v68: el estado recurrente de las capas GDN desde la 40 difiere; el primer ubatch que diverge es el primero de
  2048 del prompt, en la entrada de la capa 40.
- v69: ese ubatch calculado nodo por nodo (sin fusiones entre nodos) da salidas idénticas en las dos historias.
- v70: con `GGML_VK_DISABLE_FUSION=1`, o con una disposición nueva para cada grafo, las dos historias coinciden.

**Arreglo.** `ggml_backend_sched_reset` descarta la disposición (`ggml_gallocr_discard_layout`): cada grafo que se
asigna después de un reset se dispone para sus propias formas, así que la disposición depende solo del grafo. Un grafo
reutilizado no pasa por el reset y conserva la suya. Todos los sitios que construyen un grafo nuevo ya llaman al reset
(el decode, el prefill, el K-shift). El chequeo de depuración `GGML_SCHED_DEBUG_REALLOC` no cuenta estas
redisposiciones.

**Cadena v71:**

- Historia a contexto reducido, 1024 tokens greedy con log-probs en el slot 0: recién cargado contra tras la imagen de
  2048×2048 en el slot 1, idénticos token a token y log-prob a log-prob con MTP y lookup apagados y con los de
  producción. El decode recién cargado es idéntico al del build anterior.
- NP=1 a 40k, base (build de la fusión 3) primero y último: 51.70 / 51.13 y 51.66 / 50.99 → 51.52 / 50.99 y 51.64 /
  50.99 ms por paso: sin costo medible. El prefill de 39.5k: 89.7 → 89.6 s.
- Respuestas idénticas a la base (también en las cargas con timing), `depth_repro` + `depth_repeat` idénticos,
  `graph_diff4` 72 nodos, cero timeouts de anillo.
- Producción (NP=2, recién cargados): 51.17 / 50.90 → 51.39 / 51.04 ms por paso, respuestas idénticas a la base; probe
  idéntico a `probe-v26`, conversación texto → imagen → texto 5/5, imagen de 2048×2048 en 12.6 s. Rig NP=2, base →
  nueva: un slot solo 43.4 / 43.4 → 42.9 / 43.4 t/s, dos slots 47.6-47.9 → 48.4-49.6 t/s agregados, el slot que genera
  mientras llega un prompt de 15k 17.71 → 18.24 chunks/s. Commit 9f3e99782.
- Memoria a profundidad (cadena v72, NP=1, ctx 131072, prompt de 124.9k, MTP y lookup): con una disposición nueva por
  grafo, el buffer de cómputo podría crecer en pleno prefill si el plan fresco pidiera más que la reserva. GPU en uso
  (VRAM + GTT) con pico de 72 880 MiB en la base y 72 874 MiB en la nueva, MemFree mínimo de 33 876 y 33 912 MiB: no
  crece. Prefill 239.4 → 238.4 s, respuestas idénticas.

El defecto no vino de un error del SP1 sino de una propiedad del asignador que el SP1 dejó a la vista. Las mediciones
del SP2, el SP3 y las fusiones del SP4 compararon servidores recién cargados y siguen valiendo. Incidente de la investigación: la primera
corrida de v66 hasheaba las fuentes enteras, pesos incluidos (la tabla PLE lazy, 27 GiB); el guard de la cadena
descargó el servidor al bajar MemAvailable a 21 GiB, sin consecuencias. El volcado corregido lee por trozos de 16 MiB y
no toca pesos.

Fuentes: `~/dbg/merge/chain_v6{1..9}.{sh,out}`, `chain_v70.{sh,out}`, `chain_v71.{sh,out}`, `logprob_diff.py`,
`dump_diff.py`, `ophash_diff.py`; los parches de diagnóstico (no commiteados) en el scratchpad de la sesión:
`diag_reuse_verify.py`, `discard_layout_all.patch`.

## 36. SP4 del plan de decode: fusiones en la verificación (2026-09-29, cadenas v52-v55)

Una fusión por commit y por medición, con la exactitud del protocolo común: respuestas idénticas a la base.

### 36.1 Fusión 1: las colas del conv state sin CONT (adoptada, abc86e8be)

`build_conv_state_at` escribía la cola de cada slot de rollback con un CONT (la vista con stride hecha contigua) y un
CPY al caché. El CPY lee la vista con stride directamente: con el lookup en la ronda MTP el target guarda 8 slots, así
que son 36 capas × 8 = 288 dispatches menos por verificación.

Cadena v52, NP=1 a 40k (ctx 57344), base (build del SP3) por `LD_LIBRARY_PATH`:

| Carga | Greedy: ms por paso (t/s) | Producción: ms por paso (t/s) |
|---|---|---|
| Base (primera) | 53.91 (41.34) | 53.44 (39.31) |
| Nueva | 53.23 (41.87) | 52.89 (39.71) |
| Nueva | 53.28 (41.83) | 52.89 (39.71) |
| Base (última) | 53.92 (41.34) | 53.49 (39.27) |

**−0.66 ms por paso en greedy y −0.58 con muestreo: 41.34 → 41.85 t/s (+1.2 %) y 39.29 → 39.71 (+1.1 %)**, ~2.2 µs
por dispatch quitado. El mismo build midió 53.94 / 53.56 en la cadena v53, 30 minutos después: entre cadenas el paso se
mueve ~0.7 ms, del orden del efecto, y solo vale la comparación dentro de una cadena con la base primero y última.

Exactitud: respuestas idénticas (base contra base, nueva contra nueva, nueva contra base, también en las cargas con
timing); `depth_repro` + `depth_repeat` idénticos; `graph_diff4` 72 de 11937 nodos; probe idéntico a `probe-v26`,
conversación texto → imagen → texto 5/5, imagen de 2048×2048 en 12.8 s, cero timeouts de anillo. En producción (NP=2,
servidores recién cargados) la nueva y la base dan la misma respuesta. El tiempo de producción no quedó medido: la
carga base dio 80.25 / 82.09 ms por paso en sus dos turnos (reloj medio de 1226 MHz en el segundo), un valor atípico sin
explicar, mientras el rig NP=2 sobre ese mismo servidor salió normal; la nueva dio 54.30 / 53.91. Rig NP=2, base →
nueva: un slot solo 40.3 / 39.4 → 41.0 / 39.3 t/s, dos slots 45.3-46.0 → 45.4-45.9 t/s agregados, el slot que genera
mientras llega un prompt de 15k 16.41 → 16.64 chunks/s: sin cambio medible.

### 36.2 Fusión 2: las colas de los 8 slots en un gather y una copia (descartada)

Las 8 colas de una capa salen de un GET_ROWS sobre `x` (o sobre la historia concatenada con `x` cuando el ubatch tiene
menos de 10 tokens) con una tabla de índices por forma de ubatch, y un CPY permutado las escribe en los 8 slots: 36
GET_ROWS y 36 CPY en lugar de 288 CPY por verificación. La operación es exacta: idéntica byte a byte a las copias por
slot en CPU y en Vulkan, con decode y con prefill, 1 y 2 secuencias (`~/dbg/merge/sp2/conv_tail_test.cpp`).

En el modelo las respuestas difieren de la base (cadena v53: aceptación 563/925 contra 564/917). Con
`GGML_VK_DISABLE_FUSION=1` base y nueva dan lo mismo; con el optimizador de grafos apagado siguen distintas (v54). El
perfil de Vulkan por sumisión (v55, 128 tokens) localiza la diferencia:

- En el grafo de verificación solo cambian los conteos de las ops tocadas (CPY 336 → 84, GET_ROWS 145 → 181); las
  fusiones del backend son las mismas.
- Los ubatches de prefill de más de 32k de profundidad corren en tres sumisiones, y los cortes se mueven: 1157 / 1628 /
  582 nodos → 1070 / 1564 / 580, MULTI_ADD 32 / 48 / 17 → 30 / 47 / 18. Dos cadenas de sumas quedan partidas en un
  corte, cambia el orden de la suma, el caché del prefill difiere en redondeo y el decode diverge desde ahí.

Mecanismo (inferido, no confirmado con la impresión de splits del scheduler): la tabla de índices es la única entrada
nueva del grafo y vive en la CPU; el scheduler abre un split cuando las entradas del split llegan a su capacidad (30,
`GGML_SCHED_MAX_SPLIT_INPUTS`, `ggml/src/ggml-backend.cpp:1341`), así que una entrada más adelanta los cortes, y una
fusión del backend no cruza sumisiones.

Ganancia dentro del ruido: pares de una carga en v53 (0.00 / −0.11 ms por paso) y en v54 con perillas de entorno
(fusiones apagadas −0.84 / −0.25, optimizador apagado +0.04 / +0.29); en el perfil serializado la verificación baja
0.25 ms (CPY −421 µs, GET_ROWS +190 µs), porque el CPY permutado y los gathers cuestan casi lo que las 288 copias
chicas. Descartada: pagar una tabla construida en el grafo, o pedir que se acepte un cambio numérico, no se justifica sin
ganancia. El parche quedó en el scratchpad de la sesión (`sp4_fusion2_conv_tails_gather.patch`).

Pre-compuerta para las fusiones restantes del SP4: dos cargas cortas con `GGML_VK_PERF_LOGGER=1` (base y nueva, 128
tokens a 40k) y el diff de los conteos de ops por tipo de grafo, incluidos los ubatches de prefill partidos; toda op
fuera de las que la fusión toca debe coincidir antes de gastar la medición A B B A. Reglas de diseño que salen de aquí:
ninguna entrada nueva al grafo y ningún cambio de vecindad junto a las cadenas MULTI_ADD y RMS_NORM_MUL.

Fuentes: `~/dbg/merge/chain_v5{2,3,4,5}.{sh,out}`, `sp2/conv_tail_test.cpp`; `~/dbg/depth/budget-v5{2,3,4,5}*.jsonl`,
`srvlog-v55{b,n}.log`.

### 36.3 Fusión 3: los snapshots del GDN escritos directo en el caché (adoptada, 1d7cd0d16)

Perfil serializado del grafo de verificación por forma de nodo (cadena v56, build de diagnóstico con la forma en el
nombre del perf logger): el camino del estado GDN es el mayor bloque de ops chicas. El CPY de los snapshots cuesta 36 ×
25.6 µs = 0.92 ms por verificación (9.4 MB por capa: 3 snapshots de 128×128×48 f32) y el gather del estado 36 × 17.1 µs
= 0.62 ms. Techo medido antes de implementar (cadena v57: la misma binaria con y sin el CPY en el grafo, A B A,
`LLAMA_INPUT_TIMING`): el `gpu_wait` del ubatch de verificación baja de 39.03 / 39.00 a 37.74 ms (−1.27 ms).

Fusión del backend Vulkan GDN_SNAPSHOTS (GATED_DELTA_NET + VIEW + [VIEW] + CPY): el shader escribe cada snapshot en las
filas del destino del CPY por un binding nuevo, con los strides de slot y de secuencia del destino, y los scores de la
atención siguen en la salida del op. Sin fusión, el mismo binding apunta a la salida del op pasados los scores, con la
disposición de antes. El grafo del modelo no cambia (el de upstream), no agrega entradas y la referencia de CPU sigue
igual. El optimizador de grafos de Vulkan saca las VIEW a una segunda pasada y la del destino (una vista del caché, sin
dependencias) sale con las vistas de un set anterior: sin una regla propia los cuatro nodos nunca quedaban contiguos
(cadena v58, dos corridas con 0 fusiones; v59, impresión de los nodos). La regla mantiene juntos el op, la vista de los
snapshots y la copia, así que la secuencia fusionada tiene 3 nodos, o 4 si la vista del destino no salió antes.

Pre-compuerta (v58, tercera corrida): 2952 dispatches fusionados (36 por grafo con GDN) y, con la fusión plegada en las
ops que reemplaza, los conteos coinciden en las 290 sumisiones, incluidos los ubatches de prefill partidos.

Cadena v60, NP=1 a 40k (ctx 57344), base (build de la fusión 1) por `LD_LIBRARY_PATH`:

| Carga | Greedy: ms por paso (t/s) | Producción: ms por paso (t/s) |
|---|---|---|
| Base (primera) | 54.14 (41.16) | 53.63 (39.17) |
| Nueva | 52.09 (42.79) | 51.49 (40.80) |
| Nueva | 51.98 (42.88) | 51.48 (40.81) |
| Base (última) | 54.26 (41.08) | 53.74 (39.09) |

**−2.16 ms por paso en greedy y −2.20 con muestreo: 41.12 → 42.84 t/s (+4.2 %) y 39.13 → 40.81 (+4.3 %).** Cargas con
timing: el `gpu_wait` de la verificación baja de 38.84 a 36.23 ms (−2.6 ms), el paso de 55.34 a 52.99 ms. La ganancia
supera el techo de v57 y la diferencia no está explicada. El perfil serializado (v58) atribuye −0.94 ms a las copias y
unos −1.4 ms a las matmuls del grafo de verificación (MUL_MAT_ID iq3_s −0.91, MUL_MAT_VEC iq4_nl −0.32, q8_0 −0.17);
que sea por 36 × 9.4 MB menos de escritura sucia que desalojar del caché de la GPU es inferencia, sin verificar.

**Producción (NP=2, 262144 celdas por slot, servidores recién cargados): 54.08 / 53.64 → 51.52 / 51.50 ms por paso,
41.22 → 43.26 t/s en greedy (+4.9 %) y 39.16 → 40.79 con muestreo (+4.2 %).** Rig NP=2: un slot solo 41.2 / 41.0 →
42.4 / 41.0 t/s, dos slots 43.2-44.3 → 46.5-47.8 t/s agregados, el slot que genera mientras llega un prompt de 15k
16.64 → 17.68 chunks/s.

Exactitud: respuestas y aceptación del draft idénticas a la base (base contra base, nueva contra nueva, nueva contra
base, cargas con timing) y también con `GGML_VK_DISABLE_FUSION=1` en las dos (la ruta de escritura sin fusión,
reescrita); `depth_repro` + `depth_repeat` idénticos; `graph_diff4` 72 de 11937 nodos (evalúa nodo por nodo, así que
valida la ruta sin fusión); probe idéntico a `probe-v26`, conversación texto → imagen → texto 5/5, imagen de 2048×2048
en 12.8 s, producción nueva contra base recién cargadas idéntica, cero timeouts de anillo. Con las formas del modelo
(S_v 128, 16 cabezas q/k, 48 v, 8 slots; decode, dos secuencias y un ubatch de 2048) la salida fusionada, la sin fusión
y la del build anterior son idénticas byte a byte (`~/dbg/merge/sp4/gdn_snap_test.cpp`); casos GDN_SNAPSHOTS nuevos en
`test-backend-ops` (kv_head > 0, mem_size > n_seqs, n_tokens menor y mayor que K, dos secuencias, KDA, K = 1) y los 36
casos GATED_DELTA_NET existentes, 42/42.

Fuentes: `~/dbg/merge/chain_v5{6,7,8,9}.{sh,out}`, `chain_v60.{sh,out}`, `v56_verify_ops.txt`, `verify_wait.py`,
`opcount_gate.py`, `sp4/gdn_snap_test.cpp`; `~/dbg/depth/srvlog-v5{6d,7a1,7b,7a2,8b,8n,9d}.log`,
`budget-v60*.jsonl`.

### 36.4 Candidatas restantes: techos y descartes (2026-09-30, cadenas v73-v75)

Cada techo se midió con una sola binaria de diagnóstico, con y sin el trabajo, sobre el `gpu_wait` del ubatch de
verificación (n = 3, `LLAMA_INPUT_TIMING`, NP=1 a 40k). Umbral para implementar: 0.15 ms.

- **Vía 14, lado de lectura (el GDN lee su estado del caché, sin los 36 GET_ROWS de 3 MB; 36 × 17.1 µs en el perfil
  serializado).** Para el techo (v73, A B A) el estado se lee de la fila del slot de rollback 7, que una verificación de
  3 tokens nunca escribe, así la fusión 3 sigue activa: 36.83 / 37.57 → 37.41 ms. **Techo nulo dentro del ruido**: el
  gather se solapa con el resto del grafo. El diseño tampoco era simple: `find_slot` intercambia celdas entre
  secuencias, así que con dos secuencias en el ubatch una leería en el mismo op la celda que la otra escribe; solo era
  seguro con una secuencia y sin copias extra, y el GET_ROWS no es adyacente al GDN. Descartada.
- **Norma L2 del GDN (RMS_NORM + SCALE, 72 por verificación).** Quitar el SCALE rompe los números (v74: HTTP 500); el
  techo se midió con el op `l2_norm` de un solo dispatch (v75, pares intercalados contra la deriva): −0.13 y −0.10 ms,
  **~−0.12 ms, bajo el umbral**. La fusión pedía además una regla nueva del optimizador de grafos para dejar el SCALE
  pegado a su RMS_NORM, con el riesgo de mover vecinos que dejó la fusión 2. Descartada.
- **CONT de los mezcladores hc (96 por verificación).** Quitar 96 dispatches de su mismo costo: −0.15 ms (v74), −0.45 y
  −0.10 ms (v75), **~−0.23 ms**: se implementa (sección 36.5).

Los casos TOPK_MOE de `test-backend-ops` fallan de forma intermitente también con el build adoptado (1-2 de 416 en 2 de
3 corridas): no es de estos cambios.

Fuentes: `~/dbg/merge/chain_v7{3,4,5}.{sh,out}`, `queue_run.sh`; los diagnósticos (no commiteados) en `bin-v73diag`,
`bin-v74diag`, `bin-v75diag`.

### 36.5 CONT de los mezcladores hc: descartada; cierre del SP4 (2026-09-30, cadenas v76-v78)

El SCALE de cada mezclador hc leía una copia contigua (CONT) de las filas de su matmul conjunto down/inject. El cambio
deja que SCALE lea las filas acolchadas en el lugar: en CPU, el assert de contigüidad pasa al contrato de
`ggml_scale` (1d acolchado; se corrige además la rama con bias, que leía el origen con el stride del destino); en
Vulkan, `scale.comp` indexa con `src0_idx`/`dst_idx` como `copy.comp`, y `supports_op` deja de pedir contigüidad; casos
nuevos de `test-backend-ops` con filas acolchadas, 531/531. El cambio es exacto por construcción.

La pre-compuerta falló (v76): aceptación 67/120 → 65/123, y en los ubatches de prefill profundos faltaban cadenas del
router MoE en el perfil. El registro de nodos de Vulkan (v78, `GGML_VK_SYNC_LOGGER`) lo explica: la fusión TOPK_MOE (el
router en un dispatch) se desactiva cuando su salida se solapa en memoria con una entrada, y el solapamiento depende de
la disposición del asignador. Quitar los CONT cambia la disposición y el router de 7 a 19 capas por ubatch de prefill
pasa de sin fusionar a fusionado: otro camino de cálculo y otro redondeo. Descartada: se pedían salidas idénticas a la
base y la ganancia medida era ~0.23 ms (sección 36.4). Parche en el scratchpad de la sesión
(`sp4_cand2_scale_padded_rows.patch`).

**Tema abierto (no del SP4): la fusión TOPK_MOE depende de la disposición de memoria.** En el build adoptado, los
ubatches de prefill corren el router sin fusionar en 44-48 de las 48 capas; el decode lo fusiona en todas. Todo cambio
que mueva la disposición de un grafo de prefill (un nodo más o menos) puede prender o apagar esa fusión en algunas capas
y cambiar el redondeo: es la misma fragilidad que las secciones 35.2 y 36.2, y hace que "respuestas idénticas a la
base" rechace cambios exactos. Darle al router una decisión que no dependa de la disposición (que el kernel fusionado
admita el solapamiento fila a fila, o que el asignador no reuse esa memoria) es trabajo de backend a decidir; en tiempo
vale poco (≈6 dispatches por capa en ubatches de ~4 s).

**Cierre del SP4.** Adoptadas la fusión 1 (colas del conv state sin CONT, abc86e8be) y la fusión 3 (snapshots del GDN
directo al caché, 1d7cd0d16); descartadas la fusión 2 (colas en un gather), la vía 14 del lado de lectura, la norma L2
del GDN y el CONT de los mezcladores hc. NP=1 a 40k, sumando lo medido dentro de cada cadena (entre cadenas el paso
deriva ~0.7 ms, sección 36.1): **−0.66 (v52) − 2.16 (v60) ≈ −2.8 ms por paso**, desde ~41.3 a ~43.2 t/s en greedy y de
~39.3 a ~41.2 con muestreo (niveles aproximados, de cadenas distintas); en producción NP=2, 43.4 / 41.2 t/s (v71). La
meta del SP4 (44.2-44.5 t/s) no se alcanzó: las candidatas restantes no tenían techo o chocaban con la fragilidad de
la disposición.

Fuentes: `~/dbg/merge/chain_v7{6,7,8}.{sh,out}`, `opcount_gate.py` (ahora admite ops quitadas),
`~/dbg/depth/srvlog-v78{b,n}.log`.

### 36.6 CONT de los mezcladores hc: adoptada con la regla nueva (2026-09-30, cadena v79)

El Director decidió el 2026-09-30 que un cambio que mueve el redondeo es una optimización si sigue siendo determinista
e igual de correcto: ya no se exigen respuestas idénticas a la base sino determinismo dentro del build (probe repetido,
`depth_repeat`, `graph_diff4`, compuerta de historia) y una compuerta de calidad (`quality_gate.inc.sh`: divergencia
KL con llama-perplexity a 4096 de contexto sobre 4 chunks del wikitext-2 contra la base, a lo sumo el doble de la de
puro redondeo, que es la base con las fusiones apagadas, y acuerdo del token top-1 no menor que el de referencia menos
0.5 puntos). Con esa regla, el cambio de la sección 36.5 se volvió a medir completo (fc788ac4b).

| Carga | Greedy: ms por paso (t/s, aceptación) | Producción: ms por paso (t/s, aceptación) |
|---|---|---|
| Base (primera) | 51.28 (43.46, 564/917) | 50.77 (41.37, 536/973) |
| Nueva | 50.70 (42.48, 548/948) | 50.68 (42.14, 544/965) |
| Nueva | 50.82 (42.37, 548/948) | 50.71 (42.12, 544/965) |
| Base (última) | 51.66 (43.14, 564/917) | 51.03 (41.17, 536/973) |

**Costo por paso: −0.71 ms en greedy y −0.20 con muestreo** (cargas con timing: −1.08 / −0.41; `gpu_wait` de la
verificación 37.51 / 37.33 → 36.88 / 36.85 ms). Los t/s de esta muestra van en direcciones opuestas porque el texto
toma otro camino y la aceptación del draft cambia con él: en greedy baja (548/948 contra 564/917) y con muestreo sube
(544/965 contra 536/973). Es variación del texto, no del cambio; el costo por paso es la medida del cambio, y a igual
aceptación vale +0.4 a +1.4 % de t/s.

Compuerta de calidad: la referencia (base sin fusiones) da KL 0.0028 y top-1 igual en el 98.55 % de los tokens; la
nueva, KL ~0 y top-1 igual en el 100.000 %, perplejidad 2.2473 → +0.0012. Esa compuerta solo corría ubatches de 2048
(sección 37.6): con ubatches de 3 tokens, el camino de la verificación, a 4096 de contexto el cambio sí se distingue
de la base (cadena v79q: KL 0.0016 contra 0.0211 de referencia, top-1 igual en el 99.59 %) y pasa la compuerta. La
causa de esa diferencia en decode no se investigó; la probable es la misma que en el prefill, fusiones que dependen de
la disposición de la memoria (sección 36.5).

Determinismo: nueva contra nueva idéntica, compuerta de historia 1 (su primera corrida real), `depth_repro` +
`depth_repeat` idénticos, `graph_diff4` 72 de 11649 nodos, probe repetido idéntico (y todavía igual a `probe-v26`: el
prompt de 4.7k no llega a los ubatches profundos), conversación texto → imagen → texto 5/5, imagen de 2048×2048 en
12.6 s, cero timeouts de anillo. Producción (NP=2, recién cargados, una carga por build): 51.79 / 50.86 → 50.94 /
51.04 ms por paso; rig NP=2: dos slots 45.4-47.4 → 46.8-47.3 t/s agregados, el slot que genera mientras llega un
prompt de 15k 17.53 → 17.21 chunks/s.

**Cierre del SP4 con la fusión 1, la fusión 3 y el CONT de los mezcladores hc:** sumando lo medido en cada cadena,
−0.66 (v52) − 2.16 (v60) − 0.2 a 0.7 (v79) ≈ −3.0 a −3.5 ms por paso a 40k.

Fuentes: `~/dbg/merge/chain_v79.{sh,out}`, `quality_gate.inc.sh`, `~/dbg/depth/ppl-v79q{b,r,n}.log`.

## 37. SP6 del plan de decode: la tasa de los pesos en la verificación (2026-09-30, cadenas v80-v83)

### 37.1 Re-perfil con el build adoptado (cadena v80)

Perf logger serial (una carga NP=1 a 40k, 128 tokens greedy, 61 grafos de verificación de 3 tokens, mediana por
forma). Casi todos los matmul de la verificación ya leen a la tasa de DRAM: q5_K 10240×2560 a ~216 GB/s, la cabeza
q6_K a ~228, los iq4_nl 6144×2560, 2560×6144 y 12288×2560 a 193-212. Quedan por debajo:

| Nodo (3 tokens) | Por paso | Por nodo | Tasa |
|---|---|---|---|
| Expertos gate/up iq3_s (1280×2560, 47 capas) | 9.87 ms | 210 µs | ver 37.2 |
| Expertos down iq4_nl con el MUL fusionado (2560×640, 48) | 5.47 ms | 114 µs | ver 37.2 |
| Mezcladores hc, down 324×10240 (96) | 1.66 ms | 17.3 µs | ~108 GB/s |
| Mezcladores hc, up 10240×320 (97) | 1.36 ms | 14.1 µs | ~131 GB/s |
| Experto compartido 640×2560 (96) y 2560×640 (48) | 1.14 ms | 7.8-8.0 µs | ~115-118 GB/s |
| Router q8_0 512×2560 (48) | 0.53 ms | 11.1 µs | ~126 GB/s |

Con 3 tokens, los expertos no van por el camino de tiles sino por el mat-vec-id (hasta 8 tokens), con un dispatch
por token: el kernel lee los pesos de cada par (token, experto), 30 por capa.

### 37.2 Expertos distintos por paso

`sp6/moe_ids.cpp` evalúa texto sobre el modelo y lee, por capa MoE, los expertos que elige cada token (tensores
`ffn_moe_topk` por el callback del scheduler). Sobre 5131 tokens de respuestas del propio modelo, en ventanas de
tokens consecutivos (el lote de verificación): 2 tokens, 16.2 expertos distintos de 20 pares; 3 tokens, 21.6 de 30
(72 %); 4 tokens, ~26 de 40. La capa 47 queda fuera del promedio: en un prompt solo computa las filas de salida.

### 37.3 Techos en el modelo (cadena v81)

Build de diagnóstico bin-v81diag (no commiteado), `gpu_wait` de la verificación de 3 tokens, A B C D ×2 + A:

| Variante | Rondas | Media | Contra la base |
|---|---|---|---|
| A: base | 36.97 / 36.98 / 36.78 | 36.91 | — |
| B: expertos compartidos (el workgroup del primer token que elige un experto calcula también los tokens siguientes que lo eligieron; el resto sale) | 36.86 / 36.88 | 36.87 | −0.04 |
| C: workgroup de 4 subgrupos para m ≤ 1024 y k ≥ 8192 (mezclador hc 324×10240) | 36.47 / 36.67 | 36.57 | **−0.34** |
| D: el mismo workgroup para m ≤ 1024 y k ≥ 2048 (también 640, 512 y 48 × 2560) | 37.53 / 37.68 | 37.61 | +0.70 |

B da salidas idénticas a la base, bit a bit (por construcción: cada par corre la misma aritmética), pero no ahorra:
por nodo (perf logger) gate/up baja 210 → 205 µs y down sube 114 → 119 µs, porque los pares repetidos se calculan en
serie dentro del workgroup del primero. Las lecturas repetidas ya salían de la caché: el costo de un par repetido es
su decuantización, no la DRAM.

### 37.4 Banco fuera del modelo (`sp6/moe_bench.cpp`)

Las mismas formas con el comportamiento de memoria del modelo: 48 capas de expertos elegidos entre 512, 96 pares de
pesos de mezclador; nada queda en la MALL de un nodo al siguiente. Reproduce los mezcladores del modelo (18.3 y 14.5
µs contra 17.3 y 14.1) y los expertos a escala (gate/up 170 µs con 22 expertos distintos, 210 en el modelo).
Calibración de gate/up con 30 pares por capa: 29.4 distintos 193 µs, 22.0 distintos 170 µs, 10 distintos 138.5 µs; un
token (10 pares) 70 µs. Un par cuyo experto ya está en caché cuesta ~3.4 µs: es la decuantización del iq3_s (tabla de
la rejilla en memoria compartida, signos, escalas), repetida por cada token.

Otras variantes medidas en el banco: el camino de tiles (mul_mm_id) para los expertos de 3 tokens, 352 µs (×2.2);
el camino de tiles para el mezclador de k = 320, 25.2 µs (×1.7). Descartadas.

### 37.5 Expertos agrupados por experto: tres diseños, ninguno gana

La idea: cada experto de la verificación se lee y decuantiza una vez y se multiplica por todos los tokens que lo
eligieron (columnas del mat-vec), en vez de un par (token, experto) por workgroup. Banco, gate/up iq3_s y down
iq4_nl con el MUL fusionado, 3 tokens y 22 expertos distintos por capa (el kernel por pares: 168-170 y 110-112 µs):

| Diseño | gate/up | down | Nota |
|---|---|---|---|
| Un dispatch por número de columnas (1..n), NUM_COLS exacto, barrido serial de ids | 202 | 179 | 3× workgroups lanzados; cada uno vacío cuesta lanzamiento + barrido |
| Un dispatch, NUM_COLS = n, columnas inactivas cortadas con `break` uniforme, selección en memoria compartida | 280 | 204 | las guardas dentro de los bucles desenrollados rompen el código generado |
| Un dispatch, NUM_COLS = n, todas las columnas calculadas (las inactivas repiten la primera, sin escribir) | 258 | 238 | cada columna extra cuesta casi un par completo |

Con 2 y 4 tokens el orden es el mismo. El tercer diseño muestra la causa: en estos mat-vec el costo de un par es el
trabajo por columna (cargas del vector B en f32, signos, FMA, reducción y escritura), no la lectura ni la
decuantización de los pesos, así que agrupar tokens no puede ganar. La ruta de producto entero (B en q8_1,
`dotPacked4x8`, el shader IQ4_NL de b9c196c1c que el selector del mat-vec nunca habilitó, vía 22 de la sección 3):
down 110.6 → 109.7 µs, mezclador 324×10240 16.0 → 15.0, mezclador 10240×320 15.6 → 17.8 (el B se cuantiza en un
dispatch aparte). Tampoco. Filas por workgroup (rm 2 u 8 en vez de 4) y el camino de tiles, peores (37.4).
Los parches quedan en `~/dbg/merge/sp6/` (`shared-experts-diag.patch`, `grouped-experts-X.patch`); no entran al árbol.

### 37.6 La compuerta de calidad no veía el camino de decode

llama-perplexity procesa ubatches de 2048 tokens: los matmul van por los kernels de tiles y nunca por los mat-vec de
la verificación. La compuerta de las cadenas v79 y v82 no podía ver un cambio en esos kernels (en v82 dio KL ~0 y
top-1 100 %, lo mismo que en v79). `quality_gate.inc.sh` corre ahora dos modos, los dos obligatorios: ubatches de 2048
(4 chunks) y ubatches de 3 tokens, el lote de la verificación (2 chunks, ~2 min por corrida). En el modo decode la
referencia de puro redondeo (la base con las fusiones apagadas) es mucho mayor que en el de prompt (KL ~0.021 frente
a 0.0028; top-1 igual ~96 % frente a 98.5 %): el decode propaga el redondeo por el estado recurrente del GDN a lo largo
de los 4096 tokens. Re-medidos en ese modo, los dos cambios pasan: fc788ac4b (cadena v79q) KL 0.0016, top-1 99.59 %;
el workgroup de 37.7 (cadena v82q) KL 0.0230 contra 0.0207 de referencia, top-1 96.21 % contra 96.04 %.

### 37.7 El workgroup de 4 subgrupos: adoptado y revertido por neutro (cadenas v82-v84; 9c63b0eda, 2bfe97d00)

En RDNA3, los mat-vec con m ≤ 1024 y k ≥ 8192 usan los pipelines de 4 subgrupos que ya existían (upstream solo los
elige en NVIDIA e Intel). En qwen4exp afecta a los mezcladores hc 324×10240 y 320×10240 y a la inyección 4×10240.

| Medida | Base | Nueva |
|---|---|---|
| Mezclador 324×10240 por nodo, perf logger en el modelo / banco | 17.3 / 18.3 µs | 16.1 / 16.4 µs |
| v82, NP=1 a 40k, greedy (base primera y última) | 51.06 / 51.00 ms | 51.25 / 51.11 ms |
| v82, muestreo de producción | 51.48 / 51.20 ms | 50.98 / 50.74 ms |
| v82, `gpu_wait` de la verificación (cargas con timing) | 36.27 ms | 36.23 ms |
| v83, A B A B A B A, `gpu_wait` | 36.60 / 36.65 / 36.81 / 36.44 | 36.60 / 36.59 / 36.34 |
| v83, total de la verificación | 44.04 / 44.01 / 44.06 / 44.03 | 43.67 / 43.68 / 43.87 |
| v83, ms por paso greedy | 51.92 / 52.35 / 52.06 / 51.77 | 51.60 / 51.46 / 52.24 |

v81 a v83 sugerían −0.1 a −0.3 ms por paso (v83: −0.29 ms en el total de la verificación, −0.26 ms por paso; v82:
−0.17 ms por paso en promedio de greedy y muestreo; v81: −0.34 ms de `gpu_wait` con la perilla), con pocas rondas y
cerca del ruido entre cargas. La cadena v84 lo resolvió (abajo): **de punta a punta el cambio es neutro**; el nodo
baja 7-10 % aislado, pero eso no se ve en el paso. Compuertas (v82):
nueva contra nueva idéntica, compuerta de historia, `depth_repro` y `depth_repeat`, `graph_diff4` 72 de 11649 nodos,
probe repetido idéntico, conversación con imagen 5/5, imagen de 2048×2048 en 12.6 s, cero timeouts de anillo,
compuerta de calidad en los dos modos (37.6). Producción NP=2 recién cargada: 51.03 / 50.91 → 51.00 / 50.98 ms por
paso (greedy / muestreo); rig NP=2 sin cambio fuera del ruido.

Cadena v84 (A B ×5 + A, turnos greedy y muestreo de 512 tokens por carga; `sp6/ab_stats.py` compara cada carga nueva
con la media de las dos bases que la rodean, así la deriva se cancela; intervalo de 95 % con t de Student, 4 grados
de libertad):

| Medida | Nueva − base | 95 % |
|---|---|---|
| ms por paso, muestreo de producción | −0.03 | [−0.19, +0.14] |
| ms por paso, greedy | +0.10 | [−0.74, +0.94] |
| `gpu_wait` de la verificación | −0.00 | [−0.21, +0.20] |
| Total de la verificación | −0.10 | [−0.18, −0.01] |

Sin ganancia medible por paso: con muestreo, cualquier efecto queda por debajo de 0.2 ms (0.4 %) en uno u otro
sentido. Los t/s brutos de la cadena (+0.36 con muestreo, +0.60 en greedy) vienen del texto: el redondeo nuevo toma
otro camino y su aceptación del draft salió algo mejor (269/484 contra 267/487); no se atribuyen al cambio.

Revertido por decisión del Director (2bfe97d00): una regla neutra en una heurística que upstream retoca seguido solo
suma trabajo de merge. La librería Vulkan volvió a ser idéntica byte a byte a la del build adoptado con fc788ac4b
(bin-v79new), así que no hubo compuertas nuevas; los casos de test-backend-ops de los mezcladores (324×10240 con 1 y
3 tokens en ocho tipos, 320 y 4 filas con 3) se quedan y pasan por el camino por defecto.

### 37.8 Cierre del SP6

El plan estimaba −1.5 a −3 ms por paso con los mezcladores a 210 GB/s y los expertos gate/up a DRAM. Lo medido:
nada de punta a punta (37.7, cadena v84: −0.03 ms por paso, 95 % [−0.19, +0.14]); el único cambio adoptado se
revirtió y el SP6 cierra sin cambios de código en producción. Los mezcladores no llegan a la tasa de DRAM por la forma (un nodo de 1.9 MB con 81 workgroups
tiene poco que repartir: el workgroup grande lo lleva de ~108 a ~116 GB/s), y los expertos no tienen palanca de
kernel: su costo por par es el trabajo por columna, no la lectura de los pesos (37.5). Lo que queda en la
verificación son ~9.9 ms de gate/up y ~5.5 de down en el mat-vec-id, que solo bajarían con otro diseño de kernel (B
en memoria compartida o en f16, varias filas por hilo) o con menos pares por paso; ninguno cabe en la ventana de 3-4
días del SP6. Las herramientas quedan en `~/dbg/merge/sp6/`: `moe_ids.cpp` (expertos por token sobre el modelo),
`moe_bench.cpp` (el banco con pesos fuera de la MALL), `shared_test.cpp` (bit a bit entre kernels y contra la CPU),
`perf_shapes.py` (tiempos por forma de los grafos de verificación).

Fuentes: `~/dbg/merge/chain_v8{0,1,2,3,4}.{sh,out}`, `chain_v82q`, `chain_v79q`, `~/dbg/depth/srvlog-v8{0p,1p}.log`,
`~/dbg/merge/sp6/`.

## 38. Vía 7 (D5): split_k en el modo compacto de la FA sparse, adoptada (2026-09-30, cadena v85, 7c05f19ab)

Al cerrar el SP6 (sección 37), el perfil de la verificación a 40k dejaba la atención sparse de las 12 capas QSA en
3.7 ms por paso (309 µs por capa), creciente con la profundidad solo en la construcción de la lista. El modo compacto
(lotes chicos: decode y verificación) corre tres dispatches por nodo: la lista de celdas elegidas (la unión de las
filas del tile, desde `kv_idx`), el gather de esas celdas a regiones f16 y la atención sobre las regiones. Con
atención agrupada (GQA 12), un workgroup por token y cabeza KV: 6 por capa en una verificación de 3 tokens, en una
GPU de 40 CUs. Desglose en test-backend-ops (forma de la verificación, 32k, selección de 2051 celdas, una perilla de
diagnóstico que salta dispatches, no quedó en el árbol):

| Parte | µs por nodo |
|---|---|
| Lista (1 workgroup) | ~17 |
| Gather a f16 | ~49 |
| Atención (6 workgroups) | ~440 |
| Nodo | 507 |

Cambio: en el modo compacto, cada lista se parte hasta que los workgroups llenan los núcleos (13 partes: 2 × 40 / 6),
workgroups consecutivos toman partes consecutivas de la misma lista (`init_sparse` divide el índice de workgroup por
`k_num`) y la reducción de split_k que ya existía suma las partes en orden fijo. Perilla de apagado:
`GGML_VK_DISABLE_SPARSE_FA_COMPACT_SPLIT=1`. Banco: 509 → 125 µs por nodo a 32k, 533 → 134 a 131k.
FLASH_ATTN_EXT 5221/5221 (los casos de 1, 3, 16 y 64 filas con `kv_idx` pasan por el modo compacto con split).

Cadena v85 (NP=1 a 40k, base primero y último; la base es la librería Vulkan adoptada con fc788ac4b):

| Carga | Greedy: ms por paso (t/s, aceptación) | Muestreo de producción |
|---|---|---|
| Base (primera) | 51.42 (41.89, 548/948) | 52.69 (40.53, 544/965) |
| Nueva | 49.06 (44.37, 553/948) | 48.62 (44.57, 551/943) |
| Nueva | 48.97 (44.45, 553/948) | 48.58 (44.61, 551/943) |
| Base (última) | 50.95 (42.27, 548/948) | 50.82 (42.02, 544/965) |

**−2.1 a −2.2 ms por paso (−4.2 %)** en greedy; con muestreo −2.2 contra la base última (la primera, 52.69, es la
más alta de la cadena). Cargas con timing: `gpu_wait` de la verificación 36.69 → 34.63 ms, total 43.91 → 41.80.
Producción NP=2 recién cargada: 50.96 / 51.13 → 49.17 / 48.72 ms por paso (greedy / muestreo); rig NP=2: 48.1-49.5 →
50.4-54.3 t/s agregados. A igual aceptación, +4 a +4.5 % de t/s a 40k. El costo de la parte partida no depende de la
profundidad (la lista es de 2051 celdas por consulta), así que el ahorro en ms se mantiene a más contexto.

La reducción suma las partes en orden fijo pero distinto del anterior: los logits se mueven por redondeo. Compuertas:
nueva contra nueva idéntica, compuerta de historia, `depth_repro` y `depth_repeat`, `graph_diff4` 72 de 11649 nodos,
probe repetido idéntico (y todavía igual a `probe-v26`: el prompt de 4.7k no llega a la atención sparse),
conversación con imagen 5/5, imagen de 2048×2048 en 12.6 s, cero timeouts de anillo. Compuerta de calidad en tres
modos: los de 4096 de contexto no llegan a la atención sparse (KL ~0 en ambos), así que se agregó el modo de decode
profundo, ubatches de 3 tokens sobre un trozo de 32k con la mitad evaluada a 16-32k: referencia de puro redondeo KL
0.0116 y top-1 95.80 %, nueva KL 0.0116 y 95.74 %. La corrida base de ese modo escribe ~8 GB de logits con el modelo
cargado y bajó MemFree a 9 GiB: el guardia de `ppl_run` ahora sincroniza y suelta de la caché ese archivo cada ~5 s.

Fuentes: `~/dbg/merge/chain_v85.{sh,out}`, `quality_gate.inc.sh`, `~/dbg/depth/ppl-v85q{p,d,D}{b,r,n}.log`.

