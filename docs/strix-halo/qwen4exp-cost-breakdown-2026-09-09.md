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
| Expertos del verify (3 tokens) | 18.2 | `ggml_vk_mul_mat_vec_id_q_f16` despacha un dispatch por token (ggml-vulkan.cpp:11517): los expertos compartidos entre los 3 tokens se leen 3 veces; 3.98 GB a 219 GB/s |
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

## 3. Vías de optimización, ordenadas por ganancia esperada

### Prefill

1. **Tiles token-major en la FA sparse.** A n > 8 cada tile es 16 tokens consecutivos de una
   sola cabeza q (`get_fa_tuning_params_coopmat1`: Br = 16; la agrupación GQA solo si N ≤ 8,
   ggml-vulkan.cpp:11790) con la unión de sus 16 listas; el propio backend documenta que la
   unión cubre el 33 % de la caché a 16k, y las 12 cabezas q releen la misma unión. Con tiles
   de 12 cabezas × 1 token, la ruta GQA que ya usa el decode, cómputo y bytes de K/V bajan
   ~4.8×. Estimación: de 67 a ~15 ms por nodo a 40k, −8 s de 108. Esfuerzo medio en el
   shader del fork; el determinismo se revalida con `depth_repeat.py`.
2. **MUL_MAT_ID en prefill.** Shader entero q8_1 (`mul_mmq.comp`) con tile BN = 128 elegido
   por tokens totales (ggml-vulkan.cpp:11071), pero cada experto recibe ~40 filas: 31 % de
   utilización de columna, cargas de B con índices fuera de `_ne1` (`mul_mmq.comp:234-241`),
   barreras cada 32 de k (BK_STEP = 1), y sin hoisting de ids porque `hoist_row_ids` exige
   ≤ 256 expertos (ggml-vulkan.cpp:10969): cada workgroup vivo rebarre 20480 ids.
   Estimación: de 20 s a 9-11 s. Esfuerzo medio-alto.
3. **Mezcladores hyper-connection.** `m=320` y `k=320` caen en el tile grande con 48
   workgroups y B en f32; `m=4` en el tile 32×32 con 64 waves y 87 % de relleno; no existe
   ruta que intercambie A y B. Estimación −6 s. Esfuerzo bajo-medio: tiles para m/k
   pequeños, formulación transpuesta del inject o su fusión en `hc_gated_mean`.
4. **Los 27 s que no son GPU.** Instrumentar una línea de tiempo por ubatch (set_inputs,
   build/alloc, compute host, D2H, catch-up del draft) antes de tocar nada. Lo ya medido:
   input fill 0.15-0.25 s por ubatch, candidato `prefetch_ple_rows` (agrupar los rangos con
   `process_madvise` o emitirlos en un hilo; el prefetch del ubatch actual es redundante
   salvo en el primero del batch).
5. **`eh_proj` del draft en una sola matmul.** `ggml_reshape_2d` a [5120, 4·T] antes de
   `build_lora_mm` en `src/models/qwen4exp.cpp` (graph_mtp): de 130 a ~10 ms por ubatch.
   Trivial.
6. **IQ4_NL en la lista f16-B.** El SPIR-V `matmul_iq4_nl_f16` se genera, pero
   ggml-vulkan.cpp:5395-5406 no instancia el pipeline: los densos IQ4_NL mueven B en f32.
   El fork midió +4.5 % denso con f16-B en otros tipos. Dos líneas.
7. **Fusión de la ponderación y la suma de expertos** en el epílogo del MUL_MAT_ID de
   prefill: MUL 3.0 s + MULTI_ADD 1.65 s. Esfuerzo medio.
8. **QSA para el bloque del draft.** Atiende denso: 229 ms por ubatch a 37k, lineal con la
   profundidad; en decode su FA densa pasará de 1 ms a ~7 ms por paso a 262k. El GGUF trae
   `blk.48.indexer.*` sin usar. Requiere confirmar contra la referencia que la cabeza MTP
   usa el indexador, y un caché de indexador propio. Esfuerzo alto.
9. Menores: concat + transpose del `ssm_conv` (0.9 s); mat-muls `m=48`, `m=1` f32 y router
   f32 (0.7 s); selección QSA a granularidad de bloque y FA que consuma la lista top-k sin
   construir la máscara (~1.5 s a 40k, lineal con n_kv); GDN sin kernel chunked (76 ms por
   ubatch).

### Decode (por paso de 66 ms)

1. **Mat-vec-id agrupado por experto a n ≤ 8.** Leer cada experto único una vez para los 3
   tokens (hoy un dispatch por token con NUM_COLS = 1): ahorra el solape real, estimado
   15-30 % de 18 ms, −3 a −4.5 ms. Medir el solape con un volcado de `selected_experts`.
2. **Kernel fusionado del mezclador HC** (norm → down → silu → up → gate-mean + inject en
   dos dispatches en vez de siete): −4 a −4.5 ms.
3. **split_k en el modo compacto de la FA sparse**: 6 workgroups → 40+; −3 ms
   (ggml-vulkan.cpp:11927, assert en :12148).
4. **Router**: NUM_ROWS = 1 en el mat-vec f32 (ggml-vulkan.cpp:5898): arreglo de kernel
   −1.8 ms sin tocar el modelo; router en q8_0 en el GGUF −2.3 ms con verificación PPL/KLD.
5. **Estado GDN in place**: el op escribe los K snapshots en un tensor propio y luego 36 CPY
   de 9.4 MB, más el gather de entrada; escribir en la vista del caché y leer in place:
   −1.5 ms.
6. **Reducción de dispatches**: cont + cpy de los slots de rollback (108), scale plegado en
   `hc_down` (96, exacto porque 1/4 es potencia de dos), sigmoid × mul de la norma GDN,
   fill/set_rows/add de la máscara si la FA consume la lista: ~−2 ms.
7. Draft: MMVQ IQ4_NL compilado pero fuera del allowlist de
   `ggml_vk_get_dequantize_mul_mat_vec` (ggml-vulkan.cpp:8520-8539); ganancia pequeña.

Suma realista de 1-6: −14 ms por paso, 66 → ~52 ms, +25 % de decode. En prefill, 1-3 y 5-7
suman ~−28 s de GPU sobre 108 s; el bloque host es adicional.

## 4. Qué medir antes de implementar

- Línea de tiempo host por ubatch de prefill, producción descargada: set_inputs, build/alloc,
  compute host, copias D2H, catch-up del draft. Decide cuánto vale el punto 4 de prefill.
- Tamaño de la unión por tile y lista por token a 40k, volcando los contadores de
  `prealloc_y` (flash_attn_sparse_idx). Confirma el 6.3× del punto 1 de prefill.
- Solape de expertos entre los 3 tokens del verify, volcando `selected_experts`. Dimensiona
  el punto 1 de decode.

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
| 2 | 238 | 0.791 | 21 % |
| 3 | 40608 | 0.730 | 27 % |
| 4 | 432 | 0.533 | 47 % |

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
BN = 128 del MUL_MAT_ID (sección 3, prefill punto 2) la mediana de 21 filas ocupa el 16 % del
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
