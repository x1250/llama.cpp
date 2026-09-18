# Cruce con upstream llama.cpp (2026-09-17)

Base del fork: último sync `f3f1a8f27` (merge `00bbf6c90`, 2026-09-08); merge-base con `upstream/master`
`5eec3ad01`. Desde ahí upstream suma 273 commits: 129 ya presentes en el fork por patch-id y 144
ausentes (`git cherry master upstream/master`). Revisión de solo lectura (git show, código); nada
aplicado. Cada verdicto se revisa en el momento del merge con las reglas de CLAUDE.md (sonda a
n_ctx 135168, lazy mode, regla de splits, determinismo).

## Peligros conocidos: neutralizados, y uno nuevo

- `f3f1a8f27` (lazy off en iGPU): entró por patch-id (`6d640e6a0`) pero el hunk dañino se descartó al
  resolver el merge `00bbf6c90`; verificado por contenido en `src/llama-model.cpp:1413-1421` (solo el
  fallback de `use_mmap`). Ojo: al vivir en una resolución de merge y no en un revert, un futuro
  cambio de upstream en ese bloque volverá como conflicto: resolver manteniendo la eliminación.
- `992cb503c` (regla de splits): revertido explícitamente por `ca1c30baf`; la regla sigue viva en
  `ggml/src/ggml-backend.cpp:1341-1351`.
- Nuevo: `14a9d09f7` elimina `--mmap/--no-mmap/--mlock/--dio` (reemplazo `--load-mode`). El launcher ya
  usa `-lm mmap --lazy-mode auto` (`run-server.sh:350`); revisar cualquier script de rig que aún pase
  los flags viejos antes del merge.

## Verdictos

| Commit | Qué es | Verdicto | Valor aquí |
|---|---|---|---|
| `b0dcb8192` server: fix speculation after an image | el server pasa `n_tokens()` como posición del draft; con imágenes (mrope: una imagen ocupa nx·ny tokens pero max(nx,ny) posiciones) el draft queda desplazado cientos de posiciones tras cada imagen: RoPE del draft mal y `seq_rm` del KV del draft en la región equivocada | **tomar primero**; hunk 3 de `common/speculative.cpp` a mano (el fork divergió en DFlash); renombrar `n_past → pos0` en `speculative.h:69` y 8 sitios; `server-context.cpp:3056` → `pos_next()` | bug real en producción (MTP + mmproj). Verificar con la conversación texto → imagen → texto y el suite de determinismo |
| `481c65f09` vulkan: argsort race/OOB | carrera de lectura/escritura en `argsort.comp` (el kernel del router, 512 columnas); el kernel `large` no se usa aquí | tomar tal cual (33 líneas de shader) | rendimiento 0; corrección; el fork ya sufrió dos defectos de orden en shaders |
| `43f3dda62` ggml: ids vacío al descargar expertos | lectura fuera de rango si un tensor de ids tiene 0 elementos | tomar tal cual (4 líneas) | inalcanzable hoy (expertos en GPU); endurecimiento |
| `5cdd3d1da` KV del contexto MTP | generaliza la lista de arquitecturas | tomar tal cual | sin efecto: el fork ya cachea solo la capa nextn (`mtp_on_hybrid_qwen`) |
| `37b53fd45` qwen4exp: hc ops | extiende `dsv4_hc_pre` (gated) y `dsv4_hc_post` (sin comb) para qwen4exp, **solo CPU y CUDA; Vulkan las rechaza** (`ggml-vulkan.cpp:19695/19698` upstream) | ggml half: tomar (aplica limpio, mantiene deepseek4 alineado; sin conflicto de enums). Half de `qwen4exp.cpp`: **saltar** | adoptar el grafo de upstream perdería HC_GATED_MEAN/HC_INJECT (medidos +9.4 % pp2048, +3.8 % tg) y el `down_inject` fusionado |
| `41abbfd59` qwen4exp: rms_norm + mul | carga las normas como {n_embd, hc} para que RMS_NORM y MUL queden adyacentes | tomar con adaptación: solo `grouped_norm` del PLE (`qwen4exp.cpp:1758-1764`), en el idioma del fork (reshape del peso + forward_expand) | el fork ya fusiona las normas de los mezcladores (2.52 s por prefill); lo que falta son 2 normas del PLE, ~0.03 % |
| `311d4211b` sin V cache para el indexador | marca el cache del indexador como MLA para no reservar V | tomar con adaptación: sustituir el `n_embd_head_v_full = 1` del fork (`llama-memory-hybrid-idx.cpp:57-61`) | ~24 MiB (12 capas × 262144 × 2 × 4 B); `is_mla()` verificado sin efecto en el K del indexador |
| `50182a53f` vulkan: topk_moe en prefill vía `add_alloc_dep` | sustituye la exención `nrows == 1` por dependencias de asignación para que el router fusionado corra en cualquier lote | tomar con adaptación (a mano: `keep_pattern` de los cinco patrones topk_moe y borrar el bloque `is_topk_moe_single_row`; el `graph_optimize` del fork tiene patrones propios) | **medido en el perfil del 2026-09-16: el router ya corre fusionado en prefill y verify en la mayoría de las capas** (`TOPK_MOE_EARLY_SOFTMAX_NORM 48 x` por ubatch en varios grafos); lo no fusionado son ~3 ms por ubatch (ARGSORT 2.8 + SUM_ROWS/DIV/CLAMP) = 0.1 % del prefill y ~0.5 % del paso de decode. Bajo interés; no es bit-idéntico (reducción distinta): suite de determinismo si se toma |
| `6788edb4f` vulkan: matrices M pequeñas | (a) selector de tile solo coopmat2; (b) split_k sin la condición `m >= wg_denoms[0]`; (c) intercambio de operandos para m = 1 por mat-vec | (a) saltar; (b) **saltar: es la heurística medida negativa el 2026-09-16**; (c) tomar con adaptación, baja prioridad | (c) afecta a un nodo, `ffn_gate_inp_shexp` [2560 × 1] a n = 2048: ~50-60 ms de 80 s (< 0.1 %) |
| `fc82583e6` vulkan: sparse FA (+ `8e93a9773`) | la mitad ggml (`set_n_kv_max`) ya está en el fork; la mitad Vulkan re-barre la máscara (sin tensor de índices), **solo decode** (exige `gqa_ratio > 1`, es decir N ≤ 8), solo K/V f16, umbral 2× | **saltar la mitad Vulkan** | más estrecha que la del fork (token-major en prefill, compacto en decode, q8_0, prepass por trozos de orden fijo, umbral 8× medido). No aporta a la vía 2 bis (los índices del selector no llegan a la FA tampoco en upstream). Coste: divergencia permanente en `flash_attn_cm1.comp` / `flash_attn_base.glsl` / `get_fa_pipeline_state`; upstream usa el mismo bit 16 para su flag |
| `d4365d955` vulkan: tail BN/2 de MUL_MAT_ID | solo `mul_mm_cm2.comp` (coopmat2) | saltar (inerte en gfx1151) | 0 |
| `91f6a6cf3` vulkan: tipo de A como spec constant | un SPIR-V por kernel de matmul para todos los tipos (menos pipelines, carga más rápida); sin cambio de tiling | **saltar ahora; presupuestar una ventana propia**: es el commit que bloquea cualquier merge futuro de la ruta matmul (reexpresar el epílogo de escala, las variantes f16-B, los flags integer-dot, el tile alineado y `pipeline_dequant_mul_mat_mat_id_f16b` sobre la nueva estructura) | tiempo de carga (cada carga es una ventana de riesgo), no rendimiento |
| `28ff09582` vulkan: escrituras CPU en `cpy_tensor_async` en reposo | evita un submit por copia pequeña entre backends si el contexto está inactivo | condicional: contar las llamadas en un ubatch y un paso (`VK_LOG_DEBUG` existente); 0 llamadas → cerrar | sin cuantificar; en UMA la condición `host_coherent` se cumple más |
| `2f3fd0252` CUDA graph para el draft MTP | código compartido en `llama-context`: dos `gf_res_prev` según `n_outputs`, reuso más estricto | saltar como cambio aislado; si llega en un merge, resolver el hunk 3 manteniendo el bloque de `LLAMA_INPUT_TIMING` y correr el suite de determinismo | sin ganancia en Vulkan; toca el reuso de grafos (tres defectos de determinismo vivieron ahí) |
| `fa6769818` spec: chunk mtmd con DFlash | salta filas de imagen en DFlash | tomar por higiene (aplica limpio) | inerte (producción usa MTP) |
| `093a2f86c` common: `llama_n_rs_seq` antes de `llama_decode` | evita dos decodes de sondeo al arrancar | tomar tal cual | cosmético |
| `82d6bb284`, `56381e407`, `c069aa7f5`, `160bd031b` (router multi-modelo), `fb27a525d` (TP), arquitecturas nuevas, cmake/PCH (`3bcfeb700`, `f3a184b15`), jinja/grammar/py | — | arrastrar en el merge; reconfigurar cmake limpio tras el merge | 0 |

## Orden propuesto para el próximo merge

1. `b0dcb8192` (bug con imágenes), `481c65f09`, `43f3dda62`, `5cdd3d1da`, `fa6769818`, `093a2f86c`, ggml half de
   `37b53fd45`: sin ventana de medición, solo el suite de determinismo (con imagen) tras el build.
2. `311d4211b` y el `grouped_norm` de `41abbfd59`: pequeños, con la sonda de carga a 135168.
3. `50182a53f` y `6788edb4f`(c): una ventana de medición cada uno si se quieren los 0.1-0.5 %.
4. `91f6a6cf3`: ventana propia cuando toque un merge de la ruta matmul; antes no.
5. `fc82583e6`: decisión explícita de mantener la FA sparse del fork y resolver el conflicto a su favor.

## Aplicado (2026-09-17 noche)

Cherry-picks en `master` (con `-x`): `481c65f09` → 5a273ee63, `43f3dda62` → 9680478a9, `5cdd3d1da` → f0101a43f,
`fa6769818` → 1aa0fd128, `093a2f86c` → 099991d8f, `37b53fd45` (solo `ggml/`) → fec3ee116, `b0dcb8192` →
9361fffb2 (aplicó limpio con la fusión a tres vías: `n_past → pos0` en los 10 sitios de
`common/speculative.cpp`, `server-context.cpp` pasa `pos_next()`). Build 396.

Verificación: test-backend-ops ARGSORT 98/98 y TOP_K 525/525; rig de 40k con MTP: prefill 81.5 s (485 t/s),
decode 38-41 t/s, misma aceptación del draft (153/202), depth_repeat idéntico; graph_diff4 seq_rm 2048,631
con los mismos 72 nodos SET_ROWS y la misma distribución que la referencia; conversación texto → imagen →
texto → texto → texto dos veces contra producción: 5/5 turnos, respuestas idénticas entre corridas y
con las del 2026-09-04; repeat_probe sobre producción. Incidente de la ventana: otro proceso (ComfyUI)
tomó la GPU y 50 GiB en medio de la primera pasada (rig sin margen, graph_diff4 cortado por el kill
switch, el launcher rechazó la recarga); se repitió íntegra con la máquina libre. La puerta del rig se
bajó de 37 a 36 GiB porque el escritorio retiene ~4 GiB más que ayer; la huella real del rig es 67 GiB
(mínimo disponible 43 GiB durante la corrida, por encima de la regla de 40).
