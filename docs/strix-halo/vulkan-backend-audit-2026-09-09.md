# Auditoría de lectura del backend Vulkan para qwen4exp en gfx1151 (2026-09-09)

Lectura del código en master `c0e776ec4`, sin cambios ni ejecuciones. Referencias:
`V` = `ggml/src/ggml-vulkan/ggml-vulkan.cpp`, `S` = `ggml/src/ggml-vulkan/vulkan-shaders/`,
`M` = `src/models/`. Los números de línea corresponden a ese commit. Las cifras medidas
provienen de los perfiles descritos en `qwen4exp-cost-breakdown-2026-09-09.md`.

Contexto del dispositivo: RADV, coopmat1 (KHR), sin coopmat2, subgrupo 64, 40 CU,
`integer_dot_product` presente, `coopmat_mmq` activo por defecto en RDNA3 (`V:7198-7200`).

## 1. MUL_MAT: árbol de decisión y tabla de despacho

Orden en `ggml_vk_mul_mat` (`V:10872-10935`): (a) troceo de M si
`nbytes(src0) > maxStorageBufferRange` (`V:10882`); (b) `p021` (`V:10906-10918`) y (c) `nc`
(`V:10919-10926`), ambos exigen `src0->type == F16` y `dst->ne[1] == 1`; (d) mat-vec si
`dst->ne[1] == 1 || (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2]*src1->ne[3] == 1)`
(`V:10928-10930`), con `mul_mat_vec_max_cols = 8` (`V:399`); (e) si no,
`ggml_vk_mul_mat_q_f16` (`V:10932`).

Parámetros del mat-vec: pipeline indexado por `num_cols-1` (`V:8597-8600`); constantes de
especialización `{BLOCK_SIZE, NUM_ROWS, NUM_COLS}` (`S/mul_mat_vec_base.glsl:81-83`).
`BLOCK_SIZE = subgroup_size = 64` porque `DMMV_WG_SIZE_LARGE` solo se activa en NVIDIA/Intel
(`V:8577-8590`). `NUM_ROWS` sale de los `wg_denoms`: F32 `{1,1,1}` (`V:5898`), F16/BF16
`{2,1,1}` (`V:5899-5900`), Q8_0 `{1*rm_stdq,...}` (`V:5906`), K-quants `{rm_kq,...}`, IQ4_NL
`{rm_iq,...}` (`V:5923`); en AMD no-GCN `rm_stdq = 1`, `rm_kq = 2` (`V:5842-5852`),
`rm_iq = 2*rm_kq = 4` (`V:5862`). Reducción `SHADER_REDUCTION_MODE_SUBGROUP`, variante
`subgroup_no_shmem` (`V:5889-5895`, `S/mul_mat_vec_base.glsl:89-127`). Grilla:
`groups_x = ne01` con `wg_denoms[0] = NUM_ROWS` (`V:10472`, `V:8917`). No hay reparto de k
entre workgroups en el mat-vec.

MMVQ (B en q8_1): `quantize_y` (`V:10328`) exige `ggml_vk_should_use_mmvq`
(`V:10208-10283`): Q6_K → false salvo Intel (`V:10216-10219`); `n > 1` → true
(`V:10222-10224`); con `n == 1` en AMD: `k < 2048` → false, Q8_0 → solo GCN, resto → true
(`V:10250-10258`). El allowlist de `ggml_vk_get_dequantize_mul_mat_vec` para `b_type == Q8_1`
(`V:8520-8539`) no contiene IQ4_NL → `nullptr` → repliegue a f32-B (`V:10346-10349`).
`rm_int_n` da 4 filas en RDNA3 cuando `num_cols >= 4` (`V:5860`).

| src0 | n=1 | n=3 | n=8 | n=2048 |
|---|---|---|---|---|
| IQ4_NL | mat-vec f32-B `mul_mat_vec_iq4_nl_f32_f32`, NUM_ROWS=4, NUM_COLS=1 | idem, NUM_COLS=3 | idem, NUM_COLS=8 | coopmat f16acc `matmul_iq4_nl_f32`, tile L `{128,128,BK=32}`, B en f32 |
| Q8_0 | mat-vec f32-B, NUM_ROWS=1 | MMVQ q8_1, NUM_ROWS=1 | MMVQ q8_1, NUM_ROWS=4 | coopmat `matmul_q8_0_f16`, B convertida a f16 |
| Q5_K | MMVQ q8_1, NUM_ROWS=1 | MMVQ q8_1, NUM_ROWS=1 | MMVQ q8_1, NUM_ROWS=4 | coopmat `matmul_q5_k_f16`, B en f16 |
| Q6_K | mat-vec f32-B, NUM_ROWS=2 | idem | idem | coopmat `matmul_q6_k_f16`, B en f16 |
| F32 | mat-vec f32-B, NUM_ROWS=1 | idem | idem | `pipeline_matmul_f32` dedicado (`V:8431-8433`), sin conversión |
| BF16 | mat-vec f32-B, NUM_ROWS=2 | idem | idem | `pipeline_matmul_bf16` (`V:8438-8440`), solo si `coopmat_bf16_support` (`V:5364-5367`) |

IQ4_NL a n = 2048: coopmat f16acc con B en f32. Cadena: `quantize_y = true` (`V:9995`) →
`ggml_vk_get_mul_mat_mat_pipeline(IQ4_NL, Q8_1)` entra en la rama MMQ (`V:8457-8465`) y
`pipeline_dequant_mul_mat_mat_q8_1[IQ4_NL]` está vacío porque en dispositivos coopmat
`create_mmq_pipelines(dense)` se llama con `false` (`V:5532-5533`; con `true` solo en la rama
sin coopmat, `V:5609`) → `nullptr` → repliegue a `(IQ4_NL, F32)` con `quantize_y = false`
(`V:9999-10003`). `dense_f16b` (`V:9979-9984`) es false porque IQ4_NL no está instanciado en
`pipeline_dequant_mul_mat_mat_f16` (`V:5395-5406`), aunque el generador produce
`matmul_<tipo>_f16` para todos los tipos (`S/vulkan-shaders-gen.cpp:585-613`). Tile:
`ggml_vk_guess_matmul_pipeline` (`V:9566-9572`) → L si `m, n > 64`,
`l_mmq_wg_denoms = {128,128,1}` (`V:4759`), warptile AMD+RADV `{256,128,128,32,...}`
(`V:4748`). `split_k` por `ggml_vk_guess_split_k` (`V:9488-9528`): solo si `k >= 2048` y
`m_tiles*n_tiles <= 20` o `<= 26` (con `shader_core_count = 40`).

Router F32: a n = 3 es mat-vec `mul_mat_vec_f32_f32_f32` NUM_ROWS=1/NUM_COLS=3 sin
conversión; a n = 2048 es `pipeline_matmul_f32`. A n = 3 lee los pesos una sola vez: la carga
de A está dentro del bucle `for j in NUM_COLS` (`S/mul_mat_vec.comp:44-111`) pero es
invariante en `j` y los bindings son `readonly` (`S/mul_mat_vec_iface.glsl:8-17`); la
aritmética lo confirma (5.24 MB / 54.8 µs = 96 GB/s; tres lecturas serían 287 GB/s, por
encima del pico).

## 2. MUL_MAT_ID

Bifurcación: `ggml_vk_mul_mat_id` (`V:11551-11562`) usa `ggml_vk_use_mul_mat_vec_id`
(`V:11544-11549`) → mat-vec-id si `ids->ne[1] <= 8` (tokens).

n = 1 y n = 3 → `ggml_vk_mul_mat_vec_id_q_f16` (`V:11308-11542`):

- Pipeline `mul_mat_vec_id_iq4_nl_f32` (`V:6009`), `wg_denoms = {rm_iq=4,1,1}`, `NUM_COLS = 1`
  fijo (solo se pasan dos constantes de especialización: `V:5984`, `V:6009`; `NUM_COLS`
  conserva su valor por defecto 1 en `S/mul_mat_vec_base.glsl:83`).
- MMVQ id: `quantize_y` (`V:11345`) → `ggml_vk_get_dequantize_mul_mat_vec_id(IQ4_NL, Q8_1)`
  devuelve `nullptr` (allowlist `V:8692-8711` sin IQ4_NL) → f32-B.
- Despacho por token: `for (uint32_t expert_i1 = 0; expert_i1 < nei1; ++expert_i1)`
  (`V:11517`) emite una `vkCmdDispatch` por token con grilla `{groups_x=ne01, nei0, groups_z}`
  (`V:11536`, `V:11485`); el eje y recorre los expertos de ese token y el offset de pesos es
  `expert_id * batch_stride_a/QUANT_K` (`S/mul_mat_vec_base.glsl:66-72`).
- Consecuencia: n = 3 → 3 dispatches por nodo MoE, y los pesos de un experto elegido por
  varios tokens se releen una vez por token. `count_experts` no se usa en esta ruta.

n = 2048 → `ggml_vk_mul_mat_id_q_f16` (`V:10936-11306`):

- `mmid_f16b` (`V:11022-11026`) exige `pipeline_dequant_mul_mat_mat_id_f16b[tipo]`; la lista
  solo cubre Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K...Q6_K (`V:5519-5530`) → false para IQ4_NL →
  `quantize_y = true` (`V:11035`) → `ggml_vk_get_mul_mat_mat_id_pipeline(IQ4_NL, Q8_1)`
  (`V:8626-8634`) devuelve `pipeline_dequant_mul_mat_mat_id_q8_1[IQ4_NL].f32acc`, creado
  porque `coopmat_mmq` está activo (`V:5117`, `V:5532-5533`). Los expertos IQ4_NL en prefill
  van por el shader entero q8_1 `matmul_id_subgroup_iq4_nl_q8_1` (`S/mul_mmq.comp`), escalar
  con `v_dot4`, no coopmat. El commit `b9c196c1c` midió este camino +3.7 % en pp2048 sobre el
  coopmat id.
- Tile: `ggml_vk_guess_matmul_id_pipeline(ctx, mmp, ne01, nei1, ...)` (`V:11071`, definición
  `V:9630-9661`) usa `nei1` = tokens totales, no filas por experto → con 2048 tokens siempre
  tile L, `l_warptile_mmqid_int = {128,128,128,32,...}` (`V:4733`),
  `l_mmq_wg_denoms = {128,128,1}` (`V:4759`), 128 hilos. `aligned` es siempre false con
  `quantize_y` (`V:11079`) y CREATE_MMQ no crea variantes alineadas (`V:5070-5079`).
- Grilla: una sola dispatch `{m, nei1, n_as}` (`V:9681`) → `{ceil(M/128), ceil(n_tokens/128),
  n_as}`; el eje z recorre todos los expertos. Descarte temprano
  `if (ic*BN >= data_expert_count[expert_idx]) return;` (`S/mul_mmq.comp:114-118`).
- Reutilización de pesos: `pos_a_ib = expert_idx*batch_stride_a/BK + ir*BM*stride_a/BK`
  (`S/mul_mmq.comp:201-206`) → los pesos de un experto se leen una vez por (M-tile, bloque de
  columnas), no por token.

Agrupación de ids (`count_experts` / hoisting):

- `count_experts` se despacha antes de cada MUL_MAT_ID (`V:11206-11223`) con grilla
  `{hoist_row_ids ? 1 : n_as, 1, 1}` (`V:11223`), seguido de barrera global (`V:11275`). Se
  repite en gate_up y down aunque los `ids` sean idénticos, sobre `prealloc_split_k`.
- `hoist_row_ids` (`V:10969-10972`) exige `n_as <= 256`. El shader escribe conteos, offsets,
  total y row-ids empaquetados `(i01<<16)|i00` (`S/count_experts.comp:41-46`); el límite de
  256 viene de sus arrays compartidos `vals/offsets/cursors[BLOCK_SIZE=256]`
  (`S/count_experts.comp:26-35`). Con 512 expertos es false.
- Con hoisting el matmul solo copia `BN` ids ya agrupados (`S/mul_mm_id_funcs.glsl:73-86`).
  Sin hoisting cada workgroup vivo ejecuta `load_row_ids` (`S/mul_mm_id_funcs.glsl:7-70`), un
  barrido de hasta `nei0*nei1` ids en trozos de `BLOCK_SIZE` con dos `barrier()` por trozo,
  antes de calcular. Los shaders `matmul_id_subgroup_*` activan esa variante vía
  `MUL_MAT_ID_USE_SUBGROUPS` (`S/vulkan-shaders-gen.cpp:462-465`).

## 3. Fusiones, reordenado y submit

Patrones reconocidos, en el orden de evaluación de `ggml_backend_vk_graph_compute`
(`V:19040-19218`):

| Secuencia exacta | Línea | Condición clave |
|---|---|---|
| `ADD × N` | `V:19042` | `ggml_vk_fuse_multi_add` (`V:18803-18845`): `device->multi_add`, todo F32, mismas shapes, alineado |
| `MUL_MAT, ADD, ADD` / `MUL_MAT, ADD` | `V:19047` / `V:19053` | `mm_add_ok` (`V:18239-18260`): `ggml_nrows(mul) == 1` (solo n = 1), misma shape/stride que el bias |
| `MUL_MAT_ID, ADD_ID, MUL` / `..., ADD_ID` / `..., MUL` | `V:19058` / `V:19061` / `V:19066` | `V:18305-18348`: `ggml_vk_use_mul_mat_vec_id` (≤ 8 tokens), ids compartidos, escala `[1, n_expert_used, 1, 1]` |
| `RMS_NORM, MUL, ROPE, VIEW, SET_ROWS` | `V:19071` | `can_fuse_rms_norm_mul_rope` (`V:18770`) + `can_fuse_rope_set_rows` (`V:18619`) |
| `RMS_NORM, MUL, ROPE` | `V:19084` | contiguo, modo NEOX/NORMAL, `mul->ne[0] <= 1024` (shmem) |
| `RMS_NORM, MUL, ADD, MUL` / `RMS_NORM, MUL, ADD` | `V:19092` / `V:19097` | `V:18216-18236`: residual F32 contiguo; escala final escalar |
| `RMS_NORM, VIEW, SET_ROWS` | `V:19102` | `can_fuse_rms_norm_set_rows` (`V:18658-18685`): índices I64 contiguos y alineados |
| `RMS_NORM, MUL` | `V:19110` | `V:18178-18200`: todo F32, filas contiguas, sin broadcast si rms es operando B |
| `UNARY, MUL` | `V:19115` | `can_fuse_unary_mul` (`V:18133-18161`): unary ∈ {GELU, SIGMOID, SILU, SOFTPLUS} (`V:13802-13810`), `ggml_can_repeat` |
| `SSM_CONV, ADD, UNARY(SILU)` / `SSM_CONV, UNARY(SILU)` | `V:19124` / `V:19131` | `can_fuse_ssm_conv` (`V:18367-18419`): bias por canal, F32, alineado |
| `ROPE, VIEW, SET_ROWS` | `V:19140` | `can_fuse_rope_set_rows` (`V:18619-18656`): `ne[3] == 1`, índices I64, modos NORM/NEOX/MROPE/IMROPE |
| `MUL, SIN, SQR, MUL, ADD` (Snake) | `V:19148` | `can_fuse_snake` (`V:18687-18740`) |
| `GET_ROWS, PERMUTE, CONT, CPY, RESHAPE, ADD, TOP_K` (topk_qsa) | `V:19155` | `can_fuse_topk_qsa` (`V:18551-18617`) + aristas `V:676-683`; solo en régimen radix (`V:18610-18615`) |
| `SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE` | `V:19161` (patrón `V:640`) | `can_fuse_topk_moe(EARLY_SOFTMAX_NORM)` |
| `UNARY(SIGMOID), RESHAPE, ADD, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE` | `V:19171` (patrón `V:645`) | modo `SIGMOID_NORM_BIAS`, bias 1D |
| `UNARY(SOFTPLUS), SQRT, RESHAPE, ADD, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE` | `V:19181` (patrón `V:650`) | modo `SQRT_SOFTPLUS_NORM_BIAS` |
| `SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS` | `V:19191` (patrón `V:657`) | modo `EARLY_SOFTMAX` |
| `ARGSORT, VIEW, GET_ROWS, RESHAPE, SOFT_MAX, RESHAPE` | `V:19199` (patrón `V:660`) | modo `LATE_SOFTMAX` |
| `+ DIV, RESHAPE, SCALE` o `+ GET_ROWS, SCALE` sobre cualquier topk_moe | `V:19209-19216` | `fused_topk_moe_scale` |

Condiciones comunes de topk_moe (`V:18422-18533`): `n_expert <= 512` (`V:18514-18517`),
`soft_max` con `scale == 1`, `max_bias == 0`, sin máscara ni sinks, y
`subgroup_arithmetic && subgroup_shuffle && subgroup_require_full_support` (`V:18519-18524`).
No existe fusión `mul_mat + scale`, `mul_mat + mul`, `soft_max + X` fuera de topk_moe, ni GLU
(es op único). `hc_gated_mean` y `hc_inject` no son fusiones del backend: son ops de ggml del
fork (`ggml/include/ggml.h:588-589`), despachadas en `V:17384-17391`.

Concurrencia y reordenado: `ggml_vk_graph_optimize` (`V:19379-19559`) protege los patrones
con `keep_pattern` (`V:19467-19502`); mira `NUM_TO_CHECK = 20` nodos por delante
(`V:19513-19515`) y agrupa en un `current_set` los que no dependen de nodos pendientes
(`is_src_of`, `V:19392-19417`), con excepciones para las parejas fusionables
(`V:19546-19557`); tira hacia adelante consumidores concretos (ROPE `V:19566`, VIEW+SET_ROWS
`V:19583`, `MUL_MAT_ID+ADD_ID+MUL` `V:19620`, `MUL_MAT+ADD+ADD` `V:19636`, `SSM_CONV+...`
`V:19651`, `UNARY+MUL` `V:19665`); segunda pasada para nodos vista (`V:19688-19714`). No hay
batching de pipelines ni nodos en vuelo explícitos; la concurrencia es indirecta.

Barreras: `ggml_vk_build_graph` (`V:16889-16974`) mantiene listas
`unsynced_nodes_written`/`_read`, comprueba solape por buffer y rango en `overlaps_unsynced`
(`V:16901-16924`), e inserta barrera solo si el destino solapa lo leído/escrito o una fuente
solapa lo escrito (`V:16926-16947`); al emitirla vacía ambas listas (`V:16949-16951`). La
barrera es `ggml_vk_sync_buffers` (`V:4001-4021`): `vkCmdPipelineBarrier` global, sin buffer
memory barriers, `src = dst = ShaderRead|ShaderWrite|TransferRead|TransferWrite`. Los buffers
`prealloc_x/y/split_k` llevan flags aparte (`V:2625-2627`). En una cadena de decode casi todos
los nodos consumen el anterior, así que la mayoría recibe barrera.

Umbrales de submit (`V:18954-19004`, `V:19282-19289`): `flops_cap = 200 GFLOP`, reducido a
`2 GFLOP × shader_core_count` solo si AMD no-GCN con menos de 24 CU (`V:18965-18971`); con 40
CU se queda en 200 GFLOP; `flops_per_submit = min(flops_cap, last_total_flops/40)`
(`V:18973`), duplicado en los 3 primeros submits (`V:18998-19001`). Se emite submit si
`submitted_nodes >= device->max_nodes_per_submit` (100, `V:7178`, ajustable con
`GGML_VK_MAX_NODES_PER_SUBMIT` `V:7179-7182`), o `batch_flops >= flops_per_submit`, o último
nodo, o `almost_ready` (últimos 20 %). Flush real en `V:17415-17429` → `V:17433-17466` →
`ggml_vk_submit` (`V:3514`).

## 4. Coste fijo por dispatch

Por dispatch en host (`V:8916-8945`): un `vkUpdateDescriptorSets` (`V:8934-8935`) con un
descriptor set nuevo del pool (`V:8933`), `pushConstants` (`V:8937`), `bindPipeline`
(`V:8938`), `bindDescriptorSets` (`V:8939-8944`), `dispatch(wg0, wg1, wg2)` (`V:8945`) con
`wgN = CEIL_DIV(elements[N], pipeline->wg_denoms[N])` (`V:8917-8919`). No hay push descriptors
ni reutilización de descriptor sets. Los sets se reservan por adelantado con
`ggml_pipeline_request_descriptor_sets` (`V:3456-3463`), que crece el vector un 50 % cada vez
(`V:3475`) y crea pools de `VK_DEVICE_DESCRIPTOR_POOL_SIZE` sets (`V:3486-3489`).

No hay una barrera por nodo incondicional: solo la de la sección 3 cuando hay solape de
memoria. No existe `ggml_vk_op_needs_sync`, `same_buffer` ni «skip barrier»;
`overlaps_unsynced` es la única puerta. `ctx->s->buffer` es el command buffer del `vk_context`
actual (`V:8937-8945`); `ggml_vk_ctx_begin/end` (`V:3961-3977`) abren y cierran secuencias, y
cada secuencia es un `vk::SubmitInfo` en `ggml_vk_submit` (`V:3514`).

Sobrecoste del perf logger: con `GGML_VK_PERF_LOGGER` y sin `GGML_VK_PERF_LOGGER_CONCURRENT`,
tras cada nodo se emite `writeTimestamp(eAllCommands)` más otra `ggml_vk_sync_buffers`
(`V:19291-19298`): cada op se mide serializada, con una barrera extra y un drenaje completo.
Las cifras de perfil son cotas superiores.

## 5. Flash attention

### 5.1 Ruta para HSK = HSV = 256, K/V q8_0, GQA 12 (24 q / 2 kv)

`ggml_vk_flash_attn` (`V:11666-12266`):

- `f32acc = !fp16 || op_params[3] == GGML_PREC_F32 || k->type == BF16` (`V:11724`) → false en
  producción → acumulador f16.
- GQA: `V:11790-11797`, si `N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa` con
  `max_gqa = min(block_rows, 32)` (`V:11788`). En coopmat1 `block_rows = 16` (`V:4172`), luego
  `max_gqa = 16 >= 12` → `gqa_ratio = 12`, `N = 12`, `workgroups_y = 24/12 = 2`. A n = 2048
  no hay agrupación GQA: cada tile es Br tokens de una cabeza q.
- `get_fa_tuning_params` (`V:4213-4257`): `coopmat2 = false`, `coopmat1_fa_support = true` →
  `FA_COOPMAT1`; se degrada a `FA_SCALAR` si falla la forma 16x16x16 o la shmem
  (`V:4244-4252`), y siempre si `n_rows == 1` (`V:4254-4256`). n = 3 → tras la reagrupación
  GQA `N = 12` → coopmat1. n = 2048 → coopmat1.
- Parámetros coopmat1 (`V:4155-4183`): `Br = 16`, `Bc = 16 × 4 = 64`, `row_split = 4`
  subgrupos, `workgroup_size = 4 × subgroup_size = 256`, `d_split = min(min(sg, 8), D_lsb/4)`,
  `shmem_staging = 0`.

### 5.2 q8_0: dequantización en shader o preconversión a f16

Tres modos, elegidos en `V:11737-11774`:

1. Escaso compacto (`use_compact`), decode: `compact_candidate` (`V:11758-11762`) exige
   `neq1 <= FA_SPARSE_COMPACT_MAX_ROWS = 64` (`V:1523`), `nem2 == 1`, HSK/HSV múltiplos de 8
   y `k/v ∈ {F16, Q8_0}`. Las filas seleccionadas se reúnen y dequantizan a f16 una vez por
   `flash_attn_sparse_gather.comp` (`S/flash_attn_sparse_gather.comp:55-67`) y la atención lee
   f16 (`k_type_eff = v_type_eff = F16`, `V:11772-11773`).
2. Escaso indexado (`use_sparse` sin compact), prefill: el shader lee la caché con su tipo y
   dequantiza en vuelo (`USE_DECODE_K/V` en `S/flash_attn_base.glsl:124-125`, uso en
   `S/flash_attn_cm1.comp:288-293` vía `dequantize4`).
3. Denso con scratch f16 (`use_dequant_kv`), `V:11764-11774`: solo si no hay candidatura
   escasa, `neq1 >= 64`, caché densa; dequantiza toda la caché con
   `pipeline_dequant_transpose` (`V:12070-12077`). El comentario del fork (`V:11742-11746`)
   documenta que a 40k este scratch costaba lo que ahorraba (107 s vs 95 s).

### 5.3 split_k y tiles

- `Tr = CEIL_DIV(N, Br)` (`V:11910`).
- Con `use_compact` no hay split_k (`V:11927-11928`, `GGML_ASSERT(split_k == 1)` en
  `V:12148`). Con n = 3 la atención compacta despacha 3 × 2 = 6 workgroups.
- Con `gqa_ratio > 1 && workgroups_x <= Br`: `split_k = shader_core_count*2 /
  (wgx*wgy*wgz)` (`V:11929-11930`). Para n = 3 en modo índice: `80/6 = 13`, recalculado contra
  `alignment = Bc = 64` (`V:11938-11941`).
- El buffer de split_k se dimensiona en `V:11948` y se reduce con
  `pipeline_flash_attn_split_k_reduce` (`V:12240-12243`).

### 5.4 Estructura del FA escaso del fork

Umbral (`V:11747-11750`): `n_kv_max > 0` (op_param 4), hay máscara,
`n_kv_max * 8 <= KV`, K/V no BF16, coopmat1 y ballot disponibles. El comentario documenta la
medición: por debajo de 8 × n_kv_max la unión de celdas de los tiles cubre casi toda la caché
(52 % a 8k, 33 % a 16k con la selección de 2051 celdas).

Estructuras en `prealloc_y` (`V:11913-11921`): `counts[n_lists]`, luego
`lists[n_lists][list_stride]`, y en modo troceado `chunk_counts[n_lists][n_chunks]`.
`list_tiles = ceil(nem1/Br)`, `list_stride = ROUNDUP_POW2(min(KV, n_kv_max*Br), Bc)`
(a 40k con Br = 16 es la caché entera), `chunked = (n_lists < 32)`.

Dispatches por nodo FA:

- Decode (n = 3, GQA, compact): `flash_attn_sparse_idx` pasada 1 + barrera (`V:12112-12114`),
  pasada 2 + barrera (`V:12116-12117`; las dos pasadas fijan el orden de la lista,
  `V:12097-12103`, commit `38462e9a2`); por cada grupo de tiles y batch (`V:12160-12161`) un
  `flash_attn_sparse_gather` + barrera (`V:12175-12180`) y una dispatch de atención + barrera
  (`V:12182-12192`). Típicamente 4 dispatches y 4 barreras por nodo.
- Prefill (n = 2048, modo índice): `chunked` es false (128 tiles) → una sola dispatch de
  `flash_attn_sparse_idx` + barrera (`V:12116-12117`), y una dispatch de atención
  (`V:12256-12258`) o atención + `split_k_reduce` (`V:12233-12243`). 2-3 dispatches por nodo.

`GGML_VK_SPARSE_FA_LOG` (`V:11802-11815`) imprime una vez por forma distinta los flags
`sparse_candidate / compact_candidate / use_sparse / use_compact` y la ruta elegida.

Efecto colateral: el modo escaso invalida la caché de activaciones convertidas porque las
listas viven en `prealloc_y` (`V:12261-12265`, `prealloc_y_last_pipeline_used = nullptr`):
cada FA escaso fuerza a los matmuls posteriores a reconvertir B.

## 6. GET_ROWS, CPY, SET_ROWS, TOP_K, ARGSORT

GET_ROWS: `pipeline_get_rows[type]` y `pipeline_get_rows_f32[type]` (`V:989-990`), creados
para F32/F16/BF16 y los cuantizados (`V:6082-6110`); no cuantizados con `S/get_rows.comp`
(512 hilos), cuantizados con `S/get_rows_quant.comp` (1024) (`V:6085-6091`). Despacho
`elements = {ne00, ne10, ne11*ne12}` con clamp (`V:13292-13296`). El gather de la tabla PLE
de qwen4exp corre en la CPU (tensor lazy en host; `M/qwen4exp.cpp` `build_inp_ple`) con
lectura anticipada por `madvise`.

TOP_K: `ggml_vk_topk` (`V:15165-15296`) con dos rutas: torneo / n-ary search
(`pipeline_topk_f32[idx]`, `S/topk_nary_search.comp`) cuando
`k_min_pipeline = max(log2(k)+1, subgroup_size_log2) < num_topk_pipelines (= 11)`
(`V:15170-15171`), iterativa con doble buffer en `prealloc_x` y barrera entre pasadas
(`V:15288-15290`); radix select (`pipeline_topk_radix_f32`, `S/topk_radix_select.comp`) cuando
k es grande (`V:15172-15192`): 8 bits por pasada (4 pasadas), un workgroup por fila
(`S/topk_radix_select.comp:94`), histograma en shmem y emisión en orden fijo con
`scan_exclusive` en lugar de `atomicAdd` (`S/topk_radix_select.comp:139-143`, commit
`96f387dad`). ARGSORT es op aparte (`ggml_vk_argsort`, `V:15079`), bitónico, variante small
de un workgroup y large multipasada (`V:15093-15098`).

CPY: `ggml_vk_get_cpy_pipeline` (`V:9694`) elige contiguo / transpuesto 0↔1 / 0↔2 /
cuantización. SET_ROWS: `elements` en `V:13489-13506`.

`ggml_vk_can_fuse_topk_qsa` fusiona la expansión `get_rows` + suma de máscara + `top_k` en
una sola dispatch. Patrón `GET_ROWS, PERMUTE, CONT, CPY, RESHAPE, ADD, TOP_K` (`V:672-675`)
con aristas verificadas en `V:676-683`. Comprobaciones (`V:18551-18617`): intermedios de un
solo uso; la máscara se sigue hacia atrás por reshape/cpy hasta el f16 materializado
(`V:18578-18585`); tipos `scores` F32, `cell_blk` I32, `top_k` I32; contigüidad; geometría
exacta `[n_tps, n_blocks, n_stream] × [n_kv, n_stream]` (`V:18596-18608`). Solo en régimen
radix (`V:18610-18615`). Shader `pipeline_topk_radix_qsa` = `S/topk_radix_select.comp` con
`QSA=1` (`S/topk_radix_select.comp:11`): `gather(row, i) = score[cell_blk[i,s], t, s] +
mask[i,t,s]` (`S/topk_radix_select.comp:69-76`), materializado una vez en scratch y reutilizado
(`:78-92`). Host: `ggml_vk_topk_qsa` (`V:15298-15345`), una dispatch con 5 bindings, grilla
`{wg_denoms[0], min(nrows, límite), 1}`; se despacha desde el `case GGML_OP_GET_ROWS` de
`ggml_vk_build_graph` cuando `ctx->fused_topk_qsa` (`V:17010-17015`).

## 7. Hallazgos ordenados por ganancia esperada

1. [prefill, muy alta] `hoist_row_ids` desactivado con 512 expertos (`V:10969-10972`):
   `count_experts` con 512 workgroups reescaneando los `nei0*nei1` ids (`V:11223`,
   `S/count_experts.comp:113-134`) y cada workgroup vivo del matmul ejecutando `load_row_ids`
   con 2 `barrier()` cada 128 entradas antes de calcular. El límite de 256 viene de los arrays
   compartidos de `count_experts.comp`, no es intrínseco.
2. [prefill y decode, alta] IQ4_NL fuera de las listas f16-B (`V:5395-5406`, `V:5517-5530`):
   `dense_f16b` es false para IQ4_NL y la B viaja en f32 (el fork mide +4.5 % denso / +4.7 %
   MoE con f16-B en otros tipos, `V:4356-4362`). Y MMVQ IQ4_NL compilado pero inalcanzable:
   el commit `b9c196c1c` creó `pipeline_dequant_mul_mat_vec_q8_1_f32[w][IQ4_NL][i]` (`V:5963`)
   y `_id` (`V:6020`) sin añadir IQ4_NL a los allowlists de `V:8520-8539` y `V:8692-8711`.
   Los pipelines MMQ densos q8_1 no se crean en dispositivos coopmat (`V:5532-5533` vs
   `V:5609`), decisión medida en ese commit (437 vs 459 t/s).
3. [decode, alta] El mat-vec-id despacha por token (`V:11517-11536`) con `NUM_COLS = 1`;
   existe la maquinaria (`count_experts`, ids agrupados) para agrupar por experto y no se usa
   en esta ruta.
4. [decode, media-alta] El mat-vec F32 usa `NUM_ROWS = 1` (`V:5898`), único tipo con `{1,1,1}`.
   El router (F32, 5.24 MB por capa) lo paga 48 veces por paso.
5. [decode, media] Los proyectores de la hyper-connection son mat-vecs degenerados: con
   `NUM_ROWS = 4` para IQ4_NL el de `m = 4` lanza un solo workgroup (`V:11485`, `V:10472`).
6. [prefill, media] Elección del tile de `mul_mat_id` por tokens totales (`V:11071`) y no por
   filas por experto (`nei0*nei1/n_as`).
7. [prefill, media] `count_experts` repetido y serializado por nodo (`V:11206-11223`,
   `V:11275`) aunque gate_up y down compartan `ids`.
8. [decode, baja-media] Geometría de los kernels del fork: `hc_gated_mean` con
   `elements = {ne0, ne1, 1}` y `wg_denoms = {256,1,1}` (`V:13525-13528`, `V:6657`) → 30
   workgroups con 3 tokens, `exp()` por elemento y stream (`S/hc_gated_mean.comp:33-38`);
   `hc_inject` `{ne0, ne1, ne2}` (`V:13529-13532`) con un `exp()` por `(h,t)`; `topk_moe`
   `CEIL_DIV(n_rows, 4)` workgroups (`V:15039-15042`); `RMS_NORM` un workgroup por fila
   (`V:13263`); `multi_add` 512×512×Z sobre `ggml_nelements(dst)` (`V:13711-13718`).
9. Palancas de entorno existentes: `GGML_VK_DENSE_F16B` (`V:4354`), `GGML_VK_MMID_F16B`
   (`V:4369`), `GGML_VK_DENSE_WAVE32` (`V:5281`), `GGML_VK_MMID_WG256` / `GGML_VK_MMID_WAVE32`
   (`V:5432`, `V:5441`, solo tiles coopmat de `mul_mat_id`), `GGML_VK_DISABLE_COOPMAT_MMQ` /
   `GGML_VK_FORCE_COOPMAT_MMQ` (`V:7199-7200`), `GGML_VK_DISABLE_SPARSE_FA` / `_COMPACT`
   (`V:11740`, `V:11755`), `GGML_VK_SPARSE_FA_LOG` (`V:11803`), `GGML_VK_MAX_NODES_PER_SUBMIT`
   (`V:7179`), `GGML_VK_SYNC_LOGGER`, `GGML_VK_PERF_LOGGER[_CONCURRENT|_FREQUENCY]`
   (`V:8169-8181`).

## 8. Explicación de las mediciones

### 8.A `MUL_MAT_ID iq4_nl m=640 n=10 k=2560 n_expert=512 batch=2048`: 13.2 ms (5.1 TFLOPS)

Ruta: `ids->ne[1] = 2048 > 8` → `ggml_vk_mul_mat_id_q_f16` (`V:11554-11557`);
`mmid_f16b = false` → `quantize_y = true` (`V:11035`) → MMQ entero q8_1
`matmul_id_subgroup_iq4_nl_q8_1` (`V:5117`), shader `S/mul_mmq.comp`, tile
`l_warptile_mmqid_int = {128, 128, 128, 32, ...}` (`V:4733`), `l_mmq_wg_denoms = {128,128,1}`
(`V:4759`), `required_subgroup_size = 64`. Escalar con `v_dot4`, no coopmat.

Grilla: `{ceil(640/128) = 5, ceil(2048/128) = 16, 512}` = 40960 workgroups lanzados. Filas por
experto = `2048×10/512 = 40` frente a `BN = 128`; el descarte temprano
(`S/mul_mmq.comp:114-118`) mata todos los `ic >= 1` → 2560 workgroups vivos.

Limitantes, por peso:

1. Utilización de columna del tile 40/128 = 31 %: el bucle recorre todas las `BN` columnas
   (`S/mul_mmq.comp:252-278`) y el recorte por `_ne1` ocurre solo al almacenar
   (`S/mul_mmq.comp:296-301`): 3.2× más MACs de las útiles. Estimación: 22 TFLOPS / 3.2 =
   6.9 TFLOPS efectivos, cerca de los 5.1 medidos.
2. Cargas de B con índices basura: `S/mul_mmq.comp:234-241` lee `row_ids[buf_ib]` sin comprobar
   `buf_ib < _ne1`, y `block_b_to_shmem` solo valida el rango de k
   (`S/mul_mmq_funcs.glsl:492-516`). Las 88 columnas sobrantes leen `row_ids[]` residual →
   gathers dispersos. La ruta coopmat `S/mul_mm_funcs.glsl:673-685` sí comprueba
   `row_i < _ne1` y pone ceros.
3. `BK_STEP = 1` en MUL_MAT_ID (`S/mul_mmq.comp:88-93`) frente a 4 en el denso: dos `barrier()`
   cada 32 de k en vez de cada 128 → 160 barreras por tile con k = 2560.
4. Prólogo de row-ids sin hoisting (sección 7.1): cada workgroup vivo barre 20480 ids.
5. Presión de registros: `WMITER=2, TM=4, WNITER=4, TN=4` → 128 acumuladores por hilo.

Tráfico DRAM: A = 512 × 640×2560×4.5/8 = 472 MB leídos una vez → 1.84 ms a 256 GB/s. Los
13.2 ms son 7.2× ese suelo: el limitante es aritmético y L2, no DRAM.

`m=2560 n=10 k=640` (14.4 ms): misma ruta, grilla `{20, 16, 512}` = 163840 lanzados, 10240
vivos; con k = 640 el prólogo y las barreras se amortizan sobre 20 pasos de k.

Contraste denso: `MUL_MAT iq4_nl m=12288 n=2048 k=2560` va por coopmat `matmul_iq4_nl_f32`,
tile `{128,128}` ocupado, B contigua, sin prólogo de ids → 22 TFLOPS.

### 8.B Densos de dimensión pequeña (mezclador hyper-connection, `M/qwen4exp.cpp:356-397`)

B1 `m=320 n=2048 k=10240`, 1.75-1.8 ms (7.5 TFLOPS), `build_lora_mm(w_down, xn)`:
`ggml_vk_guess_matmul_pipeline(320, 2048)` (`V:9566-9572`) → tile grande, warptile AMD+RADV
`{256, 128, 128, 32, ...}` (`V:4748`), 256 hilos. Grilla `{3, 16, 1}` = 48 workgroups sobre
40 CU. `split_k = 1` (`V:9488-9528` exige `m_tiles*n_tiles <= 20` o `<= 26`). Limitante:
tráfico de B, cada workgroup lee 128 columnas × 10240 × 4 B = 5.24 MB, ×48 = 252 MB (los 3
M-tiles releen la B completa); estimación 290 MB / 1.775 ms = 163 GB/s. Agravante: B en f32
por la ausencia de IQ4_NL en f16-B.

B2 `m=10240 n=2048 k=320`, 1.4-1.9 ms (7 TFLOPS), `build_lora_mm(w_up, lo)`: mismo tile,
grilla `{80, 16, 1}` = 1280 workgroups, `split_k = 1` (k < 2048, `V:9501`). Con k = 320 hay
10 pasos de `BK = 32` por tile: prólogo, inicialización de acumuladores y escritura del tile
128×128 f32 (64 KB) dominan; salida escrita 84 MB.

B3 `m=4 n=2048 k=10240`, 1.5 ms (0.1 TFLOPS), `build_lora_mm(w_inject, xn)` con `w_inject`
`[10240, 4]`: `dst->ne[1] = 2048` → no entra en mat-vec (`V:10928`); `m <= 32` → tile pequeño
(`V:9566-9568`), `s_mmq_wg_denoms = {32,32,1}` (`V:4761`), `s_warptile_mmq` (`V:4713`) →
64 hilos = 1 wave64. Grilla `{1, 64, 1}` = 64 workgroups de 1 wave para 160 SIMD32: ocupación
nula y 28 de 32 filas del tile de A son relleno. Los kernels transpuestos
`mul_mat_vec_p021_f16_f32` (`V:10906-10918`) y `mul_mat_vec_nc_f16_f32` (`V:10919-10926`)
exigen `src0->type == F16` y `dst->ne[1] == 1`; no hay ruta que intercambie A y B.

### 8.C `MUL_MAT_VEC iq4_nl m=2560 n=4 k=5120 batch=2048`: 130 ms

Es `build_lora_mm(layer.nextn.eh_proj, ggml_concat(e_norm, h_norm, 0))` de la capa MTP
(`M/qwen4exp.cpp:501-502`): `e_norm` y `h_norm` son `[n_embd, hc, n_tokens]` (`M:496`), la
concatenación da `[5120, 4, 2048]`, `eh_proj` es 2D `[5120, 2560]`. La etiqueta `_VEC` es del
logger (`V:2506-2507`, sufijo cuando `node->ne[1] <= mul_mat_vec_max_cols`), no la ruta.

Ruta real: en `V:10928-10930` `dst->ne[1] = 4 != 1` y `src1->ne[2]*src1->ne[3] = 2048 != 1` →
rechaza el mat-vec (`ggml_vk_mul_mat_vec_q_f16` lo prohíbe además con
`GGML_ASSERT(ne11 == 1 || ne12 * ne13 == 1)`, `V:10321`) → `ggml_vk_mul_mat_q_f16`.
`ggml_vk_guess_matmul_pipeline(2560, 4)`: `n <= 32` → tile pequeño 32×32; grilla
`{80, 1, 2048}` = 163840 workgroups (`V:9597`). `r2 = ne12/ne02 = 2048` (`V:9953`); el shader
usa el mismo A para los 2048 batches (`S/mul_mm.comp:157-166`), releído en cada grupo z.

Tráfico (estimación): A 7.37 MB × 2048 = 15.1 GB; B 167.8 MB × 80 M-tiles = 13.4 GB; D 84 MB;
total 28.6 GB / 0.130 s = 220 GB/s. El nodo va a pico de ancho de banda pero el 99.4 % es
relectura redundante. Solo 4 de las 32 columnas del tile son reales.

### 8.D `MUL_MAT_VEC f32 m=512 n=3 k=2560`: 54.8 µs (96 GB/s)

Router MoE `ffn_gate_inp` `[2560, 512]` F32, 5.24 MB, 48 veces por paso. Ruta:
`ggml_vk_mul_mat_vec_q_f16`; `ggml_vk_get_dequantize_mul_mat_vec(F32, Q8_1)` devuelve
`nullptr` → pipeline `mul_mat_vec_f32_f32_f32` (`V:5898`), variante `subgroup_no_shmem`,
`wg_denoms = {1,1,1}`, `{BLOCK_SIZE = 64, NUM_ROWS = 1, NUM_COLS = 3}`. Grilla:
`groups_x = ne01 = 512` (`V:10472`) → 512 workgroups de 64 hilos, 3.2 waves por SIMD. Bucle:
`K_PER_ITER = 4` (`S/mul_mat_vec.comp:10-14`) frente a 8 en los cuantizados; ruta
`iter_aligned_nonquant` (`S/mul_mat_vec.comp:113-137`); 10 iteraciones por hilo.

Por qué es lento: `NUM_ROWS = 1` (una fila de A por workgroup, pocas cargas en vuelo);
`K_PER_ITER = 4`; la B se recarga por workgroup (30 KB × 512 desde L2); reducción final de 3
`subgroupAdd` amortizada sobre 10 iteraciones (`S/mul_mat_vec_base.glsl:89-127`). Contraste:
bf16 `m=512`, 2.6 MB, 15.8 µs = 165 GB/s con `NUM_ROWS = 2`; q5_K `m=10240`, 18 MB, 84 µs =
214 GB/s con 5120 workgroups; q6_K `m=248320`, 521 MB, 2.3 ms = 227 GB/s.

### 8.E Mat-vecs diminutos en decode (mezclador hyper-connection)

Ruta común: `dst->ne[1] = 3`, `src1->ne[2]*ne[3] = 1` → `ggml_vk_mul_mat_vec_q_f16`; IQ4_NL
fuera del allowlist q8_1 → `mul_mat_vec_iq4_nl_f32_f32` (`V:5923`), `wg_denoms = {4,1,1}`,
`{BLOCK_SIZE = 64, NUM_ROWS = 4, NUM_COLS = 3}`, `K_PER_ITER = 8`. Sin split en k.

| Caso | `groups_x = ne01` | Workgroups = `ceil(ne01/4)` | Iteraciones k = `ceil(k/(8×64))` | Medido |
|---|---|---|---|---|
| `m=4 k=10240` (w_inject) | 4 | 1 | 20 | 14.3 µs |
| `m=320 k=10240` (w_down) | 320 | 80 | 20 | 17.4 µs (106 GB/s) |
| `m=10240 k=320` (w_up) | 10240 | 2560 | 1, con 40 de 64 carriles activos | 16.0 µs |

`m = 4`: un único workgroup de 64 hilos en toda la GPU (`S/mul_mat_vec.comp:244-251`), 23 KB
de pesos, latencia de lanzamiento + 20 iteraciones seriadas. `m = 320`: 80 waves sobre 160
SIMD. `m = 10240, k = 320`: `num_iters = 0` corregido a 1 (`S/mul_mat_vec.comp:160-163`),
solo `tid < 40` con datos; 2560 workgroups que hacen una iteración y pagan la reducción
completa de 12 `subgroupAdd` + 12 escrituras.

Suelo por dispatch identificable: host (`V:8916-8945`) un `vkUpdateDescriptorSets` +
`pushConstants` + `bindPipeline` + `bindDescriptorSets` + `vkCmdDispatch`; GPU entre nodos
dependientes (`V:16926-16951` → `V:4001-4021`) un `vkCmdPipelineBarrier` global con máscaras
completas; con el perf logger, una barrera y un `writeTimestamp` más por nodo
(`V:19291-19298`). Los tres proyectores cuestan ~96 × 47.7 µs ≈ 4.6 ms por paso de decode
para ~3.7 MB de pesos.

## 9. GDN (`ggml_vk_gated_delta_net`, `V:14038-14092`)

Cadena verificada: `M/qwen4exp.cpp:606` → `build_layer_attn_linear` (`M/qwen4exp.cpp:1272`)
→ `build_recurrent_attn` (`M/qwen4exp.cpp:1381`, definición `M/delta-net-base.cpp:527`) →
`ggml_gated_delta_net(ctx0, q, k, v, g, b, s, K)` (`M/delta-net-base.cpp:567`) cuando
`cparams.n_rs_seq > 0` (`M/delta-net-base.cpp:545`). El grafo etiqueta `LLM_FUSED_OP_GDN_CH`
si `n_seq_tokens > 1` y `GDN_AR` si no (`M/delta-net-base.cpp:568-573`).

Geometría: `pipeline_gated_delta_net[si][kda]` (`V:12838-12852`), `si` por
`S_v ∈ {16,32,64,128}`, `kda = 1` si `g` tiene `ne[0] == S_v`. Constantes (`V:6642`)
`{S_V, KDA, SUBGROUP_SIZE, LANES_PER_COLUMN}`; el workgroup es exactamente `subgroup_size`
hilos = 1 wave64 (`S/gated_delta_net.comp:16-23`). Con `S_V = 128` y `subgroup_clustered`:
`lanes_per_column = 8` (`V:6595-6596`) → `COLS_PER_WG = 8`, `ROWS_PER_LANE = 16`; reducción
por `subgroupClusteredAdd(8)` (`S/gated_delta_net.comp:66-92`). `wg_denoms = {1, 1,
cols_per_wg}` (`V:6635`) y `elements = {H, n_seqs, S_v}` (`V:14092`) → grilla `H × n_seqs ×
16` workgroups de 1 wave.

No hay kernel de prefill troceado: `pipeline_gated_delta_net[4][2]` (`V:1152`) solo varía
por `S_V` y `KDA` (`V:6580-6644`). La misma grilla se usa a n = 3 y a n = 2048; el recorrido
de tokens es el bucle secuencial `for (uint t = 0; t < n_tokens; t++)` dentro de cada
workgroup (`S/gated_delta_net.comp:118-183`), con el estado `s_shard[ROWS_PER_LANE]` en
registros (`:111-115`, `:167`). Medido: 2.1 ms por nodo a n = 2048 (~1.03 µs por token) y
22 µs a n = 3 (~7.3 µs por token, dominado por lanzamiento y barrera). Ineficiencias:
sin paralelismo temporal en prefill (no se aplica la formulación por bloques); ocupación fija
`16·H` waves; un wave por workgroup; dos reducciones de subgrupo por token en la ruta crítica
(`:158`, `:170`) más `exp()` por elemento con `KDA == 1` (`:145-151`); escritura del estado
por token cuando `K > 1` (`:175-182`) en lugar de una sola al final (`:186-190`).

## 10. Compuertas de halo-box/strix-llama.cpp en el fork (2026-09-15)

El fork comunitario halo-box/strix-llama.cpp (Vulkan, gfx1151, RADV) publica sus kernels con la
medición en el mensaje de cada commit. Estado en nuestro `master` (los `getenv` de
`ggml-vulkan.cpp`), lo que midió halo-box y si aplica a qwen4exp. Las vías citadas son las de
`qwen4exp-cost-breakdown-2026-09-09.md`, sección 3.

| Compuerta (halo-box) | En el fork | Default aquí | Medición de halo-box (gfx1151) | Aplicación a qwen4exp |
|---|---|---|---|---|
| `GGML_VK_DENSE_F16B` (B en f16 para MUL_MAT cuantizado coopmat1; `auto` = solo ne10 == 5120) | sí, 8dc44961d (2026-08-25) | on (=1); `auto` nunca dispararía aquí (n_embd 2560) | +5..+7 % pp2048 en densos de 5120 (q6_K, q8_0), −1.2 % en un 7B, −0.5 % en un MoE; fork: +4.5 % denso, +4.7 % MoE | Lista sin IQ4_NL (Q2_K..Q8_0): actúa en wqkv q5_K, o_proj q8_0 y LM head q6_K, no en los densos IQ4_NL (vía 11) |
| `GGML_VK_MMID_F16B` (lo mismo para MUL_MAT_ID) | sí | on | fork: +6.9 % pp2048 en un MoE | Lista sin IQ4_NL: sin efecto en nuestros expertos (vía 11) |
| `GGML_VK_DENSE_WAVE32` (retile a wave32 de los coopmat cuantizados densos; `=2` también f16) | sí, con el shadow de los tiles mmid de 3865cc79a | off ("hasta medirla aquí") | GEMM densa q6_K +5.2..+10.8 %, q8_0 +5.4..+8.4 %, q4_K +0.7..+9.1 %, q4_0 −1.5..+1.8 %, f16 −6.7..+6.4 %; Qwen3-32B Q6_K pp2048 +7.2 % (ub256) / +3.9 % (ub2048), Qwen3.8-27B +5.3 / +4.8 %; PPL idéntica | Vía 6 (pre-check por variable) |
| `GGML_VK_MMID_WAVE32`, `GGML_VK_MMID_WG256` | sí | off | on por defecto allí desde 2026-08-30 (6192a050d); WG256 forma parte de 1223 → 1733 t/s pp2048 en Qwen3.6-35B (cuatro compuertas juntas) | Vía 6 |
| `GGML_VK_CONCAT_TRANSPOSE` (concat traspuesta por tiles 32×32 del estado conv de delta-net) | sí, 7f2d40ef5 (2026-08-24) | on | CONCAT 11877 → 957 µs/op (stride de 40960 B: un solo canal de memoria, 13.7 GB/s); Qwen3.8-27B pp2048 +7.2 % (ub2048); fork: +45 % pp2048 en un MoE delta-net | Activa; CONCAT 1.1 s en 34677 dispatches por corrida de 40k (32 µs) |
| `GGML_VK_FUSE_UNARY_MUL` | equivalente upstream #27220 (936849ba9) | on | 750 → 443 µs/op | Activa |
| `GGML_VK_MMID_SMALLN`, `M128`, `BM64` (tile por filas esperadas por experto sobre un prepass de listas de filas), `TILE16` | no | — | SMALLN + listas: pp512 914.7 → 1063.6 t/s (+16.3 %) en Qwen3.6-35B-A3B; MUL_MAT_ID q5_K 2.33 → 4.43 TFLOPS, q6_K 1.99 → 3.57; TILE16 negativo en gfx1151 | Vía 1: implementación de referencia; mismo diagnóstico que 8.A |
| `GGML_VK_MMID_SCALE_EPILOGUE` (escala por (experto, token) al escribir el MUL_MAT_ID de prefill; no coopmat2) | no | — | evita 134 MB de escritura y relectura por capa en Qwen3.6-35B | Vía 8 |
| Troceado por columnas del mat-vec por lotes (`GGML_VK_MMV_NO_SPLIT=1` lo desactiva; d22fa655a lo limita a q8_0 y q6_K) | no | — | Qwen3.8-Flash-Next en Vulkan, MTP n-max 4: decode 9.6 → 14.8-15.2 t/s; Qwen3.8-27B q4_K con `-b 8`: 28.8 troceado frente a 47.4 sin trocear (pierde en q4_K) | Vía 3: dato de forma; nuestro mat-vec-id ya despacha por token y los densos grandes van a 190-227 GB/s |
| `GGML_VK_FA_WAVE32` (pin a subgrupo 32 de la FA coopmat1 cuando hsv ≤ 128; solo n_rows ≥ 32) | no | — | pp2048 Qwen3-Coder-30B: +2.5 % d0, +11.3 % d32768; hsv 256 declinado (6-18 % más lento) | No aplica: HSK = HSV = 256 |
| `GGML_VK_FA_KV_CONTIG`, `GGML_VK_FA_DEQUANT` (K/V f16 con stride contiguizados; dequant una vez) | no | — | pp2048 a d8192 846.6 frente a 847.9 t/s (neutro en su modelo) | No aplica: KV q8_0 y FA sparse propia (5.4) |
| `GGML_VK_FA_TOPK_GATHER`, `GGML_VK_FA_TOPK_UNION` (gather a KV compacto para el decode sparse de DeepSeek V4; unión deduplicada para lotes 2-8) | no | — | gather: kv 32768 986 → 56 µs (17.6×), plano con el contexto; unión: 1.06-1.46× a lote 2-8 con solape del 60 % | Equivalente propio: modo compacto (filas compactas, c0e776ec4) y solape medido del 27 % a n = 3 (desglose, sección 5) |
| `GGML_VK_MAX_MB_PER_SUBMIT` (cota de bytes por submisión, 8 GiB) | no | — | evita el reset del anillo (timeout de 10 s en su kernel) con nodos sin estimación de flops | Robustez (desglose, sección 4); aquí `GGML_VK_MAX_NODES_PER_SUBMIT` cuenta nodos y `qsa_query_block` acota solo la FA |
| Kernels del indexador y de la FA sparse de DeepSeek V4 (prefill y decode) | no | — | serie de agosto (e72ffec16 … 78e31af07) | Equivalente propio (5.4); no comparados |
| ROCmFPx (tipos FPx nativos de AMD, CPU + Vulkan) | no (677ad73a8 retiró las referencias) | — | — | Tipos nuevos, fuera de alcance |

Upstream Vulkan posterior a nuestra base `6d9c82ea2` (2026-09-09) y ausente aquí: `50182a53f`
topk_moe fusionado en prefill (#28422), `6788edb4f` matrices M pequeñas para qwen (#28457),
`28ff09582` escrituras CPU en `ggml_backend_vk_cpy_tensor_async` con el contexto inactivo
(#28618), `481c65f09` carrera y OOB en argsort grande (#28705), `91f6a6cf3` constante de
especialización para el tipo A (#25773), `72797e891` etiquetas de depuración (#28101). La rama
HIP de pwilkin (kernels ggml-cuda y cambios de modelo, 2026-09-12..14) no aporta código Vulkan;
su análisis está en el desglose, sección 6.
