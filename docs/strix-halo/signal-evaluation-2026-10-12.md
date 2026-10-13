# Signal 3.8 Flash Next (agentionai): el modelo base con otra capa de salida (2026-10-12)

Evaluación pedida por el Director de https://huggingface.co/agentionai/Signal-3.8-Flash-Next-GGUF, hecha sin GPU (la
máquina estaba ocupada): cabeceras de los GGUF y filas de tensores leídas por rangos HTTP, comparadas con el checkpoint
oficial y con el GGUF de producción. Herramientas y salidas: `~/dbg/signal/`.

## 0. Resultado

- **Signal es Qwen3.8-Flash-Next con solo la capa de salida (`output.weight`, el lm_head) cambiada.** Todo lo demás es
  el checkpoint oficial: en su archivo Q8_0, cada tensor comparado es byte a byte el Q8_0 que ggml hace de los pesos
  oficiales, en las 48 capas, todos los tipos de tensor, el embedding, la tabla PLE y el mezclador final (sección 2).
  La ficha habla de "self-distillation" y "minimally invasive fine-tune"; no dice que solo se entrenó la cabeza (el
  autor sí lo dice del Signal 27B, sección 6).
- **La cabeza nueva cambia todas las filas (mediana 2.2 %), y las más cambiadas son las palabras de duda.** Las 60
  filas más cambiadas de las 248 077 son " However", " Hmm", " But", "Alternatively", " reconsider", " could", " might",
  " unsure", " Wait", " perhaps", " maybe", " wrong", " mistake", " backtrack" y similares (8-10 %); entre las que más
  bajan en su propia dirección están preámbulos y formato: "Based", "Here", "According", " **" (sección 3).
- **No arregla el modo "el mensaje del usuario está vacío".** En la petición 8, con los estados del tronco de la
  referencia y la apertura anómala forzada, la probabilidad conjunta de los 40 tokens es 0.0084 con la cabeza de
  producción y 0.0089 con la de Signal; p("The" tras `<think>\n`) 0.961 y 0.948 (sección 4). La conclusión "vacío" ya
  está en el estado oculto del tronco, que Signal no toca. No cubre req2, los estados de BF16 ni tasas muestreadas.
- **El fork de llama.cpp acepta sus tensores (no se probó una carga); en Strata los tiers publicados no corren sin
  cambios** (`token_embd` Q6_K no lo acepta el build actual). Como solo cambia la cabeza, probar Signal en producción no requiere bajar 85-95 GiB: basta
  el GGUF de producción con el `output.weight` de Signal (0.52 GB, mismo tipo y tamaño). Es una transformación de GGUF:
  decisión del Director (sección 5).
- **Las mejoras que anuncia la ficha no se pudieron medir sin GPU**, y su evidencia es delgada (sección 6).

## 1. Qué publica

| Tier | Archivo | Expertos gate/up, down | Resto |
|---|---|---|---|
| `Q8_0` | 188.2 GB | Q8_0, Q8_0 | Q8_0; normas, routers, GDN e indexador en float |
| `AP-Q4_K_XL` | 101.1 GB | Q4_K, IQ4_NL | atención, ssm_out, shared expert gate/up, hc down/inject, ssm alpha/beta, PLE key/value, `token_embd` y `output.weight` Q6_K; `output_hc_*` IQ4_NL; shared expert down y hc up Q8_0 |
| `AP-IQ4_XS` | 90.4 GB | IQ3_S, IQ4_NL | igual que AP-Q4_K_XL |
| producción (`qwen4exp-ud-small-nl3s`) | 90.5 GB | IQ3_S (IQ4_XS en la capa 2), IQ4_NL | atención IQ4_NL/Q5_K/Q8_0, hc y shared expert Q8_0, inject y ssm alpha/beta F32, `token_embd` Q8_0 |

Los cuatro archivos tienen la misma arquitectura (`qwen4exp`), los mismos 1224 tensores con las mismas formas, el mismo
tokenizer y la misma plantilla de chat que el GGUF de producción. La plantilla es la de Unsloth (termina en "Unsloth
fixes - developer role, merged system messages, tool calling"), no "the original Qwen3.8-Flash-Next template" como
dice la ficha; es la que ya usa producción, con los mismos tokens que la oficial en req2 y req8
(`empty-conversation-mode-2026-10-10.md`, sección 1). Imatrix de los AP: `bartowski-imatrix-v5-semantic.txt`, 1540
chunks. Muestreo recomendado: temperatura 0.6, top_p 0.95, top_k 20, min_p 0.05 (producción: 0.7 y 0.0). El draft MTP
que recomiendan (`agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF`) es el que usa la ruta llama.cpp de producción desde el
2026-08-27 (commit ef9f66b2); lleva su propio `output.weight`, el del modelo base.

## 2. Qué cambia respecto del modelo base

**Método.** Q8_0 no usa imatrix y redondea de una sola forma: un tensor sin cambios tiene en el tier Q8_0 de Signal
exactamente los bytes del Q8_0 que ggml hace del tensor oficial BF16. Cualquier cambio de los pesos, por chico que sea,
mueve parte de los redondeos. Las permutaciones del convertidor (cabezas de valor de la atención lineal) se aplican
como en `gguf_weights.py`.

| Comparación | Alcance | Resultado |
|---|---|---|
| Tensores enteros (`q8check.py`) | shared expert down, hc up/down/inject, shared expert gate, un experto (gate, down), o_proj de QSA, qkv de GDN | idénticos |
| Barrido por ventanas de filas (`sweep.py`) | capas 0-3, 7, 11, ..., 43, 46, 47: todos los tipos (qkv, gate, ssm_out, ssm alpha/beta, q/k/v/o, hc, shared expert, tres expertos por capa, PLE key/value), `token_embd`, mezclador final; 554 ventanas | idénticas salvo las 6 de `output.weight` |
| Barrido de las otras 32 capas, una ventana por tipo | 576 ventanas | idénticas |
| Tensores float, AP-IQ4_XS contra producción (`coverage.py floats`) | routers, normas, parámetros de GDN, indexador BF16, PLE: los 413 del mismo tipo en los dos, todas las capas | idénticos |
| Tabla PLE (`coverage.py ple_q8`) | 12 ventanas de 1024 filas en sus 320 M de filas | idénticas (control del mapeo: el IQ4_NL de producción queda a 7.6 % de las filas oficiales, el error normal del tipo) |
| `mmproj-F16.gguf` contra el local (`coverage.py mmproj`) | los 334 tensores | idénticos; solo cambian nombre y autor en los metadatos |
| `output.weight` | 6 ventanas | ningún bloque idéntico |

Control de la cabeza: el Q6_K de producción (Unsloth) reproduce exactamente los niveles que predicen las filas
oficiales del lm_head (0 de 7.6 M niveles en 9 ventanas), así que el mapeo es el correcto y la cabeza de producción es
la oficial.

**Los tiers AP usan su propio cuantizador.** Sobre tensores que no cambiaron, sus niveles Q6_K no son el redondeo al más
cercano de ggml (5.7-6.2 % de los niveles difieren; 0.8-1.0 % en Q4_K; 0 en el control con ggml): es otro
cuantizador (la ficha menciona un "per-tensor distortion solver" propio, que no es público). El error contra los pesos
oficiales es del orden del de ggml:

| Tensor | Tipo | Signal AP | ggml sin imatrix | producción |
|---|---|---|---|---|
| hc_attn_down (capa 10) | Q6_K | 0.0213 | 0.0196 | 0.0060 (Q8_0) |
| shared expert gate (capa 0) | Q6_K | 0.0237 | 0.0221 | 0.0071 (Q8_0) |
| experto 0, gate (capa 0) | IQ3_S | 0.1495 | 0.1732 | 0.1493 (IQ3_S de Unsloth) |
| experto 0, gate (capa 0) | Q4_K (AP-Q4_K_XL) | 0.0776 | 0.0750 | |
| experto 0, down (capa 0) | IQ4_NL | 0.0795 | 0.0773 | 0.0797 |

(||X - W|| / ||W||.) El tier AP-IQ4_XS lleva los mismos tipos de expertos que producción, IQ3_S calibrado con imatrix:
el que amplifica el modo en contextos largos (`empty-conversation-mode-2026-10-10.md`, secciones 9 y 10).

Según el informe del agente (muestras de 108 tensores), los tiers AP de Signal son los del modelo base de agentionai
(`agentionai/Qwen3.8-Flash-Next-AP-GGUF`) con la cabeza cambiada: 107 idénticos, `output.weight` distinto. Un cambio
del modo que venga de la cuantización AP (los expertos Q4_K de AP-Q4_K_XL, por ejemplo) se obtiene sin Signal y no se
le puede atribuir; por eso la sección 7 mide la cabeza sobre el tronco de producción.

## 3. La cabeza nueva

`lmrows.py` sobre las 248 320 filas (Signal Q8_0 contra el Q8_0 de ggml de las filas oficiales, para no contar el
redondeo, que es 0.59 %):

- Cambio por fila: mediana 2.17 %, p90 2.6 %, p99 3.5 %, p99.9 5.1 %, máximo 10.5 %; 274 filas sobre 5 %.
- Las 60 filas más cambiadas son, salvo "Based" y "Here", marcadores de duda y de vuelta atrás: " However" 10.5 %,
  "However", " Hmm", " But", "Alternatively", "But", " Alternatively", "Hmm", " reconsider", " could", " might",
  " retry", " unsure", " incorrect", " Instead", " confused", " Wait", " uncertain", " different", " mistake",
  " error", " whether", " perhaps", " rethink", " backtrack", " maybe", " doubt", " revisit", " wrong", " actually",
  " Or", " still" (8-10 %).
- El cambio es casi todo una rotación de la fila, no una escala: en su propia dirección esas filas cambian menos de
  0.7 %. Las 16 que más bajan en su propia dirección son preámbulos, formato y conjunciones: "Based" -3.2 %, "Here"
  -2.7 %, " и" -2.5 %, " **" -2.5 %, " và" -2.2 %, "According" -2.0 %, " или" -1.9 %, "\n\n\n\n" -1.8 %,
  `<tool_call>` -1.8 %, `<think>` -1.6 % y otras de menos de 1.8 %; `</think>` sube 0.7 %. El efecto sobre la frecuencia
  de llamadas a herramientas no se midió.
- El cambio está concentrado: en 2985 filas muestreadas, su dirección principal lleva 18.6 % de la energía (ruido
  i.i.d.: 0.14 %), las 64 primeras 33 %.

Es lo que dice la ficha que busca (menos "Wait", "But", "Alternatively"), hecho con una cabeza entrenada en vez de una
penalización fija; el efecto real sobre un logit depende del estado oculto, y no se midió sin GPU.

## 4. El modo "mensaje vacío" con la cabeza de Signal

Como el tronco de Signal es el del modelo base, su distribución en cada posición es el último estado oculto del modelo
base pasado por su cabeza. `headcmp.py` toma los estados de la referencia oficial (`ref_forward.py` con el GGUF de
producción sobre la petición 8, `~/dbg/strata/empty/official/dump-long/h47.npy`: las últimas 64 posiciones, cuya
continuación forzada es "<think>\nThe user hasn't asked anything substantive yet — the last message was just system
instructions setting up my role as an expert software engineer helping the user solve problems. There's no actual
task or question here."), aplica el mezclador final oficial en numpy y las tres cabezas. Validación: con la cabeza de
producción reproduce los log p de la referencia (`ref-long.tsv`) con una diferencia máxima de 4.4e-5, y la probabilidad
conjunta documentada de 0.0084.

| | Cabeza de producción (Q6_K) | Signal Q8_0 | Signal AP (Q6_K) |
|---|---|---|---|
| p("The" tras `<think>\n`) | 0.961 | 0.948 | 0.949 |
| p(" hasn") | 0.836 | 0.790 | 0.801 |
| p(" asked") | 0.924 | 0.949 | 0.944 |
| media p(primeros 10) | 0.920 | 0.918 | 0.916 |
| probabilidad conjunta de los 40 tokens | 0.0084 | 0.0089 | 0.0086 |

Las alternativas a "The" siguen siendo marginales con las tres cabezas ("I" 0.007-0.012, "Verify", "Note", "S"). En
esta petición la cabeza de Signal no cambia el modo. Lo que no cubre: req2, los estados de BF16 (con BF16 el modo en
req8 es mucho menos probable, 0.23-0.40 contra 0.87-0.96 del GGUF: `empty-conversation-mode-2026-10-10.md`,
sección 10), la tasa muestreada y la cuantización AP del tronco, que difiere de la de producción.

## 5. Compatibilidad con nuestro stack

- **Fork de llama.cpp:** mismos nombres y formas; los tensores que Signal guarda en Q6_K (hc, ssm alpha/beta) pasan por
  `build_lora_mm`, que acepta cualquier tipo. No tiene la fusión hc down+inject del GGUF de la ruta llama.cpp
  (hcdi). No se probó una carga.
- **Strata, tiers AP tal como se publican:**
  - expertos IQ3_S o Q4_K con down IQ4_NL: los aceptan los kernels agrupados (`STRATA_GU_FMTS`, `STRATA_D_FMTS`);
  - hc down/inject, ssm alpha/beta y PLE key/value en Q6_K y `output_hc_*` en IQ4_NL: el motor los lee en BF16, así
    que el pack (`tools/iq_pack.py --compat-bf16`) los convierte redondeando;
  - `token_embd` Q6_K: `NativeEmbed::load` lo rechaza (`embed_type_supported` acepta Q6_K solo con
    `STRATA_Q6K_EXPERTS`, apagado en `build-halo`) y el motor termina con error (`generate.cpp`). Haría falta
    recompilar o un `--embd-gguf`.
- **La cabeza de Signal sobre producción:** el `output.weight` de Signal AP es Q6_K de 2560 x 248 320 (521 472 000 B),
  el mismo tipo y tamaño que el de producción. Strata lo sirve de forma nativa desde el GGUF de `--native` (cualquier
  tipo de `native_mmvq_supported`, Q8_0 incluido: 675 430 400 B, 154 MB más de VRAM), y la fila del pack solo lleva la
  forma. Con los bytes reemplazados en el mismo lugar ningún offset se mueve, así que el pack actual sirve: el motor
  verifica nombre, tipo y offset de cada tensor de expertos en el archivo de `--native` (`expert_source.cpp`). En
  btrfs, una copia con reflink del GGUF de producción con esos bytes reemplazados ocupa ~0.5 GB. El draft MTP sigue
  con la cabeza base (la del pack `mtp` en Strata, la del archivo de agentionai en llama.cpp). No se probó una carga.

## 6. La ficha y la comunidad

Lo que la ficha afirma y no se pudo medir sin GPU: 26 % menos tokens de razonamiento en MATH-500 (media; la mediana baja
6 %, de 242 a 228), aceptación del draft MTP 90 % contra 69 %, Terminal-Bench 10/10 contra 8/10 con 25 % menos tiempo,
AIME 23/30 contra 25/30. Observaciones (informe de un agente sobre el repo, sus discusiones y sus commits; las citas
clave verificadas en la API de HF):

- **La evidencia es delgada.** MATH-500 son 120 problemas (119 contra 117); Terminal-Bench, 10 tareas con un intento;
  en AIME el base y Signal corrieron con muestreo y draft distintos (la ficha lo dice).
- **Las cifras de MTP y Terminal-Bench ya estaban en la ficha de la primera cabeza** (2026-09-11) y siguen iguales en
  la de la cabeza v4 (2026-09-19); la ficha no dice si se volvieron a medir. Un usuario reporta 0.554 y 0.632 de aceptación con Signal en llama.cpp mainline
  (`agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF`, discusión 1).
- **El paper que citan** (arXiv:2606.00206) existe y dice lo que la ficha resume, pero el 12-23 % es para modelos
  cuantizados; en BF16 reporta 5.6-10 %.
- **El Signal 27B, con el mismo método** (el autor: "only the output weights are changed and not the attention
  layers"), tiene reportes de bucles y de cortes en contexto largo: en RULER a 32k termina sin responder mucho más que
  el base (44 contra 16 en fwe), y el autor recomienda el base para extracción de documentos largos a 32k
  (`agentionai/Signal-3.8-27B-GGUF`, discusiones 3 y 5). Producción trabaja con contextos de agente largos.
- **Nadie reporta el modo "mensaje vacío" con Signal, ni que lo arregle.**
- **El autor** es una persona (Laurent Zuijdwijk, agention.ai, también autor de Halo Box), con el mismo hardware que
  el nuestro. Su herramienta de cuantización ("agention-infer") no es pública, y no hay BF16 de Signal Flash-Next
  publicado. La licencia es la de Qwen, byte a byte.
- Hay un PR abierto en Strata upstream (Niko1221/Strata #2041) que agrega Signal; no midió calidad contra el base.

## 7. Qué medir cuando la GPU esté libre

Sobre producción con la cabeza de Signal (sección 5), decisión del Director:
1. el modo en req2 y req8 con `sample_set.py` (25 semillas), contra producción;
2. tokens de razonamiento y respuestas sobre un conjunto fijo de peticiones de agente, a la temperatura de producción y
   a la recomendada;
3. la aceptación del draft MTP de Strata;
4. el comportamiento a profundidad (`needle_bench.py` y una sesión larga), por los cortes reportados en el 27B.

## Herramientas

| Archivo (`~/dbg/signal/scripts/`) | Qué hace |
|---|---|
| `gguf_head.py`, `compare.py` | la cabecera de un GGUF por rango HTTP; tipos y metadatos de varios archivos |
| `tensor_cmp.py`, `delta.py`, `consist.py` | tensores float byte a byte; error contra los pesos oficiales; niveles Q6_K/Q4_K contra el redondeo de ggml |
| `q8check.py`, `sweep.py`, `coverage.py` | el tier Q8_0 de Signal contra el Q8_0 de ggml de los pesos oficiales: tensores, barrido por capas, PLE, floats, mmproj |
| `lmrows.py`, `lmhead.py` | el cambio de cada fila de la cabeza; ventanas con su estructura |
| `headcmp.py` | la apertura del modo con la cabeza de producción y la de Signal sobre los estados de la referencia |
