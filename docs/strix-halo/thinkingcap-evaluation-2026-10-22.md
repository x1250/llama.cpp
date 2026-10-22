# ThinkingCap Qwen3.8-Flash-Next (BottleCap AI): el modelo base con las proyecciones densas reentrenadas (2026-10-22)

Evaluación pedida por el Director de https://x.com/BottleCapAI/status/2113300959690416219 (publicado hoy), hecha sin
GPU (la máquina estaba ocupada): metadatos de los repositorios, cabeceras de los safetensors y de los GGUF, y ventanas
de tensores leídas por rangos HTTP, comparadas con el checkpoint oficial y con el GGUF de producción. Herramientas y
salidas: `~/dbg/thinkingcap/`.

## 0. Resultado

- **ThinkingCap solo cambia las proyecciones lineales densas: 372 tensores, 5.4 GiB de los 335 GiB en BF16.** Son las
  de la atención lineal GDN (`in_proj_qkv/z/a/b`, `out_proj`, 36 capas), las de la atención QSA (`q/k/v/o_proj`,
  12 capas) y las del experto compartido (`gate/up/down_proj`, 48 capas). El cambio es chico: 0.1-0.7 % de la norma de
  cada tensor. Todo lo demás es el checkpoint oficial: los 512 expertos de las 48 capas, los routers, las
  hyper-connections, las normas y parámetros de GDN, el indexador, el embedding, la cabeza, la tabla PLE, la visión y
  el bloque MTP. Una parte está verificada por hash del archivo y los tensores grandes de los archivos cambiados por
  muestreo (sección 2). Configuración, tokenizer y plantilla de chat también son idénticos al modelo base.
- **Lo que anuncian (38.5 % menos razonamiento, −1.03 puntos) se midió en otro régimen que el nuestro.**
  - Ellos midieron en BF16 sobre vLLM, con `reasoning_effort=xhigh` y temperatura 1.0, y el modelo se entrenó en
    `xhigh`.
  - Producción usa `medium` en el 100 % de los turnos de pi desde el 2026-10-07, a temperatura 0.7.
  - En Flash-Next nadie midió `medium`. La mejor referencia es su ThinkingCap-Qwen3.8-27B, con el mismo método: en
    `medium` razona 13.5 % menos que el base en `medium`, a −0.7 puntos (11 benchmarks). Su benchmark agéntico en
    Flash-Next (τ²-bench, `xhigh`) es el de menor reducción: −22.5 % (sección 4).
- **El beneficio esperable en producción es chico.**
  - El razonamiento es ~61 % de lo que genera producción en pi, unos 325 tokens por turno en promedio (~4.9 s de
    decode).
  - Con la referencia del 27B en `medium` (−13.5 %) son ~0.7 s menos por turno en promedio y ~0.2 s en la mediana,
    sobre turnos de ~6 s. Con el −38.5 % de `xhigh` serían 1.9 y 0.7 s.
  - Es un piso: el razonamiento pasado queda en el contexto (~86k de 236k tokens en la sesión más larga), así que un
    recorte también frena el crecimiento del contexto (sección 4).
- **El eje que importa es el nivel de esfuerzo, no ThinkingCap.** En el 27B, `medium` cuesta 9.2 puntos contra `xhigh`
  en el modelo base. ThinkingCap rinde donde fue entrenado: en `xhigh` razona 37 % menos por −0.9 puntos, pero todavía
  ~2× lo de `medium` (sección 4).
- **Sus GGUF publicados no sirven para probarlo.** El IQ4_XS sigue la receta de Unsloth de la que viene producción,
  pero cuantiza a IQ4_XS los `hc_*_inject` y `ssm_alpha/beta`, que Unsloth y producción dejan en F32. Ese es el lado
  denso que la evaluación del 2026-10-07 encontró sensible. Probar ese archivo mezclaría el fine-tune con una
  cuantización peor (sección 3).
- **Integrarlo es barato y no requiere bajar el modelo.** Basta reemplazar los 372 tensores en una copia del GGUF de
  producción, cuantizados a los mismos tipos (~1.9 GiB en el archivo). Se leen los 5.4 GiB por rango HTTP, sin bajar
  los 92 GiB de los shards que cambiaron. Los offsets no se mueven. Se reconstruye solo `dense.bin` del pack de
  Strata (1.5 GB); el pack MTP, la tabla PLE, el mmproj y el vocabulario del draft se reusan tal cual. Es una
  transformación de GGUF: decisión del Director (sección 5).

## 1. Qué publica

| Repositorio | Contenido |
|---|---|
| `bottlecapai/ThinkingCap-Qwen3.8-Flash-Next` (2026-10-22, `d90c9fe`) | safetensors BF16, 131 shards, 335.4 GiB, el mismo índice de tensores que el oficial; `config.json`, `generation_config.json`, `tokenizer_config.json`, `preprocessor_config.json` y `chat_template.jinja` idénticos byte a byte a los de `Qwen/Qwen3.8-Flash-Next` |
| `bottlecapai/ThinkingCap-Qwen3.8-Flash-Next-GGUF` (`76d5cd6`) | BF16, Q8_0, Q6_K, Q4_K_M, IQ4_XS (95.5 GB) en 8 partes, `mmproj` f16; probado con llama.cpp `5f5acfe` |
| `-FP8`, `-NVFP4` | para vLLM |

- **Qué afirman:** −38.5 % de tokens de razonamiento en promedio (de −22 % a −51 % según el benchmark), con −1.03
  puntos de exactitud (89.8 % a 88.8 %). Por benchmark:

  | Benchmark | Exactitud (puntos) | Razonamiento medio |
  |---|---|---|
  | AIME26 | −0.73 | −44.4 % |
  | HMMT | −1.46 | −44.5 % |
  | LiveCodeBench | −1.00 | −40.9 % |
  | AA-LCR | −1.88 | −28.0 % |
  | RealWorldQA | −0.95 | −51.0 % |
  | τ²-bench | −0.17 | −22.5 % |
- **Configuración de su medición:**
  - vLLM 0.31.0 en BF16, 4×H200, sin decodificación especulativa;
  - `reasoning_effort=xhigh` (el default de la plantilla), temperatura 1.0, top_p 0.95, top_k 20;
  - 4 a 32 semillas por benchmark, con el modelo base medido en el mismo harness.
- **Lo que dicen de su propio error:**
  - en 5 de 6 benchmarks la diferencia de exactitud es menor que la dispersión entre semillas;
  - RealWorldQA (−0.95) es una diferencia real;
  - AA-LCR (contexto largo) es la mayor caída, y sus intervalos no la resuelven.
- **MTP:** "keeps the base model's MTP head". En vLLM aceptó 2.77 tokens por paso contra 2.72 del base.
- **Método:** no lo publican (ni SFT/RL/LoRA, ni qué pesos se entrenaron, ni datos).
- **Licencia:** Qwen Community License 1.0, la misma del modelo base.
- **Autores:** BottleCap AI (Praga; Tomáš Mikolov entre ellos). Antes publicaron ThinkingCap-Qwen3.6-27B con la misma
  idea (julio de 2026).

## 2. Qué cambia respecto del modelo base

**Método.** Primero, los hashes LFS de los shards: un shard con el mismo sha256 que el oficial es idéntico. En los
shards restantes (`st_diff.py`), las cabeceras safetensors de ambos repositorios son iguales (mismos nombres, tipos,
formas y offsets; 0 diferencias). Cada tensor se compara:
- entero si mide hasta 1 MiB;
- si es más grande, en 6 ventanas de 128 KiB repartidas por el tensor.

| Alcance | Resultado |
|---|---|
| 131 shards | 80 idénticos por sha256 (243.6 GiB): todos los `experts.gate_up_proj`, el `lm_head` y 124 de los 128 shards de la tabla PLE |
| 51 shards restantes (91.7 GiB), 1485 tensores | 372 cambiados, 1113 idénticos |

Cambiados (n = tensores; elementos BF16 distintos y ‖ΔW‖/‖W‖ en mediana):

| Tensor | n | Elementos distintos | ‖ΔW‖/‖W‖ |
|---|---|---|---|
| GDN `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` | 36 c/u | 43 / 43 / 49 / 57 % | 1.9 / 2.0 / 2.1 / 2.6 e-3 |
| GDN `out_proj` | 36 | 30 % | 1.1e-3 |
| QSA `q_proj` / `k_proj` / `v_proj` / `o_proj` | 12 c/u | 46 / 48 / 46 / 26 % | 2.2 / 2.2 / 1.9 / 1.0 e-3 |
| experto compartido `gate_proj` / `up_proj` / `down_proj` | 48 c/u | 59 / 56 / 76 % | 3.6 / 3.3 / 7.0 e-3 (máx. 1.0e-2) |

Idénticos en todas las capas:
- los expertos `down_proj` (y los `gate_up_proj` por hash), el router (`mlp.gate`) y `shared_expert_gate`;
- las hyper-connections (`input_mix_weight_up/down`, `block_inject_weight`, `hc_norm`) y el mezclador final;
- `A_log`, `conv1d`, `dt_bias` y la norma de GDN;
- el indexador de QSA y las normas q/k;
- `embed_tokens`;
- la PLE entera (`key/value_proj`, normas, conv y los 4 shards de la tabla que compartían archivo con tensores
  cambiados);
- los 27 bloques de visión y el merger;
- los 31 tensores del bloque MTP.

Lo cambiado coincide con los módulos lineales de nombre estándar (`q/k/v/o_proj`, `in_proj_*`, `out_proj`,
`gate/up/down_proj`), la lista típica de `target_modules` de un LoRA. Quedaron fuera:
- los expertos, que son parámetros 3D y no módulos lineales;
- el router (`mlp.gate`), el indexador (`index_qk_proj`) y las proyecciones de la PLE (`key/value_proj`);
- las mezclas hc;
- el MTP.

**El cambio no es de rango bajo en lo almacenado** (`delta_rank.py`, 4 tensores enteros), pero sí está concentrado:

| Tensor | Energía del cambio en la 1.ª dirección | Ruido i.i.d. | Direcciones para el 90 % |
|---|---|---|---|
| varios | 6-18 % | 0.1-0.4 % | 111 de 640, 1177-1347 de 2560 |

No se puede saber si fue un LoRA fusionado: el redondeo a BF16 de un cambio de 0.1-0.7 % es de rango completo y tapa
la estructura de un LoRA.

Un agente lo reprodujo por separado, con otro muestreo, y encontró el mismo conjunto de tensores cambiados. Con 0
diferencias:
- los down de los expertos (64 ventanas por capa);
- el bloque MTP (620 ventanas);
- los 333 tensores de visión.

**Alcance del muestreo.** Los tensores de hasta 1 MiB se compararon enteros; los más grandes de los shards cambiados
(expertos down, embedding, MTP), por ventanas. Un tensor entrenado cambia el 24-83 % de sus elementos BF16 en todo
su largo, como muestran los 372. Que ninguna ventana de los down de expertos tenga un solo elemento distinto excluye
que se hayan entrenado. Lo que no excluye es una edición confinada a una región chica; la comparación completa es un
paso opcional de la sección 5.

**Consecuencias:**
- El draft MTP de producción (el pack `mtp/rt` de Strata, construido desde los tensores oficiales; el Q8_0 de
  agentionai en llama.cpp) es exactamente el MTP de ThinkingCap. La aceptación puede cambiar igual, porque el estado
  oculto que el draft recibe viene del tronco, que sí cambió.
- La visión no cambia: el `mmproj-F16.gguf` local sirve tal cual.

## 3. Sus GGUF frente a la receta de producción

Producción viene del UD-IQ4_XS de Unsloth (`38bb39e`), recuantizado con nuestra receta (secciones 2 y 5.1 de
`strata-evaluation-2026-10-07.md`). Cabeceras leídas por rango (`hdr_cmp.py`):

- **Mismo imatrix:** el IQ4_XS de ThinkingCap usa el `imatrix_unsloth.gguf` de Unsloth (mismo dataset, 926 entradas,
  45 chunks) y los mismos tipos que el UD-IQ4_XS actual en los 1224 tensores comunes, salvo cinco grupos:

  | Tensor | Unsloth y producción | ThinkingCap |
  |---|---|---|
  | `hc_attn_inject` y `hc_ffn_inject` (96) | F32 | IQ4_XS |
  | `ssm_alpha` y `ssm_beta` (72) | F32 | IQ4_XS |
  | `ple_conv1d` | F32 | F16 |

  Los dos primeros grupos son del lado denso chico, el que en la evaluación del 2026-10-07 separó el comportamiento
  de UD del de nuestra receta anterior (secciones 3.2 y 5.1 de ese documento).
- **Trae el bloque MTP adentro** (`blk.48`, 32 tensores, `nextn_predict_layers` 1) en Q4_K/Q5_0/Q6_K/Q8_0, con
  metadatos de un convertidor más nuevo que el de producción (`block_count` 49, `recurrent_layers`). Producción carga
  el MTP aparte. No se probó si el fork carga este layout.
- **Expertos gate/up:** los bytes IQ3_S difieren de los de Unsloth aunque los pesos son idénticos. `iq3check.py`
  descuantiza 640 filas de tres expertos y las compara con las oficiales:

  | Medida | Resultado |
  |---|---|
  | Error contra el peso oficial, ThinkingCap y Unsloth | el mismo (0.149 / 0.149, 0.140 / 0.140, 0.143 / 0.143) |
  | Bloques IQ3_S idénticos | 78-99.8 % |

  Es el mismo peso con el cuantizador de otra versión de llama.cpp (desempates distintos), no un cambio del modelo.
  Los tensores Q8_0 sin cambios (`token_embd`, `output`, expertos down de las capas 2 y 47) y el router F32 son
  idénticos byte a byte (`gguf_win.py`).

Conclusión: probar el IQ4_XS publicado mezclaría tres cosas, y una diferencia no se le podría atribuir al fine-tune:
- el fine-tune;
- los inject y alpha/beta en IQ4_XS;
- el cuantizador de otra versión de llama.cpp.

## 4. Sus números frente al uso de producción

Turnos de pi con `qwen38flash` desde el 2026-10-07 (`pi_thinking.py`, sesiones de `~/.pi/agent/sessions`):

| Medida | Valor |
|---|---|
| Turnos | 346, todos en nivel `medium` |
| Temperatura (`~/.pi/agent/models.json`) | 0.7 |
| Fin del turno | 306 por llamada a herramienta, 39 por respuesta, 1 abortado; ninguno por `max_tokens` |
| Razonamiento sobre lo generado (caracteres) | 61 % del total; mediana por turno 50 % |
| Tokens generados por turno | mediana 279, p90 1248, máximo 10 053 |
| Razonamiento por turno (tokens generados × proporción de caracteres; estimación) | mediana 119, media 325 |
| Duración del turno (`durationMs`, en los 346) | mediana 6.2 s |

- **El nivel `medium` no agrega instrucción.** En la plantilla solo `xhigh` y `low` agregan una línea al sistema
  ("Reasoning effort is set to …"). Todas sus mediciones de Flash-Next son con `xhigh`. Sus propios datos muestran
  reducciones menores donde el base ya razonaba poco: τ²-bench −22.5 %, AA-LCR −28 %.
- **La referencia para `medium` es su 27B** (https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/, verificado). Es el
  mismo método, "trained at `xhigh`". Macro promedio de 11 benchmarks; Δ contra el base en `xhigh`:

  | Nivel | Base: Δ exactitud, razonamiento | ThinkingCap: Δ exactitud, razonamiento | ThinkingCap contra el base en el mismo nivel |
  |---|---|---|---|
  | `xhigh` (12 benchmarks) | 0, 15 735 | −0.86, 12 144 | −37.2 % tokens, −0.86 puntos |
  | `medium` | −9.16, 5 830 | −9.90, 5 042 | −13.5 %, −0.74 |
  | `low` | −9.71, 5 461 | −10.79, 4 779 | −12.5 %, −1.08 |

  En el modelo base, bajar de `xhigh` a `medium` le costó 9.2 puntos en el 27B. En Flash-Next la única cifra
  publicada del base por nivel es de otro fine-tune (la ficha de Swift 1.5, un solo benchmark, GPQA-Diamond):
  `xhigh` 89.80 % con 17 683 tokens, `medium` 86.36 % con 4 157, `low` 87.17 % con 3 966.
- **Lo que valdría en segundos.** Al decode de agente de producción (~67 t/s), 325 tokens de razonamiento son ~4.9 s
  por turno:

  | Reducción del razonamiento | Turno promedio | Turno mediano |
  |---|---|---|
  | −13.5 % (27B en `medium`) | −0.7 s | −0.2 s |
  | −22.5 % (τ²-bench, `xhigh`) | −1.1 s | −0.4 s |
  | −38.5 % (promedio, `xhigh`) | −1.9 s | −0.7 s |

- **El razonamiento se queda en el contexto, así que esa cifra es un piso.** La cadena completa:
  - pi reenvía el razonamiento de cada turno anterior como `reasoning_content` (`pi-ai`, `openai-completions.js`);
  - el servidor de Strata lo pasa a la plantilla (`serve/frontend.py:320`);
  - la plantilla de producción lo conserva salvo que `preserve_thinking` sea falso, y pi nunca lo manda en falso:
    solo lo pone en `true`, en el formato `qwen-chat-template`.

  En la sesión más larga (266 turnos) el razonamiento acumulado es ~86k tokens, de un contexto que llegó a 236k. Un
  recorte también frena el crecimiento del contexto, con su efecto en el decode a profundidad y en cuándo la sesión
  llega al límite:

  | Recorte | Tokens menos al final de esa sesión |
  |---|---|
  | −13.5 % | ~12k |
  | −38.5 % | ~33k |

- **Nadie comparó ThinkingCap contra el base con menos esfuerzo en Flash-Next.** El contrafactual sin costo es el
  base en `low` ("Keep your thinking brief and focused"; pi lo manda con los niveles `minimal`/`low`). Nunca se midió
  en producción.
  - La línea de esfuerzo va al comienzo del prompt de sistema. Cambiar de nivel en medio de una sesión invalida
    todo el caché de la conversación; fijarlo por sesión no cuesta nada.

## 5. Cómo llevarlo a producción (sin descargar el modelo)

Como los pesos cambiados son 372 tensores densos, un GGUF de producción con ThinkingCap es una copia del actual con
esos tensores reemplazados:

1. **Los tensores.** Leer por rango HTTP los 372 tensores BF16 (5.4 GiB) y aplicarles las permutaciones del
   convertidor (cabezas de valor de GDN, como en `gguf_weights.py`). Cuantizarlos al tipo exacto que tienen en
   producción, con el imatrix de producción (`~/dbg/imatrix/imatrix-ud.gguf`):

   | Tensor | Tipo |
   |---|---|
   | `attn_qkv`, `attn_v` | Q5_K |
   | `attn_gate`, `ssm_out`, `attn_q`, `attn_k` | IQ4_NL |
   | `attn_output` y el experto compartido | Q8_0 |
   | `ssm_alpha/beta` | F32 |

   Son ~1.9 GiB del archivo.
2. **El archivo.** Escribirlos en el mismo lugar de una copia con reflink de `qwen4exp-ud-small-nl3s.gguf`: mismos
   tipos y formas, ningún offset se mueve, y la copia ocupa ~2 GB en btrfs. Lo mismo vale para el GGUF de la ruta
   llama.cpp (`nl3s-hcdi-gu-rq8-oproj8`): sus fusiones (gate+up de expertos, hc down+inject) son de tensores que no
   cambiaron.
3. **Strata.** Los expertos se leen del GGUF por offset (`native_experts.txt`) y no cambian. `dense.bin` (1.5 GB)
   tiene las proyecciones densas y `ssm_alpha/beta` en BF16: se reconstruye con `iq_pack` sobre la copia. El pack
   MTP, el tokenizer, el vocabulario del draft de pi, la tabla PLE y el mmproj se reusan.
4. **Control.** Construir con el mismo procedimiento el mismo archivo con los 372 tensores del modelo base.
   - Producción tiene seis de esos grupos (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v`) recuantizados a Q5_K o
     IQ4_NL desde el Q8_0 de Unsloth (doble redondeo). Este modelo es sensible al lado denso.
     Sin el control, la comparación contra producción mezcla el fine-tune con el camino de cuantización.
   - El control también verifica el mapeo: sus tensores de redondeo único tienen que salir byte a byte iguales a los
     de producción. Son `attn_output` y el experto compartido (Q8_0, sin imatrix) y `ssm_alpha/beta` (F32).

Paso opcional, para certeza completa antes de reusar los tensores no cambiados: comparar enteros, por rango, los
1113 tensores no cambiados de los 51 shards en los dos repositorios. Son ~86 GiB de tráfico por lado, sin escribir a
disco ni usar la GPU.

El script de reemplazo no existe todavía. Las piezas sí:
- la lectura por rango (`st_diff.py`);
- las permutaciones (`gguf_weights.py`);
- la cuantización con imatrix (`ggml_quantize_chunk` por ctypes, como en `q8check.py` de la evaluación de Signal).

## 6. La comunidad

Informe de un agente sobre los repositorios de BottleCap, sus discusiones y benchmarks de terceros. La tabla del 27B
y el conteo de discusiones están verificados; el resto se cita con su fuente.

- **Sobre el modelo de Flash-Next no hay reportes de uso.** En la tarde del 2026-10-22, su repositorio tenía una
  discusión ("How does it compare to Swift?", sin respuesta) y el GGUF ninguna. Todo lo demás es de los
  ThinkingCap densos de 27B. BottleCap mismo dice que el MoE es "a different kind of beast"
  (`ThinkingCap-Qwen3.8-27B`, discusión 5).
- **Bucles.**
  - Varios usuarios del 3.6-27B reportan menos bucles de razonamiento que el base. La tabla de BottleCap del 3.6
    muestra menos bucles en LongBench-v2 (12.4 → 5.0 %) y AA-LCR (3.8 → 1.8 %), y más en HMMT (0 → 0.7 %).
  - A temperatura 0.1-0.2 sí entra en bucle (3.6-27B-GGUF, discusión 6). La ficha del GGUF de Flash-Next lo advierte
    ("Greedy decoding can loop"). Producción usa 0.7.
- **Uso agéntico (27B), mixto.**
  - **A favor:**
    - opencode/kilocode funcionan; el razonamiento llegó al tope de 8192 dos veces en 8 horas;
    - un turno de herramienta bajó de ~1000 tokens de razonamiento a 27.
  - **En contra:**
    - una tarea agéntica de ~1 h en LM Studio falla y el usuario volvió al base;
    - un usuario de aider-desk dice "removing extra thinking in real world scenarios is worse" (otro no lo
      reproduce);
    - anuncia que usará una herramienta MCP y no la llama hasta que se le pide.
  - **Benchmark independiente de OpenCode (akitaonrails, 3.8-27B):** 83.0 contra 80.5 del base, en 75.7 contra
    192.1 minutos. Dejó 2 defectos sin resolver; el base, 0. Una sola corrida cada uno.
- **Contexto largo (27B).**
  - Un usuario dice que el Q4_K_M "performs terribly with large contexts", sin detalles.
  - Un RULER a 131k da columnas en ambas direcciones.
  - En la tabla de BottleCap del 3.6: LongBench-v2 −2.5 puntos, AA-LCR −2.0, Claw-Eval (agéntico) −2.6.
  - El 3.6 también acortó las respuestas por error. En Flash-Next la respuesta media baja de 322 a 283 tokens.
- **MTP.** En el 3.8-27B la aceptación es igual a la del base, salvo en τ²: 3.24 contra 3.43 tokens por paso.
- **Método.**
  - Solo dicen "the training objective rewarded efficient reasoning rather than simply rewarding correctness" y que
    la intervención es "as non-invasive as possible".
  - No hay paper. No dicen si es RL o SFT ni qué congelaron. La sección 2 lo muestra en los pesos.
- **El otro fine-tune corto de Flash-Next, Swift 1.5 (ukisai, 2026-09-22).**
  - Usa penalización de tokens de sobre-razonamiento, RL y destilación on-policy con datos agénticos.
  - En una prueba de 100 preguntas sobre cosas inexistentes respondió "no sé" 65 veces contra 74 del base: parte de
    su velocidad viene de contestar donde el base seguía buscando.
  - No hay comparación directa con ThinkingCap.
- **Un defecto del modelo base que ThinkingCap heredaría** (no probado en él): Flash-Next a veces inventa tareas del
  usuario o lee mal resultados de herramientas, en `medium` sin MTP (`Qwen/Qwen3.8-Flash-Next`, discusión 68). Se
  parece al modo "mensaje vacío".

Fuentes (todas en huggingface.co salvo indicación):
- `bottlecapai/ThinkingCap-Qwen3.6-27B`, discusiones 9, 13, 19 y 21;
- `bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF`, discusiones 5, 6, 7, 9 y 13;
- `bottlecapai/ThinkingCap-Qwen3.8-27B`, discusiones 2 y 5;
- `bottlecapai/ThinkingCap-Qwen3.8-27B-GGUF`, discusiones 2, 5 y 7;
- `ukisai/Swift1.5-Qwen3.8-Flash-Next`, ficha y discusión 3;
- `Qwen/Qwen3.8-Flash-Next`, discusiones 36 y 68;
- https://bottlecapai.com/post/thinkingcap-qwen3-6-27b/ y https://bottlecapai.com/post/thinkingcap-qwen3-8-27b/;
- https://github.com/akitaonrails/llm-coding-benchmark/blob/master/docs/b300-batch-analysis.md.

## 7. Qué medir cuando la GPU esté libre

Con decisión del Director sobre la transformación (sección 5). Las mediciones comparan A (ThinkingCap) con B
(control), en la misma cadena, B primero y último. La compuerta de calidad por KL (≤ 2× el redondeo) no aplica: un
cambio de pesos la falla por construcción. Sí aplican:
- el determinismo de cada archivo (`repeat_probe`, history gate);
- la sonda de carga.

Mediciones:
1. **Tokens de razonamiento y resultado en peticiones reales de pi.** Un conjunto fijo de peticiones reproducidas de
   las sesiones, a temperatura 0.7 y con 5 semillas o más. Brazos:
   - A y B en `medium`;
   - B en `low`;
   - A en `xhigh`, si se quiere ver qué cuesta subir el esfuerzo (sección 4).
2. **Formato de las llamadas a herramientas** en esas mismas peticiones: tasa de llamadas válidas.
3. **El modo "mensaje vacío"**, con `sample_set.py` en req2 y req8 (25 semillas). Aquí el tronco cambió, a diferencia
   de Signal.
4. **Contexto largo:** `needle_bench.py` y una sesión larga, por la caída de AA-LCR.
5. **Velocidad:** aceptación del draft MTP de Strata y decode, más una petición con imagen.

Criterio que propongo:
- **adoptar** si A recorta el razonamiento en `medium` (≥ 20 % en la mediana de las peticiones de pi) sin bajar las
  llamadas válidas ni subir el modo vacío;
- **no adoptar** si B en `low` logra lo mismo: alcanza con cambiar el nivel en pi.

El 27B anticipa ~13 % en `medium` (sección 4), por debajo de ese umbral.

## Herramientas

| Archivo (`~/dbg/thinkingcap/scripts/`) | Qué hace |
|---|---|
| `st_diff.py`, `st_summary.py` | safetensors de ThinkingCap contra los oficiales: cabeceras, tensores enteros o por ventanas, resumen por tipo (`st_diff.tsv`) |
| `delta_rank.py` | espectro del cambio de tensores enteros (`delta_rank.log`) |
| `hdr_cmp.py`, `gguf_win.py`, `iq3check.py` | GGUF de ThinkingCap, de Unsloth y de producción: tipos, metadatos, bytes por ventanas, IQ3_S descuantizado contra el oficial |
| `pi_thinking.py` | razonamiento en los turnos de pi de producción |
| `~/dbg/signal/scripts/gguf_head.py` | cabecera de un GGUF por rango HTTP (`hdr/`) |
