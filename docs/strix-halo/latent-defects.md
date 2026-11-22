# Defectos latentes: registro (Strata y llama.cpp)

Por decisión del Director (2026-11-01), un defecto que no ocurre en nuestro uso no se arregla. Se registra aquí con su
síntoma, su causa y su arreglo, para reconocerlo si algún día pasa. Si aparece el síntoma, el arreglo se toma con sus
compuertas y la fila pasa a la sección de arreglados.

Los síntomas se cuentan en los registros de producción con `~/ProjectHub/llama.cpp-evidence/latent/scan.py <desde>`:
las sesiones de pi (solo los turnos de qwen38flash), el log del motor (`~/.strix-engine.log`) y los del lanzador. Al
2026-11-01, desde el 2026-10-07 (`scan-2026-11-01.out`):
- 22 sesiones y 4653 mensajes de qwen38flash;
- 19 arranques del motor, sin ningún watchdog ni reinicio;
- ningún error de pi salvo 5 cancelaciones del usuario;
- ninguna etiqueta especial citada en un mensaje del usuario o en el resultado de una herramienta;
- ninguna respuesta degenerada en "!".

## Strata (producción)

| Síntoma | Causa | Arreglo | Visto aquí |
|---|---|---|---|
| El motor deja de responder al arrancar y, a los 60 s, el watchdog lo mata. La petición siguiente lo recarga sola | El grafo "zero-doorbell" de la primera ventana de verificación espera un flag que el host sube después de lanzarlo. En el primer lanzamiento el host puede quedar bloqueado dentro de `hipGraphLaunch`. Upstream lo midió en 8 de 41 arranques en un Strix Halo | `86d10ed8`, adaptado (un hunk de ~20 líneas en `verify.cpp`) y extendido a los caminos batch (`verify.cpp:2359`, `2515`, `2540`), que producción usa con dos slots. Exacto | No, en 19 arranques |
| El texto degenera en "!!!!" y queda así en la conversación y en el caché | Tres productores de q8_1 escriben la escala sin acotar (`fused_gr.cu:202`, `verify_kernels.cu:187`, `iq_kernels.cu:2330`). Un bloque con amax > ~8.3M da inf/NaN, los logits terminan en NaN y el muestreador devuelve el id 0, `!` | `7a0e3023` + `e372d4ff`. Bits iguales en bloques normales | No |
| El primer prompt de texto después de una imagen da basura | El prefill sube los registros de pasos QSA y los ids de tokens con `cudaMemcpyAsync` desde un `std::vector` paginable (`prefill.cpp:1805,1875,2674-2675,2699`). En gfx1100 pasó 5 de 5 veces | `b97a36a4` + `7bab660b` (4 hunks, ~85 líneas en `prefill.cpp`) | No: la compuerta texto → imagen → texto pasa en cada cadena |
| Una llamada a herramienta o el pensamiento se cortan cuando un mensaje o un archivo leído contiene `<tool_call>`, `</tool_call>`, `<tool_response>` o `</think>` | El frontend tokeniza esas etiquetas citadas como tokens especiales (`serve/frontend.py:163`) | `a6830008` + `62185aab` + `0d613031` | No: ninguna etiqueta citada |
| Con dos chats a la vez, un turno tarda como si releyera el prompt entero | `pick_slot` puede mandar el turno al slot del otro chat (`serve/server.py:1368-1381`). Solo velocidad | `dd08a7d0` (choca con la cesión de nuestros commits del 2026-10-07) | Sin medir |
| Después de un error del motor, las peticiones quedan colgadas hasta 5 minutos | El servidor retiene las líneas de control hasta 300 s tras un `ERR` (`serve/server.py:1316-1323`) | `ef6c18a1` + `c5fa1b58` | No: pi no registra ningún error salvo cancelaciones |
| Al pasar los 262144 tokens, pi muestra un error en vez de compactar la conversación | El texto del error de contexto excedido no coincide con los patrones de pi (`serve/server.py:2920-2923`) | `2bf1d07c` | No |
| Las imágenes fallan con el servidor arriba más de 10 días | `systemd-tmpfiles` limpia `/tmp` a los 10 días (`/usr/lib/tmpfiles.d/tmp.conf:11`), y el directorio de trabajo de `strata-vision` está ahí (`serve/server.py:1690`) | Una línea `x /tmp/strata-vision-*` en `/etc/tmpfiles.d`. Requiere privilegios: la aplica el Director | No |
| Una petición de 1 token con una imagen en el historial falla con un 400, después de 300 s | Con dos slots, el servidor manda al camino solo (`GENI`) solo las peticiones con más de un token por generar (`serve/server.py:1141`, `alone and left > 1`). Las demás entran a un slot con `BGENI`, que el motor rechaza cuando hay imágenes (`bf7f443b`), y el `ERR` retiene la petición 300 s | Mandar al camino solo toda petición con imágenes, pida los tokens que pida (`generate_batched`) | No: pi siempre pide más de un token. Lo encontró la compuerta del top-k (2026-11-02, `topk/gates-run1.out`) |
| Respuestas peores sobre fotos JPEG que sobre la misma foto en PNG | El decodificador JPEG de mtmd (stb_image, `mtmd-helper.cpp:408`) no da los mismos píxeles que PIL, el de la referencia: 2-3 niveles como máximo, en el 2-6 % de los píxeles. En 2 de los 3 JPEG de pi, el token más afectado queda en 0.46-0.58 de similitud con la referencia; la media, en 0.995-0.999 (`image-preprocessing-2026-11-01.md`, sección 3) | Decodificar los JPEG con PIL en el servidor de Strata y pasarle un PNG a `strata-vision` (ya convierte otros formatos, `serve/server.py:1778-1810`). Se verifica con `photo_check.py` | Medido en el codificador; ningún síntoma en las respuestas |
| La misma conversación da otras respuestas en otra carga del motor (o más adelante en la misma), después de un turno cuya respuesta se cortó por `max_tokens` | La política de drafts elige el tamaño de las ventanas de lookup y de cadena con tiempos de reloj medidos (`src/spec/draft_policy.cpp`, `observe(..., round_ms)`), así que la cantidad de drafts varía entre corridas; la salida no. El turno siguiente a un corte a veces reutiliza la respuesta desde el caché y a veces la relee desde el punto de control del límite del turno, y releer redondea distinto (hipótesis: depende de cómo terminó la última ventana) | Sin diseñar. Candidatos: dejar el estado del motor en los tokens devueltos al cortar por longitud; para las compuertas, no cortar respuestas a mitad de una conversación comparada entre cargas | En pruebas (`decsel/chain-ab.sh`, 2026-11-04). En pi, 3 de 5905 respuestas de qwen38flash terminaron por longitud |

- **Menores con el mismo veredicto:** #1620 (cancelar conserva la conversación), #1666, #1700 (el borde del
  contexto), `940011da` y `53f72571` (caché de conversaciones), `f332a855` (guarda de logits no finitos).
- **Cómo aplicarlos:** los cherry-picks se simularon contra `c195c0c4`, el HEAD de entonces
  (`upstream-review-2026-10-20.md`, sección 3); producción avanzó después a `9c012d0b` (`qsa-topk-2026-11-02.md`).
  `e372d4ff`, `a6830008` y `2bf1d07c` aplican limpios; el resto lleva conflictos chicos, a mano. Las compuertas son
  las de la sección 10 de `strata-evaluation-2026-10-07.md`, con el cliente Vulkan de ruido.
- **Detalle:** `upstream-review-2026-10-20.md` (sección 1.3) y `upstream-review-2026-10-24.md` (secciones 1.1 y 3).

## llama.cpp (respaldo de qwen38flash, `ENGINE=llama`, y los modelos de llama-server)

Desde el 2026-11-19 el código sigue a upstream más seis parches del fork (`upstream-sync-2026-11-19.md`), desde el
2026-11-21 los de Inkling 2 Mini (`inkling-2-mini-2026-11-21.md`) y, desde el 2026-11-22, el marcador de medios de
cada petición (sección 9.6 de ese documento).

| Síntoma | Causa | Arreglo |
|---|---|---|
| Con `ENGINE=llama` y MTP, qwen38flash no carga: `check_tensor_dims: tensor 'blk.48.nextn.hc_head_norm.weight' not found` (2026-11-19, la prueba de humo de la sincronización) | El MTP de qwen4exp de upstream (`c061df198`, #29761) lleva la cabeza de hyper-connections en la capa NextN (`nextn.hc_head_norm/down/up`); el draft externo `Qwen3.8-Flash-Next-MTP-GGUF` lo exportó el fork para su propio puerto del MTP, sin esos tensores | Volver a exportar la cabeza MTP del checkpoint oficial con el conversor de upstream. Mientras tanto, `ENGINE=llama MTP=0` carga y responde (32.5 t/s a 32k, determinista) |
| Con inkling2mini, la misma predicción sobre el mismo caché da logits distintos, en el tamaño del redondeo, según el camino de su último token: decodificado a continuación del caché, o releído después de restaurar un checkpoint (un prompt que ya está entero en el caché). KLD de 8e-7 o menos en 56 de 60 zonas y hasta 1.9e-3 en zonas sueltas; dos restauraciones difieren entre sí en una zona de diez con ubatch 512. Medido con la sonda (2026-11-22, `inkling-2-mini-2026-11-21.md`, sección 9.3); ningún síntoma en las respuestas, y el mismo camino repite idéntico | El anillo de las capas con ventana (la ventana más un ubatch, `llama-kv-cache-iswa.cpp`): `llama_kv_cache::find_slot` reparte las celdas desde la cabeza del anillo y `state_read_meta`, al restaurar, limpia la secuencia y las pide de nuevo. La ventana queda en otras celdas según la historia, y la atención suma en orden de celda. Es código de upstream: su server puro lo muestra igual. Las zonas sueltas no están explicadas | Sin diseñar. Candidatos: un anillo indexado por posición, o restaurar en un orden fijo de celdas. Hay que revisarlos con el caché unificado, con las celdas que comparten posición y con `seq_add`. Se verifica con `depth_equiv.py` y `reread_detail.py` |
| Con dos slots activos a la vez en inkling2mini, los logits de una petición dependen, en el tamaño del redondeo, de si su token comparte ubatch con el del otro slot y de con cuál. En el slot 1, dos relecturas de una misma predicción difirieron en las 10 zonas (KLD de 2e-8 a 2e-3); en el slot 0, que leyó siempre solo, fueron idénticas bit a bit. Las respuestas no son reproducibles bit a bit; la distancia a upstream no cambia (2026-11-22, sección 9.4). Un segundo caso, visto a nivel de op y no en el modelo: con los dos slots leyendo prompt en un mismo ubatch (64 tokens o más de cada uno), la atención global de un stream da otros bytes que con ese stream solo (`fa_bits`, sección 9.7) | El caché no unificado tiene un stream por slot, y la memoria híbrida parte el lote con el `split_equal` secuencial (`llama-memory-hybrid-iswa.cpp`): un token de cada stream en un mismo ubatch, donde la atención recorre la mayor cantidad de celdas de los dos (`llama_kv_cache::get_n_kv`). El reparto es de upstream; `--decode-share`, del fork, hace que compartan en más casos. El mecanismo en los kernels no está identificado, ni si upstream puro lo muestra. Ninguna compuerta de determinismo lo ve: todas usan un slot activo a la vez. El del segundo caso sí: K/V q8_0 de dos streams de un caché sin llenar no son contiguos, la copia f16 del backend no los toma y la atención corre en el kernel sin voltear, que redondea distinto que el volteado de un stream solo | Sin diseñar: primero la medición controlada de la sección 9.4, con producción descargada. Candidato: un ubatch por stream cuando el caché no es unificado, con un costo de velocidad sin medir. El segundo caso lo quita la rama local `vk-fa-flip-q8` (`9bad70a63`): el kernel volteado lee q8_0 en el lugar. Sin adoptar: falta su medición en el modelo, con producción descargada (sección 9.7) |
| Una etiqueta de control de la plantilla citada en el texto del cliente (un mensaje, un archivo leído, el resultado de una herramienta) llega al modelo como el token de control: el modelo ve cerrarse el mensaje o abrirse otro turno dentro del texto. En inkling2mini, `<\|end_message\|>`, `<\|message_model\|>` y `<\|content_thinking\|>` citadas en un mensaje quedan cada una en un solo token (ids 200010, 200001, 200008), y el contador del chat da los 28 tokens de esa tokenización (2026-11-22, en producción y sin usar slots: `special_in_text.py`, `runs/special-in-text.out`). Ocurrió en la prueba del Director ese día: el resultado de una herramienta que leyó `/props` trae la plantilla, 26 apariciones de sus 12 etiquetas, y las respuestas siguientes de esa sesión llevan esos tokens de control en el prompt (`session_tags.py`, solo conteos). El efecto en las respuestas no se midió | El server tokeniza el prompt de chat ya renderizado con `parse_special` en todas sus partes de texto, las de la plantilla y las del cliente por igual (`process_mtmd_prompt`, y `tokenize_input_prompts` sin mmproj, desde `handle_completions_impl`). Es código de upstream, para todo modelo. Apareció al revisar el marcador de medios (sección 9.6 de `inkling-2-mini-2026-11-21.md`) | Sin diseñar. Candidato: tokenizar por partes, el texto de la plantilla con `parse_special` y el del cliente sin él. El prompt renderizado ya sale en partes marcadas (`common_chat_prompt`, `is_input`) y el server las aplana: es el paso que falta del issue #28249 de upstream, cuyo PR #32033 es un borrador. Cambia los tokens de todo prompt que cite una etiqueta: pide su ventana de medición y la decisión del Director, pendiente (sección 9.6) |
| Una petición con archivos por un camino crudo falla con 400 (`number of media markers in the prompt (N) does not match number of media files (M)`) cuando su texto contiene el marcador que publica `/props`: `/completion`, `/infill` y `/embeddings` con `prompt_string` y `multimodal_data`, y `/rerank` con imágenes. Medido con un marcador de más en `/completion` (`raw/mismatch` de `marker_check.py`, 2026-11-22). Los caminos de chat no lo tienen desde `9a62394d4`, y pi usa `/v1/chat/completions` | En los caminos crudos el marcador publicado es el mecanismo con que el cliente ubica sus archivos (`tokenize_input_subprompt` llama a `process_mtmd_prompt` con ese marcador; `format_prompt_rerank` agrega uno por archivo), y el server no distingue el que puso el cliente del que venía dentro de un texto. Es código de upstream | Sin diseñar. Candidatos: que la petición cruda declare su marcador junto a `multimodal_data`, y que el rerank escriba uno propio de la petición, como el chat (`server_prompt_media`). El primero cambia el contrato de esos caminos |

- El respaldo con `ENGINE=llama` corre ahora el qwen4exp de upstream, no el del fork: sin la FA dispersa ni el PLE lazy
  del fork, ni sus compuertas de historia y de calidad. La prueba de humo a 32k sin MTP pasó razonamiento, tool call y
  determinismo; producción (262144 × 2 slots) con este build no se midió.

## Arreglados

Una fila que se arregla pasa aquí con su commit, la fecha y el síntoma que lo disparó.

Por la sincronización con upstream del 2026-11-19 (merge `705d46bf4`): los arreglos que esta sección de llama.cpp
registraba ya estaban en upstream, y entraron con el árbol. Ninguno lo disparó un síntoma de producción.

| Síntoma | Arreglo (upstream) |
|---|---|
| Con MTP, la memoria crece con cada checkpoint: cada uno copiaba el KV entero del draft (en `apex35b`, hasta 13.6 GiB a 262k con dos slots) | `e467e5b5e`, `PARTIAL_ONLY` en el caché KV simple |
| Con dos prompts a la vez, la aceptación del draft MTP caía: el hook leía las filas NextN por índice de lote | `4453b535f` |
| Tokens aceptados de un draft después del fin de la respuesta | `d280808f5` |
| La compuerta de historia con drafts comparaba solo el token elegido en los tokens aceptados | `ead9f3205` |
| La selección QSA tomaba celdas del bloque 513 | `66e0c17ee` |
| Sin síntoma propio: `23b0202a1`, `8e1642198`, `1a3011cc0`, `436f6f89e` | en upstream |

Disparado por la prueba del build instalado (2026-11-19):

| Síntoma | Arreglo |
|---|---|
| Con DFlash2, toda petición con imagen falla con HTTP 500 (`inconsistent sequence positions` en el contexto del draft) | El draft con `dflash.rope.dimension_sections` (`upstream-sync-2026-11-19.md`, sección 5) |

Disparado por la prueba del Director con inkling2mini (2026-11-22):

| Síntoma | Arreglo |
|---|---|
| Toda petición de chat falla con 400 `Failed to tokenize prompt` desde que el historial contiene el marcador de medios que publica `/props`: el agente lo leyó con una herramienta | `9a62394d4`: el marcador es de cada petición y no se publica (`inkling-2-mini-2026-11-21.md`, sección 9.6) |

