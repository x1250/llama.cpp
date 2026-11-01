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

- **Menores con el mismo veredicto:** #1620 (cancelar conserva la conversación), #1666, #1700 (el borde del
  contexto), `940011da` y `53f72571` (caché de conversaciones), `f332a855` (guarda de logits no finitos).
- **Cómo aplicarlos:** los cherry-picks se simularon contra HEAD (`upstream-review-2026-10-20.md`, sección 3).
  `e372d4ff`, `a6830008` y `2bf1d07c` aplican limpios; el resto lleva conflictos chicos, a mano. Las compuertas son
  las de la sección 10 de `strata-evaluation-2026-10-07.md`, con el cliente Vulkan de ruido.
- **Detalle:** `upstream-review-2026-10-20.md` (sección 1.3) y `upstream-review-2026-10-24.md` (secciones 1.1 y 3).

## llama.cpp (respaldo, `ENGINE=llama`)

Producción no corre llama-server desde el 2026-10-07. Los "tomar" de `upstream-review-2026-10-20.md` (sección 2.2)
siguen sin aplicar. Los que tienen un síntoma visible:

| Síntoma | Causa | Arreglo |
|---|---|---|
| Con MTP, la memoria crece con cada checkpoint (~2 KB por token de contexto en cada uno, hasta 8 por slot) | Cada checkpoint copia el KV entero del draft: los flags `PARTIAL_ONLY` se ignoran (`llama-kv-cache.cpp:2105`) | El hunk de `llama-kv-cache.cpp` de `e467e5b5e` |
| Con dos prompts a la vez, la aceptación del draft MTP cae. La salida no cambia | El hook MTP lee las filas NextN por índice de lote y el fork las extrae en orden de ubatch (`common/speculative.cpp:1617, 1680`) | `4453b535f` |
| Tokens aceptados de un draft después del fin de la respuesta | `common/sampling.cpp:704` corta solo por desacuerdo | `d280808f5`, más el mismo corte en la variante `dists` del fork |
| La compuerta de historia con drafts compara solo el token elegido en los tokens aceptados | El servidor no devuelve log-probs de los tokens aceptados de un draft | `ead9f3205` |
| La selección QSA toma 1-3 celdas del bloque 513 en 3 de cada 4 posiciones con más de ~2k de contexto. Magnitud sin medir | Top-k de ancho 2051 sobre celdas, con la cola sesgada a 1e9, en lugar de los 512 mejores bloques más la cola de la referencia | `66e0c17ee` (sección 2.3 de esa revisión) |

- El resto de los "tomar" de esa sección (`23b0202a1`, `8e1642198`, `1a3011cc0`, `436f6f89e` y los arreglos de Vulkan
  y ggml) no tienen un síntoma propio en producción. Sus compuertas están en esa sección.
- El llama-server instalado (`~/.local/bin/llama-server`) es `dac7b4e4c`, un commit que no está en master. Antes de
  volver a usarlo como respaldo, se compila desde master y pasa sus compuertas (`CLAUDE.md`).

## Arreglados

Ninguno todavía. Una fila que se arregla pasa aquí con su commit, la fecha y el síntoma que lo disparó.
