# El preprocesado de imágenes de Qwen-VL como la referencia (2026-11-01)

`strata-vision`, el codificador de imágenes de producción, usa el preprocesado del mtmd de este fork
(`strata/tools/vision/CMakeLists.txt:41-44`). Para Qwen-VL ese preprocesado se apartaba de la referencia, y el arreglo
de upstream `f711c8f30` (#30736) lo corrige. La revisión del 2026-10-20 lo marcó para tomar ya
(`upstream-review-2026-10-20.md`, sección 0, punto 1). Se tomó el 2026-11-01.

## 0. Resultado

**Adoptado: el codificador ahora ve lo mismo que ve la referencia, y el resto de producción no cambia.**

- **El defecto lo teníamos.** De las 32 imágenes que pi le mandó a qwen38flash entre el 2026-10-07 y el 10-31, 18
  entraban distintas a la referencia:
  - 17 con una franja negra de 1 a 28 px;
  - 5 con otra cantidad de tokens, 4 de ellas también con franja.
  
  Solo las 14 cuadradas entraban iguales.
- **Con el arreglo, las 43 imágenes probadas (32 de pi y 11 sintéticas) tienen la cuadrícula de tokens de la
  referencia.** Antes, 36 de 43.
- **Con el arreglo, el codificador da lo mismo que con el preprocesado de referencia:** el token más afectado queda en
  una similitud coseno de 1.0000 en todas las PNG. Sin el arreglo, en las imágenes con franja, el token más afectado
  caía a 0.13-0.97 en las de pi y a 0.22-0.27 en las sintéticas. El autor de upstream midió 0.04-0.30 contra
  transformers en Qwen3.5.
- **Queda una divergencia aparte, en los JPEG** (sección 3). El decodificador JPEG de mtmd (stb_image) no da los mismos
  píxeles que PIL. En 2 de los 3 JPEG de pi el token más afectado queda en 0.46-0.58, con una media de 0.995-0.999. El
  redimensionado no tiene nada que ver: con el JPEG decodificado por PIL, el arreglo da 1.0000. No se observó ningún
  síntoma y queda en `latent-defects.md`.
- **El motor quedó idéntico.** La compilación recompiló también el motor de Strata desde el mismo commit (`c195c0c4`).
  Sus filas de calidad, las respuestas de imagen y las de la compuerta de historia son idénticas byte a byte a las de
  la cadena del 2026-10-31.

## 1. El defecto

| | Fork antes del arreglo | Referencia (`Qwen2VLImageProcessorFast`) |
|---|---|---|
| Tamaño objetivo | `smart_resize`: múltiplos de 32, redondeo con `std::round` (`mtmd-image.cpp:130`) | El mismo `smart_resize`, con el `round()` de Python (al par) |
| Cómo llega la imagen a ese tamaño | Conserva la proporción, la encaja adentro y rellena el resto con negro (`PAD_CEIL`, `clip-model.h:67`) | La estira directo, sin relleno |

- Un lado que no es múltiplo de 32 deja una franja negra.
- Un cociente que termina en .5 da una fila o una columna de tokens de más: 2000×1125 daba 63×35 tokens en lugar de
  62×35.
- El arreglo pone `PAD_NONE` y el redondeo al par para los proyectores Qwen2-VL, Qwen2.5-VL y Qwen3-VL (el nuestro),
  entre otros. Toca 5 archivos.
  - Al aplicarlo hubo un conflicto en `tests/test-mtmd-impl.cpp`. El contexto de upstream incluía un test de
    `longest_edge` que el fork no tiene; se tomó solo el test nuevo del commit.
  - `edc353842` (el resize de torchvision), que la revisión del 10-24 sugería mirar junto con este arreglo, es de LFM2
    y no toca Qwen.

## 2. Compuertas

Con producción descargada, todo por `gate.py`. La cadena es `~/ProjectHub/llama.cpp-evidence/imgfix/chain.sh`.

| Compuerta | Resultado |
|---|---|
| **Codificador solo** (`vision_cmp.py`): un build base de `strata-vision` sin el arreglo y el nuevo, sobre 11 imágenes sintéticas y las 32 de pi. La referencia es cada imagen redimensionada por PIL (bicúbico) al tamaño de `smart_resize`, que ningún build vuelve a redimensionar | Cuadrícula de la referencia: 43 de 43 (base: 36). Dos codificaciones seguidas idénticas: 43 de 43. La entrada ya redimensionada da lo mismo en los dos builds: 43 de 43, así que el codificador no cambió. Las 4 imágenes sintéticas alineadas a 32, idénticas entre base y nuevo. Token más afectado contra la referencia: 1.0000 en todas las PNG |
| **Motor** (`band_logpos.py` a 65536, bandas wa448, wb3008 y cr6016, con el cliente Vulkan de ruido) | Las filas de log-probs, idénticas byte a byte a las de la cadena del 2026-10-31; las 8 relecturas de cada banda, iguales |
| **Imágenes de punta a punta** (servidor de producción a 131072): texto → imagen → texto (`img_test.py`) e `img_big.py` a 896, 1536 y 2048 | Respuestas idénticas a las del 2026-10-31 (son tamaños alineados a 32, que el arreglo no toca). 2048×2048 en 9.3 s |
| **Tamaños no alineados**: `img_big.py` a 1920×1080 y 1170×2532 | Leen la barra superior, el texto de abajo y las figuras; el servidor queda sano |
| **Historia**: un servidor recién cargado contra uno que procesó antes la imagen de 2048×2048, con drafts y sin ellos | Idénticas (516 y 515 filas) y respuestas iguales; también idénticas a las del 2026-10-31 |

## 3. Los JPEG

`photo_check.py` codifica cada JPEG de tres formas con el build nuevo: el archivo (stb_image lo decodifica), la
decodificación de PIL guardada en PNG a su tamaño (el arreglo la redimensiona) y la referencia.

| Imagen | JPEG contra PNG de PIL: decodificador (peor token / media) | PNG de PIL contra la referencia: redimensionado (peor token / media) |
|---|---|---|
| 2000×1125 | 0.4589 / 0.9953 | 1.0000 / 1.0000 |
| 1333×2000 | 0.5777 / 0.9990 | 1.0000 / 1.0000 |
| 480×270 | 0.9919 / 0.9995 | 1.0000 / 1.0000 |

- Los píxeles difieren en 2-3 niveles como máximo, en el 2-6 % de los píxeles: es la precisión de la IDCT y el
  sobremuestreo del croma de cada decodificador.
- El peor token no es uno de norma casi nula (2.25 contra una mediana de 1.36), así que la diferencia es real, aunque
  la media sea de 0.995-0.999.
- Afecta a los dos builds por igual y es anterior al arreglo.
- El efecto sobre las respuestas no se midió y no hay ningún síntoma en producción: queda registrado en
  `latent-defects.md`, con el arreglo posible.

## 4. Fuera de este cambio

- El tope de tokens por imagen es 4096 (`--max-tokens` del codificador); el de la referencia, 16384. Una foto de
  3024×4032 da 4015 tokens aquí y 11844 en la referencia (`upstream-review-2026-10-20.md`). Sin propuesta.
- El llama-server de respaldo no se recompiló: el instalado es `dac7b4e4c`, que no está en master
  (`latent-defects.md`).

## Evidencia

`~/ProjectHub/llama.cpp-evidence/imgfix/`:
- `vision-synth.tsv`, `vision-pi.tsv`: por imagen, las cuadrículas y los cosenos. Las imágenes de pi no se copian:
  solo sus cifras.
- `photo.tsv`: los JPEG.
- `chain.out`, `same_as.out`: las compuertas y la comparación con el 2026-10-31.
- `build-base.sh`, `build-vision.out`: los builds.
