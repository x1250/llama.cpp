# gufo en el Strix Halo: ¿más rápido que Strata? (2026-10-31)

Revisión de escritorio de [gufo](https://github.com/gufo-org/gufo) (v0.28.0, commit `192b75a8`, 2026-10-30), pedida por
el Director: "se supone que es más rápido que Strata". Sin GPU ni descargas de modelos: el código y la documentación de
gufo, los números publicados de gufo y de Strata, y el log del motor de producción. Copias de lo citado en
`~/ProjectHub/llama.cpp-evidence/gufo/`.

## 0. Resultado

**Hoy gufo no puede reemplazar a Strata en producción, y con lo publicado no hay base para decir que sería más rápido en
nuestro uso.**

- **No hay ninguna medición de gufo contra Strata.** No la hay en la máquina de nadie, ni en el repositorio, ni en el
  hilo de Framework, ni en los hilos de Reddit que los comparan (sección 6). Las tablas de gufo comparan contra un
  llama.cpp de septiembre.
- **gufo lee el prompt más rápido; en el decode los números publicados no lo ponen arriba.** Con los mismos pesos
  (UD-Q4_K_XL), gufo publica 1481-1801 t/s de lectura de prompt; Strata upstream, 1104-1252 t/s en el promedio de un
  prompt entero. En el decode greedy publican 36-40 t/s contra 42-53 t/s de Strata. Son prompts y equipos distintos
  (sección 2).
- **En producción manda el decode.** En 2263 peticiones (2026-10-07 al 30), el motor pasó 370 min decodificando y 74 min
  leyendo prompts: 83 % contra 17 %. Aunque gufo leyera los prompts en tiempo cero, necesitaría al menos el 83 % del
  decode de Strata para empatar (sección 3).
- **No carga nuestro GGUF** (sección 4.1). Sus expertos tienen que estar en Q4_K, Q5_K, Q6_K, Q5_1 o Q8_0, y los nuestros
  están en IQ3_S e IQ4_NL.
- **Con el único GGUF que soporta, no cabe en la configuración de producción** (sección 4.2). A 262144 × 2 con MTP y
  visión la estimación da ~94 GiB antes de los buffers de cómputo, y el techo es ~91 GiB.

## 1. Qué es

- **El motor:** licencia MIT, escrito para gfx1151, con kernels HIP propios por modelo. Corre sobre ROCm 10.1 (Nix
  fijado, o la imagen de contenedor `ghcr.io/gufo-org/toolboxes/gufo-runtime`); su `QUALITY.md` del 2026-10-23 también
  lo mide con ROCm 7.2.3. Trae un parche de ROCr para que la compactación de memoria del host no desaloje sus
  colas.
- **Los modelos:** Qwen3.8 27B, Qwen3.8 Flash-Next, DeepSeek V4 Flash, ASR, TTS, imagen y video.
- **Flash-Next:**
  - el único destino soportado es el UD-Q4_K_XL de Unsloth: 4 shards, 111.4 GB, revisión `38bb39ee`;
  - MTP con el predictor shared-Q8_0 (2.79 GB) y el codificador de imágenes BF16;
  - MTP adaptativo de 1 a 7 drafts más copias del contexto (lookup), con aceptación exacta bajo muestreo;
  - varias sesiones a la vez y caché de prompts en RAM y en disco;
  - API OpenAI: chat completions y responses, herramientas, imágenes, salida estructurada y streaming;
  - contexto nativo de 262144;
  - la tabla PLE se lee del disco con `O_DIRECT`, como en Strata (`ngram.cpp`).
- **Ritmo:** cuatro releases del 2026-10-26 al 30 (v0.25 a v0.28). El 2026-10-03 su autor escribió que las llamadas a
  herramientas a veces fallaban. La v0.26.0 (2026-10-28) trae una lectura de prompt más rápida con DeltaNet por chunks
  (#758) y, en la misma versión, retira la variante de esa lectura que perdía precisión (#850).

## 2. Los números publicados

Ninguna fila de esta tabla es una medición nuestra de gufo. Las de gufo son de su documentación (`BENCHMARKS.md`,
build `fec4759d` del 2026-10-18: HTTP, greedy, sin pensamiento, pp2048 y tg128 sobre un prefijo en caché). Las de
Strata upstream son de su `docs/STRIX_HALO.md` §6 (código 0.1.40 fusionado, gfx1151 128 GB, MTP con spec 4 y
lookup-chain 3, medianas). Las de producción son de esta máquina.

| | Prompt (t/s) | Decode greedy (t/s) | Decode con muestreo (t/s) |
|---|---|---|---|
| gufo, UD-Q4_K_XL, MTP | 1735 (prefijo 0), 1619 (32K), 1481 (131K); en AR, 1801 / 1673 / 1557 | texto mixto 35.8 (prefijo 0), 36.9 (8K), 40.0 (32K), 39.9 (131K); repetitivo 105-117 | valores por defecto del modelo, 128 tokens, poca profundidad, ROCm 7.2.3 / 10.1: código 62.7 / 70.5, prosa 48.7 / 47.9, pensamiento 54.0 / 45.7; edición de archivo de 32K con pensamiento y herramientas 68.8 |
| Strata upstream, UD-Q4_K_XL | prompt entero de 8K: 1104-1169; de 128K: 1211-1252 | 8K: 50.5-52.8; 128K: 41.6-43.3 | — |
| Producción (Strata + UD-small-nl3s) | 39.5k a 1324, 125k a 1338 (`strata-evaluation-2026-10-07.md` §3.4 y §10.12); lecturas reales de ≥ 16k a 1283 | corta 78.3; 40k 48.7 / 49.3 (§3.3, §10.12) | muestreador de pi con pensamiento: 61-62 de media a 40k (§3.5); el promedio real de 1.17 M tokens generados a profundidad mediana 93k es 52.9 |

- **El "repetitivo" no es un uso real.** Al modelo se le pide repetir una palabra muchas veces para que la aceptación del
  draft llegue al 100 %. Su autor lo llama el techo teórico (hilo de Framework, 2026-09-26).
- **Lo que cambia entre las filas:** el contenido del prompt, la aceptación de los drafts, la profundidad y el equipo.
  Lo que se puede leer es el orden de magnitud. En la lectura de prompt gufo está +30-50 % arriba con los mismos pesos.
  En el decode greedy está entre el 70 % y el 96 % de Strata (36.9 contra 50.5-52.8 a 8K, 39.9 contra 41.6-43.3 a
  128K). Con muestreo, sus cifras (46-71 t/s a poca profundidad) caen en el mismo rango que las nuestras (59-67 t/s en
  la corta y a 40k; 52.9 reales a 93k): con lo publicado no se pueden ordenar.
- **Las lecturas cortas son el punto débil de Strata.** En el log de producción, las lecturas de menos de 128 tokens
  promedian 108 t/s (~0.44 s fijos por petición) y las de 128-1023, 286 t/s. gufo lee 2048 tokens a 1480-1800 t/s a
  cualquier profundidad.

## 3. Lo que pesa en nuestro uso

`engine_log_split.py` sobre `~/.strix-engine.log` (2263 peticiones; salida en `engine_log_split.out`):

| Lecturas | Peticiones | Tokens | Tiempo | t/s |
|---|---|---|---|---|
| < 128 | 1172 | 55575 | 8.6 min | 108 |
| 128-1023 | 709 | 286286 | 16.7 min | 286 |
| 1024-4095 | 225 | 423826 | 8.6 min | 823 |
| 4096-16383 | 122 | 979734 | 13.8 min | 1184 |
| ≥ 16384 | 35 | 2030098 | 26.4 min | 1283 |
| **Total lectura** | | | **74.0 min (17 %)** | |
| **Decode** | | 1173348 generados | **369.9 min (83 %)** | 52.9 |

El prefijo reutilizado tiene una mediana de 93148 tokens y un p90 de 208299.

- **Cuánto decode necesitaría gufo para empatar.** Con lecturas instantáneas, el tiempo total sería
  369.9 / d, donde d es el decode de gufo como fracción del de Strata. Empata con d ≥ 369.9 / 443.9 = 0.83. Con
  lecturas 2× más rápidas (37 min) empata con d ≥ 0.91.
- **Dónde caen los números publicados.** Greedy con los mismos pesos: 0.70-0.96. Con muestreo no se pueden ordenar
  (sección 2). El resultado depende de una medición que nadie hizo: el decode con el muestreador de pi a ~93k de
  profundidad, en la misma máquina.

## 4. Lo que impide usarlo hoy

### 4.1 Los formatos

El cargador (`src/models/qwen38_flash_next/weights.cpp`, copia en la evidencia) acepta solo estos tipos:

| Familia | Tipos aceptados |
|---|---|
| Expertos | Q4_K, Q5_K, Q6_K, Q5_1, Q8_0 |
| Denso | Q8_0, BF16, F16, F32 (el embedding y la cabeza también Q6_K) |
| Hyper-connections | Q8_0, BF16, F16, F32 |
| Tabla PLE | IQ4_NL, BF16 |

Nuestro GGUF tiene gate/up de los expertos en IQ3_S, down en IQ4_NL y las hyper-connections en IQ4_NL, así que no
carga. Hay dos caminos y ninguno es corto:

- kernels de gufo para IQ3_S e IQ4_NL. Un usuario del hilo de Framework agregó IQ4_NL en el down de los expertos en su
  propio fork; no está en gufo;
- un GGUF compatible. Es una transformación del modelo y la decide el Director.

### 4.2 La memoria con la configuración de producción

Estimación con los tamaños de los archivos:

| Componente | GiB |
|---|---|
| UD-Q4_K_XL residente: los 103.8 GiB del archivo menos la tabla PLE, que queda en el disco (~27 GiB en IQ4_NL; en BF16 no cabría en el archivo) | ~77 |
| KV: 2 × 262144 × 27.5 KiB (su cifra por token con MTP; la admisión reserva la capacidad completa) | 13.75 |
| Predictor MTP shared-Q8_0 | 2.6 |
| Codificador de imágenes | 0.84 |
| **Total, antes de los buffers de cómputo y del caché de prompts** | **~94** |

El techo es ~91 GiB: los ~111 GiB que deja el escritorio, menos el margen de 20 GiB (`strata-evaluation-2026-10-07.md`
§10.12). Su propia medición, con una sesión, 133121 de contexto y sin MTP, ya da 88.78 GiB de pico en HIP. Bajar el
contexto o los slots de producción está fuera de la orden vigente (262144 × 2).

### 4.3 El entorno

- **Instalación:** la imagen de contenedor (Docker está instalado y Podman no) o un build con Nix (está instalado).
  Cualquiera de las dos trae su propio ROCm, aparte del 7.14.1 de producción.
- **memlock:** el límite duro es 8 MB. Con ese límite gufo no puede fijar sus buffers de host a largo plazo y vuelve al
  registro normal de HIP, sin la protección contra la compactación. Subir el límite es un cambio privilegiado.
- **Mensajes de sistema a mitad de la conversación:** gufo los mueve al principio y vuelve a leer todo el historial (por
  eso no soporta Claude Code). En 3 de las 173 sesiones de pi con qwen38flash hay un mensaje de sistema después del
  primer mensaje de otro tipo. Lo vi en el registro de la sesión; no verifiqué cómo lo serializa el proveedor.

## 5. Lo que gufo hace mejor

- **Dos peticiones a la vez.** El decode conjunto de gufo con MTP sube de 35.7 t/s con una sesión a 61.3 con dos. Strata,
  con dos slots, rinde menos que llama.cpp: 42 contra 57 t/s (`strata-evaluation-2026-10-07.md` §0 y §10.9).
- **Las lecturas cortas.** Para lo que hoy lee a 108-286 t/s, la revisión del 2026-10-20 dejó propuestas dos variables de
  Strata: `STRATA_PREFILL_STREAM_MIN=128` y `STRATA_PF_SWITCH_MIN_T=2048` (`upstream-review-2026-10-20.md`). Se
  adoptaron el mismo 2026-10-31 (`short-reads-2026-10-31.md`):
  - las lecturas de 300-800 tokens, −40 a −45 %;
  - las de 2.4k-3.7k, −10 a −14 %;
  - las de menos de ~136 tokens no cambian.

## 6. Lo que dicen los usuarios

Nada de esto está verificado aquí.

- **r/LocalLLaMA 1xaco01 (2026-10-19):**
  - Un usuario corrió Strata con cuants de Unsloth en dos equipos (uno CUDA, el Strix Halo). Reporta código con espacios
    faltantes entre letras y dígitos, que costaba ~20 % más pasos de agente.
  - Con gufo, 12 horas sin ese defecto.
  - Otro usuario dice que la velocidad que anuncia gufo es casi la de Strata en 2× RTX 3080.
  - Para comprobarlo aquí habría que revisar el código que escribe pi en sus sesiones. Nuestro GGUF y nuestro camino
    HIP son otros.
- **`STRATA_ROUTE_TAIL_SKIP`,** mencionado en el mismo hilo: salta expertos de rango bajo cuando no están en la VRAM. Es
  de la versión 0.1.42 en adelante, solo en CUDA y RDNA4 (`docs/DETAILS.md` de upstream). No aplica a nosotros: la base
  es 0.1.40.1 y todos los expertos están residentes.
- **r/StrixHalo 1x0qhha (2026-10-08):** con 64 GB, Strata con un GSQ-RCO IQ2_XS le pareció "definitely an upgrade over
  gufo or llama.cpp".
- **Hilo de Framework 85083:** un plan público de comparación de gufo contra llama.cpp en sesiones de agente largas
  (2026-09-26), todavía sin resultados.

## 7. La prueba que lo decidiría

Una A/B en esta máquina con los mismos pesos (UD-Q4_K_XL, que Strata también carga) y producción descargada:

- **Prompts:** los de `bench.py` (agente de 4.8k, prosa de 39.5k, código de 41.5k), con greedy y con el muestreador de
  pi.
- **Profundidad:** un decode a ~93k, la mediana de producción.
- **Concurrencia:** dos peticiones a la vez.
- **Contexto:** el que permita la compuerta, por ejemplo 131072 × 1.
- **Costo:**
  - descargar 111.4 GB (hay 510 GB libres);
  - la imagen de gufo;
  - una ventana de GPU de 1 a 2 horas.

Si gufo gana el decode con muestreo a profundidad, todavía quedan los bloqueos de 4.1 y 4.2 antes de producción.

## 8. Fuentes

- **gufo, commit `192b75a8`:** `README.md`, `docs/models/qwen3.8-flash-next/{README,BENCHMARKS,QUALITY}.md`,
  `src/models/qwen38_flash_next/{weights,engine,ngram}.cpp`, `CHANGELOG.md`. Copias en
  `~/ProjectHub/llama.cpp-evidence/gufo/gufo-docs-192b75a8/`.
- **Strata upstream `8183c6b6`:** `docs/STRIX_HALO.md` §6 y `docs/DETAILS.md` ("Missed experts in a verify window").
- **Tamaños de UD-Q4_K_XL y del MTP:** el árbol de `unsloth/Qwen3.8-Flash-Next-GGUF` en Hugging Face, revisión
  `38bb39ee`.
- **Foros:** `~/ProjectHub/llama.cpp-evidence/gufo/web/`: hilo de Framework 85083; Reddit r/LocalLLaMA 1xaco01,
  r/StrixHalo 1x0qhha y r/Qwen_AI 1xdhff5 (un video; solo el texto del post).
- **Log de producción:** `~/ProjectHub/llama.cpp-evidence/gufo/engine_log_split.py` y `engine_log_split.out`.
