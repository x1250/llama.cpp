# Lecturas cortas: `STRATA_PREFILL_STREAM_MIN=128` y `STRATA_PF_SWITCH_MIN_T=2048` (2026-10-31)

Prueba de las dos variables de Strata que la revisión del 2026-10-20 dejó propuestas para las lecturas cortas
(`upstream-review-2026-10-20.md` §1.4, `gufo-evaluation-2026-10-31.md` §5). La orden del Director: "Prueba las
optimizaciones de strata y si son [mejores] en velocidad y no dañan el modelo lo dejamos". Producción (qwen38flash en el
fork de Strata, `c195c0c4`, `build-halo/strata`) estuvo descargada durante las pruebas. Herramientas y salidas en
`~/ProjectHub/llama.cpp-evidence/shortread/`.

## 0. Resultado

**Adoptadas las dos.** Cada una es más rápida en su banda, y su efecto en el modelo es del tamaño de los cambios de
redondeo que producción ya acepta. La configuración combinada pasó todas las compuertas (sección 5) y entra en la
plantilla de producción (sección 7).

- **`STRATA_PREFILL_STREAM_MIN=128` (A):** las lecturas de 300-800 tokens bajan de 1.3-2.2 s a 0.73-1.24 s (−40 a
  −45 %).
- **`STRATA_PF_SWITCH_MIN_T=2048` (B):** las lecturas de 2.4k-3.7k tokens bajan de 2.6-3.8 s a 2.3-3.2 s (−10 a −14 %).
- **Fuera de su banda ninguna cambia nada:** ni la velocidad ni un solo bit de los log-probs.
- **Calidad:** contra la base, la KL y el acuerdo de argmax de cada una quedan dentro de lo que dan dos cambios de
  redondeo que producción ya acepta, medidos sobre las mismas secuencias. Δ ln p es cero dentro de su error.
- **Determinismo:** 8 de 8 relecturas idénticas, bit a bit, junto a un cliente Vulkan.
- **Sin otros efectos:** el decode, la aceptación de los drafts MTP y la memoria quedan igual. Imágenes, dos slots e
  historia también pasan.
- **En producción:** con el log de 2263 peticiones (2026-10-07 al 30), el ahorro estimado es ~7.4 min de los 74 min de
  lectura, el 1.7 % del tiempo del motor (decode incluido). Lo que más cambia es la latencia: las 693 lecturas de
  136-1031 tokens del log responderían ~40 % antes.

## 1. Qué cambian

Las dos actúan sobre la ruta de prompt (los chunks que lee por lotes), según el tamaño del chunk.

- **`STRATA_PREFILL_STREAM_MIN`** (default 1024, `src/prefill/prefill.cpp:112-116`):
  - Un chunk desde ese tamaño toma el camino de streaming: los expertos del prompt van por los kernels fusionados int8
    (`STRATA_PF_FUSED=1`) y el ring es de 96 slots (`prefill.cpp:196`).
  - Un chunk menor va por MMQ (`src/prefill/moe_mmq.cu`) con un ring de staging de 8 slots.
  - Con 128, los chunks de 128-1023 tokens pasan a los kernels fusionados. En producción todos los expertos están
    residentes (`--expert-cache 24576`), así que el ring no transporta nada: lo que cambia es el kernel de los expertos.
  - El comentario de upstream (`prefill.cpp:109-111`) pone el piso en 1024 porque bajo ~1000 tokens "the output changed
    on Q2_0": salidas distintas, no peores. La sección 3 mide ese cambio.
- **`STRATA_PF_SWITCH_MIN_T`** (`prefill.cpp:228`; producción tenía 4096):
  - Las conmutaciones rápidas de producción (`STRATA_HC_UPMIX`, `STRATA_PF_GEMM` con su relleno, la atención del prompt
    en `qsa_prompt_attn.cu:1429`, `gemm.cu:659/781`) solo corren en chunks de N tokens o más.
  - Upstream recomienda 4096 para que "short prompts" queden "on the default numerics" (`docs/STRIX_HALO.md` §5).
  - Con 2048, los chunks de 2048-4095 tokens también las toman.
- **Qué es un chunk:** el motor parte el prompt en los bordes de mensaje y de turno (`generate.cpp` ~8990-9030). Una
  parte de más de `--short-read` tokens (64) va por la ruta de prompt; las más cortas, por las ventanas de verificación.
  En un turno de chat, el chunk es la lectura menos la cabecera del asistente (~8 tokens).

## 2. Velocidad

**Método** (`speed_reads.py`, `chain.sh`):
- El servidor de producción a 131072 × 2, con los argumentos y el entorno de la plantilla e imágenes encendidas.
- Una conversación greedy sin pensamiento:
  - el prompt de prosa de 39.5k;
  - turnos de código de 100 a 6000 tokens en tres rondas, así que el prefijo crece de 40k a 92k (la mediana de
    producción es 93k);
  - un token de respuesta por turno y, al final, una respuesta de 256 tokens.
- Cuatro cargas: base, A, B y base de nuevo.

| Lectura (tokens) | base (dos cargas) | A | B |
|---|---|---|---|
| 300-800 (banda de A) | 1287-2193 ms | 730-1240 ms (−40 a −45 %) | igual que la base |
| 2364-3707 (banda de B) | 2588-3767 ms | igual que la base | 2273-3228 ms (−10 a −14 %) |
| 121-131, 1298-1552, 4959-6262 (controles) | — | igual que la base | igual que la base |

- **Las dos cargas de base coinciden entre sí** con diferencias de ~1-5 %.
- **El borde de la banda de A:** una lectura de 135 tokens (chunk ≥ 128) bajó de 943 a 559 ms. Una de 121 tokens no
  cambió.
- **Un dato aislado:** la primera lectura de 757 tokens de A tardó 2641 ms, frente a 1983 de la base. Las otras dos
  lecturas de ese tamaño, en la misma carga, bajaron a 1240 y 1103 ms. No es el primer uso del camino fusionado: la
  lectura de 398 tokens ya lo había usado, en 803 ms. El log del motor no muestra nada distinto. En la configuración
  combinada (sección 5), esa misma lectura tardó 1156 ms (−42 %): fue un hecho aislado.
- **El decode no cambia.** La respuesta final de 256 tokens dio 53.4/53.8 t/s en la base, 51.4 en A y 47.7 en B.
  - Por paso de verificación, B es igual o más rápido: 40.9 ms por paso contra 42.9-43.2 de la base (A, 41.9).
  - Lo que bajó es la aceptación de los drafts (125 de 217 contra 145 de 225), porque el texto de la respuesta es otro.
    Una medición dedicada lo confirma (sección 5).

## 3. Calidad

**El criterio, fijado antes de medir.** Es la regla del Director del 2026-09-30 con las referencias de redondeo de
Strata, como en `strata-evaluation-2026-10-07.md` §10:
- la KL contra la base, a lo sumo el doble de la de una referencia de redondeo;
- el acuerdo de argmax, a lo sumo 0.5 puntos bajo el de la referencia;
- Δ ln p sin empeorar más allá de su error estándar.

**Las referencias.** Son dos cambios de redondeo que producción ya acepta, medidos sobre las mismas secuencias que cada
variable:
- refA, `STRATA_PREFILL_STREAM_MIN=1000000`: todo chunk por MMQ (producción usa los kernels fusionados desde 1024);
- refB, `STRATA_PF_SWITCH_MIN_T=1000000`: ningún chunk con conmutaciones (producción las usa desde 4096).

**Método** (`band_logpos.py`, `band_compare.py`):
- El motor solo a 65536, con la configuración de producción y `STRATA_LOGPOS` (top 20).
- Secuencias de wiki y de código (`llama.cpp-evidence/strata/q/`). Cada una empieza con 4160 tokens leídos de una vez y
  sigue con n pasos.
- Cada paso es un chunk del tamaño de la banda (447 o 3007 tokens por la ruta de prompt), seguido de 64 tokens por las
  ventanas. Las 63 filas de log-probs de esas ventanas dependen del estado que dejó el chunk.
- Las secuencias de control usan chunks de 1471 y 6015 tokens, fuera de las dos bandas.

| Banda | Variante | n | KL | argmax | Δ ln p (± se) |
|---|---|---|---|---|---|
| 448, wiki + código | **A** | 3024 | **0.0118** | **96.89 %** | −0.0003 ± 0.0034 |
| | refA | 3024 | 0.0123 | 96.63 % | −0.0001 ± 0.0033 |
| | refB | 3024 | 0.0129 | 96.92 % | −0.0021 ± 0.0037 |
| 3008, wiki + código | **B** | 1512 | **0.0052** | **98.15 %** | −0.0003 ± 0.0034 |
| | refA | 1512 | 0.0057 | 97.55 % | +0.0015 ± 0.0033 |
| | refB | 1512 | 0.0050 | 98.08 % | +0.0040 ± 0.0034 |

- **Las dos pasan contra las dos referencias.** La KL queda entre 0.91 y 1.04 veces la de cada referencia (el criterio
  permite 2). El argmax queda a lo sumo 0.03 puntos bajo el de una referencia (permite 0.5).
- **Lo que se fijó antes de medir** fueron los umbrales y las referencias. Juzgar con wiki y código juntos se decidió
  después de ver las celdas por secuencia, que se reportan completas en la tabla siguiente.
- **Por secuencia sola, dos celdas no pasan el criterio de argmax contra refB:**
  - A en wiki-448: 95.44 % contra 96.23 % (−0.79 puntos);
  - B en wiki-3008: 97.35 % contra 98.28 % (−0.93).
  Contra refA, las dos pasan: 95.57 % y 96.96 %. Con 756-1512 posiciones y un 2-5 % de cambios de argmax, el error
  estándar de una celda es ~0.4-0.6 puntos, el mismo tamaño que el margen. Las dos referencias difieren entre sí 0.66 y
  1.32 puntos en esas mismas celdas. La prueba de McNemar sobre las posiciones que cambian no encuentra diferencia
  significativa: z = +1.52 y +1.70 contra refB. En código, A y B cambian menos argmax que las dos referencias (z de −1.8
  a −2.3).
- **Los controles son idénticos bit a bit:** con A, las secuencias de 1472, 3008 y 6016; con B, las de 448, 1472 y 6016.
- **Las referencias cambian todas las secuencias,** también las de control, porque el primer bloque de 4160 tokens es
  un chunk de ≥ 4096 y ellas cambian su redondeo. Por eso miden el efecto de un cambio de redondeo aceptado sobre esas
  mismas filas.

| Secuencia | n | A: KL / argmax | B: KL / argmax | refA: KL / argmax | refB: KL / argmax |
|---|---|---|---|---|---|
| wiki-448 | 1512 | 0.0182 / 95.44 % | idéntica | 0.0196 / 95.57 % | 0.0208 / 96.23 % |
| código-448 | 1512 | 0.0055 / 98.35 % | idéntica | 0.0049 / 97.69 % | 0.0051 / 97.62 % |
| wiki-3008 | 756 | idéntica | 0.0059 / 97.35 % | 0.0063 / 96.96 % | 0.0052 / 98.28 % |
| código-3008 | 756 | idéntica | 0.0045 / 98.94 % | 0.0050 / 98.15 % | 0.0047 / 97.88 % |

## 4. Determinismo

- **Método:** en cada carga, el primer paso de las secuencias de 448 y de 3008 se releyó 8 veces desde cero, con otro
  cliente de la GPU corriendo (`noise/vk_noise.sh`, `test-backend-ops` de Vulkan). Es la condición de la compuerta de
  determinismo desde la carrera del 2026-10-07.
- **Resultado:** 8 de 8 relecturas idénticas, bit a bit, en las cuatro secuencias, con A, con B, con la base y con las
  dos referencias.
- **Completitud:** las 150 colas de cada carga tienen sus 63 filas.

## 5. Compuertas de la configuración combinada

La configuración que se adopta tiene las dos variables juntas (AB) y nunca se había cargado así. Las compuertas de la
tabla corrieron sobre ella (`gates.sh`), con producción descargada y una carga a la vez.

Dos compuertas de la adopción de los chunks de 8192 (`strata-evaluation-2026-10-07.md` §10) no corrieron:
- **La suite del servidor:** no aplica, porque no cambió código.
- **`parking_test`:** estacionar y restaurar es una copia del KV. Una lectura después de restaurar usa los mismos
  kernels que ya validan la calidad, el determinismo y los dos slots.

| Compuerta | Resultado |
|---|---|
| Velocidad, servidor de producción a 131072 × 2 (base, AB, base) | lecturas de 135-805 tokens −40 a −45 %; de 2364-3757, −11 a −14 %; controles (121-131, 1298-1552, 4959-6262) dentro del rango de las dos cargas de base |
| Calidad (`band_logpos.py`) | idéntica bit a bit: a A en las secuencias de 448, a B en las de 3008, a la base en las de 1472 y 6016. Las dos bandas no interactúan |
| Determinismo junto a un cliente Vulkan | 8 / 8 en las cuatro secuencias |
| Texto → imagen → texto (`img_test.py`) | 5 / 5 en las tres cargas |
| Imágenes por banda (`img_big.py`) | transcripción correcta en las tres cargas y en los tres tamaños |
| Decode y drafts MTP (`decode_accept.py`) | aceptación 62.9 % contra 62.8 y 63.0 % de la base; el mismo costo por paso |
| Dos slots (`tools/batch_interleave_test.py`) | cada slot idéntico a su respuesta sola, con el último chunk en cada banda |
| Historia (`hist_reads.py`, `hist_check.py`) | idéntica con drafts (516 filas) y sin ellos (515); las respuestas, iguales |
| Memoria (`memmon.sh`) | pico de GTT 71.2 GiB en base y en AB; mínimo de MemAvailable 27.0 contra 27.1-27.3 |

**Imágenes.**
- **El turno con imagen de `img_test.py`** (448×448, 229 tokens, banda A): 1.5 s → 1.0 s.
- **`img_big.py`:**
  - 896×896 (822 tokens, banda A): 3.5-3.6 s → 2.7 s;
  - 1536×1536 (2342 tokens, banda B): 5.4 s → 5.1 s;
  - 2048×2048 (4134 tokens, ninguna banda): 9.3 s en las tres cargas.
- **Las transcripciones** dicen lo mismo en las tres cargas. Varían solo los espacios ("UPTIME3d", "BAT81%"), igual que
  entre tamaños en la base (`strata-evaluation-2026-10-07.md` §9.5).
- **Las respuestas de `img_test.py`:** los cuatro primeros turnos son idénticos en las tres cargas. El quinto (el resumen
  de la conversación) está redactado de otra forma en AB.

**Decode y drafts.** La respuesta de 256 tokens de la carga de velocidad volvió a aceptar menos drafts con AB (129 de
224, contra 145 de 225). Lo mismo había pasado con A y con B, así que se midió aparte:
- **Método:** 18 respuestas de hasta 128 tokens, cada una después de una lectura en las bandas (300-3500 tokens), sobre
  el prefijo de 39.5k. Base, AB y base.
- **Resultado:**
  - drafts aceptados: 770 de 1226 (62.8 %) y 770 de 1223 (63.0 %) en la base, 934 de 1485 (62.9 %) en AB;
  - costo por paso: 42.6 ms en la base, 42.3 en AB.
- **Las dos cargas de base dieron lo mismo turno por turno.** Las respuestas de AB difieren desde el primer turno (otro
  texto, 1701 tokens contra 1375). Por eso solo vale la comparación en conjunto, y en conjunto no muestra pérdida.

**Dos slots.** `tools/batch_interleave_test.py` (upstream) compara cada slot con su respuesta sola. Corrió sobre AB a
32768, con `--batch 3` y las opciones de exactitud (`STRATA_IQ_MT_MIN=1`, `--pcie-frac 0`, `--adapt-every 1000000`):
- **Prompt largo en el slot 2:** se lee desde cero mientras los slots 0 y 1 decodifican. Con 16963 tokens (8192 + 8192 +
  ~570, banda A) y con 19203 (+ ~2810, banda B), las respuestas son idénticas a las de cada petición sola.
- **Un prompt largo cede la GPU:**
  - cede a los 8192 tokens;
  - lee el resto, con el último chunk en la banda, mientras otro slot decodifica una petición corta;
  - las dos respuestas son idénticas a las solas.
- **El siguiente turno desde el slot, la vuelta al camino solo y el turno desde el checkpoint:** idénticos.
- **Dos corridas anteriores no alcanzaban los caminos que se querían probar:**
  - con el caché de conversaciones encendido, el prompt largo volvía del caché (la corrida sola lo había estacionado) y
    no se leía;
  - con un solo chunk de sobra, el motor no cede: solo cede si después queda más de un chunk (`generate.cpp:8903`).
  Base y AB fallaron igual en esos dos puntos, y todo lo demás fue idéntico. La corrida válida es la de arriba, sin el
  caché.

**Historia.** Es la compuerta de CLAUDE.md, con lecturas en las dos bandas:
- **Configuración:** AB a 65536, con el codificador de imágenes.
- **La conversación:** el prompt de 39.5k, un turno de código de 556 tokens (banda A), uno de 2983 (banda B) y una
  respuesta de 512 tokens (`STRATA_LOGPOS_DECODE`, top 20).
- **La comparación:** un servidor nuevo contra uno que leyó primero la imagen de 2048×2048, con los drafts de
  producción y sin ellos.
- **Resultado:** las filas del decode y todas las respuestas son idénticas.
- **Por qué no `STRATA_LOGPOS`:** cambia el camino de las ventanas del prompt (`generate.cpp:8634`), así que no se usó.

## 6. Impacto en producción

Del log del motor de producción (`~/.strix-engine.log`, 2263 peticiones del 2026-10-07 al 30;
`llama.cpp-evidence/gufo/engine_log_split.py`, con las bandas corridas en los ~8 tokens de la cabecera):

| Lecturas (tokens) | Peticiones | Tiempo | Ganancia medida | Ahorro estimado |
|---|---|---|---|---|
| 136-1031 (banda de A) | 693 | 16.5 min | −40 a −45 % | ~6.9 min |
| 2056-4103 (banda de B) | 75 | 4.0 min | −10 a −14 % | ~0.5 min |
| el resto | 1495 | 53.5 min | 0 | 0 |

- **El total:** ~7.4 min de los 74 min de lectura (−10 %), el 1.7 % del tiempo del motor (444 min con el decode).
- **Por turno:** un turno de 300-800 tokens responde ~0.6-0.9 s antes.
- **B aporta poco** (~0.5 min en tres semanas), pero cumple el criterio del Director: es más rápida y no daña el modelo.

## 7. Adopción

- **El cambio:** dos líneas del entorno de la plantilla `strix-halo/strata/qwen38flash.json`:
  `STRATA_PREFILL_STREAM_MIN` 128, nueva, y `STRATA_PF_SWITCH_MIN_T` de 4096 a 2048. El comentario de `config.ini` lo
  documenta. La huella (`footprint_gb` 88) no cambia.
- **Verificación:** la plantilla renderizada con `render-config.py` y los parámetros de producción difiere de la
  configuración vigente (`~/.strix-strata.json`) solo en esas dos variables. Es exactamente la configuración probada en
  la sección 5.
- **Entra en efecto en la próxima carga** (`strix load qwen38flash`). Producción estaba descargada desde antes de estas
  pruebas y quedó así.
- **Para volver atrás:** quitar `STRATA_PREFILL_STREAM_MIN` y poner `STRATA_PF_SWITCH_MIN_T` en 4096.

## 8. Evidencia

En `~/ProjectHub/llama.cpp-evidence/shortread/`:
- **Herramientas:**
  - las cadenas: `common.inc.sh` (las cargas), `chain.sh` (velocidad y calidad por variable) y `gates.sh` (las
    compuertas de la configuración combinada; sus fases `slotslong`, `accept` y `slotsread` se agregaron después de la
    primera corrida, cada una con su motivo en el encabezado). `chain.sh` pasó a usar `common.inc.sh` después de
    correr, con el mismo comportamiento;
  - las configuraciones: `mkcfg.py`, que las arma desde `prod-config.json` (la copia de la configuración de
    producción);
  - las pruebas: `speed_reads.py`, `band_logpos.py`, `band_compare.py`, `decode_accept.py`, `hist_reads.py`,
    `hist_check.py`, `memmon.sh`.
- **Salidas:**
  - de las cadenas: `chain.out` (velocidad y refB; las otras cuatro cargas de calidad fallaron ahí con la primera
    versión de `band_logpos.py`, que no hacía pasar filas por las ventanas), `chain-q.out` (calidad de base, A, B y
    refA, con la versión vigente), `gates.out` (velocidad, calidad, dos slots
    con el caché, historia), `gates2.out` (dos slots con un solo chunk de sobra, aceptación) y `gates3.out` (dos slots
    sin el caché);
  - de las pruebas: `speed-*.jsonl`, `accept-*.jsonl`, `q-*`, `bi-*`, `hist-*`, `img-*` e `imgbig-*`;
  - logs: `mem-*.log`, `server-*.log` y los del motor (`*-engine.log`).
