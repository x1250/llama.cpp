# Strata en el Strix Halo: velocidad, memoria, calidad y el pensamiento ajeno (2026-10-07)

Evaluación de [Strata v0.1.40](https://github.com/Niko1221/Strata) (motor propio para Qwen3.8-Flash-Next, HIP/ROCm,
gfx1151 experimental) frente al fork en producción, pedida por el Director: "no más lento que Strata, sin el bug, no más
de 58 GB residentes". Fork del motor: `x1250/Strata` (privado), clon en `~/ProjectHub/strata`, build con ROCm 7.14.1 de
TheRock en `~/rocm/install` (`~/dbg/strata/build.sh`). Herramientas y salidas: `~/dbg/strata/`.

## 0. Resultado

**Strata (fork, 3 cambios) con el GGUF UD-small decodifica 16-18 % más rápido que el llama.cpp de producción, lee el
prompt 2.2-2.5× más rápido y ocupa la misma memoria;** el GGUF UD-small además está más cerca del modelo (la mitad de
distancia a UD-IQ4_XS, perplejidad ~1 % mejor) y arranca el pensamiento en tema mucho más seguido.

| | llama.cpp, producción | Strata + UD-small |
|---|---|---|
| Prompt 40k / 125k | 618 / 537 t/s | 1379 / 1338 t/s |
| Decode medio, greedy sin pensamiento (corto, prosa y código 40k, dos turnos) | 49.7 t/s | 57.7 t/s (+16 %) |
| Decode medio, muestreador de pi con pensamiento (los mismos prompts) | 51.7 t/s | 61.2-61.8 t/s (+18 %) |
| Decode a 125k, greedy (primer / segundo turno) | 40.9 / 45.6 | 46.2 / 48.4 (con el GGUF de producción) |
| GTT a 65k de contexto | 66.7 GiB (mmproj incluido) | 66.2 GiB (sin codificador de imágenes) |
| Pesos sin la PLE (la definición de `config.ini`) | 56.5 GiB | 57.4 GiB |
| Fidelidad del GGUF: KL contra UD-IQ4_XS, medido en llama.cpp (wiki 4k / código 4k) | 0.156 / 0.073 | 0.085 / 0.036 |
| Primer pensamiento del incidente: en tema / genérico / ajeno | 57 / 43 / 0.4 % (1000) | 98 / 2 / 0 % (400) |

Los 3 cambios del fork (`x1250/Strata`, local): 310bfe4 (las páginas del archivo se liberan a medida que los
expertos llegan a la GPU: sin esto la carga lleva MemFree a cero), 2485459 (la lectura de las hyper-connections en el
IQ4_NL del GGUF: −6 ms por ventana) y la configuración (`--mtp-hnorm stream`, la norma del draft por stream como la
nuestra; un subconjunto de vocabulario del draft hecho con las salidas de pi; `--mtp-window 8192`). Configuración
reproducible: `~/dbg/strata/prod/strata-qwen38flash-strix.json` y `env.sh`.

**El pensamiento ajeno al contexto no es un defecto del motor y no se demuestra que baje:** aparece en los dos motores
a ~0.4 % en el contexto del incidente, y cero en 400 muestras con UD-small no alcanza para afirmar una mejora. Lo que
UD-small sí cambia, en los dos motores, es la fidelidad al modelo y el arranque genérico ("I need to investigate this
further…", de ~43 % a 2-17 %). Los tensores densos chicos de nuestra receta (routers, hyper-connections, compuertas de
la GDN, experto compartido, PLE key/value y embedding, cuantizados más que en UD; cuál de ellos pesa no está aislado)
alejan al modelo de UD; devolverlos a su formato de UD cuesta 0.93 GiB. La posición del incidente es un casi empate
entre "I" y "The" que la numérica de cada motor también mueve (sección 5), así que lo robusto es la fidelidad (sección 6).

**Pendiente para reemplazar producción (sección 8):** determinismo de Strata entre cargas, pi a través de su servidor
(plantilla y tool calls), codificador de imágenes en GPU, dos slots a 262144.

## 1. Cómo corre aquí sin duplicar los expertos ni usar la CPU

- Por defecto Strata copia todos los expertos a un arena de RAM fijada y además llena la caché de expertos de la GPU con
  `MemAvailable - 6 GiB` (`device_free_bytes`, `src/core/expert_cache.cpp`): en memoria unificada son dos copias.
  Los expertos que la caché no tiene los calcula la CPU (AVX-512): prohibido aquí (CLAUDE.md, dos apagones).
- Configuración usada: `--mmap-experts` (el GGUF leído en su lugar), `--expert-cache 24576` (los 48 × 512 expertos en
  la GPU, sin aciertos fallidos: la CPU no pasó de 23 % en ninguna corrida), `--adapt-swaps 0`, `--no-prefill-borrow`.
- Parche del fork (commit 310bfe4): `FileExpertSource::release` era solo de Windows; en Linux ahora desmapea
  (`MADV_DONTNEED`) y saca de la caché de páginas (`POSIX_FADV_DONTNEED`) cada experto apenas llega a la VRAM, con el
  mismo interruptor `STRATA_FILE_RELEASE=1`. Sin él, llenar 55 GiB de caché apila otros 55 GiB de caché de páginas: la
  firma del freeze #9. Con él: llenado en 19 s, MemFree estable en 19-20 GiB, la caché de páginas no crece.
- La tabla PLE (26.8 GiB) se lee del SSD con `O_DIRECT` en cada token: no ocupa RAM ni caché de páginas.

## 2. Los GGUF

- UD-IQ4_XS de Unsloth (`38bb39e`): expertos 55.4 GiB (gate/up IQ3_S salvo la capa 2 en IQ4_XS; down IQ4_NL salvo 5
  capas en Q8_0), denso 5.0 GiB.
- El de producción (`nl3s-hcdi-gu-rq8-oproj8`): expertos 53.5 GiB, denso 3.0 GiB. Strata lee el layout del checkpoint:
  `scripts/qwen4exp-split-merged.py` deshace las fusiones gate+up y down+inject (rangos de bytes). Strata no tiene
  kernels de GPU para gate/up en IQ4_NL (`STRATA_GU_FMTS`, `iq_kernels.cu`), y nuestra capa 2 es IQ4_NL/IQ4_NL:
  `scripts/qwen4exp-take-tensors.py` toma esa capa (gate/up IQ4_XS) del archivo de Unsloth. Resultado:
  `NL3S-strata/qwen4exp-nl3s-rq8-oproj8-l2xs.gguf`, pack `strata-packs/nl3s-rq8-oproj8-l2xs`.
- El pack de Strata (`iq_pack --compat-bf16`) redondea a BF16 las proyecciones que el motor lee así: 196 tensores en
  UD, 412 en el nuestro (hc, inject, routers y ssm_alpha/beta, que el nuestro guarda en IQ4_NL/Q8_0 y UD en Q8_0/F32).

## 3. Velocidad (mismos prompts, greedy, 256 tokens, contexto 65536, una carga por configuración)

`~/dbg/strata/bench.py`: el prompt de agente de 4.8k, prosa de 39.5k y código de 41.5k (los de los rigs del fork),
leídos enteros y luego una segunda pregunta en la misma conversación. Strata en su configuración rápida de Strix
(`--spec 4 --mtp --lookup-chain 3 --mtp-q4 all --prefill 16384`, KV int8, los interruptores de la sección 5 de su
`docs/STRIX_HALO.md`); llama.cpp con los argumentos de producción (MTP n-max 2, lookup, PLE lazy, visión), NP=1.

| | Strata + nuestro GGUF | Strata + UD-IQ4_XS | llama.cpp (producción) |
|---|---|---|---|
| Prompt 40k prosa / código | 1333 / 1387 t/s | 1388 / 1399 | 622 / 614 |
| Decode 40k prosa / código | 40.4 / 38.5 t/s | 37.1 / 39.4 | 45.0 / 45.7 |
| Decode segundo turno prosa / código | 40.8 / 41.8 | 43.1 / 40.7 | 46.9 / 50.7 |
| GTT pico | 65.3 GiB | 72.7 GiB | 66.7 GiB |

El prompt de Strata es 2.2× el nuestro; su decode, 10-17 % menor. Por ronda de verificación cuestan lo mismo (45 ms
contra 47 ms a 40k): la diferencia está en los tokens aceptados por ronda (1.83 contra 2.13; aceptación del draft 52 %
contra 57 %). Su draft MTP usa expertos Q2_0 (el runtime solo acepta ese formato), proyecciones Q4_0 (`--mtp-q4 all`) y
un subconjunto de 106k tokens que deja fuera el 6.6 % de lo que el modelo genera en las sesiones de pi.

### 3.1 Palancas del decode sin tocar el motor (barridos s1 y s2, `sweep.py`, una carga por configuración)

Decode en t/s, primer turno / segundo turno (drafts aceptados en el primero), prompt corto de código y los de 40k:

| Configuración | Corto | Prosa 40k | Código 40k | Media |
|---|---|---|---|---|
| Base (primera / última de la cadena) | 66.8 / 43.8 · 67.1 / 43.7 | 37.6 / 41.3 · 39.8 / 40.4 | 37.9 / 41.6 · 37.2 / 40.0 | 44.8 · 44.7 |
| Vocabulario del draft de pi (`--mtp-draft-vocab`) | 65.6 / 49.8 | 42.2 / 43.4 | 42.0 / 44.3 | 47.9 |
| Ídem, otra carga | 66.5 / 49.5 | 42.9 / 44.3 | 40.0 / 45.0 | 48.0 |
| + draft en Q8_0 (sin `--mtp-q4`) | 64.9 / 49.1 | 39.6 / 46.7 | 44.3 / 43.0 | 47.9 |
| + `--spec-min-p` 0.3 / 0.2 | 65.3 / 48.2 · 64.7 / 47.5 | 40.6 / 42.2 · 41.6 / 40.2 | 42.0 / 43.6 · 37.8 / 46.7 | 47.0 · 46.4 |
| `--spec 2` (min-p 0 / 0.2) | 48.9 / 44.9 · 49.1 / 44.9 | 38.7 / 40.9 · 40.2 / 40.2 | 41.1 / 42.2 · 41.3 / 43.5 | 42.8 · 43.2 |
| `--spec 3` (min-p 0.3) | 57.8 / 49.8 | 41.4 / 39.9 | 43.3 / 47.9 | 46.7 |

El subconjunto de vocabulario construido con lo que el modelo genera en las sesiones de pi (`tools/draft_vocab.py
--corpus`, 8 MB de salidas, cobertura 0.99: 113171 ids, 0.55 % de los tokens de pi fuera contra 6.74 %) da +7 %
estable; el resto no mejora. Con él la aceptación iguala a la nuestra (58 % contra 57 %), pero la ronda sigue
costando lo mismo y el decode a 40k queda 5-10 % por debajo: lo que falta está en el costo de la ventana.

### 3.2 La norma del estado del draft y el perfil de la ventana (barridos s2, s3 y p1)

`--spec 2` y `--spec 3` pierden (el prompt corto cae de 66 a 49 t/s: los drafts largos pagan en código predecible).
`--mtp-window 8192` suma ~2 %. La palanca es `--mtp-hnorm stream`: el draft de Strata normaliza por defecto el estado
oculto con una sola RMS sobre los cuatro streams de hyper-connection (la lectura de vLLM); la nuestra, por stream
(`src/models/qwen4exp.cpp`, la corrección medida de eb635675b, que subió la aceptación 0.64 → 0.72). Con la norma por
stream sube la aceptación de Strata (código 40k: 113 → 129 drafts aceptados de ~240):

| Decode, t/s (primer / segundo turno) | Corto | Prosa 40k | Código 40k | Media |
|---|---|---|---|---|
| Strata, vocabulario pi + `--mtp-window 8192` | 66.6 / 49.8 | 43.8 / 45.3 | 41.8 / 47.6 | 49.1 |
| Ídem + `--mtp-hnorm stream` | 67.4 / 52.3 | 43.9 / 47.5 | 44.2 / 48.0 | 50.6 |
| llama.cpp (producción, misma sesión) | 60.2 / 51.2 | 44.4 / 46.5 | 45.8 / 50.4 | 49.7 |

Perfil por fases de la ventana a 40k (`STRATA_VERIFY_PROFILE=1`, `STRATA_DECODE_TIMING=1`; prosa, 2.66 tokens por
ventana): 47 ms = verificación 41.6 + draft 5.4. Contra el reparto de nuestra verificación (sección 28.3 del cost
breakdown): expertos 10.8 ms (nuestros 16.8: los kernels agrupados de Strata leen más rápido) y hyper-connections
~11.8 ms (las nuestras ~3.4 más parte de 2.3): el pack ensancha nuestras proyecciones hc IQ4_NL a BF16 (~1.26 GB por
ventana contra ~0.36), y la lectura nativa de Strata (`STRATA_HC_Q8`) solo existía para Q8_0. El fork la extiende a
IQ4_NL (los cuatro kernels de la lectura parametrizados por formato, el cargador y la verificación;
`gr_parity` compara la lectura cuantizada de Q8_0 e IQ4_NL contra la referencia con los pesos decuantizados).

### 3.3 La lectura hc en IQ4_NL (commit 2485459 del fork de Strata, barrido s5)

`gr_parity` compara la lectura cuantizada contra la referencia con los pesos decuantizados (Q8_0 e IQ4_NL, inject
cuantizada y en BF16, T 1, 3 y 8: error relativo 1.4e-7 a 4.7e-7, con los kernels por defecto y los de `STRATA_TSUM`;
la lectura con los nibbles intercalados, la trampa, queda a 0.60-0.76). En el modelo, la misma configuración con la
lectura BF16 primero y último:

| Decode, t/s (primer / segundo turno) | Corto | Prosa 40k | Código 40k | Media | Prompt 40k |
|---|---|---|---|---|---|
| Strata, hc BF16 (primera / última) | 67.5 / 52.1 · 67.3 / 52.5 | 44.1 / 47.0 · 44.2 / 47.1 | 42.9 / 46.6 · 43.0 / 46.7 | 50.0 · 50.1 | 1355 |
| **Strata, hc IQ4_NL** | **74.6 / 57.4** | **48.7 / 48.5** | **49.3 / 54.7** | **55.5** | 1354 |
| llama.cpp (producción) | 60.2 / 51.2 | 44.4 / 46.5 | 45.8 / 50.4 | 49.7 | 618 |

La ventana a 40k baja de 41.6 a 35.3 ms de verificación (hyper-connections de ~11.8 a ~6.4 ms). Con esto Strata
supera a nuestro llama.cpp en todas las celdas de decode (+4 a +24 %) y lee el prompt 2.2× más rápido.

### 3.4 A 125k

El prompt de prosa de 124878 tokens (wikitext-2, `depth_prompt125.txt`) leído entero y una segunda pregunta,
contexto 131072, una carga por motor (`depth125.sh`; Strata en la configuración de 3.3):

| | Strata | llama.cpp (producción, NP=1) |
|---|---|---|
| Lectura del prompt | 1338 t/s (93 s) | 537 t/s (233 s) |
| Decode, primer / segundo turno | 46.2 / 48.4 t/s | 40.9 / 45.6 t/s |
| GTT pico / MemFree mínimo | 66 / 33 GiB | 69 / 33 GiB |

### 3.5 Con el muestreador de pi y pensamiento (barrido s7)

`BENCH_SAMPLING=pi` (T 0.7, top-k 20, top-p 0.95, min-p 0, semilla 1, pensamiento medium), los mismos prompts:

| Decode, t/s (primer / segundo turno) | Corto | Prosa 40k | Código 40k | Media |
|---|---|---|---|---|
| Strata + UD-small, `--spec-min-p 0.5` (primera / última) | 63.5 / 67.0 · 62.8 / 67.3 | 57.0 / 61.1 · 59.2 / 63.0 | 59.9 / 58.9 · 59.7 / 59.0 | 61.2 · 61.8 |
| Ídem, `--spec-min-p 0.2` / `0` | 64.5 / 64.6 · 65.7 / 64.3 | 56.8 / 60.5 · 56.6 / 59.5 | 60.6 / 56.6 · 59.7 / 57.1 | 60.6 · 60.5 |
| llama.cpp, producción | 57.5 / 50.3 | 50.1 / 53.8 | 48.6 / 50.1 | 51.7 |
| llama.cpp, UD-small (layout separado, sin las fusiones gate+up y down+inject) | 51.0 / 44.3 | 47.7 / 51.6 | 46.4 / 48.8 | 48.3 |

Bajo muestreo la ventaja se mantiene (+18 % de media) y el umbral de confianza del draft por defecto sigue siendo el
mejor. Un primer cálculo desde los logs del estudio de muestreo daba paridad (48.0 contra 48.2 t/s) porque llama.cpp
cortaba en `</think>` (~16 tokens por petición) y Strata generaba 60. En llama.cpp, UD-small en su layout
separado decodifica 7 % menos que el GGUF de producción (que lleva la fusión down+inject, decode +15 %, y la de
gate+up, sección 15 del cost breakdown y su tabla de estado); una versión fusionada recuperaría parte.

## 4. Calidad: los mismos tokens en los dos motores

Logprobs teacher-forced por posición (top 20): Strata con su `STRATA_LOGPOS` (`strata_logpos.py`, el motor por stdin),
llama.cpp con `logpos.cpp` (fuera del árbol, mismo formato). Secuencias de wikitext-2 y de código: 4096 tokens
puntuados por la ruta de decode, y 16384 de prefijo por la ruta de prompt con 1024 puntuados por la de decode.
Referencia de ruido: llama.cpp con `GGML_VK_DISABLE_FUSION=1`.

| Secuencia | Δ ln p Strata (± se) | KL Strata | KL piso | argmax Strata / piso |
|---|---|---|---|---|
| wiki 4k | +0.0004 ± 0.0049 | 0.028 | 0.019 | 96.4 / 97.2 % |
| código 4k | +0.0019 ± 0.0027 | 0.009 | 0.006 | 97.9 / 98.3 % |
| wiki 16k+1k | −0.0080 ± 0.0080 | 0.019 | 0.013 | 94.3 / 95.7 % |
| código 16k+1k | −0.0094 ± 0.0075 | 0.011 | 0.002 | 97.8 / 99.5 % |

La perplejidad es la misma dentro del error en las cuatro: el redondeo BF16 del pack no degrada nuestro GGUF. La
numérica difiere más que nuestro redondeo puro (1.5× el KL del piso; 4.6× en código profundo).

## 5. El pensamiento ajeno de la sección 47

- El camino exacto del incidente (prompt de 19857 tokens, los 37 tokens del pensamiento puntuados por la ruta de
  decode en ambos motores, `inc_score.py`): 'A' 8.9 % en llama.cpp y 0.25 % en Strata (fuera de lo que admite el
  muestreador de pi); probabilidad conjunta de los 7 primeros tokens 5.5e-10 contra 3.5e-13.
- Frecuencia del primer pensamiento en ese contexto, muestreador de pi (T 0.7, top-k 20, top-p 0.95), semillas
  distintas por muestra (`strata_sample.py`, `llama_sample.py`; clasificación de `classify.py` revisada a mano):

| Motor y GGUF | Muestras | En tema | Genérico | Ajeno al contexto |
|---|---|---|---|---|
| llama.cpp + el de producción | 1000 | 569 | 427 | 4 |
| Strata + el de producción, hc del pack en BF16 | 1000 | 434 | 530 | 36 |
| Strata + el de producción, hc leídas en IQ4_NL (sección 3.3) | 400 | 197 | 203 | 0 |
| Strata + UD-small (5.1) | 400 | 393 | 7 | 0 |
| llama.cpp + UD-small | 400 | 330 | 70 | 0 |
| Strata + híbrido: nuestros expertos, todo el lado denso de UD | 400 | 328 | 63 | ~9 |
| Strata + híbrido, el prompt leído por las ventanas de verificación | 400 | 164 | 236 | 0 |
| llama.cpp + el híbrido | 400 | 388 | 12 | 0 |
| Strata + UD-IQ4_XS | 800 | 784 | 13 | 3 |
| llama.cpp + UD-IQ4_XS | 400 | 395 | 5 | 0 |

Clasificación (`classify.py`, revisada a mano): en tema, la primera línea nombra lo pedido (leer la guía H3, la
referencia); genérico, "I need to investigate / check this further. Let me check the details." y variantes vagas
("Just need to confirm and summarize concisely"); ajeno al contexto, lo que habla de otra tarea: "This is a pure logic
question… cats can swim", "A2A: Translate the previous working memo into Japanese" (llama.cpp), "I will maintain the
designated assistant's identity…" (Strata con las hc del pack, 28 de sus 36), "I need to check what the `verify`
function does…" (Strata con el híbrido).

**Lo ajeno al contexto (el incidente) no se demuestra que baje:** con la tasa de producción (0.4 %), cero en 400
muestras ocurre por azar ~20 % de las veces. Lo demostrado es otra cosa: la fidelidad al modelo (sección 6) y el arranque
genérico, que cae de ~43 % a 2-17 % con UD-small. Las tasas elevadas de Strata con las hc del pack (3.6 %) y con el
híbrido (~2 %) muestran que la numérica puede abrir modos ajenos propios.

El GGUF es lo que más mueve el arranque, en los dos motores: con el de Unsloth el modelo arranca en tema casi
siempre, con nuestra receta la mitad de las veces cae en relleno. Nuestra receta difiere de UD en el lado denso
(IQ4_NL/Q5_K contra Q8_0, routers e inject en Q8_0/IQ4_NL contra F32, embedding IQ4_NL contra Q8_0), en las 5 capas de
down que UD deja en Q8_0 y en la capa 2.

Con la lectura hc nativa desaparece el modo "designated assistant" de Strata: lo producía el redondeo a BF16 de
nuestras hc IQ4_NL en el pack. El híbrido (`qwen4exp-ud-dense-nl3s-experts.gguf`, 58.5 GiB sin la PLE) recupera la
mayor parte del comportamiento de UD: el lado denso de nuestra receta explica la degradación.
En llama.cpp el híbrido recupera casi todo (97 %), y UD completo lo deja en 99.8 %. La carga del híbrido se
verificó por memoria (GTT pico 65.0 GiB contra 63.3 con el de producción en la misma configuración: los +1.9 GiB del
lado denso de UD). Strata queda por debajo de llama.cpp con el mismo híbrido (82 % en tema). La hipótesis de que lo causa su
camino de prompt (las hc redondeadas a BF16 del pack; la lectura nativa de 3.3 es solo del decode) se midió y no se
sostiene: con el prompt leído por las ventanas de verificación el mismo híbrido da 41 % en tema (400 muestras, 0
ajenos). La posición es un casi empate entre "I" (relleno) y "The" (en tema) y la ruta numérica también la mueve: un
solo contexto no alcanza para decir que un GGUF no tiene el fenómeno; la medida robusta es la fidelidad (sección 6).

### 5.1 La variante con los tensores densos chicos de UD

`qwen4exp-ud-small-nl3s.gguf` (`prep_udsmall.sh`): el GGUF de producción (capa 2 de UD) con los tensores densos
chicos de UD-IQ4_XS: routers (F32), proyecciones e inject de las hyper-connections (Q8_0 / F32), `ssm_alpha/beta`
(F32), el experto compartido (Q8_0), `ple_key/value` y el embedding (Q8_0); las proyecciones grandes (attn_qkv,
attn_gate, ssm_out, la atención QSA) quedan las nuestras. 57.42 GiB sin la PLE (+0.93 sobre el de producción, 0.30 de
ellos el embedding, que se lee por filas).

Muestras: tabla de la sección 5.

Decode de Strata con los tres GGUF (barrido s6, la configuración de 3.3, el de producción primero y último):

| Decode, t/s (primer / segundo turno) | Corto | Prosa 40k | Código 40k | Media | Prompt 40k |
|---|---|---|---|---|---|
| El de producción (primera / última) | 74.7 / 57.5 · 76.1 / 57.6 | 48.6 / 48.8 · 48.6 / 48.6 | 49.5 / 56.1 · 49.5 / 56.2 | 55.9 · 56.1 | 1357 |
| **UD-small** | **77.3 / 65.8** | **48.2 / 50.3** | **49.2 / 55.5** | **57.7** | 1379 |
| Híbrido (todo el lado denso de UD) | 67.2 / 49.1 | 47.3 / 49.2 | 49.0 / 48.5 | 51.7 | 1385 |
| llama.cpp, producción (sección 3.1) | 60.2 / 51.2 | 44.4 / 46.5 | 45.8 / 50.4 | 49.7 | 618 |

UD-small es la más rápida de las tres (el modelo enfocado acepta más drafts: segundo turno corto 161 de 209 contra
139 de 225) y la que mejor arranca: los tensores densos chicos cuantizados de más en nuestra receta explican el
arranque degradado, y devolverlos a su formato de UD cuesta 0.93 GiB.

## 6. Fidelidad contra UD-IQ4_XS

UD-IQ4_XS (60.4 GiB sin la PLE) es la cuantización más fiel que cabe; los demás GGUF contra él en llama.cpp, mismos
tokens y rutas que la sección 4 (`q_llama_one.sh`, `q_compare.py`):

| GGUF (pesos sin PLE) | KL wiki 4k | código 4k | wiki 16k+1k | código 16k+1k | Δ ln p wiki 4k (± se) |
|---|---|---|---|---|---|
| Producción (56.5 GiB) | 0.156 | 0.073 | 0.106 | 0.042 | −0.026 ± 0.012 |
| UD-small (57.4 GiB) | 0.085 | 0.036 | 0.061 | 0.022 | −0.014 ± 0.009 |
| Híbrido (58.5 GiB) | 0.032 | 0.010 | 0.023 | 0.007 | +0.003 ± 0.005 |
| Piso de ruido de llama.cpp (sin fusiones, contra su base) | 0.019 | 0.006 | 0.013 | 0.002 | — |

UD-small contra producción, mismos tokens: Δ ln p positivo en las ocho mediciones (llama.cpp +0.0115 / +0.0070 /
+0.0099 / +0.0148; Strata +0.0078 / +0.0083 / +0.0041 / +0.0123, cada una ~1 se). Strata con UD-small queda a la misma
distancia de llama.cpp con UD-small que en la sección 4.

En el contexto del incidente, el arranque cambia con el GGUF y con la ruta numérica: llama.cpp 57 % (producción) →
83 % (UD-small) → 97 % (híbrido) → 99.8 % (UD); Strata 49 % → 98 % → 82 % → 98 %; Strata con el híbrido y el prompt
leído por las ventanas de verificación en vez del camino de prompt, 41 %. Ningún pensamiento ajeno al contexto en 400
muestras con UD-small en ninguno de los dos motores, lo que con la tasa de producción (0.4 %) no alcanza para afirmar
que bajen.

## 7. Fuentes

`~/dbg/strata/`: `chain.sh`, `sweep.py` y `sweep*.sh` (velocidad), `bench.py`, `memmon.sh`, `llama_chain.sh`,
`depth125.sh`, `logpos.cpp`, `strata_logpos.py`, `strata_engine.py`, `q_*.sh`, `q_compare.py` (calidad),
`inc_score.py`, `strata_sample.py`, `llama_sample.py`, `bug_sample.sh`, `classify.py` (pensamiento), `prep_*.sh`
(GGUF y packs), `verify_split.py` (bytes), `test_gr.sh`; salidas `bench-*.jsonl`, `mem-*.log`, `engine-*.log`,
`q/`. GGUF en `~/ProjectHub/strix-halo/models/Qwen3.8-Flash-Next-GGUF/NL3S-strata`, packs en `strata-packs/`. El
archivo dividido (`NL3S-split`, 89.5 GB) se borró el 2026-10-07: se regenera desde producción con
`scripts/qwen4exp-split-merged.py`.

## 8. Pendiente antes de adoptar

- **Determinismo.** La misma configuración greedy dio drafts aceptados distintos entre cargas (s5-iq4nl contra
  s6-ours: código 40k 126/246 contra 125/238 y 148/259 contra 144/219; s6-ours contra s6-ours2: prosa 40k, segundo
  turno, 125/243 contra 125/246): el flujo de tokens greedy cambió entre cargas, contra la regla del fork. Compuerta
  pendiente: la misma petición 5 veces en una carga y en dos cargas nuevas, ids de tokens comparados (Strata documenta
  `--prompt-cache 0 --pcie-frac 0` para repeticiones idénticas byte a byte).
- **El servidor de Strata con pi.** Su plantilla rinde ~50 tokens más que llama-server para los mismos mensajes (4808
  contra 4758, sin comparar), y en el bench emitió una tool call en una petición sin tools. El estudio del pensamiento
  le pasó ids crudos al motor: pi a través del servidor de Strata no está probado.
- **Imágenes.** El codificador de Strata en AMD solo corre en la CPU (`strata-vision`); hace falta compilarlo contra
  `mtmd` con Vulkan, como el nuestro, y probar la imagen de 2048×2048.
- **Dos slots a 262144** (`"parallel": 2`, `--batch-mtp`): sin medir la memoria ni la concurrencia.
- **Memoria.** "No más de 58 GB residentes" se cumple con la definición de `config.ini` (pesos sin la PLE: 57.4 GiB);
  lo residente de Strata en la GPU con UD-small es ~60 GiB (caché de expertos 53.5, proyecciones nativas 2.5, pack BF16
  1.4, copia Q8_0 de las hc 0.6, cabeza y su copia empaquetada 1.0, draft 1.1) más 0.6 de embedding mapeado en el host,
  frente a ~59.4 de llama.cpp (56.5 + draft 2.05 + mmproj 0.86). A 65k de contexto el GTT es 66.2 contra 66.7, con el
  mmproj cargado solo en llama.cpp.
- Commits locales, sin push: llama.cpp 4a1004d96, b417ddd39 y este documento; Strata 310bfe4 y 2485459. GGUF de ~90
  GB en `NL3S-strata` (l2xs, UD-small, híbrido); `NL3S-split` ya se borró (sección 7).

## 9. Las compuertas (2026-10-07, tarde)

Configuración de producción (UD-small, ajustes del draft) a 65536 de contexto, producción descargada, una carga a la
vez por la compuerta. Herramientas en `~/dbg/strata/`: `det.py` y `det_compare.py` (el texto de las respuestas por el
servidor), `det_logpos.py` y `det_tail.py` (log-probs con 9 decimales de los últimos 64 tokens leídos por las ventanas
de verificación tras leer el prompt, comparados entre lecturas), `logpos_chain.sh` y `queue14`-`queue23`.

### 9.1 Determinismo

- **Con la caché de prompts, como en producción:** las mismas peticiones (agente 4.7k ×5, prosa 40k ×3, otra
  conversación en medio, el muestreador de pi con semilla ×3) dieron textos idénticos dentro de una carga y entre dos
  cargas nuevas.
- **Sin la caché** (`--prompt-cache 0`, cada petición lee su prompt entero): la prosa de 40k divergió en el carácter 179
  en 1 de 3 lecturas, y también con ventanas de un token (`--spec 2` sin MTP ni lookup): el origen no es el draft.
- **La política de drafts de Strata** (`DraftPolicy`, una por proceso) decide la cadena de lookup y los sufijos con el
  tiempo medido de cada ronda y lo aceptado en peticiones anteriores: los drafts propuestos varían entre peticiones
  idénticas (agente: 160, 156, 150, 153 y 145 propuestos, 114 aceptados siempre). Con `STRATA_LOOKUP_CHAIN_FIXED=1
  --suffix-draft 0` son fijos. No cambia el texto greedy mientras el modelo sea determinista (las ventanas de
  verificación dan las mismas filas), pero es estado entre peticiones.
- **La lectura del prompt no es determinista desde el segundo chunk de 16384:**

  | Lectura (log-probs de la cola de 64) | Idénticas a la primera |
  |---|---|
  | 15k, un chunk | 8 / 8 |
  | 32.8k, dos chunks | 1 / 8 y 7 / 8 (dos cargas) |
  | 40k, tres chunks | 4 / 8 |
  | 40k sin MTP | 1 / 8 |
  | La cola a 32.8k leída 8 veces sobre un mismo estado (solo ventanas de verificación) | 8 / 8 |
  | 1024 tokens por ventanas de verificación | 8 / 8 en todas las configuraciones |

  La diferencia no es de último bit: Δlogp de hasta 0.51 en la cola, la misma magnitud que separa la numérica rápida de
  Strata de la lenta (0.065 de media): el modelo amplifica una diferencia mínima a lo largo de 40k de recurrencia.
- **Es una carrera que depende del tiempo, no un kernel con su propio orden.** Con 8 lecturas, apagar PF_FUSED, PA_FAST,
  el indexador por lotes (`STRATA_INDEXER_PER_TOKEN=1`) o el scorer de matrices (`STRATA_SELECT_OLD=1`) la esconde
  (8 / 8 cada uno); la misma configuración cambia de tasa entre cargas; y `STRATA_PF_STEP_SYNC=1` (el host espera a la GPU
  al final de cada capa del prompt) da 8 / 8 a la misma velocidad (1475 contra 1465 t/s). En la lectura que divergió con
  `STRATA_STATE_HASH_GDN=1`, los estados GDN de las capas 0 a 13 eran iguales y distintos desde la 14. Todos los
  expertos están en la caché de la GPU (el stream de copia del prompt no copia nada).
- **Pendiente:** el primer paso que diverge, con una suma por capa y mitad calculada en la GPU sin esperas del host
  (`STRATA_PF_CHECKSUM`, escrita en el fork, sin compilar todavía); la compuerta de historia (una imagen de 2048×2048
  en el otro slot) necesita imágenes en GPU y dos slots. `STRATA_PF_STEP_SYNC=1` sería una medida provisoria sin
  costo medido, no la corrección.

### 9.2 pi a través del servidor de Strata

- La plantilla es la misma en producción, UD-small y el pack de Strata (sha256 `12827f24…`).
- Los ~50 tokens de más del bench (4808 contra 4758) vienen del prompt del bench, no de pi: `agent_prompt.txt` es una
  conversación ya renderizada pegada como texto del usuario, y Strata codifica el texto de los tokens de control dentro
  de un mensaje como texto (#537), llama-server como tokens de control; con el mismo tratamiento son 4758. La tool call
  "sin tools" es el modelo continuando ese texto, que el parser de Strata reconoce aunque la petición no traiga tools
  (llama-server no lo hace).
- **Un defecto real, corregido en el fork:** el servidor de Strata quitaba el envoltorio `{"type": "function",
  "function": …}` de las tools antes de la plantilla, que las escribe con `tojson`; la petición real de pi del incidente
  (`~/dbg/glitch-1006/request.json`, 14 tools) daba 19 731 ids contra 19 857 de llama-server. Con 9c6c243 el render
  recibe las tools como las mandan los clientes (como transformers y llama-server) y la misma petición da exactamente
  los ids de llama-server. Suite del servidor: 454 tests, OK.
- **Pendiente:** una sesión de pi en vivo contra el servidor de Strata (streaming y tool calls de vuelta a pi).

### 9.3 Dos slots (`--batch 2`, el `"parallel": 2` del servidor)

| | 65k | 131k |
|---|---|---|
| Sesión por slot (log del motor) | 1.00 GiB | 1.89 GiB |
| GTT máximo (solo a 65k: 66.2) | 68.3 GiB | 71.0 GiB |
| MemAvailable mínimo | 43.9 GiB | 41.2 GiB |
| Una petición sola, agente / corta | 66.6 / 77.7 t/s | 67.2 / 77.8 t/s |
| Dos a la vez, agente + corta | 24.7 + 26.3 t/s (36.7 juntas) | 24.7 + 26.2 t/s (36.5 juntas) |
| Una corta mientras se leen 40k | 6.2 t/s | 6.0 t/s |

- Las sesiones crecen en línea con el contexto (0.89 GiB cada 65536 celdas): a 262144 serían ~3.7 GiB cada una, tres
  sesiones (la del camino solo y dos slots), ~76 GiB de GTT, ~77.5 con el codificador de imágenes (1.4 GiB): ~36 GiB de
  MemAvailable, contra los ~22 de producción con llama.cpp. Medido hasta 131k (la compuerta de 40 GiB no admite cargar
  262144 × 2 para una prueba); 262144 es extrapolado.
- Dos peticiones a la vez decodifican sin drafts (un token por ventana): ~25 t/s cada una, menos en conjunto que una
  sola. `--batch-mtp` (una propuesta MTP por slot) hace caer el motor al admitir la segunda petición con `--mtp-q4 all`
  ("mtp: unsupported native MMVQ GGML type"): sin probar sin `--mtp-q4`. Las respuestas en un slot no son las del camino
  solo (otra composición de ventanas, sin los ajustes de exactitud de BATCHING.md).
- Pendiente: la misma medición de llama.cpp con dos peticiones a la vez, para comparar.
