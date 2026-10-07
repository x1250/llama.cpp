# Strata en el Strix Halo: velocidad, memoria, calidad y el pensamiento ajeno (2026-10-07)

Evaluación de [Strata v0.1.40](https://github.com/Niko1221/Strata) (motor propio para Qwen3.8-Flash-Next, HIP/ROCm,
gfx1151 experimental) frente al fork en producción, pedida por el Director: "no más lento que Strata, sin el bug, no más
de 58 GB residentes". Fork del motor: `x1250/Strata` (privado), clon en `~/ProjectHub/strata`, build con ROCm 7.14.1 de
TheRock en `~/rocm/install` (`~/dbg/strata/build.sh`). Herramientas y salidas: `~/dbg/strata/`.

## 0. Resultado

**Strata (fork, 3 cambios) con el GGUF UD-small supera al llama.cpp de producción en todo lo medido, con la misma
memoria:** decode +16 % de media (+7 a +29 % por celda; +13 % a 125k), lectura del prompt 2.2-2.5×, perplejidad ~1 %
mejor, la mitad de distancia a UD-IQ4_XS, y el arranque del pensamiento del incidente en tema el 98 % de las veces
(contra 49-58 % con el GGUF de producción).

| | llama.cpp, producción | Strata + UD-small |
|---|---|---|
| Prompt 40k / 125k | 618 / 537 t/s | 1379 / 1338 t/s |
| Decode medio (corto, prosa y código 40k, dos turnos) | 49.7 t/s | 57.7 t/s |
| Decode a 125k (primer / segundo turno) | 40.9 / 45.6 | 46.2 / 48.4 (con el GGUF de producción) |
| Pesos sin la PLE | 56.5 GiB (+2.05 draft, +0.86 mmproj) | 57.4 GiB (+0.9 draft) |
| GTT a 65k de contexto | 66.7 GiB | 66.2 GiB |
| KL contra UD-IQ4_XS (wiki 4k / código 4k) | 0.156 / 0.073 | 0.085 / 0.036 |
| Primer pensamiento del incidente en tema | 58 % (1000) | 98 % (400) |

Los 3 cambios del fork (`x1250/Strata`, local): 310bfe4 (las páginas del archivo se liberan a medida que los
expertos llegan a la GPU: sin esto la carga lleva MemFree a cero), 2485459 (la lectura de las hyper-connections en el
IQ4_NL del GGUF: −6 ms por ventana) y la configuración (`--mtp-hnorm stream`, la norma del draft por stream como la
nuestra; un subconjunto de vocabulario del draft hecho con las salidas de pi; `--mtp-window 8192`). Configuración
reproducible: `~/dbg/strata/prod/strata-qwen38flash-strix.json` y `env.sh`.

**El pensamiento ajeno al contexto no es un defecto del motor:** aparece en los dos, y lo que más mueve el arranque
del pensamiento es el GGUF. Los tensores densos chicos de nuestra receta (routers, hyper-connections, compuertas de la
GDN, experto compartido, PLE key/value y embedding, cuantizados más que en UD; cuál de ellos pesa no está aislado)
alejan al modelo de UD; devolverlos a su formato de UD (UD-small, +0.93 GiB) mejora la perplejidad, reduce la distancia
a la mitad y cambia el arranque del pensamiento. La
posición del incidente es un casi empate entre "I" y "The": su frecuencia la mueve también la numérica del motor
(sección 5), así que lo robusto es la fidelidad (sección 6), no un solo contexto.

**Pendiente para reemplazar producción (decisión del Director):** codificador de imágenes en GPU (el `strata-vision`
de Strata solo trae CPU en AMD; compilarlo contra `mtmd` con Vulkan), dos slots a 262144 (`"parallel": 2` con
`--batch-mtp`, sin medir), la integración con pi por la API de Strata y las compuertas de determinismo equivalentes a
las del fork.

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

| Motor y GGUF | Muestras | En tema | Relleno genérico | Ajeno al contexto |
|---|---|---|---|---|
| llama.cpp + el de producción | 1000 | 582 (58 %) | 414 (41 %) | 4 (0.4 %) |
| Strata + el de producción (capa 2 de UD) | 1000 | 437 (44 %) | 530 (53 %) | 33 (3.3 %) |
| Strata + UD-IQ4_XS | 800 | 784 (98 %) | 13 (1.6 %) | 3 (0.4 %) |
| Strata + el de producción, hc leídas en IQ4_NL (sección 3.3) | 400 | 197 (49 %) | 203 (51 %) | 0 |
| Strata + híbrido: nuestros expertos, el lado denso de UD | 400 | 358 (89.5 %) | 34 (8.5 %) | 8 (2 %) |
| llama.cpp + el híbrido | 400 | 388 (97 %) | 12 (3 %) | 0 |
| llama.cpp + UD-IQ4_XS | 400 | 399 (99.8 %) | 0 | 1 (un relleno breve que vuelve al tema) |
| Strata + híbrido, el prompt leído por las ventanas de verificación | 400 | 172 (43 %) | 228 (57 %) | 0 |

"Relleno genérico" es "I need to investigate this further. Let me check the details." (sin relación con lo pedido);
lo ajeno al contexto, frases como "This is a pure logic question… cats can swim", "A2A: Translate the previous
working memo into Japanese" (llama.cpp) o "I will maintain the designated assistant's identity…" (Strata con nuestro
GGUF, 28 de sus 33).

El GGUF es lo que más mueve el arranque, en los dos motores: con el de Unsloth el modelo arranca en tema casi
siempre, con nuestra receta la mitad de las veces cae en relleno. Nuestra receta difiere de UD en el lado denso
(IQ4_NL/Q5_K contra Q8_0, routers e inject en Q8_0/IQ4_NL contra F32, embedding IQ4_NL contra Q8_0), en las 5 capas de
down que UD deja en Q8_0 y en la capa 2.

Con la lectura hc nativa desaparece el modo "designated assistant" de Strata: lo producía el redondeo a BF16 de
nuestras hc IQ4_NL en el pack. El híbrido (`qwen4exp-ud-dense-nl3s-experts.gguf`, 58.5 GiB sin la PLE) recupera la
mayor parte del comportamiento de UD: el lado denso de nuestra receta explica la degradación; lo que falta hasta UD
(98 %) lo ponen sus 5 capas de down en Q8_0 (60.4 GiB).
En llama.cpp el híbrido recupera casi todo (97 %), y UD completo lo deja en 99.8 %. La carga del híbrido se
verificó por memoria (GTT pico 65.0 GiB contra 63.3 con el de producción en la misma configuración: los +1.9 GiB del
lado denso de UD). Strata queda por debajo de llama.cpp con el mismo híbrido (89.5 %). La hipótesis de que lo causa su
camino de prompt (las hc redondeadas a BF16 del pack; la lectura nativa de 3.3 es solo del decode) se midió y no se
sostiene: con el prompt leído por las ventanas de verificación el mismo híbrido da 43 % en tema (400 muestras, 0
ajenos). La posición es un casi empate entre "I" (relleno) y "The" (en tema) y la ruta numérica también la mueve: un
solo contexto no alcanza para decir que un GGUF no tiene el fenómeno; la medida robusta es la fidelidad (sección 6).

### 5.1 La variante con los tensores densos chicos de UD

`qwen4exp-ud-small-nl3s.gguf` (`prep_udsmall.sh`): el GGUF de producción (capa 2 de UD) con los tensores densos
chicos de UD-IQ4_XS: routers (F32), proyecciones e inject de las hyper-connections (Q8_0 / F32), `ssm_alpha/beta`
(F32), el experto compartido (Q8_0), `ple_key/value` y el embedding (Q8_0); las proyecciones grandes (attn_qkv,
attn_gate, ssm_out, la atención QSA) quedan las nuestras. 57.42 GiB sin la PLE (+0.93 sobre el de producción, 0.30 de
ellos el embedding, que se lee por filas).

| | Muestras | En tema | Relleno | Ajeno al contexto |
|---|---|---|---|---|
| Strata + UD-small | 400 | 393 (98.3 %) | 7 (1.8 %) | 0 |
| llama.cpp + UD-small | 400 | 331 (83 %) | 69 (17 %) | 0 |

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

En el contexto del incidente, el arranque cambia con el GGUF y con la ruta numérica: llama.cpp 58 % (producción) →
83 % (UD-small) → 97 % (híbrido) → 99.8 % (UD); Strata 49 % → 98 % → 89.5 % → 98 %; Strata con el híbrido y el prompt
leído por las ventanas de verificación en vez del camino de prompt, 43 %. Ningún pensamiento ajeno al contexto en 400
muestras con UD-small en ninguno de los dos motores.

## 7. Fuentes

`~/dbg/strata/`: `chain.sh`, `sweep.py` y `sweep*.sh` (velocidad), `bench.py`, `memmon.sh`, `llama_chain.sh`,
`depth125.sh`, `logpos.cpp`, `strata_logpos.py`, `strata_engine.py`, `q_*.sh`, `q_compare.py` (calidad),
`inc_score.py`, `strata_sample.py`, `llama_sample.py`, `bug_sample.sh`, `classify.py` (pensamiento), `prep_*.sh`
(GGUF y packs), `verify_split.py` (bytes), `test_gr.sh`; salidas `bench-*.jsonl`, `mem-*.log`, `engine-*.log`,
`q/`. GGUF en `~/ProjectHub/strix-halo/models/Qwen3.8-Flash-Next-GGUF/NL3S-split` y `NL3S-strata`, packs en
`strata-packs/`.

