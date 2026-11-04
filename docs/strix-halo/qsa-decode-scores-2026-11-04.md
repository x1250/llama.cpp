# El puntaje QSA del decode sin ds_bpermute (2026-11-04)

Sigue a `qsa-score-format-2026-11-04.md` (sección 8): la selección QSA del decode pesaba ~4.5 % del tiempo de decode
de producción y no se sabía cuánto era puntaje y cuánto top-k. Se midió, y el puntaje tenía un costo que no era de
memoria: sus reducciones entre hilos pasaban por la LDS.

## 0. Resultado

**Adoptado:** Strata `473cc674`, en producción desde el 2026-11-04. Es exacto: los mismos bits.

- **Qué cambia:** el kernel que puntúa los bloques en cada ventana de verificación (`block_scores_multi_kernel`) suma
  cada producto punto con la misma mariposa de 32 hilos, pero sus movimientos entre hilos son DPP y `permlanex16`
  en lugar de `ds_bpermute` (`s26_tsum.cuh`, ya usado en cuatro archivos de kernels).
- **Velocidad:**
  - el puntaje de una ventana, por capa, a 246k: 68 → 40 µs con 1 consulta, 192 → 88 con 3, 320 → 140 con 5 (en el
    bench, con las claves en la caché de la GPU);
  - en el motor, puntaje + top-k por ventana: −26 % a 125k, −32 % a 192k, −26 % a 226k;
  - en producción, con su distribución de profundidad: ~1.2 % del tiempo de decode, ~1 % del tiempo del motor
    (estimado desde la corrida perfilada, sección 3).
- **Exactitud:** los puntajes son bit a bit los del kernel por consulta (1 474 575 comparaciones). En el motor, las
  respuestas y las filas de log-probs son idénticas a producción en todas las compuertas (sección 4).

## 1. Dónde estaba el tiempo

`qsa_select_bench` con un modo nuevo, `decode`: las dos llamadas de una ventana tal como las hace la ventana (el
puntaje sin conteo de bloques activos y el top-k por la regla de capacidad), con la capacidad de producción (262144
celdas, 65538 bloques), cada una medida sola. Por capa:

| Contexto | Puntaje, 1 / 3 / 5 consultas | Top-k |
|---|---|---|
| 65k | 20 / 53 / 88 µs | 28-37 µs |
| 131k | 36 / 105 / 170 µs | 57-60 µs |
| 246k | 68 / 192 / 320 µs | 76-87 µs |

- **El puntaje crece con las consultas de la ventana.** El kernel lee cada bloque una sola vez para todas las
  consultas: el tráfico de memoria no crece con ellas.
- **El ensamblador lo explica:** por cada bloque y cada consulta, las 4 cabezas suman su producto punto con 5 pasos
  de `__shfl_xor_sync`, que HIP compila a `ds_bpermute_b32` (20 por bloque y consulta, cada uno esperando al
  anterior a través de la LDS).
- **Las ventanas de producción verifican 3.1 tokens en promedio:** 1 222 935 drafts en 572 235 ventanas (log del
  motor, del 2026-10-07 al 2026-11-04).

## 2. El cambio

`d = s26ts::tplain<16>(d)` en lugar del bucle de `__shfl_xor_sync`. Suma los mismos pares en el mismo orden
(desplazamientos 16, 8, 4, 2, 1; la suma de punto flotante conmuta, así que x + y y y + x dan los mismos bits). Sus
movimientos son `permlanex16` para el 16 y DPP `row_xmask` para el resto en AMD, y `__shfl_xor_sync` en CUDA. Los
otros dos kernels del puntaje (el de la lectura por consulta y el del bloque de cola) no cambian: el primero sirve de
referencia bit a bit.

## 3. Velocidad

**El bench** (`bench-ab.sh`: el binario anterior primero y último, el nuevo entre medio):

| Contexto, consultas | Antes | DPP | Antes (otra vez) |
|---|---|---|---|
| 65k, 5 | 88.1 µs | 38.7 µs | 88.2 µs |
| 131k, 3 | 104.6 µs | 47.4 µs | 104.6 µs |
| 196k, 5 | 251.3 µs | 115.1 µs | 250.8 µs |
| 246k, 1 | 67.9 µs | 39.9 µs | 67.7 µs |
| 246k, 5 | 320.4 µs | 140.5 µs | 312.4 µs |

En cada punto, los puntajes del kernel nuevo son bit a bit los del kernel por consulta, y los ids del top-k los de la
referencia. Los tiempos del bench son con las claves en la caché de la GPU: a 246k una consulta lee 31.5 MB en 68 µs,
~460 GB/s, más que el pico de la memoria; en el motor las claves de cada capa no quedan en caché entre ventanas, y el
efecto real es el de la tabla siguiente.

**En el motor** (`chain-prof.sh`: la conversación de `depth_profile.py`, de 0 a ~248k en lecturas de ~16k y respuestas
de 64 tokens, con `STRATA_VERIFY_PROFILE`; producción primero y último):

| Profundidad | Producción | DPP | Producción (otra vez) |
|---|---|---|---|
| 66k | 1.50 ms | 1.22 ms | 1.51 ms |
| 125k | 2.68 ms | 1.97 ms | 2.68 ms |
| 160k | 2.63 ms | 2.22 ms | 2.63 ms |
| 192k | 3.93 ms | 2.68 ms | 3.92 ms |
| 226k | 4.15 ms | 3.07 ms | 4.14 ms |

(puntaje + top-k por ventana de verificación, con ~2.7 tokens por ventana en esta conversación). Las dos cargas de
producción coinciden dentro de 0.01 ms. En la carga DPP, las ventanas de los turnos a 0-50k fueron más lentas en
etapas que el cambio no toca (la GEMV q8 de las capas GDN 10.7 ms contra 3.8, la recurrencia, el router): otro
cliente de la GPU en esos minutos.

**En producción:** con el ahorro por ventana de cada profundidad y la distribución del decode de producción
(`topk/decode_depth.out`: 7 % del decode bajo 32k, 19 % a 32-64k, 18 % a 64-96k, 15 % a 96-128k, 27 % a 128-192k,
14 % pasados 192k), ~1.2 % del tiempo de decode: 2.6 % pasados 192k, 1.8 % entre 128k y 192k. Es una estimación: el
perfil apaga el solapamiento del stream compartido, y en la corrida libre (`chain-ab.sh`) la velocidad de decode quedó
dentro del ruido (+3 % a 192k, +0.6 % a 226k, ±0.5 % a menor profundidad). Las ventanas de producción verifican 3.1
tokens (2.5-2.9 en esta conversación), lo que la sube; el solapamiento puede bajarla.

## 4. Compuertas

El cambio da los mismos bits, así que se exigió lo mismo que al top-k: todo idéntico a producción (`gates.sh`, 163840
× 2 slots salvo donde se indica).

- **Conversación a profundidad** (`topk_gate.py`: 39.5k de prosa, una imagen de 2048×2048, código hasta 139k celdas,
  una respuesta de 256 tokens y cuatro repeticiones): producción, DPP, DPP después de la imagen en el otro slot y
  producción otra vez, idénticas en respuestas y filas en cada paso.
- **A 146k junto a otro cliente de la GPU:** DPP idéntica a producción.
- **Dos slots** (`batch_interleave_test.py`, lecturas de 17400 y 19700 tokens): cada slot idéntico a su corrida sola.
- **Historia sin drafts** (`hist_reads.py`, 65536 × 2): DPP idéntica a producción (515 filas) y a sí misma después de
  la imagen. La mitad con drafts es la conversación de arriba.
- **Siete respuestas de 256 tokens hasta 225k** (`chain-ab.sh`, un slot): DPP idéntica a producción en las siete.

## 5. Lo que queda

- **El top-k del decode:** 76-87 µs por capa a 246k, ~1 ms por ventana a 226k. Corre en un solo grupo de hilos por
  consulta, así que con 1-5 consultas usa 1-5 de las 40 unidades de cómputo. En CUDA, Strata lo reparte en un cluster
  de 8 bloques (sm_90); HIP no tiene clusters. Un top-k que use más unidades daría ~0.7 ms por ventana a 226k
  (~1 % del decode), con los mismos ids por construcción.
- **Producción no se repite entre cargas** (sección 6).

## 6. Hallazgo: producción no se repite entre cargas

- **Lo observado** (`chain-ab.sh`): la misma conversación de siete turnos con respuestas cortadas a 256 tokens dio en
  la segunda carga de producción otras respuestas en los turnos a 192k y 226k. Los turnos 0-4 dieron el mismo texto.
- **Los drafts varían, la salida no** (`chain-drafts.sh`): la misma petición de 33.5k, tres veces en cada una de dos
  cargas, propuso 229, 227 y 224 drafts en la primera carga y 230, 232 y 224 en la segunda; las seis respuestas son
  idénticas. La política de drafts (`src/spec/draft_policy.cpp`) elige el tamaño de las ventanas de lookup y de cadena
  comparando costos por tamaño, y esos costos son medias móviles del tiempo de reloj de cada ronda (`observe(...,
  round_ms)`), aprendidas durante todo el proceso. Es por diseño: la verificación mantiene la salida.
- **Dónde se vuelve una diferencia de salida:** el turno siguiente a una respuesta cortada por `max_tokens` a veces
  reutiliza la respuesta entera desde el caché (base2, turno 5: 160689 tokens) y a veces vuelve al punto de control
  del límite del turno y la relee (base, turno 5: 160426 = fin del prompt − 7). Releer por la ruta de prompt redondea
  distinto que el KV que escribieron las ventanas de verificación, y desde ahí las respuestas divergen. Que la
  reutilización dependa de cómo terminó la última ventana (drafts aceptados más allá del corte) es la hipótesis que
  explica los logs; no está verificada en el código.
- **En producción:** de las 5905 respuestas de qwen38flash en las sesiones de pi, 3 terminaron por longitud (las demás,
  por llamada a herramienta, fin de turno, error o cancelación). Queda en `latent-defects.md`. Para las compuertas:
  las conversaciones de varios turnos comparadas entre cargas no deben cortar respuestas por `max_tokens` a mitad de la
  conversación, o deben fijar la política (`STRATA_LOOKUP_CHAIN_FIXED` fija solo la cadena).

## 7. Adopción

- **Strata:** `0ffc0065` (el modo `decode` del bench) y `473cc674` (el cambio) en `main` del fork, publicados.
- **El binario:** `strix-halo/build-strata.sh`. En el código de máquina del objeto de producción, el kernel del
  puntaje del decode tiene 0 `ds_bpermute`, 4 `permlanex16` y 16 movimientos DPP `row_xmask` (4 cabezas × 5 pasos).
- **Sin cambio de configuración.** Producción se recargó a las 18:47 con 262144 × 2.
- **Para volver atrás:** `git revert 473cc674`, `build-strata.sh` y recargar.

## Evidencia

`~/ProjectHub/llama.cpp-evidence/decsel/`:
- el bench: `bench-decode.sh`, `bench-decode.out` (el reparto), `bench-ab.sh`, `bench-ab.out`,
  `qsa_select_bench.base` (el binario anterior), `build.out`, `build-dpp.out`;
- el motor: `chain-ab.sh`, `depth_decode.py`, `chain-ab.out`, `ab-*.jsonl`; `chain-prof.sh`, `chain-prof.out`,
  `prof-*`;
- las compuertas: `gates.sh`, `gates.out`, `g-*`, `bi-dpp-*`, `hist-*`;
- los drafts: `chain-drafts.sh`, `chain-drafts.out`, `drafts-*`;
- `build-tests.sh` y `build-tests.out` (`arch_defaults_test`, pendiente del cambio f16x2: pasa);
- enlaces a `idxprec/mkcfg.py` y a `shortread/` (`hist_reads.py`, `hist_check.py`, `speed_reads.py`).
