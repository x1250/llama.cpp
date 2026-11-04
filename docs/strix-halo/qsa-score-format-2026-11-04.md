# El puntaje del indexador QSA en fp16 de dos piezas (2026-11-04)

El Director preguntó qué se podía extraer de Latent Sparse Attention (LSA, de Prefrontal-AI; entre sus autores, Torsten
Hoefler de ETH Zúrich; arXiv 2610.42147, código Apache 2.0). La parte que aplica a producción es una: la selección de
bloques tolera menos precisión de la que Strata le da. Se midió dónde está el tiempo, se probó un formato más barato y
pasó todas las compuertas.

## 0. Resultado

**Adoptado:** `STRATA_SELECT_WMMA_FORMAT=f16x2`, en producción desde el 2026-11-04.

- **Qué cambia:** el puntaje de bloques que usa la lectura de prompts (no el decode) multiplica cada valor partido en
  dos piezas fp16 (3 productos WMMA) en lugar de tres piezas bf16 (6 productos).
- **Velocidad:**
  - la fase "qsa select" (puntaje + top-k) de la lectura baja 26-30 % entre 17k y 226k celdas;
  - con los mismos tokens en las tres cargas (sección 4): leer 180k desde cero, +7.0 %; chunks de 3008 entre 183k y
    229k, +4.8 %; hasta 47k, dentro del ruido (+0.7 % en total);
  - **en producción**, sobre las lecturas del log del motor: ahorra 1.2 % del tiempo de lectura, 0.2 % del tiempo del
    motor. Las lecturas que empiezan pasados los 128k ganan 2.4 %. La ganancia es chica porque casi todas las lecturas
    profundas de pi son cortas y en ellas pesa poco la selección.
- **Calidad:** contra producción, la KL agregada es 1.06 veces la de la referencia (el kernel FP32, el default de
  Strata: otro redondeo de la misma selección) y el argmax queda 0.21 puntos debajo. El criterio permite 2 veces y
  0.5 puntos (sección 5).
- **Determinismo:** idéntico a sí mismo entre cargas, después de una imagen en el otro slot (con y sin drafts), a 146k
  junto a otro cliente de la GPU, en dos slots y en las relecturas.
- **El decode no cambia:** el único llamador del puntaje WMMA es la lectura (`prefill.cpp:2377`); el decode usa el
  kernel FP32 (`qsa_block_scores`).

## 1. Qué se tomó de LSA y qué no

LSA agrupa las claves en un espacio latente aprendido y cada consulta puntúa grupos en lugar de claves. Sus cifras
grandes (72.4× en lectura y 42.5× en decode) son de una capa de atención de Qwen3.8-27B a 10M tokens contra atención
completa; contra atención dispersa (GLM-5.3-Flash), en el modelo entero, empata hasta 1M y gana ~2× a 10M.

| Idea de LSA | Por qué no aplica aquí |
|---|---|
| Agrupar claves en un espacio latente destilado | Hay que destilar un mapa para Qwen3.8-Flash en un cluster (H200, Slurm) y reemplaza el indexador que Qwen entrenó con el modelo. Debajo de ~256k el propio LSA vuelve al puntaje denso |
| La variante sin entrenamiento (PCA) | Reduce dimensiones del puntaje; cambia la selección mucho más que bajar la precisión, para una ganancia parecida |
| Búsqueda en árbol | Paga desde ~5M tokens |
| No escribir la matriz de puntajes | A nuestra profundidad son ~5 MB por token, ~3 % del ancho de banda |
| Contexto repartido entre GPUs | Hay una |

Lo que sí aplica: LSA guarda los centroides en bf16 (fp8 en el árbol) y puntúa con multiplicaciones de matrices de
baja precisión. Strata hacía lo contrario: su puntaje WMMA (`STRATA_SELECT_WMMA=1`, en producción) parte cada fp32 en
tres piezas bf16 y suma seis productos para reproducir FP32 (`qsa_select.cu`).

## 2. Dónde está el tiempo

Producción a 262144 con un slot, una conversación que crece de a ~16k tokens (`depth_profile.py`), con los perfiles de
fases del motor (`STRATA_PREFILL_TIMING`, `STRATA_VERIFY_PROFILE`). El perfil del decode apaga el solapamiento del
stream compartido, así que sus porcentajes son una estimación:

| Lectura desde | "qsa select" en la lectura | Puntaje + top-k en la ventana de decode |
|---|---|---|
| 0 | 1.3 % | 1.4 % |
| 50k | 5.6 % | 3.1 % |
| 99k | 10.3 % | 4.7 % |
| 143k | 14.1 % | 6.1 % |
| 176k | 16.8 % | 7.0 % |
| 226k | 20.1 % | 8.3 % |

- La lectura de 34k de esa corrida dio 10.7 %, un caso suelto: en las tres cargas de la sección 4 con el puntaje de
  producción, la misma lectura dio 4.4 %.
- El resto de las fases no crece con el contexto (la atención QSA, por ejemplo, queda en 8.4-9.6 %).
- Con `qsa_select_bench` (sintético, lotes de 256 consultas como la lectura): a 245k, el puntaje tarda 3.25 ms y el
  top-k 0.98 ms; el puntaje es la parte grande.

## 3. El cambio

`STRATA_SELECT_WMMA_FORMAT` elige el formato de los operandos del puntaje WMMA:

- `bf16x3`, el default: tres piezas bf16, seis productos, nivel FP32;
- `f16x2`: dos piezas fp16, x = hi + lo con ~22 bits, y los tres productos de orden ≤ 1; es el método de la atención
  de la lectura en HIP (`STRATA_HIP_WMMA`), que ya parte en mitades fp16 alta y baja.

Las claves y consultas del indexador salen de una norma RMS, así que |x| ≤ √128 por la mayor ganancia de la norma:
2.08 en este modelo (`indexer.q_norm` y `indexer.k_norm` de las 12 capas), ~24 contra el máximo fp16 de 65504.

`qsa_select_bench`, lotes de 256 consultas, capacidad 262144:

| Contexto | bf16x3 | f16x2 | fp16 de una pieza | Top-k |
|---|---|---|---|---|
| 34.8k | 0.47 ms | 0.30 ms | 0.25 ms | 0.16 ms |
| 131k | 1.70 ms | 1.03 ms | 0.77 ms | 0.55 ms |
| 246k | 3.25 ms | 2.16 ms | 1.78 ms | 0.98 ms |

- f16x2 pasa la compuerta de precisión del bench contra FP64 (a 246k, 2.8e-7 de la escala; bf16x3 2.2e-7) y, con
  las claves sintéticas, elige las mismas celdas que el kernel FP32 en las 256 consultas a toda profundidad.
- Una sola pieza fp16 falla esa compuerta (~1e-4 de la escala) y cambia la selección de 17-31 de 256 consultas. No
  entró en el código; en el motor ahorraba ~1 % más de lectura a 226k.

## 4. Velocidad

**En el motor** (`chain-ab.sh`): la conversación de la sección 2 con producción primero y último, la rama con
`bf16x3` (idéntica a producción: las 15 respuestas iguales) y con `f16x2`.

| Lectura desde | Producción: "qsa select" | f16x2 |
|---|---|---|
| 17k | 368 ms | 272 ms |
| 99k | 1395 ms | 989 ms |
| 176k | 2345 ms | 1653 ms |
| 226k | 3437 ms | 2505 ms |

Las dos corridas de producción coinciden en esa fase dentro de ~1 %; f16x2 la baja 26 % a 17k y 27-30 % desde 34k.
Las velocidades de lectura de esta prueba no sirven para comparar: las respuestas de f16x2 son otras desde el primer
turno, así que los turnos siguientes leen otro texto y otros expertos.

**Con los mismos tokens** (las lecturas forzadas de la compuerta de calidad, sin el cliente de ruido):

| Lectura | Producción | f16x2 | Velocidad |
|---|---|---|---|
| 180k desde cero | 146.7 s | 137.1 s | +7.0 % |
| 16 chunks de 3008, de 183k a 229k | 47.8 s | 45.6 s | +4.8 % |
| 38 chunks de 3008 y 6016, de 7k a 47k | 134.4 s | 133.5 s | +0.7 % |

El kernel FP32 (el default de Strata), en la misma prueba, lee los chunks profundos 25 % más lento que producción.

**En producción** (`weight_prod.py`): f16x2 ahorra 2.36e-7 ms por token leído y por celda de profundidad (29 % del
"qsa select"). Aplicado a las 1726 lecturas de más de 64 tokens del log del motor (`~/.strix-engine.log`, 2736
peticiones del 2026-10-07 al 2026-11-04; las lecturas de 64 tokens o menos van por las ventanas de verificación, que no
cambian):

| Lecturas que empiezan en | Lecturas | Tiempo | Ahorro |
|---|---|---|---|
| 0-64k | 776 | 53.5 min | 1.0 % |
| 64k-128k | 512 | 14.0 min | 1.3 % |
| 128k-192k | 262 | 8.1 min | 2.4 % |
| 192k+ | 176 | 5.0 min | 2.4 % |

Total: 1.05 min de 86.3 de lectura (1.2 %), 0.2 % de lectura más decode.

## 5. Calidad

**El criterio, fijado antes de medir:** la regla del Director del 2026-09-30, con el método de
`short-reads-2026-10-31.md` (sección 3): la KL contra producción a lo sumo el doble de la de una referencia de
redondeo, el argmax a lo sumo 0.5 puntos debajo, Δ ln p sin empeorar más allá de su error estándar. La referencia es
el kernel FP32 del puntaje (`STRATA_SELECT_WMMA=0`), el default de Strata: otro redondeo de la misma selección.
Producción usa el WMMA desde que pasó a Strata (strix-halo `f6b98ca`, 2026-10-07); la referencia mide cuánto movió
las filas ese cambio.

**Método** (`chain-quality.sh`, `band_logpos.py`): el motor solo a 262144, filas de log-probs (top 20) de las
ventanas que siguen a cada chunk leído por la ruta de prompt; wiki y código en chunks de 3008 y 6016 hasta ~47k, y
wiki + código en chunks de 3008 de 180k a 229k.

| Secuencia | n | f16x2: KL / argmax / Δ ln p | Referencia: KL / argmax / Δ ln p |
|---|---|---|---|
| wiki-3008 | 756 | 0.0047 / 97.49 % / +0.0045 | 0.0040 / 97.88 % / −0.0030 |
| código-3008 | 756 | 0.0042 / 98.28 % / −0.0029 | 0.0040 / 98.15 % / −0.0000 |
| wiki-6016 | 441 | 0.0036 / 97.96 % / −0.0067 | 0.0029 / 98.41 % / −0.0012 |
| código-6016 | 441 | 0.0065 / 98.19 % / +0.0009 | 0.0068 / 97.96 % / +0.0071 |
| 180k-229k | 1008 | 0.0057 / 97.82 % / −0.0021 | 0.0055 / 98.21 % / −0.0009 |
| **Total** | 3402 | **0.0049 / 97.91 % / −0.0010 ± 0.0023** | 0.0047 / 98.12 % / −0.0002 ± 0.0023 |

Pasa: KL 1.06 veces la de la referencia, argmax −0.21 puntos, Δ ln p −0.0009 con error estándar 0.0033. Cada
secuencia pasa por separado.

## 6. Determinismo e historia

Con la rama y `f16x2`:

- **Relecturas** (`chain-quality.sh`, junto al cliente de ruido Vulkan): wiki-3008 y código-3008, 8 de 8 idénticas.
  En la secuencia profunda el script marca "0 of 2" en las tres cargas, producción incluida: la caché de
  conversaciones restauró 163840 tokens del prefijo y el script exige leerlo desde cero. Las filas son 63 de 63
  idénticas en cada relectura, después de leer 19k tokens a 164-183k.
- **Conversación a profundidad** (`gates.sh conv`, la de las compuertas del top-k: 39.5k de prosa, una imagen de
  2048×2048 a esa profundidad, código hasta 139k celdas, una respuesta de 256 tokens y la misma pregunta cuatro veces
  más; 163840 × 2 slots): f16x2, f16x2 después de la imagen en el otro slot y f16x2 otra vez dan respuestas y filas
  idénticas en cada paso, y las cinco respuestas a la última pregunta son iguales en cada carga.
- **A 146k junto a otro cliente de la GPU** (`gates.sh deep`, 318 iteraciones de `test-backend-ops` durante la
  corrida): idéntica a la corrida sin el cliente.
- **Dos slots** (`batch_interleave_test.py`, lecturas de 17400 y 19700 tokens mientras dos slots decodifican): cada
  slot idéntico a su corrida en solitario, también la lectura que cede el paso y sigue.
- **Historia en sus dos mitades** (`chain-hist.sh`, 65536 × 2, la conversación de `hist_reads.py`: 39.5k de prosa,
  lecturas de 556 y 2983 tokens, una respuesta de 512): un servidor nuevo y uno que leyó antes la imagen de 2048×2048
  en el otro slot dan las mismas 515 filas y las mismas respuestas, con los drafts de producción y sin drafts.

## 7. Adopción

- **Strata:** `3f6fb965` en `main` del fork (la rama `select-wmma-terms`, fast-forward). Se publicó
  `9c012d0b..3f6fb965` junto con dos commits de visión del mismo día de otra sesión (`4f60e029`, `03d9fd8f`), con el OK
  del Director.
- **El binario:** `strix-halo/build-strata.sh`. Antes de cambiar la configuración, el build de producción con f16x2
  repitió los tres primeros turnos de la corrida que pasó las compuertas con respuestas idénticas (`chain-verify.sh`).
- **La configuración:** `STRATA_SELECT_WMMA_FORMAT=f16x2` en `strata/qwen38flash.json` y la nota en `config.ini`
  (strix-halo `d1b21df`). Producción se recargó a las 17:14 con 262144 × 2; el motor tiene la variable en su entorno.
- **Para volver atrás:** sacar la variable de la plantilla y recargar. `bf16x3` es el default y da los mismos bits
  que antes del cambio (las 15 respuestas de la sección 4), sin recompilar.

## 8. Lo que queda: la selección en el decode

El decode no se tocó. Su puntaje + top-k pesa 4.7 % de la ventana a 99k y 8.3 % a 226k (con el perfil, sección 2);
con la distribución de profundidad del decode de producción (`topk/decode_depth.out`) es ~4.5 % del tiempo de decode,
~3.7 % del tiempo del motor (lo que ahorra f16x2 es 0.2 %). El puntaje del decode (`qsa_block_scores`, FP32) lee las
claves agrupadas en fp32: 29 MB por capa y por ventana a 226k. Bajar ese tráfico requiere guardar las claves con menos
bits, como LSA. Guardarlas en una pieza fp16 lo reduce a la mitad, pero ese formato falla la compuerta de precisión del
bench; si pasara la de calidad, el techo sería ~1-1.5 % del decode. Lo que falta saber primero es cuánto de ese 4.5 %
es puntaje y cuánto top-k, que el perfil no separa.

## Evidencia

`~/ProjectHub/llama.cpp-evidence/idxprec/`:
- el perfil: `chain-profile.sh`, `depth_profile.py`, `profile_table.py`, `profile-table.out`, `cfg-prof-engine.log`;
- el bench: `build.sh`, `build.out`, `bench.sh`, `bench.out`;
- el motor: `chain-ab.sh`, `ab_compare.py`, `chain-ab.out`, `ab-*.jsonl`, `cfg-ab-*-engine.log`;
- la producción pesada: `weight_prod.py`, `weight-prod.out`;
- la calidad: `chain-quality.sh`, `deep.ids.txt`, `quality_pool.py`, `quality-pool.out`, `q-*`;
- el determinismo: `gates.sh`, `gates.out`, `g-*`, `bi-f16x2-*`, `chain-hist.sh`, `chain-hist.out`, `hist-f16x2-*`;
- la adopción: `build-final.out` y `bench-final.*` (el código final de la rama), `build-prod.out`, `chain-verify.*`,
  `verify.jsonl`, `reload.out`;
- los enlaces a `shortread/` (`hist_reads.py`, `hist_check.py`, `speed_reads.py`) y `mkcfg.py`, que arma cada
  configuración desde la de producción.
