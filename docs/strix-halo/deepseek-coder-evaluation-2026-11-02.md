# DeepSeek-V4.1-Flash Coder de ISTA-DASLab en esta máquina (2026-11-02)

El Director pidió probar `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF` (revisión `54229d38`) con el motor
de DeepSeek de Strata upstream (`strata-dsv41`, `c62d5e3a`, versión 0.1.55). ISTA-DASLab es el laboratorio de Dan
Alistarh en el Institute of Science and Technology Austria, autor de GSQ y RCO.

## 0. Resultado

**Corre y el motor calcula bien el modelo, pero no reemplaza a qwen38flash.** Responde a la mitad de su velocidad,
lee prompts a un tercio y en español se rompe.

| | DeepSeek Coder (Strata, DSpark) | qwen38flash (producción) |
|---|---|---|
| Decode, código corto / código de ~42k | 35.9 / 25.9 t/s | 71.7 / 50.3 t/s |
| Lectura de prompt, ~5k / ~42k | 272 / 449 t/s | 1073 / 1335 t/s |
| GTT con el modelo cargado | 79.2 GiB (contexto de 65536) | 77 GiB (262144 × 2 slots) |
| Español | cambia al inglés a mitad de la oración | correcto |
| Validar un RUT chileno, con razonamiento | algoritmo inventado, en inglés, 4 min | correcto, en español, 53 s |

- **Corrección del motor:** Strata contra llama.cpp en CPU, sobre los mismos pesos y los mismos 4995 tokens, coincide
  en el token más probable el 95.9 % de las veces, con KL medio 0.012. Los dos caminos de cálculo de Strata difieren
  entre sí lo mismo (95.9 %, 0.012). Es redondeo, amplificado por los 2 bits de los pesos (sección 3).
- **Calidad:** ISTA reporta 77.0 en SWE-bench Verified (el modelo base: 81.2), en inglés y en código. En español el
  modelo pasa al inglés a mitad de la oración, con y sin razonamiento, y llega a meter palabras en polaco. En una tarea
  práctica (validar un RUT chileno) inventó el algoritmo y respondió en inglés (sección 5).
- **El motor de upstream tenía dos problemas en Linux sobre una APU.** El loader dejaba los 66 GB de expertos en la
  caché de páginas mientras la GPU ya tenía 72 GB fijados, el patrón del freeze #9; quedó corregido en la rama de
  evaluación. Además, una caché de expertos en RAM que el motor dimensiona solo duplicaba expertos que ya estaban en
  la GPU; se apaga por configuración (sección 6).

## 1. El modelo

- **Qué es:** DeepSeek-V4.1-Flash-1017 (40 capas, 384 expertos enrutados más uno compartido, 6 activos por token,
  atención comprimida, Engram en las capas 1 y 14, contexto nativo de 1M) podado con RCO a 192 expertos por capa,
  elegidos con datos de código, y cuantizado con GSQ.
- **Los archivos:**
  - Transformer: 67.2 GiB en 5 shards. Expertos IQ1_M (27.7 GiB), IQ2_XXS (22.8) e IQ2_XS (11.0); denso mayormente
    Q5_K.
  - Tablas Engram: 103 GiB en IQ4_NL, el sexto shard. Se leen unas filas por token y quedan en disco.
  - Proyector de visión: 0.9 GiB.
  - El drafter DSpark (3 bloques, 7.4 GiB en Q8_0) no viene en el repo de ISTA: es el de Unsloth, como en la receta de
    Strata.
- **La descarga:** los 183 GB verificados contra los SHA-256 de LFS (`verify.out`).
- **Bytes por token:** 7.7 GB (denso 4.2, expertos enrutados 2.1, compartido 1.0, cabeza 0.5). Con el bus de 212 GB/s,
  el techo de decode sin drafter es ~28 t/s.

## 2. El motor en Linux

Strata solo publica `strata-dsv41` para Windows en Strix Halo: no hay paquete de Linux. Se compiló de las fuentes de
upstream en un worktree aparte (`~/ProjectHub/strata-wt-dsv41`, rama `dsv41-upstream`), con la misma configuración
HIP que nuestro build de producción más `STRATA_BUILD_DSV41=ON`. El motor de producción no se tocó.

- **Paridad de kernels (`ds4_iq_test`):** IQ2_XXS, IQ1_M e IQ2_XS pasan todas las filas. Falla una sola, IQ4_XS
  wmma contra el escalar (6.7e-3 contra una tolerancia de 3e-3), en un tipo que el Coder no usa.
- **El pack:** la receta de setup de Strata para un set de ISTA (`pack.sh`). Son punteros a los GGUF, sin copiar
  pesos. `--pack-info`: 40 capas × 192 expertos, todos los tipos con kernel de GPU.
- **La configuración:** la que setup escribe para Strix Halo (`DS4_ASYNC_T=4`, prompts en bloques de 8192), más:
  - `--experts 6` fijo. En automático, el motor usa 5 de los 6 expertos en una GPU integrada, y eso cambia el modelo.
  - `--ssd-cache 0` (sección 6).

## 3. Corrección: Strata contra llama.cpp en CPU

El set de calidad de Strata (`data/quant_quality/v1`: código, documentación, inglés, otros idiomas y una transcripción
de agente) son 5 segmentos de 1000 tokens. Es el mismo eval de 4995 posiciones que usa ISTA. Lo tokenizó el
tokenizador del propio modelo, el de llama.cpp (`mkids.py`). El de Strata (`quant_quality.py ids`) arma su tokenizador
sin el `pre` del GGUF y corta el texto de DeepSeek con el pre-tokenizador de Qwen.

- **Strata:** `strata-dsv41 --score` (`score-strata.sh`): los 6 expertos, sin drafter, por sus dos caminos (ventanas
  de 16 tokens, que usa el decode, y bloques de 128, que usa la lectura de prompts).
- **Referencia:** llama.cpp upstream (b12030) en CPU, un lote por segmento (`score_ids.cpp`).

| Comparación (las 4995 posiciones) | Top-1 igual | KL medio | Perplejidad |
|---|---|---|---|
| CPU contra Strata, ventanas de 16 | 95.86 % | 0.0122 | 4.693 → 4.663 |
| CPU contra Strata, bloques de 128 | 95.76 % | 0.0105 | 4.693 → 4.687 |
| Strata, ventanas de 16 contra bloques de 128 | 95.88 % | 0.0123 | 4.663 → 4.687 |

Llama.cpp y Strata difieren lo mismo que los dos caminos de Strata entre sí. Como referencia de escala, Strata publica
KL 0.170 entre el set 2.00 de ISTA y el modelo original.

| Segmento (Strata, ventanas de 16) | Perplejidad | Top-1 contra el texto |
|---|---|---|
| Código | 2.24 | 81.8 % |
| Documentación | 11.98 | 45.8 % |
| Inglés | 6.19 | 53.0 % |
| Otros idiomas | 6.24 | 62.3 % |
| Agente | 2.13 | 80.9 % |

## 4. Velocidad

El protocolo de la evaluación de Strata (`bench.py`): cada prompt leído entero y luego una segunda pregunta en la
misma conversación; greedy, sin razonamiento, 256 tokens por respuesta. Contexto de 65536, con un servidor por
configuración.

| Decode, t/s (primer / segundo turno) | Sin drafter | DSpark | Drafts aceptados (DSpark) |
|---|---|---|---|
| Código corto | 22.0 / 20.9 | 35.9 / 28.2 | 173/192 · 144/173 |
| Agente, 5.1k | 21.0 / 21.1 | 24.9 / 29.1 | 107/123 · 142/164 |
| Prosa, 37.9k | 20.8 / 20.8 | (no se corrió) | |
| Código, 42.7k | 20.8 / 20.7 | 25.9 / 30.6 | 125/157 · 153/166 |

| Lectura de prompt | Sin drafter | DSpark |
|---|---|---|
| Agente, 5.1k | 273 t/s | 272 t/s |
| Prosa, 37.9k | 433 t/s | (no se corrió) |
| Código, 42.7k | 450 t/s (95 s) | 449 t/s |

- **El decode no cae con la profundidad:** 21 t/s de 0 a 43k sin drafter.
- **DSpark acepta entre 80 % y 92 % de los borradores** y sube el decode entre 19 % y 63 %.
- **La segunda pregunta reutiliza la conversación:** lee 27 tokens.
- **La lectura** es mejor que la medida por Strata en Windows (177 t/s a 32k), donde ese set no cabía en memoria y
  leía 23 GB del SSD.
- **Arranque:** 25-27 s.

### 4.1 Contra qwen38flash

Es el mismo bench, corrido el mismo día sobre el motor y la configuración de producción (262144 × 2 slots, MTP; cadena
`qwen/chain.sh`). Los prompts tienen otra cantidad de tokens porque los tokenizadores son distintos.

| | DeepSeek Coder, DSpark | qwen38flash |
|---|---|---|
| Decode, código corto (primer / segundo turno) | 35.9 / 28.2 t/s | 71.7 / 58.5 t/s |
| Decode, código de ~42k | 25.9 / 30.6 t/s | 50.3 / 52.1 t/s |
| Decode, prosa de ~38k (DeepSeek sin drafter) | 20.8 / 20.8 t/s | 48.0 / 48.3 t/s |
| Lectura, agente (~5k) | 272 t/s | 1073 t/s |
| Lectura, código de ~42k | 449 t/s (95 s) | 1335 t/s (31 s) |

En el prompt de agente, qwen38flash respondió con llamadas a herramientas: 167 y 86 tokens en lugar de 256, así que
su decode ahí no es comparable.

## 5. Calidad en el uso real

- **El bench, con la pregunta en español:**
  - En el prompt de agente sin drafter respondió "Analisis del the text" y siguió en inglés.
  - Con DSpark respondió "El texto anterior esté un mensaje de un agent (assistant) dentro a coding-agent harness".
  - Los demás prompts respondieron en inglés.
- **Pregunta corta en español** (`probe.py`, "explica qué es una tabla hash"):
  - Sin razonamiento empieza en español y pasa al inglés: "almacenía clave–value pairs and uses a hash function…".
  - Con razonamiento (esfuerzo 50, el de la ficha de ISTA) hace lo mismo y mete una palabra en polaco: "almacenuje
    wartości asociados with keys".
- **Tarea de código en español:** validar un RUT chileno, con pruebas.
  - Pensó 7082 tokens (4 minutos a 31.5 t/s) y respondió en inglés.
  - El algoritmo está mal: un RUT de 11 dígitos con pesos [3, 2, 7, 6, 4, 8, …]. El real tiene 7-8 dígitos y pesos 2-7
    cíclicos desde la derecha.
  - Además rechaza la "k" minúscula.
  - qwen38flash, con la misma pregunta y razonamiento "medium": respuesta en español, el módulo 11 correcto, la "k"
    minúscula aceptada, 3750 tokens en 53 s.
- **No es Strata:**
  - El servidor tokeniza los prompts de la prueba igual que llama.cpp (26 y 57 ids idénticos).
  - La misma pregunta, en modo chat y greedy, por llama.cpp upstream en CPU (`cpu-explain.sh`), sin nada de Strata en
    el camino, también cambia de idioma y mezcla francés: "usando una función hash pour convertir a cada key into an
    index (bucket) in an array…".
- **La causa probable:** la poda. RCO conservó la mitad de los expertos, elegidos con datos de código. La perplejidad
  en otros idiomas está bien (6.24), pero generar texto fuera del código perdió capacidad. Strata ya medía en el set
  IQ1_M de ISTA (1.75 bpw, sin podar), contra el modelo original, +14 % de perplejidad en otros idiomas y +53 % en
  prosa en inglés.

## 6. Memoria y lo que hubo que resolver en el motor

| | Sin drafter | DSpark |
|---|---|---|
| GTT con el modelo cargado | 71.7 GiB | 79.2 GiB |
| MemAvailable con el modelo cargado | 31.4 GiB | 23.2 GiB |
| MemFree mínimo en el bench | 13.9 GiB | 11.7 GiB |

1. **El loader dejaba los expertos en la caché de páginas** (corregido, commit `14d42f1a` en la rama de evaluación).
   - **Qué pasaba:** Strata etiqueta todo experto de un GGUF con el formato MXFP4 (`ds4_model.cu:1960`), así que la
     subida va por `load_experts_mx`. Ese camino libera las páginas de los expertos recién después del último. El
     motor reserva antes los 72 GB en la GPU, y en esta APU esa memoria es la misma RAM.
   - **Lo que se vio:** en la primera carga, MemFree cayó de 25 a 5 GiB a un tercio de la subida y el fusible de la
     compuerta la cortó, dos veces.
   - **La corrección:** cada experto libera sus páginas apenas sube, como ya hacía el otro camino de carga del motor.
     MemFree mínimo: 23.7 GiB, con la caché plana durante toda la subida.
   - **Lo que queda:** 0.93 de los 61.5 GiB de rangos de expertos siguen en caché al terminar. Son las páginas que
     comparten partes vecinas no alineadas (`fadv2.py`, `incore_experts.py`).
2. **La caché de expertos en RAM ("RAM store") duplicaba la GPU** (apagada con `--ssd-cache 0`).
   - **Cómo se dimensiona:** en automático, la RAM libre menos 12 GB (13.8 a 22.4 GB acá).
   - **Cuándo se llena:** al arrancar, en cuanto existe un perfil de expertos. El primer servidor escribió ese perfil
     al cerrarse, y en el arranque siguiente el motor leyó 702 expertos (5.5 GB) en 1.3 s.
   - **Lo que pasó:** el fusible cortó la carga con DSpark en ese paso.
   - **Por qué sobra acá:** los 8064 expertos ya están en la GPU y el carril de CPU está vacío, así que solo copiaba
     expertos a la misma memoria unificada.
   - **El bench sin drafter** corrió antes de este ajuste: la store quedó configurada pero vacía, porque todavía no
     había perfil (el RSS del motor se mantuvo en 4.8 GiB). Sus cifras no cambian.
3. **Las filas del Engram se leen por la caché de páginas.** No se corrigió: es caché limpia que el kernel recupera.
   La lectura directa solo existe en Windows (`ds4_model.cu:2867`). La caché del Engram llegó a 8.8 GiB después de
   ~100k tokens. Una sesión larga la hace crecer hasta que el kernel la recupera, y una carga posterior necesita
   `gate.py settle` antes.

Si DeepSeek quedara corriendo fuera de una corrida de la compuerta, `gate.py` y el ram-guard tendrían que reconocer
`strata-dsv41`, que hoy no reconocen. No hace falta mientras no se adopte.

## 7. Si fuera yo

No lo adoptaría. Producción con qwen38flash es mejor en lo que usamos todos los días:
- responde más rápido;
- lee los prompts largos de pi en una fracción del tiempo;
- trabaja en español.

El 77.0 de SWE-bench del Coder es en inglés y en código, y lo poco que pierde ahí se paga con todo lo demás.

Lo que me haría cambiar de opinión: que en tareas reales de pi en inglés resuelva lo que qwen38flash no resuelve. Esa
prueba cuesta poco: el pack y el motor quedan listos, el servidor arranca en 27 s y la cadena es `bench-chain.sh`.

## Evidencia

`~/ProjectHub/llama.cpp-evidence/dsv41/` (su fila en el README de la carpeta), entre otros:
- el build (`build-dsv41.sh` y su `.out`) y la paridad (`iq_test.out`);
- el pack (`pack.sh`, `pack.out`) y la verificación de la descarga (`verify.sh`, `verify.out`);
- los puntajes (`set.ids.json`, `score-*.tsv`, `compare-*.txt`);
- el bench y la prueba en español (`bench-*.jsonl`, `probe-*.jsonl`, `chain-*.out`, `server-*.log`, `engine-*.log`),
  la referencia de qwen38flash (`qwen/`), la comparación de tokenizadores (`tokcmp.py`) y la respuesta de llama.cpp en
  CPU (`cpu-explain.sh`, `.out`);
- las trazas de memoria (`memmon-*.log`).

La corrección del loader es `14d42f1a` en `~/ProjectHub/strata-wt-dsv41`, rama `dsv41-upstream`, sin push.
