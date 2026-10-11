# qwen38flash y "el mensaje del usuario está vacío": un modo del propio modelo que la cuantización amplifica (2026-10-10)

Investigación pedida por el Director tras una sesión de pi con Strata en producción en la que el pensamiento del modelo
abrió con "The user's message is empty — there's no actual question or content beyond the conversation start marker"
sobre una petición con un mensaje real. Herramientas y salidas: `~/dbg/strata/empty/` (reproductor, réplica de sesiones,
muestreo) y `~/dbg/strata/empty/official/` (la referencia con el código oficial).

## 0. Resultado

- **No es un defecto de Strata ni de llama.cpp.** Con los mismos pesos, el código oficial del modelo (transformers
  5.17.0) produce el modo y los dos motores le son fieles (sección 5). El render y la plantilla dan los mismos tokens
  que la plantilla oficial del checkpoint (sección 1).
- **Es un modo del propio modelo.** El checkpoint oficial en BF16 abre el pensamiento con él en el primer turno
  afectado de la sesión con probabilidad 0.43 ("The user hasn't asked anything substantive yet", "The user's
  message is empty — there's no", el texto exacto de pi), igual que el GGUF de producción (0.46) (sección 10). Aparece
  en turnos de agente cuyo último mensaje "user" es una respuesta de herramienta: el modelo concluye que el usuario no
  pidió nada o que solo hay instrucciones de sistema. Las pasadas de referencia usan la tabla PLE del GGUF (IQ4_NL);
  con las filas oficiales de la tabla el inicio se mantiene: "The user hasn't asked" 0.39 en req2 (sección 10).
- **En el contexto largo de req8 la cuantización de los expertos lo amplifica.** Con los siete pensamientos anteriores
  que conserva la plantilla, BF16 abre con el modo 0.23-0.40 y el GGUF 0.87-0.96 (sección 10). No depende del
  pensamiento anómalo del turno 2: quitar cualquiera de los siete por separado deja el modo (sección 3). Las secciones 7 y 9
  la localizan con una medida forzada (gate/up IQ3_S cerca de tres cuartos, down IQ4_NL el resto, repartida en todas
  las capas); el control nulo de la sección 9 dice si esa medida sigue la precisión.
- **No hay arreglo de motor.** Las palancas (sección 11) son decisiones del Director: más precisión en los expertos
  (lo reduce en req8, no el inicio, y no cabe en la memoria de producción) o no conservar los pensamientos anteriores
  (lo elimina en req8 por una razón que no se aisló; no toca el inicio).

## 1. El síntoma

La sesión (`~/.pi/agent/sessions/--home-daniel-ProjectHub-media-stack--/2026-10-10T01-37-19-261Z_01a12375-….jsonl`, copia
en `~/dbg/strata/empty/session-media-stack-1010.jsonl`) tiene 11 turnos del asistente. Tres abren el pensamiento con el
modo: el turno 2 ("…the user hasn't responded"), el 8 ("The user's message is empty — … conversation start marker") y
el 9 (el mismo texto). El modelo cita frases que no están en el prompt ("expert software engineer", "Always invoke a
function call", "Understood. I will follow these instructions": 0 apariciones).

El render no es la causa: las peticiones reconstruidas con el código del servidor de Strata (`render.py`) dan
exactamente los tokens que pi registró (entrada + caché leída), cada prompt es prefijo exacto del siguiente y el
mensaje del usuario está. La plantilla del pack de Strata es la de Unsloth (fusiona mensajes de sistema, acepta el rol
developer, convierte el esfuerzo "high" en "xhigh", valida los argumentos de las llamadas), no la del checkpoint
oficial; con la oficial, req2 y req8 dan los mismos tokens (8970 y 13323, idénticos uno a uno).

Tasa en el campo (expresión regular sobre los primeros 300 caracteres del pensamiento, todas las sesiones de pi):

| Modelo y motor | Turnos con el modo | Sesiones con alguno |
|---|---|---|
| qwen38flash, llama.cpp antes del arreglo de la atención del 2026-09-09 | 32 / 869 | 2 de 17 |
| qwen38flash, llama.cpp 2026-09-09 a 10-06 | 2 / 2342 | 2 de 87 |
| qwen38flash, Strata 2026-10-07 a 10-10 | 3 / 356 | 1 de 8 (los 3 en esta sesión) |

## 2. El reproductor

`repro_set.py <puerto> <tag>`: 14 prompts de esa sesión, greedy, 40 tokens, esfuerzo medium, clasificados por
`classify.py` (ANOM / sane / other): la petición 8 con 0, 1, 2, 3, 4, 6, 8, 12, 24 y 48 palabras de relleno al final
del prompt de sistema (perturbaciones mínimas que mueven el empate), la petición 8 sin los pensamientos anteriores (con
0 y 12 de relleno), la 9 y la 2.

| Motor | GGUF | Rellenos ANOM / 10 | sin pensamientos +0 / +12, req9, req2 |
|---|---|---|---|
| Strata, producción | `qwen4exp-ud-small-nl3s` | 9 (+1 other) | sane, sane, sane, sane |
| llama.cpp | el mismo | **10** | sane, sane, sane, sane |
| llama.cpp | `qwen4exp-nl3s-hcdi-gu-rq8-oproj8` (el anterior de llama.cpp) | 6 | sane ×4 |
| llama.cpp | Unsloth UD-IQ4_XS | 7 (+1 other) | sane, sane, **ANOM**, sane |
| llama.cpp, selección densa (`indexer.top_k` 32768) | `qwen4exp-ud-small-nl3s` | 10 | sane, other, sane, sane |

No es un defecto de Strata: llama.cpp con los mismos pesos lo reproduce más; tampoco de la atención dispersa (con la
selección densa sigue) ni exclusivo de nuestra receta (UD-IQ4_XS completo también).

## 3. `preserve_thinking` no es el arreglo

La plantilla del modelo conserva por defecto el pensamiento de los turnos anteriores (la del GGUF y la oficial; la
ficha del modelo lo dice). Quitarlo hace desaparecer el modo en la petición 8 (muestreo a T 0.7 y 1.0, 25 semillas:
25/25 con el pensamiento, 0/25 sin él), pero:

- la petición 2 no tiene pensamientos anteriores que quitar y abre con el modo 9/25 (T 0.7) y 8/25 (T 1.0);
- quitar cualquiera de los 7 pensamientos de la petición 8 por separado deja el modo 7/7;
- sin los pensamientos, la petición 8 con 4 de relleno inventa otra cosa ("otelclaw… 127.0.0.1:43104", ausente del
  prompt).

Cambia el prompt que el modelo espera, mueve el empate y no toca la causa. El Director lo descartó ("Quiero un fix, no
un workaround"); la implementación quedó en un stash de los dos repos ("preserve_thinking (not adopted)").

## 4. El GGUF contra el checkpoint oficial

Con el código oficial (transformers 5.17.0, `qwen4_exp`) y el checkpoint `Qwen/Qwen3.8-Flash-Next` leído por rangos HTTP
(`official_hf.py`), las transformaciones del conversor deshechas en `gguf_weights.py` (normas centradas en cero, las
cabezas de valor de la GDN de agrupadas a intercaladas, `A_log`, q/k del indexador y gate/up concatenados):

- los tensores F32/BF16 chicos son exactos; los Q8_0, error relativo 0.006;
- las proyecciones grandes: in_proj_qkv 0.043, in_proj_z 0.091, out_proj 0.089, QSA q/k/v/o 0.087 / 0.104 / 0.048 /
  0.006; los expertos gate/up 0.14-0.15 (IQ3_S), down 0.076-0.079 (IQ4_NL);
- la tabla PLE: correlación por fila 0.997, cociente de normas 0.996 (0.988-1.002), error 0.078 (IQ4_NL); sus
  proyecciones y el embedding, 0.005: sin errores de escala;
- los metadatos de la PLE (multiplicadores, tamaños de vocabulario por cabeza, semilla 1234) coinciden con los que da
  la configuración oficial.

## 5. La referencia oficial con los pesos del GGUF

`ref_forward.py`: los módulos oficiales sin cambios, una capa a la vez sobre la secuencia entera, fp32 en la GPU, la
atención con la matemática eager oficial en bloques de 512 consultas (para no materializar 13k × 13k) y la tabla PLE
leída por filas del GGUF.

| Prompt corto (1900 tokens, 899 posiciones) | KL | Coincidencia del argmax |
|---|---|---|
| Referencia contra llama.cpp | 0.156 | 88.65 % |
| Referencia contra Strata | 0.098 | 91.21 % |
| llama.cpp contra Strata | 0.146-0.157 | 89.43 % |

Sobre la petición 8 con la continuación anómala de Strata, el argmax de la referencia es el token de la continuación en
las 21 posiciones impresas; la probabilidad conjunta de los 40 tokens es 0.0084 en la referencia, 0.0206 en llama.cpp y
0.0159 en Strata (KL sobre esas posiciones 0.058 y 0.067). **Con estos pesos, el código oficial produce el modo: los
dos motores son fieles.**

## 6. BF16 contra GGUF en el código oficial

`ref_pair.py`: la misma pasada con los pesos BF16 del checkpoint (descargados capa a capa mientras corre la anterior)
y con los del GGUF; la tabla PLE es la del GGUF en las dos (sus 320 M de filas no se pueden traer fila a fila). Regla
fijada antes del resultado (`pair_score.py`): una secuencia es anómala con unos pesos si su argmax sigue la
continuación anómala 5 tokens o más y el tercer token (lo que sigue a "The user") tiene p > 0.5.

| Continuación | p(3.º) BF16 → GGUF |
|---|---|
| req8+0 | 0.121 → **0.775** |
| req8+1 | 0.180 → **0.657** |
| req8+2 | 0.053 → 0.271 |
| req8+3 | 0.312 → 0.016 |
| req8+4 | 0.298 → **0.667** |
| req8+12 | 0.240 → 0.364 |
| req2 | 0.058 → **0.647** |

BF16 0/7, GGUF 4/7. En el prompt corto, GGUF contra BF16: KL 0.43, argmax 83.1 %, PPL 3.40 → 3.48. Entre corridas la
referencia varía hasta ~0.01 en p(3.º) (cómputo no determinista de la GPU), muy por debajo de la diferencia.

Estas cifras son las de **las continuaciones que escribió el GGUF**, forzadas: miden cuánto prefiere cada modelo esas
aperturas, no con qué frecuencia el modelo BF16 abriría con el modo por su cuenta (sección 10).

## 7. Bisección de la cuantización

`ref_variants.py`: la referencia con conjuntos de pesos mezclados en una pasada (lado denso BF16 o GGUF; expertos BF16,
GGUF o cuantizados desde BF16 con el cuantizador de ggml). Media de p(primeros 10 tokens) sobre las 4 continuaciones
anómalas: cuánto prefiere el modelo la apertura exacta.

| Pesos | Anómalas / 4 | Media p(primeros 10) | KL corto contra BF16 |
|---|---|---|---|
| BF16 | 0 | 0.034 | — |
| GGUF de producción | 4 | 0.246 | 0.425 |
| Denso BF16 + expertos del GGUF | 1 | 0.104 | 0.381 |
| Denso del GGUF + expertos BF16 | **0** | **0.049** | 0.232 |
| GGUF con gate/up IQ4_XS (sin imatrix) | 1 | 0.171 | 0.364 |

**Sobre estas continuaciones, lo que más pesa es la cuantización de los expertos.** Con los expertos sin cuantizar la
preferencia vuelve al nivel de BF16 aun con el lado denso cuantizado; con el denso sin cuantizar y los expertos del
GGUF baja a la mitad. El lado denso pesa menos. La sección 10 corrige el alcance: el modelo BF16 abre con el modo por su
cuenta; lo que esta tabla mide es cuánto lo amplifica cada parte de la cuantización. Los gate/up de los expertos (IQ3_S, 3.44 bpw) son los de Unsloth copiados tal cual
(`qwen4exp-cost-breakdown-2026-09-09.md`, sección 7); los down, IQ4_NL.

Recalibrar no basta. Error ponderado por el imatrix (capa 5, expertos 0-3) contra BF16:

| Cuantización de gate/up | Error |
|---|---|
| IQ3_S de producción (Unsloth) | 0.0168-0.0185 |
| IQ3_S desde BF16 con nuestro imatrix | 0.0160-0.0175 |
| IQ3_S sin imatrix | 0.027-0.029 |
| IQ4_XS con o sin imatrix | 0.0054-0.0062 |

El IQ3_S de Unsloth ya está calibrado. IQ4_XS baja el error 3× y en la bisección solo llega a mitad de camino.

## 8. Lo que admiten los motores

Strata (build de producción, HIP gfx1151): cada capa puede tener su tipo de expertos (`native_experts.txt`, una línea
por capa); gate y up comparten tipo; down admite solo tipos de bloque de 32 (sus filas son de 640: IQ4_NL, Q8_0, Q5_0,
Q4_0). Para gate/up: IQ3_S e IQ4_XS por la ruta fusionada completa; Q8_0 por MMQ (una capa no fusionada cambia el
dimensionado del prefill de todo el pack: hay que medir su costo); Q4_K/Q5_K con prefill FP16 salvo build con
`STRATA_MMQ_KQUANTS`; Q6_K rechazado al cargar. llama.cpp Vulkan tiene `MUL_MAT_ID` para Q5_K, Q6_K, Q8_0 e IQ4_XS.

## 9. Localización

Misma medida que la sección 7, con el lado denso del GGUF en todas y la referencia "denso del GGUF + expertos BF16":

| Pesos de los expertos | Anómalas / 4 | Media p(primeros 10) | KL corto contra la referencia |
|---|---|---|---|
| BF16 (la referencia; 0.049 en la pasada de la sección 7) | 0 | 0.053 | — |
| gate/up BF16, down del GGUF | 1 | 0.106 | 0.138 |
| gate/up Q5_K con imatrix, down del GGUF | 0 | 0.092 | 0.139 |
| BF16 en las capas 0-23, GGUF en 24-47 | 2 | 0.146 | 0.151 |
| GGUF en las capas 0-23, BF16 en 24-47 | 0 | 0.097 | 0.229 |
| GGUF completo (sección 7) | 4 | 0.246 | — |

- Gate/up lleva cerca de tres cuartos del exceso; down (IQ4_NL) el cuarto restante.
- Q5_K en gate/up queda al nivel de gate/up sin cuantizar: la diferencia entre los dos (0.092 contra 0.106) es la
  resolución de la medida, no una mejora.
- Ninguna mitad de las capas lo concentra: la segunda pesa más (0.093 de exceso contra 0.044) y juntas dan más que la
  suma (0.193).
- req2 aporta ~1e-10 en todas las variantes: la media son, en la práctica, las tres continuaciones de la petición 8.

Costo de las recetas que la medida señala (gate/up en Strata: IQ3_S 0.672 GiB por capa; IQ4_XS +0.159, Q5_K +0.403;
down IQ4_NL 0.439, Q5_0 +0.098): Q5_K en las 48 capas +19.3 GiB, solo en la segunda mitad +9.7, IQ4_XS en todas +7.6
(llega a mitad de camino, sección 7), down Q5_0 en todas +4.7. Producción deja ~3 GiB sobre el margen.

**El GGUF de Strata y el anterior de llama.cpp tienen los mismos expertos.** Diferencias de tipos entre
`NL3S/qwen4exp-nl3s-hcdi-gu-rq8-oproj8.gguf` y `NL3S-strata/qwen4exp-ud-small-nl3s.gguf` (además del layout gate+up
fusionado, que no cambia valores): gate/up de la capa 2 IQ4_NL → IQ4_XS (Strata no tiene kernels de gate/up IQ4_NL), y
el lado denso chico subido a los tipos de UD (sección 5.1 de la evaluación de Strata: routers F32, experto compartido,
hc, `ssm_alpha/beta`, PLE key/value y embedding en Q8_0/F32). Los expertos de las otras 47 capas tienen los mismos
tipos y los mismos bytes (comprobado en la capa 30: gate, up y down). El GGUF de Strata es más preciso en todo
salvo la capa 2, y en llama.cpp reproduce más (10/10 contra 6/10, sección 2): **más bits no bajan el modo de forma
monótona**.

Las cifras de esta sección y de la 7 son de continuaciones que escribió el GGUF de producción en un casi empate:
cualquier perturbación de esos pesos tiende a bajarles la probabilidad aunque no mejore nada. El control nulo
(gate/up re-cuantizado a IQ3_S desde BF16, mismos bits que producción) decide si miden precisión o distancia al redondeo
de producción: PENDIENTE.

## 10. Generación libre con BF16

`ref_decode.py`: el código oficial generando con su propia caché (`DynamicCache`) y los expertos leídos bajo demanda
(GGUF: filas del archivo; BF16: rangos del checkpoint). Verificado contra la pasada por capas con los mismos pesos:
expertos bajo demanda idénticos al módulo oficial (`test_ondemand.py`, diferencia 0); el decode con caché, forzado
sobre req8+0 y req8+1, KL 0.009 / 0.004 y un argmax distinto en 80 posiciones (un empate 0.362 / 0.362), del orden del
ruido de la propia referencia entre corridas (KL hasta 0.006, |Δ log p| hasta 0.55).

La medida (`opening_tree`): el árbol de la apertura del pensamiento con la distribución de cada modelo, todas las rutas
de 3 tokens con probabilidad ≥ 0.01 entre los 8 más probables de cada nodo, cada hoja continuada greedy hasta 10
tokens y clasificada (`classify.py`); la masa cortada se informa aparte. Cada modelo expande su propio árbol: no hay
continuaciones elegidas por uno de los dos. req2 es la prueba del inicio (no hay pensamientos anómalos anteriores en su
contexto); req8 es el contexto largo con siete pensamientos conservados (entre ellos el anómalo del turno 2, que por
sí solo no es la causa: sección 3). Los pesos BF16 llevan, como en todas las pasadas, la tabla PLE del GGUF.

Regla fijada antes de los números BF16 (masa ANOM del GGUF: req2 0.46 con 0.49 cortada; req8 0.87-0.96):
- inicio propio del modelo si BF16 da ANOM ≥ 0.30 en req2 (dos tercios del GGUF);
- inicio causado por la cuantización si BF16 da ANOM + cortada ≤ 0.15;
- entre medio, no concluyente con τ 0.01;
- req8 con la misma proporción (propio ≥ 0.6, cuantización ≤ 0.3), leído aparte.

| Petición | GGUF: ANOM / cortada | BF16: ANOM / cortada |
|---|---|---|
| req2 | 0.460 / 0.490 | **0.429** / 0.371 |
| req8+0 | 0.873 / 0.127 | **0.402** / 0.325 |
| req8+3 | 0.938 / 0.062 | **0.231** / 0.331 |
| req8+12 | 0.958 / 0.042 | **0.398** / 0.173 |

Las hojas del GGUF en req2: "The user just sent system instructions setup. There's" (0.316), "The user hasn't asked
anything yet — the last" (0.144); en req8 las tres familias del campo ("hasn't asked anything", "just sent the system
instructions", "message is just the system instructions").

**El inicio del modo es del modelo.** En req2 el checkpoint oficial BF16 abre con "The user hasn't asked anything
substantive yet — the" (0.342) y "The user's message is empty — there's no" (0.088, el texto exacto de la sesión de
pi): ANOM 0.429 contra 0.460 del GGUF, sobre el umbral fijado (0.30). El resto de sus hojas son sanas (0.168: "Also
check the skill location…", "I should approach this with a direct answer", "Answer the question…") u otras (0.032).
La cuantización cambia la redacción dominante ("just sent system instructions" en el GGUF, "hasn't asked anything" en
BF16); con la masa cortada (0.37 y 0.49) la medida no dice si cambia también la frecuencia.

**La tabla PLE oficial.** Todas las pasadas de referencia, BF16 incluido, leen la tabla PLE del GGUF (IQ4_NL): sus
320 M de filas (~102 GB en BF16) no se traen por rangos fila a fila. `ple_official.py` calcula con el módulo oficial
las filas que leen req2 y req8+0 con sus aperturas (132 921), recorre la tabla oficial completa en 768 pedidos de
128 MB (27 min, sin disco) y guarda solo esas (43 MB; correlación por fila con las del GGUF 0.997, la de la sección 4).
`ref_variants.py` con `"ple"` las usa en lugar de las del GGUF (`ref_forward.OfficialRows`). Las aperturas de BF16,
forzadas con las dos tablas (`ple_compare.py`, `out-ple/`):

| | Tabla del GGUF | Tabla oficial |
|---|---|---|
| req2: P("The"), P(" user"), P(" hasn" \| "The user") | 0.532, 0.972, 0.751 | 0.562, 0.941, 0.762 |
| req2: "The user hasn't asked" (conjunta) | 0.37 | 0.39 |
| req2: "The user's message is empty — there's no" (conjunta) | 0.031 | 0.026 |
| req8+0: P("The") | 0.674 | 0.942 |
| req8+0: tercer token tras "The user" | has 0.31, hasn 0.25, asked 0.19, just 0.16 | just 0.34, hasn 0.23, asked 0.16, has 0.14 |

El inicio no depende de la tabla: con la oficial, req2 abre con "The user hasn't asked" 0.39 y "The user's message is
empty" 0.03, al menos lo mismo que con la del GGUF. En req8+0 la tabla sí pesa, y la oficial lleva al modelo hacia la
apertura del GGUF ("The" 0.94 contra 0.93 del GGUF): con la tabla oficial BF16 abriría req8 con el modo al menos tanto
como en el árbol de arriba (con la oficial, "The user has just sent system instructions establishing me as" 0.11
forzada, más el prefijo "The user hasn" 0.22, que en todos los árboles siguió con el modo). La cifra exacta de BF16
completo en req8 no se midió; la amplificación por los expertos de abajo está medida con la tabla del GGUF fija, la
que conservaría cualquier receta de producción.

**En req8 la cuantización de los expertos lo amplifica.** En el contexto largo de req8, BF16 también abre con el modo (0.23-0.40; en req8+0 es su apertura greedy: "The user has just sent system instructions
establishing me as"), pero el GGUF lo hace 0.87-0.96. Aun sumándole a BF16 toda la masa cortada (0.56-0.73), queda por
debajo del GGUF. El resto de las hojas de BF16 en req8 son sanas o neutras ("The user asked only to fix the AGENTS.md",
"The user just wants the content", "El archivo quedó con 3 líneas"). Por la regla fijada, req8 queda entre los dos
umbrales: ni todo del modelo ni todo de la cuantización; la cuantización multiplica por 2-4 una tendencia que el
modelo ya tiene en ese contexto.

## 11. Qué se puede hacer

- **En los motores:** nada. Strata y llama.cpp reproducen el código oficial con los mismos pesos (sección 5); la
  diferencia de tasa en el campo (Strata 3/356 turnos, los tres en una sesión; llama.cpp 2/2342) es una sesión contra
  dos (1 de 8 contra 2 de 87; Fisher unilateral p = 0.23).
- **El inicio:** es del modelo (BF16 0.43 en req2; con la tabla PLE oficial, "The user hasn't asked" 0.39). Ninguna
  cuantización de los expertos lo quita.
- **req8, con precisión:** subir los expertos lo acerca a BF16 (0.23-0.40 en vez de 0.87-0.96). Sobre la
  medida forzada, IQ4_XS en gate/up de todas las capas (+7.6 GiB) recupera la mitad y Q5_K (+19.3 GiB) casi todo
  (secciones 7 y 9); producción deja ~3 GiB sobre el margen. Un canje a igual tamaño por capa (Strata ya acepta
  IQ3_XXS e IQ2_S en gate/up para bajar) choca con que el efecto está repartido en todas las capas; por experto (los
  más usados arriba: el 25 % más usado recibe el 68 % del tráfico en el corpus del imatrix) exige dos tipos por capa en
  los dos motores. Ninguno se midió.
- **req8, sin los pensamientos anteriores:** no conservarlos (sección 3) elimina el modo en req8 (25/25 → 0/25 con
  el GGUF), aunque ninguno de ellos por separado es la causa, y no toca el inicio (req2 9/25); sin pensamientos, req8
  con 4 de relleno inventa otra cosa. Quedó en un stash de los dos repos ("preserve_thinking (not adopted)"); el
  Director lo había descartado como workaround.
- **Aceptarlo y reportarlo a Qwen** como modo de falla del modelo en bucles de herramientas, con req2 como caso.

## Herramientas

| Archivo (`~/dbg/strata/empty/`) | Qué hace |
|---|---|
| `render.py` | reconstruye las peticiones de pi de una sesión y las renderiza con el código del servidor de Strata |
| `classify.py` | la expresión regular del modo, compartida |
| `repro_set.py`, `sample_set.py`, `session_replay.py`, `replay_score.py` | reproductor greedy, muestreo, réplica de sesiones |
| `official/official_hf.py`, `official/gguf_weights.py` | el checkpoint por rangos HTTP; el GGUF como tensores oficiales |
| `official/ref_forward.py`, `ref_pair.py`, `ref_variants.py` | la referencia oficial por capas: GGUF, BF16 contra GGUF, conjuntos de pesos mezclados |
| `official/ref_decode.py`, `decode_score.py` | generación con la caché oficial y los expertos bajo demanda; el árbol de la apertura |
| `official/ple_official.py`, `ple_compare.py` | las filas oficiales de la tabla PLE que leen unas secuencias; la comparación de las dos tablas |
| `official/variants_score.py`, `pair_score.py` | las medidas sobre las continuaciones |
