# Camino de decode de qwen4exp: plan por subproyectos (2026-09-28)

Objetivo: bajar los milisegundos del paso de decode a 40k sin cambiar la salida. Punto de partida: 61.7 ms por paso
con MTP n-max 2 y ~2.3 tokens por paso, ~37 t/s (`qwen4exp-cost-breakdown-2026-09-09.md`, secciones 28.7 y 28.9).
Este plan no toca los tokens por paso: las palancas de aceptación (sección 29.7, filas 4-6) se multiplican aparte.
Cada subproyecto (SP) se hace en días distintos, con su propia ventana de medición y su commit.

## 1. Dónde está el paso hoy (40k, NP=1)

| Tramo | ms por paso | GPU | Fuente |
|---|---|---|---|
| Verificación antes del primer submit: set_inputs 1.81 (agrupación QSA 1.20), scheduler 0.8, grabación hasta el primer submit 0.38 | ~3.0 | parada | 28.6, 28.7, 28.8 |
| Verificación desde el primer submit | 46.5 | ocupada | 28.6 |
| · pesos (mat-vec y expertos, ~5.9 GB a ~174 GB/s) | 33.9 | | 28.3 |
| · atención (FA sparse y top-k QSA) | 4.8 | | 28.3 |
| · ops chicos (~2300 dispatches; 2327 sincronizaciones para 3269 nodos) | 8.9 | | 28.3 |
| Hook del draft sobre los 3 tokens | 1.78 | 0.38 | 28.2 |
| Dos pasos del draft de 1 token | 7.71 | 2 × 2.96 | 28.2 |
| Huecos de host: muestreo del target 1.27, clon del sampler 0.21, post 0.22, sin atribuir ~1.4 | 3.08 | parada | 28.2 |

Host en serie con la GPU: ~9 ms por paso. De eso, los dos grafos del draft se reconstruyen en cada paso (build 0.18,
alloc y split 0.47, `ggml_vk_graph_optimize` 0.60 tras la sección 28.9: 1.25 ms) y la máscara de la atención densa
del draft se llena tres veces por paso (0.51 ms, sección 28.8). Las cifras vienen de cargas distintas (la de timing
agrega ~2 ms); el reparto es el de la sección 28.

## 2. Descartado: replay de command buffers

Desde el primer submit la GPU no espera al host y la grabación de la verificación termina 35 ms antes que la GPU
(sección 28.6). Re-ejecutar command buffers grabados solo recuperaría los 0.38 ms previos al primer submit; los 9.4 ms
de grabación por paso (sección 28.8) son CPU solapada con la GPU salvo en los grafos chicos del hook y del draft, que
cubre el SP1. Los escenarios de 43 / 47 / 55 t/s de la sección 29.4 contaban el replay y quedan corregidos por la
sección 5 de este plan.

## 3. Protocolo común

- Día 1 de cada SP: lectura del código y decisión de diseño entre las opciones listadas, sin GPU.
- Build con `strix-halo/build-llama.sh`, producción descargada (`strix unload`, y
  `pgrep -x "llama-(server|bench|cli|perplexit)"` vacío). Nunca editar fuentes durante un build.
- Cargas NP=1 (ctx 57344, MTP n-max 2, feature set de producción) con el rig `~/dbg/merge/decode_budget.py`: una a
  la vez, compuerta de 40 GiB, MemFree ≥ huella y corte si baja de 6 GiB. Base (copia de `build/bin` por
  `LD_LIBRARY_PATH`, verificada en `/proc/<pid>/maps`) primero y último en la misma cadena.
- Medida: ms por paso sin instrumentar, greedy y muestreo de producción (temp 0.7, top_k 20, top_p 0.95), dos turnos
  de 1024 tokens con `ignore_eos` sobre el prefijo de 39.5k; una carga con `LLAMA_INPUT_TIMING` y `LLAMA_SPEC_TIMING`
  para atribuir; relojes y potencia con `power_sampler.py`.
- NP=2: dos slots generando a ~40k con `--decode-share 0.4` (rig de la sección 25.3), t/s por slot y agregado.
  Producción es NP=2: los grafos del draft y las entradas QSA cambian de forma con dos slots.
- Exactitud: respuestas idénticas a la base en greedy y en muestreo, misma aceptación del draft, probe idéntico a la
  referencia vigente, `graph_diff4` con los 72 nodos conocidos, `depth_repeat` a 40k, conversación texto → imagen →
  texto, cero timeouts de anillo; `img_big.py` 2048×2048 solo si se toca el camino de visión.
- Producción recargada separada de la cadena (`setsid -f bash -c "./strix load qwen38flash"`, luego `/health`).
- Un commit por cambio adoptado; una subsección del documento de costos por ventana; push a los dos repos al adoptar.

## 4. Subproyectos

### SP1. Grafos del draft sin reconstrucción

- Evidencia: 0 de 482 grafos del draft reutilizados (28.2); 1.25 ms por paso en el camino crítico (28.8, 28.9).
- Causa: `llama_context` guarda un solo grafo previo (`gf_res_prev`, `src/llama-context.cpp:1377-1407`) y el contexto
  del draft alterna el hook (3 filas por slot que genera) y el paso del draft (1 fila por slot), así que reconstruye
  los dos en cada paso. El estado del scheduler también es de un solo grafo: `ggml_vk_graph_optimize` y el split
  corren al asignar (`ggml/src/ggml-backend.cpp:1458-1473`), así que guardar el grafo sin su asignación no alcanza.
- Opciones de diseño: (a) una caché de resultados de grafo por parámetros de reuso, con el estado del scheduler
  guardado por entrada o un scheduler por entrada; (b) topologías unificadas entre el hook y el paso. Debe cubrir las
  formas de NP=2 (hook de 3 o 6 filas, paso de 1 o 2 filas) y los drafts más cortos.
- Archivos: `src/llama-context.{h,cpp}`, `src/llama-graph.cpp`; `ggml/src/ggml-backend.cpp` según el diseño.
- Hecho: en régimen estable `LLAMA_INPUT_TIMING` marca `reused = 1` en el hook y en el paso del draft, con NP=1 y NP=2;
  compuertas del protocolo.
- Estimación: −1.2 ms por paso (−2 %). Riesgo medio-bajo. 2-3 días.

### SP2. Entradas del paso incrementales

- Evidencia (85 W): `set_input_qsa` cuesta 1.20 ms por paso a 40k con la GPU parada (28.8); recorre todas las celdas
  en cada ubatch y por stream (`src/llama-memory-hybrid-idx.cpp:442`, el comentario del código mide ~865 µs a 33k). La
  máscara de la atención densa del draft cuesta 0.51 ms por paso (28.8).
- Evidencia con la potencia nueva (sección 32 del documento de costos, cadena v39): las entradas de la verificación
  cuestan 0.88 ms por paso a 40k y 2.20 a 125k; las de los tres grafos del draft, 0.19 y 0.53. La parte que crece con
  la profundidad (QSA y máscaras KQ del target y del draft, sin reparto medido en el decode) es 0.77 ms por paso a 40k
  (1.4 %) y 2.43 a 125k (3.8 %): la cota superior del SP2, +1.4 % de t/s a 40k y hasta +4 % a 125k.
- Opciones de diseño: agrupación por bloques persistente entre ubatches, actualizada solo en las celdas que cambian
  (filas nuevas, rollback del draft, `seq_rm`, celdas de imagen en orden de rango); máscaras KQ del target y del draft
  escritas solo en las columnas nuevas. La máscara del draft desaparece si entra la vía 16 (IndexShare) o el SP5: no se
  cuenta dos veces.
- Archivos: `src/llama-memory-hybrid-idx.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`.
- Día 1: separar en el decode el costo de la agrupación QSA y el de las máscaras (medición por entrada en los
  ubatches chicos, o `perf` si el Director baja `kernel.perf_event_paranoid` a 2) para elegir qué se hace incremental.
- Hecho: set_inputs medido antes y después a 40k y a 125k con el rig de la cadena v39 (`decode_budget.py` con
  `DEPTH_PROMPT=depth_prompt125.txt`, MTP, lookup y visión cargados), más `depth_repeat`; compuertas del protocolo con
  énfasis en texto → imagen → texto y NP=2.
- Estimación: hasta −0.77 ms por paso a 40k y −2.43 a 125k (medido como cota, sección 32). Riesgo medio: la
  agrupación QSA tuvo defectos con imágenes (`n_dirty < n_dirty_max`, 2026-09-04). 2-3 días.

### SP3. Muestreo y huecos de host del server

- Evidencia: el muestreo del target cuesta 1.27 ms por paso (3 posiciones) frente a 0.14 ms por posición en el
  microbench (28.6). El llenado de 248320 candidatos cuesta 0.18-0.22 ms por posición en el server frente a 0.077 con
  los logits en memoria normal; en el server viven en el buffer de salida de Vulkan, con tipo de memoria sin verificar
  (28.8). `partial_sort` del top-k 0.38 ms; clon del sampler 0.21, post 0.22, ~1.4 ms sin atribuir (28.2); scheduler
  0.8 ms antes del `graph_compute` de Vulkan (split de CPU: `get_rows` de `token_embd` y de la tabla PLE, 28.7).
- Día 1: atribución (`LLAMA_SPEC_TIMING`, perf con pilas DWARF, tipo de memoria del buffer de salida).
- Opciones: logits en memoria host cacheada; llenado y top-k más baratos con el mismo orden de empates; clon evitado;
  split de CPU del scheduler más corto.
- Restricción: el muestreo debe quedar bit-idéntico (mismos candidatos, mismo orden de empates) o el probe cambia.
- Atribución hecha el 2026-09-29 (sección 34 del documento de costos, perf con la línea de tiempo del paso): 1.64 ms
  de huecos de host por paso a 40k. Cambios: el clon del sampler sin el vector de candidatos (−0.14 ms), `seq_rm` por
  el índice de posiciones de la secuencia (−0.09 ms con ctx 57k, ~−0.4 en producción con 262k celdas por slot), llenado
  y top-k del muestreo en una pasada con el mismo orden de empates (−0.2 a −0.3), la limpieza del command pool de
  Vulkan fuera del camino crítico (−0.10 a −0.13).
- Estimación: −0.55 a −0.65 ms por paso con ctx 57k (+1.0-1.2 %) y −0.85 a −1.0 en producción (+1.5-1.8 %). Riesgo
  bajo salvo el top-k (orden de empates). 2 días.

### SP4. Fusiones en el grafo de verificación

- Evidencia: 8.9 ms de ops chicos en ~2300 dispatches y 2327 sincronizaciones para 3269 nodos, casi sin concurrencia
  (28.3). Candidatas: fila 17 de la sección 3 (cont + cpy de los slots de rollback, 108 nodos; scale plegado en
  `hc_down`, 96; sigmoid × mul de la norma GDN; fill / set_rows / add de la máscara QSA), estado GDN in place (vía 14:
  GET_ROWS + CPY, 2.5 ms por paso, sección 24.3), indexador QSA con relu + suma + bias en un kernel (fila 15 de la
  sección 13).
- Fuera de este SP sin decisión del Director: las fusiones de pesos que transforman el GGUF (z + alfa + beta, gate y
  up del experto compartido; fila 17 de la sección 13), con el procedimiento del freeze #10.
- Método: una fusión por commit y por medición. El perfil serializado sobreestima los ops chicos (sección 20.2): solo
  cuentan los ms de punta a punta. Cada fusión suma en el mismo orden que el camino que reemplaza.
- Estimación: −2 a −4 ms por paso en total. Riesgo medio por fusión. ~1 día por fusión, 4-6 días.

### SP5. Draft dentro de la pasada de verificación (spike y decisión)

- Idea (TensorFold, sección 29.2): la cabeza MTP absorbe todas las filas verificadas y saca el primer draft de cada
  fila en la misma pasada de GPU; al aceptar, el primer draft de la ronda siguiente ya está calculado.
- Hoy: hook de 1.78 ms (0.38 de GPU) y primer paso del draft de ~3.9 ms de pared; el hook sincroniza el contexto del
  draft porque dos contextos con trabajo en vuelo bloquean el dispositivo Vulkan (`common/speculative.cpp:1666`, en
  `process`). El hook calcula solo las filas de KV (recorte de la sección 19): sacar el draft de cada fila agrega la
  capa MTP completa y la cabeza sobre 3 filas.
- Restricciones: el draft vive en otro GGUF (`-md`) y en otro `llama_context` con su propia KV; rollback y
  checkpoints; NP=2; visión (mrope).
- Spike (2 días, sin producción): (a) la capa MTP cargada en el modelo y el grafo del target; (b) el grafo del draft
  enviado en la misma cola sin sincronización de host. Decisión después de SP1-SP3, con la ganancia recalculada
  (el SP1 ya quita parte del host del hook y del draft).
- Estimación: −2 a −3 ms por paso sobre SP1. Riesgo alto. 5-8 días si se aprueba.

### SP6 (opcional, fuera de la palanca 1). Tasa de los pesos en la verificación

- Mezcladores hc: 3.37 ms a 107 GB/s (324 filas × k 10240, 81 workgroups; candidato a split-k). Expertos gate/up:
  11.2 ms a ~160 GB/s, sin ganancia por forma de kernel hasta hoy (sección 26.1). El microbench no sirve con tensores
  en la MALL: se mide en el modelo.
- Estimación: −1.5 a −3 ms por paso. Riesgo medio. 3-4 días.

## 5. Resultado esperado (estimación)

| Tras | ms por paso | t/s a ~2.3 tokens por paso | Frente a hoy |
|---|---|---|---|
| Hoy | 61.7 | ~37 | — |
| SP1 + SP2 + SP3 | 57.3-58.3 | 39.5-40 | +6-8 % |
| + SP4 | 53.3-56.3 | 41-43 | +10-16 % |
| + SP5 | 50.3-54.3 | 42-46 | +14-23 % |
| + SP6 | 47.3-52.8 | 44-49 | +17-30 % |

Estimaciones hechas en el régimen de 85 W; desde el 2026-09-28 el paquete corre a 117-130 W y el paso base mide 55.4 ms
(sección 30 del documento de costos), así que los ms absolutos de esta tabla quedan corridos. Todo bajo el límite de
potencia (sección 28.4): quitar tiempo ocioso de la GPU sube la potencia media y el SMU baja
relojes, así que cada SP puede recuperar menos de lo que quita; solo cuentan los ms medidos de punta a punta. El SP2
gana más con la profundidad. Las palancas de aceptación (draft muestreado con la regla p/q, lookup en la ronda MTP,
aceptación typical) multiplican estos números por el cambio en tokens por paso.

## 6. Orden, calendario y estado

Orden revisado el 2026-09-28 con el presupuesto medido a 117-130 W (sección 31.4 del documento de costos): SP4, SP6, SP5 y, al final, SP2 y SP3, porque el host en serie bajó a ~2.6 ms por paso. El lookup en la ronda MTP (fila 4 de la sección 29.7) quedó adoptado antes que estos SP.

Revisado el 2026-09-29 con la cota del SP2 medida a 40k y a 125k (sección 32.3): SP4, SP6, SP2, SP5 y SP3. El SP2 pasa delante del SP5 porque su ganancia está medida y crece con el contexto de los agentes (hasta +4 % a 125k), cuesta 2-3 días y no necesita decisión del Director; el SP5 es un spike de riesgo alto con estimaciones del régimen de 85 W.

El estado se actualiza al cerrar la ventana de cada SP, con el commit y la subsección del documento de costos que
registra su medición.

| Orden | SP | Días | Condición | Estado |
|---|---|---|---|---|
| 1 | SP1 | 2-3 | — | hecho el 2026-09-28 en un día (e579358ba, sección 30 del documento de costos): −1.17 ms por paso en greedy y −0.90 con muestreo a 40k; NP=2 sin cambio medible; con NP=2 los grafos se reutilizan (sección 31.2) |
| 2 | SP4 | 4-6 | una fusión por día | en curso (sección 36 del documento de costos): fusión 1 adoptada el 2026-09-29 (abc86e8be), −0.6 ms por paso a 40k (+1.1-1.2 %); fusión 2 (colas en un gather) descartada: ganancia dentro del ruido y agrega una entrada al grafo que mueve los cortes de sumisión del prefill profundo; fusión 3 (snapshots del GDN directo al caché, vía 14 lado de escritura) adoptada el 2026-09-29 (1d7cd0d16): −2.2 ms por paso a 40k (+4.2-4.3 %), producción +4.2-4.9 %; salidas idénticas |
| 3 | SP6 | 3-4 | opcional | pendiente |
| 4 | SP2 | 2-3 | — | hecho el 2026-09-29 en un día (100f88835, b3e7e2b3f; sección 33 del documento de costos): −0.46 ms por paso a 40k (+0.9 % t/s, bajo su cota de 0.77) y −2.0 a −2.4 ms a 125k (+3.2-3.9 %); salidas idénticas; NP=2 sin cambio medible |
| 5 | SP5 | 2 de spike + 5-8 | decisión del Director tras el spike | pendiente |
| 6 | SP3 | 2 | — | hecho el 2026-09-29 (f5149b386, 2ed7aac92, 7b0e8cf98; sección 35 del documento de costos): NP=1 a 40k −0.93 ms por paso (+1.7 %), producción a 40k −1.4 a −1.9 ms (+2.6-3.6 %); salidas idénticas a la base. El SP1 hacía que la salida dependiera de la historia del servidor (sección 35.1): la causa era el asignador de ggml, que reutilizaba la disposición del grafo anterior; resuelto el 2026-09-29 (9f3e99782, sección 35.2) |
