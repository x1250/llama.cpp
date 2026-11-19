# Sincronización con upstream: el código sigue a upstream, más los parches que todavía pagan (2026-11-19)

Hasta el 2026-11-19 el fork estaba 1794 commits detrás de upstream y llevaba 433 commits propios, casi todos para
qwen38flash (qwen4exp), que corre en Strata desde el 2026-10-07. El Director decidió que el código siga a upstream y
que solo se reapliquen los parches propios que dan velocidad medida en los modelos de llama-server; los de determinismo,
a criterio de Claude. Este documento registra la medición que eligió esos parches, los ports y la verificación.

## 0. Resultado

El árbol es upstream `08c252e22` (merge `705d46bf4`, que toma el lado de upstream en todo archivo salvo
`docs/strix-halo/` y `CLAUDE.md`) más seis commits:

| Commit | Parche del fork | Por qué |
|---|---|---|
| `975a47b64` | cargador acotado (`a7b9fa7ec`/`92a3a2f29` + `f10bbc727`) | seguridad: sin él, la MemFree mínima de una carga de Apex bajó a 8.5 GiB (freeze #9) |
| `69c286726` | `--decode-share` (`c89eba7d8`) | con 2 slots, el que genera no se detiene mientras el otro lee un prompt |
| `066393004` | layout nuevo tras cada reset del scheduler (`9f3e99782`) | determinismo según la historia del servidor |
| `7ca65bbaa` | primera fila de la máscara KQ en una pasada (`b3e7e2b3f`) | host del paso de decode a contexto largo (medido en el fork) |
| `11c3e8ab8` | snapshots de la gated delta net directo al caché (`1d7cd0d16`) | la verificación especulativa de los modelos con GDN: +12.9 ms por paso en upstream |
| `b9b9ef6a3` | celdas enmascaradas fuera del producto de la FA coopmat1 (`b4aa73cc3`) | determinismo entre peticiones idénticas |

Con eso, el costo por paso de los tres modelos de llama-server queda a menos de 2 % del fork, la aceptación del draft
igual, las peticiones idénticas dan los mismos tokens y log-probs, y la MemFree mínima de la carga igual a la del fork
(secciones 4 y 5).

## 1. Upstream puro contra el fork

Mismo lanzador (`strix load`), mismo modelo, solo cambia el binario (`PATH`); muestreo de producción, 512 tokens.
Evidencia: `~/ProjectHub/llama.cpp-evidence/daisyrose27b/` (`ab-upstream.sh`, `runs/speed-*-up.jsonl`).

| Modelo | Código | Prosa | Español | Thinking | Dos a la vez |
|---|---|---|---|---|---|
| DaisyRose-27B (DFlash2) | 39.2 → 34.7 | 16.2 → 13.9 | 14.4 → 13.5 | 19.7 → 19.1 | 27.4 → 26.0 |
| Apex-35B (MTP n2) | 76.4 → 69.7 | 58.5 → 53.6 | 59.0 → 55.0 | 74.1 → 67.6 | 79.8 → 75.6 |
| Qwen3.8-27B (DFlash2) | 39.4 → 35.8 | 17.8 → 14.8 | 15.8 → 13.7 | 21.0 → 19.1 | 26.0 → 25.4 |

Además, con upstream tres peticiones greedy idénticas (`cache_prompt` off, después de otra) difirieron desde el primer
token en los tres modelos, y la MemFree mínima durante la carga bajó: Apex 40 → 8.5 GiB, DaisyRose 33 → 20, Qwen3.8-27B
41 → 27.

## 2. Dónde estaba la diferencia de velocidad

- **No en los kernels.** `llama-bench` sin draft: decode de un token igual (DaisyRose 7.63 / 7.63 t/s; Apex 55.0 /
  54.6) y prefill de 512 igual o mejor en upstream (Apex 1491 → 1625 t/s). Lotes de 2, 3, 4 y 8 tokens iguales.
  `llama-batched-bench` con 8 secuencias (la cabeza de salida con N = 8, el caso que `llama-bench` no ejercita):
  10.97 / 10.95 s. El tuning de Vulkan del fork para gfx1151 no hace falta.
- **No en el tamaño del contexto.** Apex a 32768 de contexto da el mismo costo por paso que a 262144, en los dos builds.
- **No en el draft.** Las estadísticas del especulativo (`-lv 4`): el draft de DFlash2 cuesta 13.2 ms por llamada en
  el fork y 13.7 en upstream.
- **Es trabajo de GPU, no espera del host.** El tiempo de motor del proceso (`drm-engine-*` de amdgpu en
  `/proc/<pid>/fdinfo`) iguala al de reloj por paso en los dos builds: DaisyRose 160–162 ms (fork) contra 172–174
  (upstream), GPU ocupada todo el paso (`gpu_split.py`).
- **Está en el grafo de verificación.** Con `GGML_VK_PERF_LOGGER` en el servidor (`perf_graphs.py`), el grafo del target
  de 8 filas cuesta 148.7 ms en el fork y 161.6 en upstream; los del draft, iguales. La diferencia es la copia de los
  snapshots de la gated delta net al caché recurrente en cada verificación: CPY 0.9 → 8.0 ms y la GDN sin la escritura
  directa (2.4 → 2.8 ms). El fork la fusionaba (`GDN_SNAPSHOTS`, `1d7cd0d16`). El +6 ms que el perf logger atribuye al
  matmul m=6144 en upstream se compensa con el −6 ms del matmul vecino: es atribución entre nodos concurrentes.

Con la fusión portada, el grafo de 8 filas cuesta 146.8 ms y el paso de DaisyRose 159.4–159.7 ms (fork 159.6–159.8).

## 3. Los ports

Cada uno sobre la estructura actual de upstream; donde upstream ya lo resolvía, no se portó.

- **Cargador.** Las tres rutas de lectura de upstream (mmap, read con buffers de staging y direct IO con lectura
  alineada) sueltan la copia del page cache de cada tensor; la ventana de read-ahead de 256 MiB y `drop_fragment`
  quedan como en el fork. `seq_rm` por el índice de posición (`2ed7aac92`) no se portó: upstream ya lo hace.
- **`--decode-share`.** Aplica sin cambios.
- **Layout tras el reset.** Además de lo del fork, el callback de reasignación que upstream agregó
  (`sched_graph_realloc`) no cuenta las re-disposiciones por diseño.
- **Máscara KQ.** La ruta rápida se limita a las máscaras cuyas columnas son las celdas: upstream agregó un mapa de
  filas (`kv_rows`, caché unificado multi-secuencia) que la invalidaría.
- **`GDN_SNAPSHOTS`.** El backend de upstream se partió en varios archivos (`ggml-vulkan-types.h`,
  `ggml-vulkan-push-constants.h`, `ggml-vulkan-common.h`); el shader aplica sin cambios. Tests: los casos del fork más
  las formas de verificación de Qwen3.5/3.8-27B (48 cabezas de valor, 8 filas, 2 slots) y Qwen3.6-35B, 8/8;
  `GATED_DELTA_NET` 43/43 y el test de fusión de upstream 5/5.
- **FA coopmat1.** El shader de upstream tiene listas dispersas y tiles de varias filas; las columnas inactivas de la
  lista dispersa también quedan fuera. El cambio del fork del acumulador de salida a f32 no se portó: es de precisión,
  no de determinismo. `FLASH_ATTN_EXT` 5693/5693. Con KV q8_0 upstream usa este shader (no el flip) en todo lote de más
  de una fila: prompts y verificaciones.

Descartados: el tuning de matmul y mat-vec para gfx1151 (sección 2), `--spec-draft-adaptive` (el lanzador ya no lo
pasa), qwen4exp, QSA y la FA dispersa del fork (qwen38flash corre en Strata), el resto de los fixes de determinismo
de qwen4exp (top-k radix, listas dispersas).

## 4. Verificación del port

Fork primero y último en la misma cadena (`ab-port.sh`, `fa-check.sh`, `acc_test.py`).

| | Fork | Port |
|---|---|---|
| DaisyRose, ms por paso greedy / muestreo | 160.0 / 157.5 | 160.6 / 160.5 |
| DaisyRose, aceptados por paso greedy / muestreo | 2.74 / 2.41 | 2.71 / 2.41 |
| Apex, ms por paso greedy / muestreo | 32.8 / 33.3 | 33.4 / 33.4 |
| Apex, aceptados por paso greedy / muestreo | 1.22 / 1.14 | 1.22 / 1.22 |
| Determinismo (3 peticiones idénticas) | OK | OK en los tres modelos |
| MemFree mínima en la carga (DaisyRose / Apex / Qwen3.8-27B) | 33 / 41 / 41 GiB | 33.7 / 40.7 / 36.2 GiB |

`acc_test.py`: seis prompts greedy y dos con el muestreo de producción y cuatro semillas, 256 tokens cada uno. La
aceptación más baja que mostró el A/B de una muestra por categoría (Apex 1.45 contra 1.53 aceptados por paso en
código) era varianza del muestreo: con más muestras, igual.

`--decode-share` (Apex, NP=2, prefijos de ~39k, `np2_rig.py s2`): mientras el otro slot lee 14 990 tokens, el slot que
genera recibe 20.4 chunks por segundo con 0.4 y 1.5 sin el parche; el prompt ajeno tarda 17.9 s en vez de 11.0.

## 5. Adopción

- **Instalado**: master `b9b9ef6a3` compilado con `build-llama.sh --install` (`~/.local/bin/llama-server`, bibliotecas en
  `build/bin`). `build-llama.sh` exporta ahora `HF_UI_VERSION` con la etiqueta de upstream más reciente del árbol
  (`b12532`): la UI web precompilada se descarga por versión, y el número de build del repo no coincide con ninguna
  release; sin versión caía a "latest", de formato más nuevo, y el build fallaba.
- **DaisyRose-27B** quedó como `daisyrose27b` (opción 8 de `strix`): Q8_0 propio, DFlash2 con el draft del Qwen3.8-27B,
  262144 × 2 slots (`models/DaisyRose-27B-GGUF/README.md` en strix-halo, con la receta y la medición: Q8_0 KLD 0.0051
  contra el BF16; el Q6_K_L propio, 0.0125 y sin ganancia de velocidad).
- **Presupuesto de memoria corregido**: el `kv_gb` 16 de `qwen38-27b` contaba solo el KV. Medido con `srv_mem.py`, el
  servidor ocupa 50.5-50.8 GiB al cargar y ~55 tras unas peticiones, 10-15 GiB más de lo que reservaba la compuerta;
  con los checkpoints (12 por slot de ~150 MiB) y la caché de prompts (8 GiB) el techo es ~38 GiB sobre los pesos.
  `qwen38-27b` y `daisyrose27b` (la misma arquitectura) quedan con `kv_gb` 38, como `apex35b`.
- **Imágenes con DFlash2**: con el build instalado, toda petición con imagen a `daisyrose27b` o `qwen38-27b` fallaba
  con HTTP 500 (`llama_process(ctx_dft) failed`, `inconsistent sequence positions: X = 3, Y = 68`). El draft salta las
  filas de una imagen (comparten una posición M-RoPE) y upstream deja que el contexto del draft salte el hueco solo si el
  draft usa M-RoPE; su conversor escribe para eso `dflash.rope.dimension_sections = [head_dim/2, 0, 0, 0]` cuando el
  target es M-RoPE (`conversion/qwen.py`, `DFlashModel`), pero nuestro `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` se convirtió
  sin ellas y cargaba con NeoX, que exige posiciones consecutivas. El fork lo resolvía empaquetando las escrituras del
  draft en un flujo continuo (`6695d6d53`, reemplazado después por el salto de upstream). Arreglo: el mismo GGUF con
  esa clave (`draft_add_mrope.py`: 81 tensores idénticos byte a byte, ninguna otra clave cambia; el original queda como
  `Qwen3.8-27B-DFlash2-Q4_K_M.no-rope-sections.gguf.orig`). Con la rotación solo en la dimensión temporal, el texto
  rota igual que con NeoX: la aceptación quedó idéntica (DaisyRose 2.710 / 2.408 aceptados por paso, Qwen3.8-27B
  2.728 / 2.336), y la imagen de 2048 × 2048 (19 s) y la conversación texto → imagen → texto (5/5) pasan en los dos.
- **Gemma 4 31B no arrancaba** (crash en el `GET_ROWS` del warmup, que el manejador de crash dejaba colgado): el cargador
  portado soltaba las páginas de los tensores que el host lee del mapeo cuando el upstream actual los hace pasar por la
  rama de subida (ya ubicados dentro del mapeo, o un peso compartido por un tensor duplicado en otro contexto, como la
  salida atada al embedding). Arreglado en `0374d50a6`; gemma31b, gemma12b y qwen38-27b-unc pasan razonamiento, tool
  call, determinismo y las dos pruebas de imagen con el binario instalado. Los `kv_gb` de qwen38-27b-unc (16 → 49, KV
  f16) y gemma12b (16 → 22) también se corrigieron desde lo medido.
- **Respaldo de qwen38flash (`ENGINE=llama`)**: corre el qwen4exp de upstream. Sin MTP carga y responde (prueba de humo a
  32k: razonamiento, tool call, determinismo, 32.5 t/s); con MTP no carga, porque el draft externo que exportó el fork no
  tiene la cabeza de hyper-connections que el MTP de upstream exige (`latent-defects.md`).
- **Defectos latentes**: los arreglos que el registro tenía pendientes para llama.cpp ya estaban en upstream
  (`e467e5b5e`, `4453b535f`, `d280808f5`, `ead9f3205`, `66e0c17ee` y cuatro sin síntoma) y entraron con el árbol.
- **No corridas**: la compuerta de historia y la de calidad en cinco modos (definidas para qwen38flash, que corre en
  Strata) y la medición de producción de qwen38flash con `ENGINE=llama`.

Evidencia: `~/ProjectHub/llama.cpp-evidence/daisyrose27b/` (cadenas `ab-upstream.sh`, `prof.sh`, `prof-small.sh`,
`batched.sh`, `step-split*.sh`, `gpu-split.sh`, `perf-graphs.sh`, `port-check.sh`, `ab-port.sh`, `fa-check.sh`,
`flash-llama*.sh`, `measure2.sh`, `installed-check.sh`, `draft-fix-check.sh`; herramientas `gpu_split.py`,
`perf_graphs.py`, `acc_test.py`, `draft_add_mrope.py`;
salidas en `runs/`).
