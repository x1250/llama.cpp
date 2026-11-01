# KTransformers frente a esta máquina (2026-11-01)

KTransformers (`kvcache-ai/ktransformers`, revisado en `c987331` del 2026-10-28) reparte un modelo MoE entre la GPU y la
CPU: los expertos viven en la RAM del sistema y los calcula la CPU, y la atención y las capas densas van en la GPU. El
Director preguntó por un post que lo presenta como la forma de correr modelos de frontera en una GPU de 24 GB.

## 0. Resultado

**No aplica a esta máquina y ninguna de sus piezas se traslada a Strata.**

- **El bus de memoria no da más con la CPU leyendo junto a la GPU** (sección 2). La GPU sola lee 212 GB/s; con la CPU
  leyendo a la vez, la GPU baja a 182 y la CPU suma 31: 213 en total. Lo que lee la CPU se lo quita a la GPU. Repartir
  expertos con la CPU no agrega ancho de banda, y el decode está limitado por el ancho de banda.
- **El problema que resuelve no existe aquí.** Su caso es una GPU con poca VRAM y mucha RAM aparte. En el Strix la GPU
  ve los 124 GiB de memoria y el modelo entero está a su alcance.
- **No corre en nuestra GPU.** La parte de GPU de `kt-kernel`, el motor actual sobre SGLang, es solo CUDA (SM 80-90,
  `kt-kernel/README.md`). El soporte ROCm de marzo de 2025 (`doc/en/ROCm.md`) es del framework anterior, hoy en
  `archive/`: una beta probada en una Radeon 7900 XTX con un EPYC, sin Marlin.
- **Sí corre nuestro modelo.** Su código cita Qwen3.8-Flash-Next (`kt-kernel/operators/avx2/mxfp4-moe.hpp:117`) y la
  ficha oficial del modelo lo lista entre los motores. Lo hace con una GPU NVIDIA y la CPU de un servidor.
- **El margen que existe solo se alcanza desde la GPU.** Los kernels de expertos leen a 160-172 GB/s y las lecturas
  densas grandes a 210-227 en la misma GPU (`qwen4exp-cost-breakdown-2026-09-09.md`, vía 3). Llevar los expertos a ese
  techo es trabajo de kernels de GPU, no de reparto con la CPU.

## 1. Componente por componente

| Optimización | Qué hace | Aquí |
|---|---|---|
| Kernels de expertos en CPU (AMX, AVX-512, AVX2, BLIS) | Calcula los expertos en la CPU | No suma ancho de banda (sección 2). Sus kernels más rápidos son AMX, y esta CPU no lo tiene: `/proc/cpuinfo` trae `avx512f`, `avx512_bf16` y `avx512_vnni`, sin `amx` |
| Ubicación de expertos por frecuencia, con actualización dinámica (`--kt-expert-placement-strategy`) | Decide qué expertos viven en la VRAM | No aplica: todos están en memoria de la GPU |
| Prefill dual (`--kt-gpu-prefill-token-threshold`) | Sobre un umbral de tokens, sube los pesos de la CPU a la GPU capa por capa | Ya ocurre: todo el prefill corre en la GPU |
| Ejecución por nodos NUMA | Reparte hilos y memoria por socket | No aplica: un solo nodo NUMA (`/sys/devices/system/node`) |
| Expertos diferidos (`--kt-max-deferred-experts-per-token`) | Suma la salida de algunos expertos en la capa siguiente para que la CPU y la GPU trabajen en paralelo | Cambia la matemática del modelo y solo rinde si la CPU participa |
| Atención dispersa en CPU y caché de prefijos (framework archivado) | Contexto largo con poca VRAM; reuso de prompts | Ya existen: la atención dispersa del modelo (QSA) en la GPU y el caché de conversaciones de Strata |

Hay además un costo que la tabla no muestra: la CPU y la GPU comparten la potencia del paquete. Un render de Blender
solo en CPU bajó los relojes de la GPU de ~2.84 a ~2.3 GHz (`qwen4exp-cost-breakdown-2026-09-09.md`, cadena v36).

## 2. La medición del bus

- **Método:** dos lectores sin modelo, con producción descargada.
  - La GPU lee un buffer de 8 GiB de `hipMalloc` (en esta APU vive en GTT, como los pesos de Strata) con vectores de
    16 bytes. Se probaron 4, 16 y 64 bloques por CU y se usó 64.
  - La CPU lee otro buffer de 8 GiB con OpenMP (`-O3 -march=native`), con 16 y con 32 hilos.
  - El orden fue GPU sola, CPU sola, las dos a la vez y otra vez la GPU sola.
  - Al correr juntas, la CPU arranca 6 s antes y termina después. Se promedian solo las pasadas de la CPU que caen
    dentro de la corrida de la GPU (`overlap.py`).

| Prueba | GPU | CPU | Total |
|---|---|---|---|
| GPU sola (primera) | 212.3 GB/s | — | 212.3 GB/s |
| CPU sola, 16 hilos | — | 112.0 GB/s | 112.0 GB/s |
| CPU sola, 32 hilos | — | 104.3 GB/s | 104.3 GB/s |
| Las dos, CPU con 16 hilos | 182.1 GB/s | 30.6 GB/s | 212.7 GB/s |
| Las dos, CPU con 32 hilos | 182.2 GB/s | 30.9 GB/s | 213.1 GB/s |
| GPU sola (última) | 212.0 GB/s | — | 212.0 GB/s |

- La CPU sola lee 112 GB/s, en línea con los 124 que midió Chips and Cheese en lecturas con los dos CCD.
- El total con las dos es el de la GPU sola. El bus ya está saturado con la GPU.
- Los kernels densos de Vulkan llegan a 227 GB/s (`qwen4exp-cost-breakdown-2026-09-09.md`), así que el techo práctico
  está en 212-227.

## 3. Lo que dice el post, contra la fuente

| Afirmación | Fuente |
|---|---|
| DeepSeek-R1, 671B, en una GPU de 24 GB | Es cierto, pero con 382 GB de RAM: Q4_K_M con 14 GB de VRAM (`doc/en/DeepseekR1_V3_tutorial.md`). No entra en 128 GB |
| De 3 a 28 veces más rápido | Son datos de febrero de 2025, frente a llama.cpp en CPU, en dos Xeon de 32 núcleos. El 28× es la lectura del prompt con kernels AMX (286.55 contra 10.31 t/s) y el 3× es el decode (13.69 contra 4.51 t/s). Las dos cifras usan 6 expertos por token en lugar de los 8 del modelo (`archive/README_LEGACY.md`). El post es del 2026-07-20 |
| Fine-tuning de DeepSeek-V3 en 4 RTX 4090; 6-12 veces más rápido que ZeRO-Offload | Está en su README, sin las condiciones de la prueba. También es CUDA |
| Kimi-K2, GLM y Qwen3-Next sin trabajo extra | Es cierto: están en su lista de modelos |

## Evidencia

`~/ProjectHub/llama.cpp-evidence/ktransformers/bus/`: `bw_gpu.hip` y `bw_cpu.c` (los lectores), `run.sh` (la cadena,
corrida con `gate.py`), `overlap.py` (los promedios al correr juntas), `run.out` y `overlap.out` (los resultados) y
las pasadas de cada corrida (`gpu-*.txt`, `cpu-*.txt`, `both-*.txt`).
