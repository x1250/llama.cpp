# Cruce con upstream: Strata y llama.cpp (2026-10-24)

Revisión de solo lectura de lo que upstream agregó desde la revisión del 2026-10-20
(`upstream-review-2026-10-20.md`), pedida por el Director: no se compiló, cargó ni aplicó nada. Los cherry-picks se
simularon con `git merge-tree --write-tree --merge-base=<c>^ HEAD <c>`, que no toca los árboles de trabajo. Quedan
fuera DeepSeek-V4.1 (`dsv41`, descartado por el Director el mismo día), la app, los instaladores y las rutas de otras
GPU.

| | Nuestro | Upstream | Distancia |
|---|---|---|---|
| Strata (producción) | fork `c195c0c4`, sin cambios desde el 10-20 | `8183c6b6` (2026-10-24, motor 0.1.49) | 783 commits desde `e88ee39d` (172 first-parent); 117 tocan `src/` e `include/` fuera de `dsv41` |
| llama.cpp (respaldo e investigación) | master `8df47a308` | `8e5a58df8` (2026-10-24) | 104 commits desde `dd002b91f` |

## 0. Resultado

1. **Nada nuevo acelera producción.** La release 0.2.3 anuncia Strix Halo +6 a +19 % (`up:docs/release-notes-0.2.3.md:5`,
   Windows, IQ3_S) y lo atribuye a tres cambios que producción ya tiene: las hc leídas en 8 bits (`STRATA_HC_Q8=1`),
   chunks de 8192 (`--prefill 8192`) y la tabla hipBLASLt (`STRATA_HIPBLASLT_TUNING`), los tres en
   `strix-halo/strata/qwen38flash.json`.
2. **Un defecto de upstream que nuestro HEAD también tiene: el cuelgue de la primera ventana de verificación**
   (`86d10ed8`, sección 1.1).
   - Upstream midió 8 de 41 arranques colgados sin el arreglo y 0 de 46 con él, en un Strix Halo con IQ3_S
     (`up:docs/release-notes-0.2.2.md:47`).
   - Aquí no se observó. `~/.strix-engine.log` (2026-10-07..10) registra 17 arranques con ese grafo; 14 llegaron a
     decodificar y ninguno terminó en el watchdog. Con la tasa de upstream, eso tendría ~5 % de probabilidad
     (0.8^14), así que nuestra exposición parece menor. El defecto está en el código igual.
   - Si ocurre, el costo no son los 60 s del watchdog. La siguiente petición vuelve a cargar el motor sin pasar por
     `gate.py`, con lo que esté corriendo en ese momento.
   - El arreglo es exacto: solo cambia el orden en el host.
   - **Recomiendo tomarlo, adaptado, y extenderlo a los caminos batch**, que producción usa con dos slots y que
     upstream no arregló.
   - **Registrado, no aplicado (2026-11-01):** no se observó en 19 arranques (2026-10-07 a 10-31). Por decisión del
     Director, un defecto que no ocurre aquí se registra en `latent-defects.md` y no se arregla.
3. **Un merge de Strata cambiaría un caso raro para pi, sin interruptor** (`70f42407`, sección 1.2).
   - Las llamadas a una herramienta no declarada volverían como texto.
   - Hoy pi recibe "Tool X not found" y el modelo reintenta; con el cambio, el turno terminaría con el markup crudo.
   - En las sesiones de pi con qwen38flash no hay ningún "Tool … not found".
   - Sería un commit más del fork, que se suma a los defaults que la sección 1.5 de la revisión anterior pide fijar.
4. **Los ítems de la revisión del 10-20 siguen igual:** upstream no los cambió, y los cherry-picks simulados dan el
   mismo resultado (sección 1.3).
5. **llama.cpp: un solo hallazgo útil, en las compuertas y no en producción** (`ead9f3205`, sección 2). El servidor
   del fork no devuelve log-probs de los tokens aceptados por un draft. Por eso, en la compuerta de historia, la
   pasada con los drafts de producción solo compara qué token salió en esos tokens. La pasada sin drafts sí compara
   todo.

## 1. Strata

### 1.1 El cuelgue de la primera ventana (`86d10ed8`)

**El defecto en HEAD:**
- La ventana de verificación con todos los expertos residentes usa el grafo "zero-doorbell" (`src/core/verify.cpp:2744`).
  Producción lo usa: la línea "100% VRAM resident: zero-doorbell graph" aparece en cada arranque del log.
- Ese grafo espera un solo flag del host, el de las filas PLE. El kernel lo espera girando, sin timeout
  (`wait_flag_ge_kernel`, `src/kernels/cuda/verify_kernels.cu:951-954`).
- `Verifier::run` lanza el grafo (`verify.cpp:1726`) y sube el flag después de que la llamada vuelve
  (`verify.cpp:1737-1746`).
- Según el reporte de upstream, en el primer lanzamiento de un grafo el host queda dentro de `hipGraphLaunch`
  esperando un lock del runtime, mientras el runtime espera a la GPU, que espera el flag (comentario de
  `up:src/core/verify.cpp:272-279`).
- Si ocurre, el watchdog aborta el motor a los 60 s (`src/program/generate.cpp:7200-7230`, `STRATA_WATCHDOG_S`).
- **Ese es el costo menor.** La siguiente petición vuelve a cargar el motor sola: "the engine had stopped ...;
  starting it again" (`serve/server.py:2480-2486`).
  - La recarga no pasa por `gate.py`. La única guarda es `min_free_vram_mib` (`serve/server.py:2466-2474`), y nuestra
    config no la fija.
  - Es la carga completa de producción, a 262144 × 2, mientras puede estar corriendo otro trabajo pesado (un render,
    ComfyUI).

**El arreglo de upstream:**
- En el primer lanzamiento de cada grafo, lee las filas PLE y sube el flag antes de lanzar.
- Modo 2 por defecto en HIP (`STRATA_VERIFY_EARLY_PLE_FLAG=0/1/2`).
- Leer primero cuesta ~1.4 % de decode según upstream, pero solo en el primer lanzamiento de cada grafo.
- Es exacto: los mismos kernels con los mismos datos.

**Lo que falta en upstream y en el fork:** los caminos batch también lanzan antes de subir el flag
(`run_slot_rows`, `batch_launch` y `batch_poll`; `verify.cpp:2359`, `2515`, `2540`). Producción los usa con dos slots y
`--batch-mtp`. Al extender el arreglo hay dos condiciones, verificadas en el código:
- el flag solo lo reinicia el host (`verify.cpp:1644`, `2333`);
- la marca de "ya lanzado" de un grafo batch se borra cuando se desaloja el grafo (`verify.cpp:2284`).

**Cherry-pick:** un hunk en conflicto en `verify.cpp`, ~20 líneas a mano.

**Compuertas:**
- N arranques en frío, primero con una petición sola y después con dos a la vez, contando las ventanas que se
  detienen;
- `batch_test` y la compuerta de historia como regresión.

La compuerta de calidad no hace falta: el arreglo no cambia la numérica.

### 1.2 Lo que un merge encendería o cambiaría

| Commit | Qué | Efecto aquí | Cómo conservar lo de hoy |
|---|---|---|---|
| `70f42407` (#3075, #3441) | una llamada a una herramienta no declarada vuelve como texto (`up:serve/frontend.py:965-970`, `1460`) | pi responde "Tool X not found" y reintenta (`pi-agent-core/dist/agent-loop.js:487-494`). Sin tool calls, en cambio, el turno se cierra (`:158-165`) | no hay interruptor: un commit del fork |
| `034422c4` | `nice +5` al motor con Radeon integrada en Linux (`up:serve/server.py:3291-3294`) | se enciende en gfx1151 | `STRATA_ENGINE_NICE=0` en el `env` de `qwen38flash.json` |
| `9fe2ad3f` + `4c953b42` | "modo 8" del layout de expertos en gfx1151 | solo alcanza a `blk.2`, la única capa con gate/up IQ4_XS: las otras 47 van por `STRATA_EXPERT_V2` (`up:src/kernels/cuda/iq_kernels.cu:5085`). Las ganancias publicadas (−6 a −25 %) salen de `native_expert_bench`, que no aplica los defaults de gfx1151 | `STRATA_GFX1151_EXP8=0` |
| `86d10ed8` | sección 1.1 | se enciende en HIP | se deja encendido |

El resto de los defaults nuevos no se activa en gfx1151 con Linux: el iGPU 780M, NUMA, la sonda de disco (solo con
`--resident-budget-gib`) y la caché automática.

### 1.3 Los ítems de la revisión del 10-20

- **Sin cambios upstream:** las seis correcciones de su sección 1.3, el top-k (`82922f55`, `6ef23b89`, `b6ccf873`),
  `8824e217` (#1620), `14b9dd2e` (#1666), `940011da` y `53f72571`. Los cherry-picks simulados contra el mismo HEAD
  dan lo mismo que entonces.
- **Extendido:** `f332a855` (la guarda de logits no finitos) por `894f133b`, que reintenta la lectura con
  `--short-read 0`. Depende de la lógica de la guarda, que HEAD no tiene, y es el cherry-pick más pesado: `server.py`
  4 hunks, `generate.cpp` 3.
- **Sin respaldo:** el "#1700 (borde del contexto)" que cita esa revisión no aparece en ningún commit, mensaje ni
  archivo de upstream.
- **Las imágenes con dos slots siguen rotas upstream.** El servidor manda imágenes a los slots con `BGENI`
  (`up:serve/server.py:1931`) y el motor lo acepta (`up:src/program/generate.cpp:12000-12016`). Nuestro `bf7f443` sigue
  siendo necesario.
- **La cesión no cambió:** los diffs no tocan `pick_slot`, `read_part` ni el yield del motor.

### 1.4 Visto y descartado

- **DFlash2** (`7f97c35a` y siguientes): un drafter alternativo al MTP, opt-in. La release dice "Auto means MTP,
  because no card has shown DFlash2 winning everywhere yet" (`up:docs/release-notes-0.2.3.md`). No hay cifras de
  Strix Halo.
- **Otros cambios del motor:**
  - `STRATA_SPEC_COST` (`68781055` y siguientes), la ventana MTP según el costo: opt-in.
  - `aec327ee`, la lectura hc fusionada: no corre con `--native`, que fuerza la rama nativa de `gr_read`
    (`src/program/generate.cpp:2121`).
  - `6d4874a4` (HC_Q8 por defecto) y `4ed8eb02`/`c62d9801` (la tabla hipBLASLt por defecto): ya los tenemos por
    variable de entorno.
- **Otras plataformas:** `b1594fbb` (el hotfix 0.2.1.2 de Strix Halo), `051db74c`, las colas (`3f1fa67f` y
  siguientes), `ab61f892` y `aece46db` son solo de Windows, solo de gfx12 o solo de la CLI.
- **Servidor, sin efecto para pi ni para nuestra config:**
  - `8bf178fe`: alias `ns__tool`; las herramientas de pi no tienen puntos.
  - `828a2559`: tope de tokens por imagen. Ya damos 4096 y el mtmd del fork limita.
  - `76186bb5`: recrea el directorio del codificador si se borra; en Linux el arreglo queda a medias.
  - `c330caf4`, `fccfd343`, `c40a03e6`, `a0b6d1cd`, `c6a20b7e`, `2f5f7c11` y `5c8dd322`: cosméticos, o en rutas
    que no usamos.
- **Dato de calidad de HC_Q8, que producción ya usa** (`up:docs/AMD_HIP.md`, RTX A4000, IQ3_XXS): KL 0.0251 contra
  0.0228 del redondeo puro. Peor en agente: 0.0779 contra 0.0745.

## 2. llama.cpp

| Commit | Qué | Veredicto |
|---|---|---|
| `ead9f3205` (#27196) | log-probs de los tokens aceptados por un draft | **tomar, adaptado.** El fork tiene el mismo hueco: `tools/server/server-context.cpp:4135` ("TODO: set result.probs"). `decode_budget.py` pide `top_logprobs` 4 y `logprob_diff.py` compara esas listas: en la pasada con drafts de la compuerta de historia, los tokens aceptados por el draft comparan `[] == []`. Conflicto en `server-context.cpp`, ~10 líneas. El fork ya guarda los índices del lote de verificación (`slot.spec_i_batch`, `:329`, usados en `:4029-4035`): el port es conservar esa lista hasta el envío de los tokens y llamar a `populate_token_probs` con cada índice |
| `f711c8f30` (revisión del 10-20) | resize de Qwen-VL como transformers | sigue pendiente. Ningún commit posterior tocó el camino de Qwen en `mtmd-image.cpp` |
| `edc353842` (#30943) | resize con los pesos int16 de torchvision (`RESIZE_ALGO_FLAG_TORCH`), asignado solo a LFM2 y d1-omni | a considerar junto con `f711c8f30`. La referencia de Qwen usa torchvision: `Qwen2VLImageProcessor(TorchvisionBackend)`, BICUBIC, `tvF.resize(antialias=True)` (transformers 5.5.3: `models/qwen2_vl/image_processing_qwen2_vl.py:92-94`, `image_processing_backends.py:251`). Nuestro bicúbico de Pillow difiere en 1-2 niveles en ~0.25 % de los valores (medición del commit en LFM2; en Qwen sin medir) |
| `d454a204e` (#30640), `5c3114268` (#30965) | op fusionada `ggml_top_k_mask` para la selección QSA | saltar. Solo CPU y CUDA, sin Vulkan; el fork arma su propia lista para la FA sparse (`src/models/qwen4exp.cpp:1006-1065`) |
| `d9ee007a0`, `1ab09d632`, `4912ac24f` | drafters DFlash sobre qwen4exp | saltar: el drafter publicado para Flash Next no mejora al MTP (documento de costes, 29.3) |
| `8baabd53b` (#30861) | escala por experto y token del down MoE | saltar: solo Mistral4, Mellum y Flint |
| `8e5a58df8` (#30656) | precisión de src1 y del acumulador en Vulkan | arrastrar en el merge: sin precisión fijada se comporta igual que antes |
| `7894e8b92` | `build_inp_temporal_merge` pasa a la clase base del mtmd | arrastrar: renombre |
| resto | HIP/CUDA, otros modelos, server (rerank, `/v1/messages`, TTS, realtime, router), SYCL/OpenCL/Metal/Hexagon, CI | inertes aquí |

## 3. Pendiente con la GPU

La lista de la sección 3 de la revisión del 10-20 sigue vigente. Cambios:
- **Se agrega `86d10ed8`, adaptado y extendido a los caminos batch** (sección 1.1), en el lote de cherry-picks de
  Strata.
- **`f711c8f30` puede ir con el resize de torchvision para Qwen** (`edc353842`). Si se mide, en la misma ventana de
  imágenes.
- **`ead9f3205` va con los cherry-picks de llama.cpp.** Se verifica con la compuerta de historia con drafts: los
  tokens aceptados deben traer `top_logprobs`.
- **Si se hace el merge de Strata:** a los defaults de la sección 1.5 de esa revisión se suman `STRATA_ENGINE_NICE=0`,
  `STRATA_GFX1151_EXP8=0` y un commit del fork que revierta lo de las herramientas no declaradas de `70f42407`.

**Riesgo lateral, fuera de upstream:**
- `systemd-tmpfiles` limpia `/tmp` a los 10 días (`/usr/lib/tmpfiles.d/tmp.conf:11`).
- El directorio de trabajo de `strata-vision` está en `/tmp` (`serve/server.py:1690`).
- No pasó: hay directorios del 10-07 que siguen ahí.
- Una línea `x /tmp/strata-vision-*` en `/etc/tmpfiles.d` lo cerraría. Es un cambio con privilegios: lo decide y lo
  aplica el Director.

## Fuentes

- Revisiones de los agentes, con la evidencia por ítem:
  - `~/ProjectHub/llama.cpp-evidence/upstream-1024/strata-engine.md`;
  - `~/ProjectHub/llama.cpp-evidence/upstream-1024/strata-serve.md`;
  - `~/ProjectHub/llama.cpp-evidence/upstream-1024/llama.md` (notas de llama.cpp).
- Los puntos centrales se verificaron contra el código antes de escribir este documento:
  - el orden del lanzamiento en `verify.cpp:1718-1746` y el kernel de espera;
  - el reintento de pi;
  - el interruptor de `nice`;
  - el origen del +6-19 %;
  - la referencia de Qwen en transformers;
  - el `TODO` del servidor del fork.
