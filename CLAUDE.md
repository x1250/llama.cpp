IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# Resource safety on the Strix Halo desktop (mandatory)

- Before ANY model load or GPU-heavy run: compute the footprint (weights + KV + compute buffers) and require `MemAvailable - footprint >= 40 GiB`. Abort otherwise.
- One load at a time: `pgrep llama-` must be empty before starting; monitor MemAvailable while it runs.
- qwen4exp: lazy mode (`-lzm auto`, 66 GiB) is the only safe config with the desktop up. `-lzm off` / `LAZY=0` pins 93 GiB, leaves ~15 GiB and froze the machine (2026-09-02, freeze #7). Do not run it unless the Director closes the desktop apps first.
- Benchmarks: put every test of one config in ONE llama-bench invocation (one load); separate loads only for env knobs.
- After a process exits, its GTT pages come back to `MemAvailable` asynchronously: the gate must wait (bounded) for the margin, not assume it. A run with host-visible (write-combined) Vulkan memory (`GGML_VK_PREFER_HOST_MEMORY=1`) leaves its pages (~66 GiB) parked in the kernel TTM page pool, outside `MemAvailable`, until memory pressure; `pkexec sysctl -w vm.drop_caches=3` releases them (it also runs the slab shrinkers). Avoid that env on this desktop.
- Never edit sources while a build is running: make picks up the edit mid-build and the binary silently differs from what the bench labels say (2026-09-02: an unplanned graph edit entered a build and produced two hours of contradictory MMQ measurements). Rebuild, then edit.
- Every measurement chain re-runs the baseline config in the same chain (first and last); the Vulkan graph optimizer's concurrency makes results sensitive to graph node order, so a source change outside the backend can move numbers by 10%.
