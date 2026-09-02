IMPORTANT: Ensure you’ve thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

# Resource safety on the Strix Halo desktop (mandatory)

- Before ANY model load or GPU-heavy run: compute the footprint (weights + KV + compute buffers) and require `MemAvailable - footprint >= 40 GiB`. Abort otherwise.
- One load at a time: `pgrep llama-` must be empty before starting; monitor MemAvailable while it runs.
- qwen4exp: lazy mode (`-lzm auto`, 66 GiB) is the only safe config with the desktop up. `-lzm off` / `LAZY=0` pins 93 GiB, leaves ~15 GiB and froze the machine (2026-09-02, freeze #7). Do not run it unless the Director closes the desktop apps first.
- Benchmarks: put every test of one config in ONE llama-bench invocation (one load); separate loads only for env knobs.
