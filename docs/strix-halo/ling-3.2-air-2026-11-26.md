# Ling-3.2-air on llama-server, and the rope of its MLA layers (2026-11-26)

Ling-3.2-air (inclusionAI, arch `bailingmoe3`) was installed as `ling32air` in strix-halo: 42 layers (35 KDA, 7 MLA at
layers 5, 11, ..., 41), 512 experts with 8 active, an MTP head (blk.42, unused: MTP is off), vision, 262144 of context
x 2 slots with the manufacturer's sampling (temperature 1.0, top_p 0.95, top_k 20, thinking on). In the Director's
first agent sessions in pi the model degenerated at long context with both GGUFs it ran on. There were two llama.cpp
defects: the rope of the MLA layers (fixed in the fork by `3e6dd9641`, section 2) and the Ling 3 parser keeping the
newline before a tool call in the content, which grew by one newline per turn (fixed by `d3c2177fd`, section 5).
Evidence: `~/ProjectHub/llama.cpp-evidence/ling32air/`.

## 1. Symptom

- bartowski Q4_K_S, ~30k of context: an answer ended in 386 newlines and a broken tool call.
- aj9o9 L (experts IQ4_XS/Q5_K, the rest Q8_0), ~43k of context: the reasoning repeated a block of five paragraphs
  verbatim seven times (41 lines, 10 distinct) until the Director cancelled it at 2852 tokens.

Two quantizations from two authors failing the same way pointed away from the weights.

## 2. Cause (upstream issue #33079, open on 2026-11-26, no PR)

The reference (`modeling_bailing_moe_v3_vl.py`, `rope_interleave=True` by default, not overridden by `config.json`)
writes each head's 64 rotary channels pair-interleaved, `[x0, y0, x1, y1, ...]`, and rotates the adjacent pairs
`(2j, 2j+1)` at frequency `j` (`apply_rotary_pos_emb_interleave`). A VL GGUF carries
`rope.dimension_sections = [8, 12, 12, 0]`, so `llama_model_rope_type()` selects `LLAMA_ROPE_TYPE_MROPE`
(`src/llama-model.cpp`, "VL files carry mrope sections"), and ggml's M-RoPE rotates the pairs `(i, i + 32)`. Every
rotated pair mixed two channels of different pairs. At position 0 the rotation is the identity and nothing differs, so
the error grows with the distance between tokens, which matches a model that works at short context and loops past
15-45k. The defect entered with the Ling 3.0 VL support (`f830688e9`, PR #29151) and affects every Ling 3.x VL GGUF.
Text-only files carry no sections and use NORM rope, which rotates the adjacent pairs as the reference does.

There is no rope scaling to account for: `config.json` has no `rope_scaling` (so no yarn mscale in the softmax scale),
the GGUF none either; `rope_theta` 6e6, `qk_rope_head_dim` 64 fully rotated, the attention scale `1/sqrt(192)` as in the
reference.

## 3. Fix (`3e6dd9641`)

`bailingmoe3_rope_input()` in `src/models/bailingmoe3.cpp` reorders `q_pe` and `k_pe` to `[x0, x1, ..., y0, y1, ...]`
(a 4D view `[2, 32, heads, tokens]`, permute, cont) before `build_rope` when the model uses M-RoPE, in the main graph
and in the MTP graph. q and k get the same order, so at text positions q.k is the reference's. The cost is one copy
of 64 channels per head in 7 layers.

It is a fork patch kept on every merge until upstream fixes #33079. The alternative of zeroing the sections in the
GGUF (NORM rope) would be exact for text too but would take the 2D positions from images.

## 4. Verification

| Check | Before | After |
|---|---|---|
| q.k against `apply_rotary_pos_emb_interleave` (numpy, 200 random pairs, positions up to 262144) | max diff 2.8e+01 | 8.9e-15 |
| llama-perplexity, wikitext-2, one 32768 chunk scored at 16-32k, aj9o9 L, ub 512, FA on | 17.44 +/- 0.46 | 14.85 +/- 0.39 |
| The looping pi request (38.4k-token prompt, production sampling, 3 replays) | looped (the Director's session) | 3/3 end on their own: 9-10 reasoning lines, all distinct, no repeated paragraph |
| The newline turn of the bartowski session (aj9o9 L, the history before it as the fixed parser returns it, 3 replays) | 386 newlines (the Director's session) | 3/3 a tool call in 238-268 tokens, longest newline run 2 |

`rope.sh` (the perplexity pair, the build in between, the install only if the perplexity drops), `parser.sh` (the parser
build, test-chat and install), `pi_replay.py` (re-sends a pi session's request, the last one or a given turn's, with the
history's contents optionally stripped before their tool calls; prints only structural measures, never its text) and
`nl_diag.py` (the same request as a raw completion with no grammar, the probabilities at the newline run) are in the
evidence folder; outputs in `runs/`.

### 4.1 Replay of the bartowski session

With only the rope fixed, the replay of that session's last request (its history includes the 386-newline turn) and of
the 386-newline turn itself both still filled their replies with newlines (runs of 7585 and 7919). A raw completion of
the templated prompt, with no grammar, did the same: the model put probability 1.000 on every newline, so the tool-call
grammar was not the cause. The history was: before every tool call the content ended in a newline run that grew from 1
to 7 over the session's 18 calls. Section 5 explains it.

## 5. The newline before a tool call (`d3c2177fd`)

The Ling 3 template renders an assistant turn as its content, then `\n<tool_call>` for each call. The dedicated parser
(`common/parsers/ling3.cpp`) took the content up to `<tool_call>`, so the newline the model writes before a call stayed
in the content. pi sends the content back and the template adds its own newline: the history showed one newline more
than the model wrote, the model copies the spacing of its history, and the run grew by one every few turns. At ~25k of
context the attractor took the whole reply. The parser now stops the content before the whitespace that precedes a
tool call (a negative lookahead for whitespace plus `<tool_call>`); whitespace with no call after it stays content.
`test-chat` covers one and three newlines before a call and trailing whitespace with no call. Upstream's parser is
the same (no report or PR on 2026-11-26).

The Director's existing pi sessions keep the grown runs (and the loops) in their history: a model reading them copies
them again, so they are not reusable; new sessions start clean.

## 6. What still differs from the reference: images

At image positions the sections differ besides the pairs. The reference's `apply_video_rope` puts T on the 8 lowest
frequencies (pairs 24-31) and alternates H and W over pairs 0-23; ggml's M-RoPE with `[8, 12, 12]` puts T on pairs 0-7,
H on 8-19 and W on 20-31. An image's 2D positions therefore reach the model on other frequency bands than it was
trained with. This occurs in our use whenever an image is sent to ling32air (the mmproj is loaded). The fix that keeps
ggml's kernels: reorder the channels so each M-RoPE section holds the reference's pairs of that section, and give the
rope a frequency-factor tensor that sets each reordered pair back to its reference frequency
(`ggml_rope_multi` takes `freq_factors`). Not done yet: it awaits the Director's decision.

## 7. Memory

`srv_mem.py` with production loaded (262144 x 2, KV f16, mmproj on the GPU): 77.9 GiB at load (72.3 of weights, 0.8 of
mmproj, 4.9 of KV and buffers), 78.0 after the replays. The checkpoints (8 per slot, ~73 MiB each) and the prompt cache
(4 GiB) add up to ~5.1 GiB more: `kv_gb` 11 in `config.ini` (it was 18, an estimate, which kept the launcher's gate
~7 GiB above what the server uses). The load dropped MemAvailable from 107.6 to 27.4 GB.
