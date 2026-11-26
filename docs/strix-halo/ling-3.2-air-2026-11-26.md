# Ling-3.2-air on llama-server, and the rope of its MLA layers (2026-11-26)

Ling-3.2-air (inclusionAI, arch `bailingmoe3`) was installed as `ling32air` in strix-halo: 42 layers (35 KDA, 7 MLA at
layers 5, 11, ..., 41), 512 experts with 8 active, an MTP head (blk.42, unused: MTP is off), vision, 262144 of context
x 2 slots with the manufacturer's sampling (temperature 1.0, top_p 0.95, top_k 20, thinking on). In the Director's
first agent sessions in pi the model degenerated at long context with both GGUFs it ran on. There were two llama.cpp
defects: the rope of the MLA layers (fixed in the fork by `3e6dd9641`, section 2) and the Ling 3 parser keeping the
newline before a tool call in the content, which grew by one newline per turn (fixed by `d3c2177fd`, section 5). Images
reached the model on other rope bands and at other positions than the reference's (fixed by `46ad2807f` and
`d2f46fb65`, section 6), and videos in a generic layout (fixed by `c2c45429e`, section 7). Evidence:
`~/ProjectHub/llama.cpp-evidence/ling32air/`.

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

## 6. Images (`46ad2807f`, `d2f46fb65`)

After the rope fix two differences from the reference remained at image positions. Both occur whenever an image is
sent to ling32air (the mmproj is loaded), and the Director sends images.

- **The bands** (`46ad2807f`, `src/models/bailingmoe3.cpp`). The reference's `apply_video_rope` gives the 32 rotary
  pairs, from the highest frequency, to H and W alternating pair by pair over the first 24 (H on the even ones) and to
  T on the last 8. ggml's M-RoPE with `[8, 12, 12]` gives contiguous slots to T (0-7), H (8-19) and W (20-31). The q
  and k channels now go to the slot of their section (`bailingmoe3_rope_input`: three strided views and two
  concats), and the MLA layers get a `rope_freqs` tensor made at load (`load_tensors`) that gives each slot the
  frequency of its pair. `get_rope_params` sets the rope base to 1 for those layers, so the frequency comes from one
  division and not from `theta_scale^i` (see below); a `--rope-freq-base` override does not apply to them, and YaRN is
  off (its ramp follows the slot order; the reference has no rope scaling). The sections are checked at load.
- **The positions** (`d2f46fb65`, `tools/mtmd/mtmd.cpp`). The reference's `get_rope_index` puts an image at
  `t = s`, `h = s + row - (ny-1)//2`, `w = s + col - (nx-1)//2`, and the text after it goes on at `s + 1`
  ("body-diagonal symmetry", as SGLang's `BailingMRotaryEmbedding`). mtmd gave `LING3VL` the Qwen layout
  (`h = s + row`, `w = s + col`, the text after at `s + max(nx, ny)`), so each image also moved the positions of all
  the text after it. A position type of its own (`MTMD_POS_TYPE_LING3VL`, like SenseNova U1's) gives the reference's.
  Near the start of the context x and y are negative; `mtmd_decoder_pos` carries them in two's complement to the
  batch's `llama_pos`.

The frequency from one division. A first version kept the rope base and gave the factors `theta_scale_f32^i / freq`,
so that they cancelled the f32 rounding of `theta_scale`: in numpy (which rounds a power correctly) its angles matched
the plain rope's. On the kernels they did not: the CPU builds `theta_scale^i` from i products and Vulkan from a
`pow`, a systematic phase drift that grows with the distance between q and k, and that version had the high
frequencies on the high slots, where it is largest. `rope_angle` (the angle of every slot of `ggml_rope_multi` at text
positions, against the exact angle of its pair; mean signed error over the top 1024 positions, pairs 2-3, Vulkan):

| Layout | Drift at 32k (rad) | Drift at 262k (rad) | Max error at 262k (rad) | wikitext PPL, 32k |
|---|---|---|---|---|
| Plain rope in its own slots (`3e6dd9641`) | -0.0021 / -0.0014 | -0.017 / -0.011 | 0.024 | 14.85 |
| Factors over `theta_scale^i` (first version) | -0.0044 / -0.0039 (up to -0.0084 on pair 1) | -0.036 / -0.031 (-0.068) | 0.086 | 14.89 |
| Rope base 1, factor `1/freq` (`46ad2807f`) | -0.0002 / -0.0005 | -0.0015 / -0.0037 | 0.024 | **14.75** |

Pairs 0 and 1 drift the same in the plain rope and in the adopted layout (-0.0013 at 32k): the frequency there is
exact in both and the drift is Vulkan's `sin`/`cos` of large arguments. The perplexity is deterministic (14.7478 twice, and the same with
`GGML_VK_DISABLE_FUSION=1`), and its order follows the drift. That the plain rope drifts more than one division is
upstream code for every RoPE model, recorded in `latent-defects.md`.

| Check | Before | After |
|---|---|---|
| q.k at image positions against `apply_video_rope` + `apply_rotary_pos_emb_interleave` (numpy, double, positions near 200k, `img/mrope_check.py`) | max diff 14 | 1.4e-14 |
| The positions mtmd gives a prompt with three non-square images (20x40, 28x14, 3x3 merged tokens) against a port of `get_rope_index` (`img/pos_check`, the projector loaded on the CPU, the text model with `no_alloc`) | 1224 of 1234 differ | 0 differ |
| `test-backend-ops` ROPE and CONCAT, Vulkan against the CPU | | 510/510, 236/236 |
| 148 spatial questions on synthetic non-square images, greedy, thinking off (`img/img_ab.py`): exact answers, mean probability of the gold answer (teacher-forced) | 110/148, 0.640 | **127/148, 0.719** |
| of those, the cell of a red circle in a 4x8 / 8x4 grid | 30/64, 0.465 | 46/64, 0.608 |
| the 3x4 grids | 33/36, 0.896 | 33/36, 0.895 |
| five digits in order along a non-square image | 23/24, 0.957 | 24/24, 0.997 |
| left/right, above/below (the probability splits between "Left" and "left", so only the count measures position) | 24/24 | 24/24 |
| A three-turn conversation, two images, one slot (`img/img_turns.py`) | right, cache reused | right, cache reused, no position error |
| The 2048x2048 screenshot (`img_big.py`) | | 13.2 s, transcribed, health ok |

The A/B ran against production (262144 x 2), the before on build `b2491-3e6dd9641`, the after on this one. The chain
is `imgrope.sh` (build, `test-backend-ops`, the perplexity, install, reload) and `pplnoise.sh` (the perplexity
repeated and without fusions); outputs in `runs/` (`img-*.log`, `imgturns-*.log`, `poscheck-*.log`,
`rope-angle.log`, `ppl-imgrope*.log`, `ppl-C-*.log`).

The references disagree on the image positions. The HF modeling file that ships with the weights and SGLang center
the image; inclusionAI's vLLM fork (`inclusionAI/vllm-ling-v3`, `get_mrope_input_positions`, written in September,
images only) uses the Qwen layout, the one mtmd had. An A/B decided it (`posab.sh`, the same 148 questions, the bands
fixed in both):

| Build | Exact | Mean gold probability | 4x8 grid |
|---|---|---|---|
| Before both fixes | 110/148 | 0.640 | 30/64 |
| Bands fixed, Qwen positions (vLLM's) | 122/148 | 0.708 | 41/64 |
| Bands fixed, centered positions (HF, SGLang; `d2f46fb65`) | **127/148** | **0.719** | **46/64** |

The centered positions have the higher gold probability in all four tasks; most of the gain is the bands'.

## 7. Video (`c2c45429e`)

Ling 3 VL took mtmd's default video template: a "Video:" prefix, `[XmY.YYs]` stamps every 5 s and frames at the
generic rate. The model's processor (`processing_bailingmm2.py`, with `use_interleaved_frame_timestamp: true` in
`preprocessor_config.json`, and `bailingmm_utils_video.py` v1, its default) lays a video out as
`<|video_start|>` + `<X.X seconds><|vision_start|>` frame pair `<|vision_end|>` per pair + `<|video_end|>`, with:

- 2 fps, at least 1 frame and at most 32, an even number (`v1_smart_nframes`), the frames of
  `np.linspace(0, n - 1, k, dtype=int)` (truncated; Qwen3-VL rounds);
- 4718592 pixels shared by the frames, counted in pairs, at least 43008 per frame (`v1_fetch_video`, its
  `smart_resize`): a 10 s 1080p video gives 20 frames of 896x512, a 20-minute one 32 frames of 704x384;
- each pair stamped at the mean of `i / fps` of its two frames, the fps of the sampled frames over the source (not the
  source frames' own times); an odd last frame is padded with a copy and stamped as itself.

`MTMD_VIDEO_TEMPLATE_LING3VL` gives that layout (`mtmd_video_config_init`, `mtmd_image_preprocessor_ling3vl`, the video
helper's indices and stamps), with `<|video_start|>`/`<|video_end|>` as Ling's video markers. Each pair is placed as an
image (one temporal position, H/W centered, `d2f46fb65`): with interleaved stamps every pair is its own
`<|vision_start|>` segment, as SGLang's `bailing_3drope_get_input_positions_tensor` treats them. The HF modeling file
reads the flag from `text_config`, where `config.json` does not set it, and would then give the frames text positions;
SGLang's comment says the per-frame segments are the intended layout.

| Check | Before | After |
|---|---|---|
| `test-mtmd-impl` `test_video_ling3vl`: frame counts, sizes and sequences of five videos from the reference's own functions (`img/ref_video.py`) | | 28 assertions passed |
| The prompt mtmd builds for two encoded videos against the processor's, and its positions (`img/vid_seq_check.py`) | | identical, 0 positions differ |
| 40 synthetic videos, greedy, thinking off (`img/vid_ab.py`): the second at which a square appears, answer within 1 s | 6/8, probability 0.18 | **8/8, 0.99** |
| the same as a multiple choice (the options' probabilities after "The black square appears at second") | 2/8, share 0.30 | **8/8, 0.87** |
| the direction of a moving dot, answered in one word | 0/16 (prose) | 8/16 |
| the same as a multiple choice | 16/16, 0.97 | 15/16, 0.87 |
| the order of three colors (answer and multiple choice) | 16/16 | 16/16 |
| A two-turn conversation on one video | | right, the second turn reads 25 tokens over 2330 cached |

The direction's multiple choice is not a measure for "up": the four options take 2-23 % of the next token in both
builds (the model says "upward"), and the one item it misses answers "towards the top" in prose. A 6 s video takes
~1400 tokens with the processor's sampling (12 frames) and took ~2950 with the generic one. The chain is `vidchain.sh`
(build, unit tests, the image positions byte-identical to the build before, reload).

## 8. Memory

`srv_mem.py` with production loaded (262144 x 2, KV f16, mmproj on the GPU): 77.9 GiB at load (72.3 of weights, 0.8 of
mmproj, 4.9 of KV and buffers), 78.0 after the replays. The checkpoints (8 per slot, ~73 MiB each) and the prompt cache
(4 GiB) add up to ~5.1 GiB more: `kv_gb` 11 in `config.ini` (it was 18, an estimate, which kept the launcher's gate
~7 GiB above what the server uses). The load dropped MemAvailable from 107.6 to 27.4 GB.
