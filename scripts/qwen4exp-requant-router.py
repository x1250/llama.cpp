"""qwen4exp-requant-router.py <in.gguf> <out.gguf>: a GGUF whose MoE routers (blk.N.ffn_gate_inp.weight, F32
[n_embd, n_expert]) are quantized to Q8_0. The router is read in full on every decode step (48 x 5.2 MB at F32), and
Q8_0 reads 3.8x fewer bytes; build_moe_ffn takes any weight type. Every other tensor and every metadata field is copied
verbatim (gguf_stream_copy.py). The logits change by the quantization, so the file is gated by perplexity before use."""
import logging
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gguf-py"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf                      # noqa: E402
from gguf.quants import quantize  # noqa: E402
import gguf_stream_copy as gsc    # noqa: E402

logging.basicConfig(level=logging.INFO)
src, dst = sys.argv[1], sys.argv[2]
reader = gguf.GGUFReader(src, "r")
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
assert arch == "qwen4exp", arch

RX = re.compile(r"^blk\.\d+\.ffn_gate_inp\.weight$")
plan = []
n_router = 0
for t in reader.tensors:
    if not RX.match(t.name):
        plan.append(gsc.verbatim(t))
        continue
    assert t.tensor_type == gguf.GGMLQuantizationType.F32 and len(t.shape) == 2, (t.name, t.tensor_type.name, t.shape)
    rows = np.asarray(t.data, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0]))   # one row per expert
    q = quantize(rows, gguf.GGMLQuantizationType.Q8_0)
    plan.append(gsc.TensorPlan(t.name, t.shape, gguf.GGMLQuantizationType.Q8_0, [("bytes", q.tobytes())]))
    n_router += 1

assert n_router > 0, "no router found"
logging.info(f"{len(reader.tensors)} tensors, {n_router} routers quantized to Q8_0")
gsc.write(gguf, reader, src, dst, plan)
