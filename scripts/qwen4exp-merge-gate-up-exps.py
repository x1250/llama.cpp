"""qwen4exp-merge-gate-up-exps.py <in.gguf> <out.gguf>: a GGUF whose per-layer pair (blk.N.ffn_gate_exps.weight,
blk.N.ffn_up_exps.weight), both [n_embd, n_ff, n_expert] of the same type, is replaced by blk.N.ffn_gate_up_exps.weight
[n_embd, 2*n_ff, n_expert]: for every expert its gate rows followed by its up rows, the layout build_moe_ffn splits back
into the gate and up views (src/llama-graph.cpp, merged gate_up path). A byte concatenation per expert, no requantization;
every other tensor and every metadata field is copied verbatim. The loader accepts either layout
(create_tensor_gate_up_exps). The data is copied inside the kernel (copy_file_range) and both files are dropped from the
page cache as the copy advances, so the process stays small and MemFree high."""
import sys, os, re, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gguf-py"))
import gguf

logging.basicConfig(level=logging.INFO)
src, dst = sys.argv[1], sys.argv[2]
assert not os.path.exists(dst), dst
reader = gguf.GGUFReader(src, "r")
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
assert arch == "qwen4exp", arch
align = int(reader.fields["general.alignment"].contents()) if "general.alignment" in reader.fields else gguf.GGUF_DEFAULT_ALIGNMENT
writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
writer.data_alignment = align

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

tensors = {t.name: t for t in reader.tensors}
RX = re.compile(r"^blk\.(\d+)\.ffn_gate_exps\.weight$")
plan = []; merged = 0   # (name, pieces [(src_offset, n_bytes)], element shape, type, total bytes)
for t in reader.tensors:
    m = RX.match(t.name)
    if m:
        up = tensors[f"blk.{m[1]}.ffn_up_exps.weight"]
        assert t.tensor_type == up.tensor_type and list(t.shape) == list(up.shape) and len(t.shape) == 3, (t.name, t.tensor_type, up.tensor_type, t.shape, up.shape)
        n_expert = int(t.shape[2])
        assert t.n_bytes % n_expert == 0
        per_expert = t.n_bytes // n_expert
        pieces = []
        for e in range(n_expert):
            pieces.append((t.data_offset + e * per_expert, per_expert))
            pieces.append((up.data_offset + e * per_expert, per_expert))
        shape = [int(t.shape[0]), 2 * int(t.shape[1]), n_expert]
        plan.append((f"blk.{m[1]}.ffn_gate_up_exps.weight", pieces, shape, t.tensor_type, 2 * t.n_bytes))
        merged += 1
        continue
    if re.match(r"^blk\.\d+\.ffn_up_exps\.weight$", t.name):
        continue
    plan.append((t.name, [(t.data_offset, t.n_bytes)], [int(x) for x in t.shape], t.tensor_type, t.n_bytes))

for name, pieces, shape, ttype, nbytes in plan:
    writer.add_tensor_info(name, list(reversed(shape)), np.dtype(np.float32), nbytes, raw_dtype=ttype)
logging.info(f"{len(reader.tensors)} tensors in, {merged} pairs merged, {len(plan)} out, alignment {align}")

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
writer.fout[0].flush()
out_fd = os.open(dst, os.O_WRONLY)
in_fd = os.open(src, os.O_RDONLY)
pos = os.fstat(out_fd).st_size
pad = (-pos) % align
if pad:
    os.pwrite(out_fd, bytes(pad), pos); pos += pad
written = 0; since_drop = 0
for name, pieces, shape, ttype, nbytes in plan:
    for off, n in pieces:
        done = 0
        while done < n:
            c = os.copy_file_range(in_fd, out_fd, n - done, off + done, pos)
            if c <= 0:
                raise RuntimeError(f"copy_file_range returned {c} at {name}")
            done += c; pos += c
    pad = (-nbytes) % align
    if pad:
        os.pwrite(out_fd, bytes(pad), pos); pos += pad
    written += nbytes; since_drop += nbytes
    if since_drop >= (2 << 30):
        os.fsync(out_fd)
        os.posix_fadvise(out_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(in_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        since_drop = 0
        logging.info(f"{written / 2**30:.1f} GiB written")
os.fsync(out_fd)
os.posix_fadvise(out_fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.posix_fadvise(in_fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(out_fd); os.close(in_fd)
writer.fout[0].close()
logging.info(f"done: {written} tensor bytes, file {os.path.getsize(dst)} bytes")
