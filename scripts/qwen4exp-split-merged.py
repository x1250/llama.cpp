"""qwen4exp-split-merged.py <in.gguf> <out.gguf>: the inverse of qwen4exp-merge-gate-up-exps.py and
qwen4exp-merge-hc-inject.py, for engines that read the checkpoint's own tensor layout (Strata). Every
blk.N.ffn_gate_up_exps.weight [n_embd, 2*n_ff, n_expert] becomes blk.N.ffn_gate_exps.weight and blk.N.ffn_up_exps.weight
[n_embd, n_ff, n_expert] (per expert: its gate rows, then its up rows), and every blk.N.hc_{attn,ffn}_down_inject.weight
[hc_dim, hc_lr + hc] becomes blk.N.hc_{attn,ffn}_down.weight [hc_dim, hc_lr] and blk.N.hc_{attn,ffn}_inject.weight
[hc_dim, hc] (its first hc_lr rows, then the last hc). Byte ranges of whole rows, no requantization; every other
tensor and every metadata field is copied verbatim. The data is copied inside the kernel (copy_file_range) and both
files are dropped from the page cache as the copy advances, so the process stays small and MemFree high."""
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
hc = int(reader.fields[f"{arch}.hyper_connection.count"].contents())
hc_lr = int(reader.fields[f"{arch}.hyper_connection.low_rank"].contents())
writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
writer.data_alignment = align

for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

GATE_UP = re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$")
DOWN_INJECT = re.compile(r"^blk\.(\d+)\.hc_(attn|ffn)_down_inject\.weight$")
plan = []; n_gate_up = 0; n_hc = 0   # (name, pieces [(src_offset, n_bytes)], element shape, type, total bytes)
for t in reader.tensors:
    m = GATE_UP.match(t.name)
    if m:
        assert len(t.shape) == 3 and int(t.shape[1]) % 2 == 0, (t.name, t.shape)
        n_expert = int(t.shape[2])
        assert t.n_bytes % (2 * n_expert) == 0
        half = t.n_bytes // (2 * n_expert)               # one expert's gate (or up) rows
        shape = [int(t.shape[0]), int(t.shape[1]) // 2, n_expert]
        for part, first in (("gate", 0), ("up", 1)):
            pieces = [(t.data_offset + (2 * e + first) * half, half) for e in range(n_expert)]
            plan.append((f"blk.{m[1]}.ffn_{part}_exps.weight", pieces, shape, t.tensor_type, half * n_expert))
        n_gate_up += 1
        continue
    m = DOWN_INJECT.match(t.name)
    if m:
        assert len(t.shape) == 2, (t.name, t.shape)
        rows = int(t.shape[1])
        assert rows == hc_lr + hc and t.n_bytes % rows == 0, (t.name, t.shape, t.n_bytes)
        row = t.n_bytes // rows
        lr = hc_lr
        plan.append((f"blk.{m[1]}.hc_{m[2]}_down.weight", [(t.data_offset, lr * row)], [int(t.shape[0]), lr],
                     t.tensor_type, lr * row))
        plan.append((f"blk.{m[1]}.hc_{m[2]}_inject.weight", [(t.data_offset + lr * row, hc * row)], [int(t.shape[0]), hc],
                     t.tensor_type, hc * row))
        n_hc += 1
        continue
    plan.append((t.name, [(t.data_offset, t.n_bytes)], [int(x) for x in t.shape], t.tensor_type, t.n_bytes))

for name, pieces, shape, ttype, nbytes in plan:
    writer.add_tensor_info(name, list(reversed(shape)), np.dtype(np.float32), nbytes, raw_dtype=ttype)
logging.info(f"{len(reader.tensors)} tensors in, {n_gate_up} gate_up and {n_hc} down_inject split, {len(plan)} out, "
             f"alignment {align}")

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
