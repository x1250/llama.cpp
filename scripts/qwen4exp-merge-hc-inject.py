"""qwen4exp-merge-hc-inject.py <in.gguf> <out.gguf>: qwen4exp GGUF with every pair (blk.N.hc_{attn,ffn}_down.weight [hc_dim, hc_lr],
blk.N.hc_{attn,ffn}_inject.weight [hc_dim, hc]) replaced by blk.N.hc_{attn,ffn}_down_inject.weight [hc_dim, hc_lr + hc]:
the rows of both, concatenated (same type and row length: a byte concatenation, no requantization). Every other tensor
and every metadata field is copied verbatim. The loader accepts either layout.
The metadata is built with gguf-py; the tensor data is copied file to file with copy_file_range (no user-space buffers)
and both files are dropped from the page cache as the copy advances, so the process stays small and MemFree high."""
import sys, os, re, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gguf-py"))
import gguf
from gguf.quants import quant_shape_to_byte_shape

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

# the output tensor list: (name, [(src_offset, n_bytes), ...] pieces to concatenate, element shape, type)
tensors = {t.name: t for t in reader.tensors}
RX = re.compile(r"^blk\.(\d+)\.hc_(attn|ffn)_down\.weight$")
plan = []; merged = 0
for t in reader.tensors:
    m = RX.match(t.name)
    if m:
        inj = tensors[f"blk.{m[1]}.hc_{m[2]}_inject.weight"]
        assert t.tensor_type == inj.tensor_type and t.shape[0] == inj.shape[0], (t.name, t.tensor_type, inj.tensor_type)
        shape = [int(t.shape[0]), int(t.shape[1]) + int(inj.shape[1])]
        plan.append((f"blk.{m[1]}.hc_{m[2]}_down_inject.weight", [(t.data_offset, t.n_bytes), (inj.data_offset, inj.n_bytes)], shape, t.tensor_type, t.n_bytes + inj.n_bytes))
        merged += 1
        continue
    if re.match(r"^blk\.\d+\.hc_(attn|ffn)_inject\.weight$", t.name):
        continue
    plan.append((t.name, [(t.data_offset, t.n_bytes)], [int(x) for x in t.shape], t.tensor_type, t.n_bytes))

for name, pieces, shape, ttype, nbytes in plan:
    # tensor_shape in numpy (reversed) order, as the writer expects; dtype irrelevant with raw_dtype set (not uint8: no byte-shape conversion)
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
data_start = pos
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
logging.info(f"done: {written} tensor bytes, file {os.path.getsize(dst)} bytes, data at {data_start}")
