"""qwen4exp-take-tensors.py <base.gguf> <donor.gguf> <out.gguf> <name-regex>: the base GGUF with every tensor whose name
matches the regex (a full match) replaced by the donor's tensor of the same name, with the donor's type, shape and bytes;
every other tensor and every metadata field come from the base verbatim. The donor may be shard 1 of a split
(<name>-00001-of-0000N.gguf): its shards are read beside it. A name the regex matches must exist in both files, and the
regex must match at least one. The data is copied inside the kernel (copy_file_range) and the files are dropped from the
page cache as the copy advances, so the process stays small and MemFree high."""
import sys, os, re, logging
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../gguf-py"))
import gguf

logging.basicConfig(level=logging.INFO)
base_path, donor_path, dst, pattern = sys.argv[1], sys.argv[2], sys.argv[3], re.compile(sys.argv[4])
assert not os.path.exists(dst), dst
reader = gguf.GGUFReader(base_path, "r")
arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
assert arch == "qwen4exp", arch
align = int(reader.fields["general.alignment"].contents()) if "general.alignment" in reader.fields else gguf.GGUF_DEFAULT_ALIGNMENT

donor_files = [donor_path]
m = re.match(r"^(.*)-00001-of-(\d{5})\.gguf$", donor_path)
if m:
    donor_files = [f"{m[1]}-{i:05d}-of-{m[2]}.gguf" for i in range(1, int(m[2]) + 1)]
donor = {}   # name -> (file index, tensor)
for i, path in enumerate(donor_files):
    for t in gguf.GGUFReader(path, "r").tensors:
        donor[t.name] = (i, t)

writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
writer.data_alignment = align
for field in reader.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

plan = []   # (name, file index (0 = base, 1 + i = donor shard i), src offset, element shape, type, bytes)
taken = 0
for t in reader.tensors:
    if pattern.fullmatch(t.name):
        assert t.name in donor, f"{t.name} is not in the donor"
        i, d = donor[t.name]
        plan.append((t.name, 1 + i, d.data_offset, [int(x) for x in d.shape], d.tensor_type, int(d.n_bytes)))
        logging.info(f"{t.name}: {t.tensor_type.name} {t.n_bytes} B -> {d.tensor_type.name} {d.n_bytes} B from the donor")
        taken += 1
        continue
    plan.append((t.name, 0, t.data_offset, [int(x) for x in t.shape], t.tensor_type, int(t.n_bytes)))
assert taken > 0, "the regex matches no tensor"

for name, fi, off, shape, ttype, nbytes in plan:
    writer.add_tensor_info(name, list(reversed(shape)), np.dtype(np.float32), nbytes, raw_dtype=ttype)
logging.info(f"{len(plan)} tensors, {taken} from the donor, alignment {align}")

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
writer.fout[0].flush()
out_fd = os.open(dst, os.O_WRONLY)
in_fds = [os.open(base_path, os.O_RDONLY)] + [os.open(p, os.O_RDONLY) for p in donor_files]
pos = os.fstat(out_fd).st_size
pad = (-pos) % align
if pad:
    os.pwrite(out_fd, bytes(pad), pos); pos += pad


def drop_cache():
    os.fsync(out_fd)
    for fd in [out_fd] + in_fds:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


written = 0; since_drop = 0
for name, fi, off, shape, ttype, nbytes in plan:
    done = 0
    while done < nbytes:
        c = os.copy_file_range(in_fds[fi], out_fd, nbytes - done, off + done, pos)
        if c <= 0:
            raise RuntimeError(f"copy_file_range returned {c} at {name}")
        done += c; pos += c
    pad = (-nbytes) % align
    if pad:
        os.pwrite(out_fd, bytes(pad), pos); pos += pad
    written += nbytes; since_drop += nbytes
    if since_drop >= (2 << 30):
        drop_cache()
        since_drop = 0
        logging.info(f"{written / 2**30:.1f} GiB written")
drop_cache()
for fd in [out_fd] + in_fds:
    os.close(fd)
writer.fout[0].close()
logging.info(f"done: {written} tensor bytes, file {os.path.getsize(dst)} bytes")
