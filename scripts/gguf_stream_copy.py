"""gguf_stream_copy.py: write a GGUF from a plan of tensors whose data comes from byte ranges of a source GGUF or from
new bytes, with every metadata field of the source copied verbatim. File ranges are copied inside the kernel
(copy_file_range) and both files are flushed and dropped from the page cache every 2 GiB and at the end, so the process
stays small and MemFree high (the rules after freeze #10 in CLAUDE.md). Used by the qwen4exp transform scripts."""
import logging
import os

import numpy as np


class TensorPlan:
    """One output tensor: name, element shape in ggml order (ne0 first), ggml type, and its data as a list of pieces,
    each ("file", offset, n_bytes) from the source or ("bytes", data)."""
    def __init__(self, name, shape, ttype, pieces):
        self.name = name
        self.shape = [int(x) for x in shape]
        self.ttype = ttype
        self.pieces = pieces
        self.n_bytes = sum(p[2] if p[0] == "file" else len(p[1]) for p in pieces)


def verbatim(t):
    """The plan of a source tensor copied as is."""
    return TensorPlan(t.name, t.shape, t.tensor_type, [("file", t.data_offset, t.n_bytes)])


def write(gguf, reader, src, dst, plan, drop_every=2 << 30):
    assert not os.path.exists(dst), dst
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    align = int(reader.fields["general.alignment"].contents()) if "general.alignment" in reader.fields else gguf.GGUF_DEFAULT_ALIGNMENT
    writer = gguf.GGUFWriter(dst, arch, endianess=reader.endianess)
    writer.data_alignment = align

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)

    for p in plan:
        writer.add_tensor_info(p.name, list(reversed(p.shape)), np.dtype(np.float32), p.n_bytes, raw_dtype=p.ttype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    writer.fout[0].flush()

    out_fd = os.open(dst, os.O_WRONLY)
    in_fd = os.open(src, os.O_RDONLY)
    pos = os.fstat(out_fd).st_size
    pad = (-pos) % align
    if pad:
        os.pwrite(out_fd, bytes(pad), pos)
        pos += pad

    def drop():
        os.fsync(out_fd)
        os.posix_fadvise(out_fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.posix_fadvise(in_fd, 0, 0, os.POSIX_FADV_DONTNEED)

    written = 0
    since_drop = 0
    for p in plan:
        for piece in p.pieces:
            if piece[0] == "file":
                _, off, n = piece
                done = 0
                while done < n:
                    c = os.copy_file_range(in_fd, out_fd, n - done, off + done, pos)
                    if c <= 0:
                        raise RuntimeError(f"copy_file_range returned {c} at {p.name}")
                    done += c
                    pos += c
            else:
                data = piece[1]
                os.pwrite(out_fd, data, pos)
                pos += len(data)
        pad = (-p.n_bytes) % align
        if pad:
            os.pwrite(out_fd, bytes(pad), pos)
            pos += pad
        written += p.n_bytes
        since_drop += p.n_bytes
        if since_drop >= drop_every:
            drop()
            since_drop = 0
            logging.info(f"{written / 2**30:.1f} GiB written")
    drop()
    os.close(out_fd)
    os.close(in_fd)
    writer.fout[0].close()
    logging.info(f"done: {len(plan)} tensors, {written} tensor bytes, file {os.path.getsize(dst)} bytes")
