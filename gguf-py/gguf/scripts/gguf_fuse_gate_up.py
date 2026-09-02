#!/usr/bin/env python3
# Fuse the MoE gate/up expert tensors of an existing GGUF into gate_up tensors without requantizing.
#
# The fused tensor has the layout the converter writes with --fuse-gate-up-exps: for every expert the
# gate rows followed by the up rows, [n_embd, 2 * n_ff, n_expert]. The model then runs one mul_mat_id
# per layer instead of two. Quantized blocks are copied byte for byte; everything else is copied unchanged.
from __future__ import annotations

import logging
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf

logger = logging.getLogger("gguf-fuse-gate-up")

GATE_RE = re.compile(r'^blk\.(\d+)\.ffn_gate_exps\.weight$')
UP_RE   = re.compile(r'^blk\.(\d+)\.ffn_up_exps\.weight$')
FUSED   = 'blk.{}.ffn_gate_up_exps.weight'


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        # virtual fields and fields written by GGUFWriter itself
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def fuse(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> int:
    tensors = {t.name: t for t in reader.tensors}
    plan: list[tuple[gguf.ReaderTensor, gguf.ReaderTensor | None]] = []
    n_fused = 0
    total_bytes = 0

    for t in reader.tensors:
        if UP_RE.match(t.name):
            continue  # written together with its gate tensor
        m = GATE_RE.match(t.name)
        if m is None:
            if t.name.endswith('.ffn_gate_up_exps.weight'):
                raise ValueError(f'{t.name}: the model already has fused gate_up tensors')
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            plan.append((t, None))
            total_bytes += t.n_bytes
            continue

        up = tensors.get(f'blk.{m.group(1)}.ffn_up_exps.weight')
        if up is None:
            raise ValueError(f'{t.name}: no matching ffn_up_exps tensor')
        if up.tensor_type != t.tensor_type or list(up.shape) != list(t.shape) or len(t.shape) != 3:
            raise ValueError(f'{t.name}: gate {t.tensor_type.name} {list(t.shape)} and up {up.tensor_type.name} {list(up.shape)} do not match')

        # reader data is [n_expert, n_ff, row] (bytes for quantized types); the fusion doubles the row axis
        gate_shape = t.data.shape
        fused_shape = (gate_shape[0], 2 * gate_shape[1], gate_shape[2])
        writer.add_tensor_info(FUSED.format(m.group(1)), fused_shape, t.data.dtype, 2 * t.data.nbytes, t.tensor_type)
        plan.append((t, up))
        total_bytes += 2 * t.n_bytes
        n_fused += 1

    if n_fused == 0:
        raise ValueError('no ffn_gate_exps/ffn_up_exps pairs found')

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)
    for t, up in plan:
        if up is None:
            data = t.data
        else:
            # one layer of temporary memory at a time; the rest streams from the input mapping
            data = np.concatenate((t.data, up.data), axis=1)
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
        bar.update(data.nbytes)
        del data
    writer.close()
    return n_fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse ffn_gate_exps/ffn_up_exps into ffn_gate_up_exps tensors (no requantization)")
    parser.add_argument("input",  type=Path, help="GGUF file to read (single file, not a split)")
    parser.add_argument("output", type=Path, help="GGUF file to write")
    parser.add_argument("--dry-run", action="store_true", help="list the pairs that would be fused and exit without writing")
    parser.add_argument("--force", action="store_true", help="overwrite the output if it exists")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.output.exists() and not args.force and not args.dry_run:
        logger.error(f'{args.output} exists (use --force to overwrite)')
        sys.exit(1)
    if args.input.resolve() == args.output.resolve():
        logger.error('input and output must be different files')
        sys.exit(1)

    reader = gguf.GGUFReader(args.input, 'r')
    split = reader.fields.get(gguf.Keys.Split.LLM_KV_SPLIT_COUNT)
    if split is not None and split.contents() > 1:
        logger.error('split models are not supported: merge the shards first (gguf-split --merge)')
        sys.exit(1)

    if args.dry_run:
        tensors = {t.name: t for t in reader.tensors}
        pairs = [(t, tensors.get(f'blk.{GATE_RE.match(t.name).group(1)}.ffn_up_exps.weight')) for t in reader.tensors if GATE_RE.match(t.name)]
        for gate, up in pairs:
            state = 'ok' if up is not None and up.tensor_type == gate.tensor_type and list(up.shape) == list(gate.shape) else 'MISMATCH'
            logger.info(f'{gate.name}: {gate.tensor_type.name} {[int(x) for x in gate.shape]} + up -> {state}')
        logger.info(f'{len(pairs)} pairs, {sum(2 * g.n_bytes for g, _ in pairs) / 2**30:.1f} GiB of expert data')
        return

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    copy_metadata(reader, writer)
    n_fused = fuse(reader, writer)
    logger.info(f'fused {n_fused} gate/up expert pairs into {args.output}')


if __name__ == '__main__':
    main()
