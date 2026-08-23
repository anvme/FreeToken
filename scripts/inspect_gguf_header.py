"""Dump GGUF header (KV metadata + tensor table) from a possibly-partial .gguf file.

Uses gguf.GGUFReader with _build_tensors patched to skip tensor *data* reads, so only
the header + KV section + tensor-info table (all at file start) must be present.
"""
from __future__ import annotations

import argparse
from collections import Counter

import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.gguf_reader import GGUFReader, ReaderTensor


def _build_tensors_no_data(self, start_offs, fields):
    tensors = []
    for field in fields:
        _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
        name = str(bytes(name_data), encoding="utf-8")
        ggml_type = GGMLQuantizationType(raw_dtype[0])
        n_elems = 1
        for d in dims.tolist():
            n_elems *= int(d)
        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        n_bytes = n_elems * type_size // block_size
        tensors.append(
            ReaderTensor(
                name=name,
                tensor_type=ggml_type,
                shape=dims,
                n_elements=n_elems,
                n_bytes=n_bytes,
                data_offset=int(start_offs + offset_tensor[0]),
                data=None,
                field=field,
            )
        )
    self.tensors = tensors


GGUFReader._build_tensors = _build_tensors_no_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--all-tensors", action="store_true", help="print every tensor row")
    args = ap.parse_args()

    reader = GGUFReader(args.path)
    print(f"=== GGUF header: {len(reader.tensors)} tensors ===")

    arch = None
    for name, field in reader.fields.items():
        if name == "general.architecture":
            arch = field.contents()
    print(f"general.architecture = {arch!r}")

    print("=== metadata (arch-prefixed + general/ggml) ===")
    for name, field in reader.fields.items():
        if name.startswith(("tokenizer.", "training.", "file.")):
            continue
        if arch and not (name.startswith(arch + ".") or name.startswith(("general.", "ggml.", "context.", "alignment"))):
            continue
        val = field.contents()
        if isinstance(val, str):
            if len(val) > 200:
                val = val[:200] + f"... ({len(val)} chars)"
        elif isinstance(val, list) and len(val) > 20:
            val = f"[{val[0]!r}, {val[1]!r}, ...] (len={len(val)})"
        print(f"  {name} = {val!r}")

    rows = []
    for t in reader.tensors:
        sh = tuple(reversed([int(d) for d in t.shape]))
        rows.append((t.name, sh, t.tensor_type, t.n_bytes, t.data_offset))
    rows.sort(key=lambda r: (r[4]))

    by_type = Counter(r[2] for r in rows)
    print("=== tensor type histogram (arch-specific) ===")
    for tt, cnt in sorted(by_type.items(), key=lambda kv: int(kv[0])):
        total = sum(r[3] for r in rows if r[2] == tt)
        print(f"  {str(tt.name):8s} ({int(tt)}): {cnt} tensors, {total/2**30:.3f} GiB")

    print("=== tensor table (offset-ordered) ===")
    expert = 0
    for name, sh, tt, nb, off in rows:
        if "ffn_exps" in name or ".ffn.exps." in name:
            expert += 1
            continue
        if expert and args.all_tensors:
            print(f"    [{expert} expert tensors so far]")
            expert = 0
        # ggml fastest-dims: token_embd is [vocab, hidden] in ggml -> torch [hidden, vocab]
        print(f"  {name:64s} {str(sh):28s} {str(tt.name):6s} {nb/1e6:12.3f} Mbytes")
    if expert:
        print(f"  ... {expert} further expert tensors")


if __name__ == "__main__":
    main()
