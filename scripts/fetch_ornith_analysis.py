"""Sparse-fetch the diagnostic tensors of the HF-hosted Ornith Q8_0 GGUF (HTTP
byte ranges -- ~30 MB total, no full download) and run the Part A convention
analysis locally.

    .venv/bin/python scripts/fetch_ornith_analysis.py

Outputs:
  ornith_slices/*.bin   raw tensor bytes (kept for re-verification)
  console               norm/scalar stats + PASS/SUSPECT verdicts + Q8_0
                        dequant sanity (independent numpy implementation)
"""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np
import gguf
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
from gguf.gguf_reader import GGUFReader

URL = "https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF/resolve/main/Ornith-1.5-35B-Q8_0.gguf"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "ornith_slices"
HEADER_BYTES = 4_000_000

# (name, kind) -- kind: "f32" | "q8" (full tensor) | ("qs", n_rows) (first n rows)
WANT: list[tuple[str, object]] = [
    ("output_norm.weight", "f32"),
    ("blk.0.attn_norm.weight", "f32"),
    ("blk.40.attn_norm.weight", "f32"),
    ("blk.0.post_attention_norm.weight", "f32"),
    ("blk.40.post_attention_norm.weight", "f32"),
    ("blk.3.attn_q_norm.weight", "f32"),   # full-attn layers: 3,7,11,...,39
    ("blk.3.attn_k_norm.weight", "f32"),
    ("blk.15.attn_q_norm.weight", "f32"),
    ("blk.15.attn_k_norm.weight", "f32"),
    ("blk.39.attn_q_norm.weight", "f32"),
    ("blk.0.ssm_norm.weight", "f32"),
    ("blk.14.ssm_norm.weight", "f32"),   # blk.14 is GDN (full-attn = 3,7,11,15,...)
    ("blk.0.ssm_a", "f32"),
    ("blk.14.ssm_a", "f32"),
    ("blk.0.ssm_dt.bias", "f32"),
    ("blk.14.ssm_dt.bias", "f32"),
    ("blk.0.ssm_conv1d.weight", "f32"),
    ("blk.3.ffn_gate_inp.weight", "f32"),          # router (256, 2048) F32
    ("blk.3.ffn_gate_inp_shexp.weight", "f32"),    # shared-expert gate (2048,)
    ("blk.0.ssm_alpha.weight", "q8"),              # (32, 2048) small Q8_0
    ("blk.3.attn_q.weight", "q8"),                 # (8192, 2048)
    ("blk.0.ffn_gate_exps.weight", ("qs", 1024)),  # experts 0..1 gate
    ("blk.0.ffn_up_exps.weight", ("qs", 1024)),    # experts 0..1 up
    ("blk.0.ffn_down_exps.weight", ("qs", 4096)),  # experts 0..1 down
    ("token_embd.weight", ("qs", 4)),              # 4 embedding rows
]


# ---- header-only GGUFReader (tensor table without data reads) -------------
def _build_tensors_no_data(self, start_offs, fields):
    from gguf.gguf_reader import ReaderTensor

    tensors = []
    for field in fields:
        _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
        name = str(bytes(name_data), encoding="utf-8")
        ggml_type = GGMLQuantizationType(raw_dtype[0])
        n_elems = 1
        for d in dims.tolist():
            n_elems *= int(d)
        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        tensors.append(
            ReaderTensor(
                name=name,
                tensor_type=ggml_type,
                shape=dims,
                n_elements=n_elems,
                n_bytes=n_elems * type_size // block_size,
                data_offset=int(start_offs + offset_tensor[0]),
                data=None,
                field=field,
            )
        )
    self.tensors = tensors


GGUFReader._build_tensors = _build_tensors_no_data


def fetch_range(start: int, end_incl: int) -> bytes:
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end_incl}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 206, f"range request not honoured: {r.status}"
        return r.read()


def dequant_q8_0_np(raw: bytes, out_shape) -> np.ndarray:
    b = np.frombuffer(raw, np.uint8)
    nblk = len(b) // 34
    blk = b[: nblk * 34].reshape(nblk, 34)
    d = np.frombuffer(blk[:, :2].tobytes(), dtype="<u2").astype(np.float16).astype(np.float32)
    q = blk[:, 2:34].astype(np.int8)
    return (d[:, None] * q).reshape(out_shape)


def stats(a: np.ndarray) -> str:
    a = a.astype(np.float64)
    return (f"mean={a.mean():+.5f} std={a.std():.5f} min={a.min():+.5f} "
            f"max={a.max():+.5f} first4={np.round(a.ravel()[:4], 5)}")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    hdr_path = OUT / "header.part"

    def build_reader() -> GGUFReader:
        return GGUFReader(str(hdr_path))

    size = HEADER_BYTES
    if not hdr_path.exists() or hdr_path.stat().st_size < size:
        print(f"fetching header ({size // 1024} KB) ...")
        hdr_path.write_bytes(fetch_range(0, size - 1))
    while True:
        try:
            reader = build_reader()
            break
        except (IndexError, Exception) as e:  # truncated header: grow it
            size *= 2
            if size > 256_000_000:
                raise RuntimeError(f"header larger than 256 MB? ({e})")
            print(f"  header truncated, refetching {size // 1024} KB ...")
            hdr_path.write_bytes(fetch_range(0, size - 1))
    by_name = {t.name: t for t in reader.tensors}
    print(f"tensor table: {len(by_name)} tensors")

    slices: dict[str, bytes] = {}
    for name, kind in WANT:
        t = by_name.get(name)
        if t is None:
            print(f"  MISSING {name}")
            continue
        nb = t.n_bytes
        rows = 0
        if isinstance(kind, tuple) and kind[0] == "qs":
            rows = kind[1]
            block, type_size = GGML_QUANT_SIZES[t.tensor_type]
            n_fast = int(t.shape[0])
            row_bytes = n_fast // block * type_size
            nb = rows * row_bytes
        bin_path = OUT / f"{name.replace('.', '_')}.bin"
        if bin_path.exists() and bin_path.stat().st_size == nb:
            print(f"  have  {name:34s} [{nb / 1e6:8.3f} MB]")
            slices[name] = bin_path.read_bytes()
            continue
        print(f"  fetch {name:34s} [{nb / 1e6:8.3f} MB] @ {t.data_offset}")
        slices[name] = fetch_range(t.data_offset, t.data_offset + nb - 1)
        bin_path.write_bytes(slices[name])

    print()
    print("=" * 76)
    print("PART A: convention analysis (local)")
    print("=" * 76)
    problems = 0

    def verdict(norm: str, v: np.ndarray, expect: str) -> None:
        nonlocal problems
        mean, std = float(v.mean()), float(v.std())
        ok = (0.5 < mean < 1.6 and std < 1.0) if expect in ("plus_one", "raw_1") else True
        if expect == "neg_all":
            ok = bool((v < 0).all())
        if not ok:
            problems += 1
        print(f"  {norm:34s} [{expect}] {stats(v)}{'   <<<<< SUSPECT' if not ok else ''}")

    for name in ["output_norm.weight", "blk.0.attn_norm.weight", "blk.40.attn_norm.weight",
                 "blk.0.post_attention_norm.weight", "blk.40.post_attention_norm.weight"]:
        v = np.frombuffer(slices[name], "<f4")
        verdict(name, v, "plus_one")
    for name in ["blk.3.attn_q_norm.weight", "blk.15.attn_q_norm.weight", "blk.39.attn_q_norm.weight",
                 "blk.3.attn_k_norm.weight", "blk.15.attn_k_norm.weight"]:
        v = np.frombuffer(slices[name], "<f4")
        verdict(name, v, "plus_one")
    for name in ["blk.0.ssm_norm.weight", "blk.14.ssm_norm.weight"]:
        if name in slices:
            v = np.frombuffer(slices[name], "<f4")
            verdict(name, v, "raw_1")

    for name in ["blk.0.ssm_a", "blk.14.ssm_a"]:
        if name not in slices:
            continue
        v = np.frombuffer(slices[name], "<f4")
        ok = bool((v < 0).all())
        if not ok:
            problems += 1
        rec = np.log(-v)
        print(f"  {name:34s} [all<0] {stats(v)}{'   <<<<< SUSPECT' if not ok else ''}")
        a_vals = -v  # a = -exp(A_log) => A = -a
        print(f"  {'':34s} recovered A_log=log(-a): mean={rec.mean():+.3f} std={rec.std():.3f} "
              f"(=> A in [{a_vals.min():.3f}, {a_vals.max():.3f}])")

    for name in ["blk.0.ssm_dt.bias", "blk.14.ssm_dt.bias", "blk.3.ffn_gate_inp_shexp.weight"]:
        if name not in slices:
            continue
        v = np.frombuffer(slices[name], "<f4")
        print(f"  {name:34s} [info]  {stats(v)}")
    v = np.frombuffer(slices["blk.3.ffn_gate_inp.weight"], "<f4").reshape(256, 2048)
    print(f"  {'blk.3.ffn_gate_inp.weight':34s} [info]  {stats(v)} (256x2048 router)")
    v = np.frombuffer(slices["blk.0.ssm_conv1d.weight"], "<f4").reshape(-1, 4)
    print(f"  {'blk.0.ssm_conv1d.weight':34s} [info]  per-channel stats below")
    print(f"  {'':34s} ch0={np.round(v[0], 4)} ch1={np.round(v[1], 4)} "
          f"ch8191={np.round(v[-1], 4)} all-finite={np.isfinite(v).all()}")

    print()
    print("-" * 76)
    print("Q8_0 payloads (independent numpy dequant sanity)")
    print("-" * 76)

    def q8_report(name: str, shape) -> None:
        w = dequant_q8_0_np(slices[name], shape)
        zero_blocks = 0
        if "exps" in name:
            # count all-zero 32-elem blocks (a scale==0 block is degenerate)
            b = np.frombuffer(slices[name], np.uint8).reshape(-1, 34)
            zero_blocks = int((b[:, :2] == 0).any(axis=1).sum()) if False else \
                int(((b[:, :2] == 0).sum(axis=1) == 2).sum())
        print(f"  {name:34s} {stats(w)}")
        if zero_blocks:
            print(f"  {'':34s} all-zero scale blocks: {zero_blocks} <<<<< check")

    q8_report("blk.3.attn_q.weight", (8192, 2048))
    q8_report("blk.0.ssm_alpha.weight", (32, 2048))
    q8_report("blk.0.ffn_gate_exps.weight", (1024, 2048))
    q8_report("blk.0.ffn_up_exps.weight", (1024, 2048))
    q8_report("blk.0.ffn_down_exps.weight", (4096, 512))
    q8_report("token_embd.weight", (4, 2048))

    # Cross-check: dequant the same expert slice two ways (row order) and verify
    # expert 0 gate row 0 vs expert 1 gate row 0 differ (no duplicated rows).
    g = dequant_q8_0_np(slices["blk.0.ffn_gate_exps.weight"], (1024, 2048))
    same = float(np.abs(g[0] - g[512]).max())
    print(f"  expert0-gate-row0 vs expert1-gate-row0 max|Δ| = {same:.5f} "
          f"({'OK distinct' if same > 1e-4 else 'SUSPECT duplicated!'})")

    print()
    print(f"{'PROBLEMS FOUND: ' + str(problems) if problems else 'ALL CONVENTION CHECKS PASS'}")


if __name__ == "__main__":
    main()
