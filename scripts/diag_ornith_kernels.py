"""Numerical self-test of the native-GGUF Q8_0 GPU path against a numpy reference.

The qwen35moe adapter keeps every projection in its packed ggml layout and relies on
the borrowed llama.cpp kernels to dequantize *inside* the GEMM. Before this file, the
repo only ever ran Q4_0 dense projections + a Q6_K embedding (gemma4), so the Q8_0
MMVQ / MMQ / MoE-MMVQ paths and the packed-row *fusion* (qkv, GDN in_proj, shared
gate_up) are exercised for the first time by Ornith. This script checks each of them
in isolation, so a garbage-output bug can be attributed (or not) to the kernels
without loading 35 GB or running a forward pass.

    .venv/bin/python scripts/diag_ornith_kernels.py /path/to/Ornith-1.5-35B-Q8_0.gguf

Checks, each PASS/FAIL with the observed error:

  1. dequant      ggml_dequantize(packed) vs an independent numpy Q8_0 dequant.
                  Exact equality is expected (same arithmetic, d*q in fp32).
  2. gemv/gemm    fused_mul_mat_gguf(x, W) vs x @ dequant(W).T for 1 row (MMVQ path,
                  what decode uses) and 32 rows (MMQ path, what prefill uses).
                  The kernels quantize the activations to q8_1, so a small relative
                  error is expected -- ~1e-3, not ~1e-1.
  3. fusion       the loader concatenates packed rows of several tensors along dim 0
                  and calls the result one weight. Verify the fused GEMM's output
                  slices equal the per-part GEMMs.
  4. moe          fused_experts_gguf over real blk.0 expert banks (first few experts)
                  vs a dense reference. Covers ggml_moe_a8_vec + silu_and_mul + the
                  gate/up bank halves.

Only a few MB of the file are touched (plus the expert slice for check 4).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

# --- reference Q8_0 dequant (independent of the repo's implementation) -----------
# Q8_0 block = 2-byte fp16 scale d + 32 int8 codes; w = d * q.


def deq_q8_0_np(raw: np.ndarray) -> np.ndarray:
    """``raw`` uint8 [..., n*34] -> float32 [..., n*32]."""
    flat = np.ascontiguousarray(raw).reshape(-1, 34)
    d = flat[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1, 1)
    q = flat[:, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(*raw.shape[:-1], -1)


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float(), ref.float()
    denom = ref.abs().max().clamp_min(1e-9)
    return float((got - ref).abs().max() / denom)


_FAILED: list[str] = []


def report(name: str, err: float, tol: float, extra: str = "") -> None:
    ok = err <= tol and not (err != err)  # NaN-safe
    tag = "PASS" if ok else "FAIL"
    if not ok:
        _FAILED.append(name)
    print(f"  [{tag}] {name:<44s} max_rel_err={err:.3e} (tol {tol:.0e}) {extra}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", help="path to the Q8_0 GGUF")
    ap.add_argument("--experts", type=int, default=8, help="experts to test in check 4")
    ap.add_argument("--layer", type=int, default=0, help="GDN layer for the expert/in_proj checks")
    ap.add_argument("--full-layer", type=int, default=3, help="full-attention layer for qkv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required (these are the borrowed ggml CUDA kernels)")
        return 2

    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q8_0, row_bytes
    from freetoken.models.gguf.reader import iter_gguf_tensors

    dev = torch.device("cuda")
    L, F = args.layer, args.full_layer
    want = {
        f"blk.{F}.attn_q.weight",
        f"blk.{F}.attn_k.weight",
        f"blk.{F}.attn_v.weight",
        f"blk.{L}.attn_qkv.weight",
        f"blk.{L}.attn_gate.weight",
        f"blk.{L}.ssm_beta.weight",
        f"blk.{L}.ssm_alpha.weight",
        f"blk.{L}.ffn_gate_shexp.weight",
        f"blk.{L}.ffn_up_shexp.weight",
        f"blk.{L}.ffn_gate_exps.weight",
        f"blk.{L}.ffn_up_exps.weight",
        f"blk.{L}.ffn_down_exps.weight",
    }
    print(f"reading tensor table of {args.gguf} ...")
    T = {t.name: t for t in iter_gguf_tensors(args.gguf) if t.name in want}
    missing = want - set(T)
    if missing:
        print(f"missing tensors (wrong file or layer flags?): {sorted(missing)}")
        return 2
    for n, t in sorted(T.items()):
        if t.ggml_type != GGML_Q8_0:
            print(f"{n} is ggml type {t.ggml_type}, not Q8_0 -- this script targets the Q8_0 file")
            return 2

    def packed(name: str) -> torch.Tensor:
        return T[name].packed().to(dev)

    def ref_dense(name: str) -> torch.Tensor:
        """numpy-reference dequant of a 2-D tensor -> [out, in] float32 on GPU."""
        t = T[name]
        w = deq_q8_0_np(t._raw)  # [rows, in]
        return torch.from_numpy(w).to(dev).reshape(t.shape)

    # ---------------------------------------------------------------- 1. dequant
    print("\n1. ggml_dequantize vs numpy reference")
    from freetoken.kernel.gguf import ggml_dequantize

    for name in (f"blk.{F}.attn_q.weight", f"blk.{L}.attn_qkv.weight"):
        t = T[name]
        out, inn = t.shape
        got = ggml_dequantize(packed(name), GGML_Q8_0, out, inn, torch.float32)
        report(f"dequant {name}", rel_err(got, ref_dense(name)), 0.0, f"shape={tuple(t.shape)}")

    # ------------------------------------------------------------- 2. gemv / gemm
    print("\n2. fused_mul_mat_gguf vs dense reference (MMVQ: 1 row, MMQ: 32 rows)")
    for name in (
        f"blk.{F}.attn_q.weight",
        f"blk.{F}.attn_k.weight",
        f"blk.{L}.attn_qkv.weight",
        f"blk.{L}.ssm_alpha.weight",
        f"blk.{L}.ffn_gate_shexp.weight",
    ):
        t = T[name]
        out, inn = t.shape
        w_ref = ref_dense(name)
        qw = packed(name)
        for rows, path in ((1, "MMVQ/decode"), (32, "MMQ/prefill")):
            x = torch.randn(rows, inn, device=dev, dtype=torch.bfloat16)
            got = fused_mul_mat_gguf(x, qw, GGML_Q8_0)
            ref = x.float() @ w_ref.T
            report(f"{path:<12s} {name}", rel_err(got, ref), 5e-2, f"out={out}")

    # -------------------------------------------------------------- 3. fusion
    # The loader emits one qweight per fused group by concatenating each part's packed
    # rows along dim 0 (legal only because every part shares the input dim, hence the
    # same row_bytes). Verify the fused GEMM's row slices match the per-part GEMMs.
    print("\n3. packed-row fusion (concat along dim 0) reproduces the per-part GEMMs")
    groups = {
        "self_attn.qkv_proj": [
            f"blk.{F}.attn_q.weight", f"blk.{F}.attn_k.weight", f"blk.{F}.attn_v.weight",
        ],
        "linear_attn.in_proj": [
            f"blk.{L}.attn_qkv.weight", f"blk.{L}.attn_gate.weight",
            f"blk.{L}.ssm_beta.weight", f"blk.{L}.ssm_alpha.weight",
        ],
        "shared_expert.gate_up_proj": [
            f"blk.{L}.ffn_gate_shexp.weight", f"blk.{L}.ffn_up_shexp.weight",
        ],
    }
    for gname, parts in groups.items():
        inn = T[parts[0]].shape[1]
        assert all(T[p].shape[1] == inn for p in parts), f"{gname}: parts differ in input dim"
        rb = row_bytes(inn, GGML_Q8_0)
        assert all(T[p].row_bytes == rb for p in parts), f"{gname}: row_bytes mismatch"
        fused = torch.cat([packed(p) for p in parts], dim=0)
        splits = [T[p].shape[0] for p in parts]
        for rows in (1, 32):
            x = torch.randn(rows, inn, device=dev, dtype=torch.bfloat16)
            got = fused_mul_mat_gguf(x, fused, GGML_Q8_0)
            ref = torch.cat(
                [fused_mul_mat_gguf(x, packed(p), GGML_Q8_0) for p in parts], dim=-1
            )
            report(f"fuse[{rows:>2}] {gname}", rel_err(got, ref), 1e-6, f"splits={splits}")

    # ------------------------------------------------------------------- 4. MoE
    print(f"\n4. fused_experts_gguf over real blk.{L} expert banks (first {args.experts} experts)")
    from freetoken.moe.fused_gguf_q import fused_experts_gguf

    E = args.experts
    gate_t, up_t, down_t = (
        T[f"blk.{L}.ffn_gate_exps.weight"],
        T[f"blk.{L}.ffn_up_exps.weight"],
        T[f"blk.{L}.ffn_down_exps.weight"],
    )
    n_exp, I, H = gate_t.shape
    assert down_t.shape == (n_exp, H, I), f"down_exps shape {down_t.shape} != {(n_exp, H, I)}"
    rb_h, rb_i = row_bytes(H, GGML_Q8_0), row_bytes(I, GGML_Q8_0)

    # Bank layout the adapter builds: gate rows in [:, :I], up rows in [:, I:].
    gate_p = gate_t.packed().reshape(n_exp, I, rb_h)[:E].to(dev)
    up_p = up_t.packed().reshape(n_exp, I, rb_h)[:E].to(dev)
    gate_up_q = torch.cat([gate_p, up_p], dim=1).contiguous()  # [E, 2I, rb_h]
    down_q = down_t.packed().reshape(n_exp, H, rb_i)[:E].to(dev).contiguous()

    g_ref = torch.from_numpy(deq_q8_0_np(gate_t._raw)).to(dev).reshape(n_exp, I, H)[:E]
    u_ref = torch.from_numpy(deq_q8_0_np(up_t._raw)).to(dev).reshape(n_exp, I, H)[:E]
    d_ref = torch.from_numpy(deq_q8_0_np(down_t._raw)).to(dev).reshape(n_exp, H, I)[:E]

    torch.manual_seed(0)
    for tokens, top_k in ((1, 8), (16, 8)):
        x = torch.randn(tokens, H, device=dev, dtype=torch.bfloat16)
        ids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(tokens)])
        ids = ids.to(torch.int32)
        w = torch.softmax(torch.randn(tokens, top_k, device=dev), dim=-1)
        got = fused_experts_gguf(x, gate_up_q, down_q, w, ids, "silu", "q8_0")

        ref = torch.zeros(tokens, H, device=dev, dtype=torch.float32)
        xf = x.float()
        for t_i in range(tokens):
            for k in range(top_k):
                e = int(ids[t_i, k])
                gate = xf[t_i] @ g_ref[e].T
                up = xf[t_i] @ u_ref[e].T
                ref[t_i] += float(w[t_i, k]) * ((torch.nn.functional.silu(gate) * up) @ d_ref[e].T)
        report(f"moe tokens={tokens:<3d} top_k={top_k}", rel_err(got, ref), 5e-2, f"E={E} I={I} H={H}")

    print()
    if _FAILED:
        print(f"FAILED {len(_FAILED)} check(s): {_FAILED}")
        return 1
    print("all Q8_0 kernel checks passed -- the packed GEMM/MoE path is numerically sound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
