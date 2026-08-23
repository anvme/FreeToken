"""Diagnose the Ornith Q8_0 garbage-output bug ON THE GPU SERVER.

    python scripts/diag_gguf_q8_0.py /home/stanislaw/models/ornith/.../Ornith-1.5-35B-Q8_0.gguf
    python scripts/diag_gguf_q8_0.py <path> --skip-kernels     # conventions only

Part A: convention dump. NOTE: Part A already PASSED on the actual file bytes
(scripts/fetch_ornith_analysis.py, sparse-fetched locally): norms are stored as
effective scales (1-centered to ~2.6, never 0-centered), ssm_a = -exp(A_log)
with A in [0.02, 70], Q8_0 payloads dequant to sane weights. Keep as a sanity
check if this ever needs rerunning.

Part B: kernel A/B -- the borrowed GGUF CUDA kernels vs the repo's pure-torch
reference dequant (dequant_q8_0). Q8_0 noise is ~1e-3..1e-2 relative; an
O(1)/NaN result means the flagged kernel instance is the bug.

If all checks pass: the bug is runtime-side. Bisect order:
  1) ft shell --model <gguf> --cuda-graph-max-bs 0   (eager; the GGUF kernel
     module was JIT-compiled DURING graph capture on the first run)
  2) set _MMVQ_SAFE = 0 in python/freetoken/layers/gguf.py  (decode linears on
     the MMQ GEMM path) and retest
  3) (2) fixes it -> the Q8_0 MMVQ GEMV instance is the bug.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

GGML_Q8_0 = 8


def _by_name(reader) -> dict:
    return {t.name: t for t in reader.tensors}


def _raw(reader, names: dict, name: str, max_rows: int | None = None) -> np.ndarray:
    """Packed bytes as a writable uint8 array, row-major [rows, row_bytes].

    ``reader.data`` is a uint8 memmap; a slice keeps it 1-D, so reshape here.
    (Do NOT rely on np.asarray(..., dtype=) to reinterpret a memmap slice.)"""
    t = names[name]
    nb = t.n_bytes
    if max_rows is not None:
        from gguf.constants import GGML_QUANT_SIZES

        block, type_size = GGML_QUANT_SIZES[t.tensor_type]
        n_fast = int(t.shape[0])
        nb = max_rows * (n_fast // block * type_size)
    buf = np.ascontiguousarray(
        np.asarray(reader.data[int(t.data_offset) : int(t.data_offset) + nb]).view(np.uint8)
    )
    ggml_dims = [int(d) for d in t.shape]  # fastest dim first
    rows = max_rows if max_rows is not None else max(1, int(np.prod(ggml_dims[1:])))
    row_bytes = nb // rows
    assert buf.shape == (rows * row_bytes,), (name, buf.shape, rows, row_bytes)
    return buf.reshape(rows, row_bytes)


def _stats(a: np.ndarray) -> str:
    a = a.astype(np.float64)
    return (f"mean={a.mean():+.5f} std={a.std():.5f} min={a.min():+.5f} "
            f"max={a.max():+.5f} first4={np.round(a.ravel()[:4], 5)}")


# --------------------------------------------------------------------------- #
# Part A: conventions
# --------------------------------------------------------------------------- #

def part_a(reader) -> int:
    print("=" * 76)
    print("PART A: convention dump")
    print("=" * 76)
    names = _by_name(reader)
    bad = 0

    def f32(name: str) -> np.ndarray | None:
        if name not in names:
            print(f"  {name:34s} MISSING")
            return None
        t = names[name]
        n = t.n_bytes // 4
        buf = np.asarray(reader.data[int(t.data_offset) : int(t.data_offset) + t.n_bytes],
                         np.uint8)
        v = np.frombuffer(buf, dtype="<f4")
        assert v.shape == (n,), (name, v.shape, n)
        return v

    # The runtime (GemmaRMSNorm) multiplies by the RAW stored value, so the
    # file must hold the EFFECTIVE scale. A 0-centered (~0 mean) distribution
    # would mean raw gamma was stored -> every norm collapses to ~0 -> dead model.
    def check(name: str, kind: str) -> None:
        nonlocal bad
        v = f32(name)
        if v is None:
            bad += 1
            return
        mean, std = float(v.mean()), float(v.std())
        if kind == "scale":
            ok = abs(mean) > 0.3 or std > 0.3  # not 0-centered gamma
        elif kind == "gdn_norm":
            ok = 0.3 < mean < 1.8
        elif kind == "ssm_a":
            ok = bool((v < 0).all())
        else:
            ok = True
        if not ok:
            bad += 1
        extra = ""
        if kind == "ssm_a" and (v < 0).all():
            al = np.log(-v)
            extra = f"  A=-a in [{v.min() * -1:.3f}, {v.max() * -1:.3f}], A_log mean={al.mean():+.2f}"
        print(f"  {name:34s} [{kind}] {_stats(v)}{'   <<<<< SUSPECT' if not ok else ''}{extra}")

    for n in ["output_norm.weight", "blk.0.attn_norm.weight", "blk.40.attn_norm.weight",
              "blk.0.post_attention_norm.weight", "blk.40.post_attention_norm.weight"]:
        check(n, "scale")
    for n in ["blk.3.attn_q_norm.weight", "blk.15.attn_q_norm.weight", "blk.39.attn_q_norm.weight",
              "blk.3.attn_k_norm.weight", "blk.15.attn_k_norm.weight"]:
        check(n, "scale")
    for n in ["blk.0.ssm_norm.weight", "blk.14.ssm_norm.weight"]:
        check(n, "gdn_norm")
    for n in ["blk.0.ssm_a", "blk.14.ssm_a"]:
        check(n, "ssm_a")
    for n in ["blk.0.ssm_dt.bias", "blk.14.ssm_dt.bias", "blk.3.ffn_gate_inp_shexp.weight"]:
        v = f32(n)
        if v is not None:
            print(f"  {n:34s} [info] {_stats(v)}")
    v = f32("blk.0.ssm_conv1d.weight")
    if v is not None:
        w = v.reshape(-1, 4)
        print(f"  {'blk.0.ssm_conv1d.weight':34s} [info] finite={np.isfinite(w).all()} "
              f"ch0={np.round(w[0], 4)}")
    return bad


# --------------------------------------------------------------------------- #
# Part B: kernels
# --------------------------------------------------------------------------- #

def _relerr(y, yref) -> float:
    y = np.asarray(y, dtype=np.float64)
    yref = np.asarray(yref, dtype=np.float64)
    return float(np.max(np.abs(y - yref) / (np.abs(yref) + 1e-2)))


def q81_act_ref(x_bf16: "torch.Tensor", packed: "torch.Tensor") -> "torch.Tensor":
    """Exact reference for the ggml Q8_1(x)-Q8_0(w) kernel math.

    The kernel quantizes the 32-elem blocks of x to Q8_1 (fp16 row-block scale)
    and does per-block integer dots x d_w(fp16) x d_x(fp16), accumulated in
    fp32. Reproducing that (including the fp16 scale rounding) makes a healthy
    kernel agree to ~1e-4, while a real kernel/stride bug shows up as O(1).
    x: [R, K] bf16; packed: [M, M//32*34] uint8. Returns [R, M] fp32."""
    import torch

    M, rb = packed.shape
    nb = rb // 34
    b = packed.view(M, nb, 34).contiguous()
    d_w = b[:, :, :2].view(torch.float16).float()          # [M, nb]
    q_w = b[:, :, 2:34].to(torch.int8).float()             # [M, nb, 32]
    x = x_bf16.float()
    xb = x.reshape(-1, nb, 32)
    d_x = xb.abs().amax(dim=-1, keepdim=True) / 127.0
    q_x = torch.round(xb / d_x.clamp_min(1e-45)).to(torch.int8).float()
    d_x = d_x.squeeze(-1).to(torch.float16).float()       # [R, nb]
    dot = torch.einsum("rnb,mnb->rnm", q_x, q_w)          # [R, nb, M]
    return (dot * d_x.unsqueeze(2) * d_w.T.unsqueeze(0)).sum(1)


def part_b(model_path: str) -> int:
    import torch

    from freetoken.kernel.gguf import (
        ggml_dequantize,
        ggml_moe_a8_vec,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
    from freetoken.models.gguf.dequant import dequant_q8_0
    from freetoken.layers.activation import silu_and_mul

    print()
    print("=" * 76)
    print("PART B: kernel A/B vs exact Q8_1-quantized reference (CUDA required)")
    print("=" * 76)
    if not torch.cuda.is_available():
        print("  no CUDA -- skipped")
        return 0

    import gguf

    reader = gguf.GGUFReader(model_path)
    names = _by_name(reader)
    dev = "cuda"
    bad = 0
    E, I, H = 2, 512, 2048  # test experts / moe intermediate / hidden
    rb_h = H // 32 * 34  # 2192
    rb_i = I // 32 * 34  # 544

    def packed(name: str, max_rows: int | None = None) -> torch.Tensor:
        return torch.from_numpy(_raw(reader, names, name, max_rows)).to(dev).contiguous()

    def check(tag: str, y_kernel: torch.Tensor, y_ref: torch.Tensor) -> None:
        nonlocal bad
        err = _relerr(y_kernel.cpu().numpy(), y_ref.cpu().numpy())
        ok = err < 5e-3
        if not ok:
            bad += 1
        print(f"  {tag:34s} max rel err={err:.6f}  {'OK' if ok else 'FAIL <--'}")

    # ---- one Q8_0 2D linear: blk.3.attn_q.weight (8192, 2048) --------------
    t0 = names["blk.3.attn_q.weight"]
    m, n = int(t0.shape[1]), int(t0.shape[0])  # torch (out, in)
    w = packed("blk.3.attn_q.weight")
    assert w.shape == (m, n // 32 * 34), (w.shape, m, n)
    wref = dequant_q8_0(w.cpu(), torch.bfloat16).reshape(m, n).to(dev)

    d = ggml_dequantize(w, GGML_Q8_0, m, n, torch.bfloat16)
    err = float((d.float() - wref.float()).abs().max().item())
    ok = err <= 1e-3
    print(f"  ggml_dequantize      (Q8_0)  max|diff|={err:.6f}  {'OK' if ok else 'FAIL <--'}")
    bad += not ok

    x1 = torch.randn(1, n, dtype=torch.bfloat16, device=dev)
    x8 = torch.randn(8, n, dtype=torch.bfloat16, device=dev)

    check("ggml_mul_mat_vec_a8 (MMVQ)", ggml_mul_mat_vec_a8(w, x1, GGML_Q8_0, m).float(),
          q81_act_ref(x1, w))
    check("ggml_mul_mat_a8 (MMQ)", ggml_mul_mat_a8(w, x8, GGML_Q8_0, m).float(),
          q81_act_ref(x8, w))

    # ---- grouped expert GEMV: experts 0..1 of blk.0, same bank layout as
    #      the offload banks (gate_up [E, 2I, rb_h] = per expert [gate I; up I];
    #      down [E, H, rb_i]) ----
    g = packed("blk.0.ffn_gate_exps.weight", E * I)  # [2I, rb_h]
    u = packed("blk.0.ffn_up_exps.weight", E * I)
    dn = packed("blk.0.ffn_down_exps.weight", E * H)  # [2H, rb_i]
    assert g.shape == (E * I, rb_h) and dn.shape == (E * H, rb_i), (g.shape, dn.shape)
    gate_up_q = torch.stack(
        [torch.cat([g[e * I : (e + 1) * I], u[e * I : (e + 1) * I]], dim=0) for e in range(E)],
        dim=0,
    ).contiguous()  # [E, 2I, rb_h]
    down_q = dn.view(E, H, rb_i).contiguous()  # [E, H, rb_i]

    topk = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
    gu_out = ggml_moe_a8_vec(x1, gate_up_q, topk, 2, GGML_Q8_0, 2 * I, 1)  # [2, 2I]
    inter = silu_and_mul(gu_out)  # [2, I]
    moe_out = ggml_moe_a8_vec(inter, down_q, topk, 1, GGML_Q8_0, H, 2)  # [2, H]

    # reference, stage by stage (row r uses expert topk[r])
    gu_ref = torch.stack([q81_act_ref(x1, gate_up_q[topk[0, r]].reshape(2 * I, -1))[0]
                          for r in range(2)], dim=0)
    inter_ref = torch.nn.functional.silu(gu_ref[:, :I]) * gu_ref[:, I:]
    down_ref = torch.stack([q81_act_ref(inter_ref[r : r + 1].bfloat16(),
                                        down_q[topk[0, r]].reshape(H, -1))[0]
                            for r in range(2)], dim=0)
    check("ggml_moe_a8_vec gate_up stage", gu_out.float(), gu_ref)
    check("ggml_moe_a8_vec down stage", moe_out.float(), down_ref)

    # info: how much of the gap to the *bf16* math is plain Q8 activation noise
    y_bf16 = (x1.float() @ wref.T.float())
    print(f"  [info] bf16-math floor, attn_q MMVQ shape: "
          f"{_relerr(y_bf16.cpu().numpy(), q81_act_ref(x1, w).cpu().numpy()):.5f} "
          "(expected ~1e-2..5e-2; this is activation-quantization noise, not a kernel bug)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--skip-kernels", action="store_true")
    args = ap.parse_args()

    import gguf

    reader = gguf.GGUFReader(args.model)
    names = _by_name(reader)
    needed = {
        "output_norm.weight", "blk.0.attn_norm.weight", "blk.40.attn_norm.weight",
        "blk.0.post_attention_norm.weight", "blk.40.post_attention_norm.weight",
        "blk.3.attn_q_norm.weight", "blk.15.attn_q_norm.weight", "blk.39.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight", "blk.15.attn_k_norm.weight",
        "blk.0.ssm_norm.weight", "blk.14.ssm_norm.weight",
        "blk.0.ssm_a", "blk.14.ssm_a", "blk.0.ssm_dt.bias", "blk.14.ssm_dt.bias",
        "blk.0.ssm_conv1d.weight", "blk.3.ffn_gate_inp_shexp.weight",
        "blk.3.attn_q.weight",
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight",
    }
    missing = needed - set(names)
    if missing:
        print(f"WARNING: tensors not in file: {sorted(missing)}")

    bad_a = part_a(reader)
    bad_b = 0 if args.skip_kernels else part_b(args.model)

    print()
    print("=" * 76)
    if bad_a == 0 and bad_b == 0:
        print(
            "ALL CHECKS PASS -> bug is runtime-side. Next A/B on the server:\n"
            "  1) ft shell --model <gguf> --cuda-graph-max-bs 0   (eager; the GGUF\n"
            "     kernel module was JIT-compiled DURING graph capture on first run)\n"
            "  2) if still broken: set _MMVQ_SAFE = 0 in python/freetoken/layers/gguf.py\n"
            "     (decode linears onto the MMQ GEMM path) and retest\n"
            "  3) (2) fixes it -> the Q8_0 MMVQ GEMV kernel is the bug."
        )
    elif bad_b > 0:
        print("KERNEL MISMATCH FOUND -> the flagged Q8_0 kernel instance is the bug.")
    else:
        print("CONVENTION MISMATCH FOUND -> fix the mapping in models/qwen3_5_moe/gguf.py.")
    sys.exit(1 if (bad_a or bad_b) else 0)


if __name__ == "__main__":
    main()
