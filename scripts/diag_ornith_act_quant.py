"""Measure the ggml activation quantization error on REAL activations, not Gaussians.

Every ggml matmul in this engine quantizes the *activations* to q8_1 before multiplying:
``ggml_mul_mat_vec_a8`` / ``ggml_mul_mat_a8`` for the dense projections, and
``ggml_moe_a8_vec`` for the routed experts. q8_1 packs 32 values per block behind one
shared fp16 scale, so a single large element in a block costs every other element in that
block its precision. Transformer hidden states are famous for exactly that shape -- a few
"massive activations" orders of magnitude above the rest.

diag_ornith_kernels.py measured this path with ``torch.randn`` inputs and got ~5e-3, which
looked fine. Gaussian inputs have no outliers, so that number says nothing about the
inputs the model actually produces. This script reruns the same comparison on activations
dumped from a real forward (FREETOKEN_DUMP_LAYER=all) and prints, side by side:

  * the activation's outlier profile (max/rms per row, kurtosis) vs a Gaussian of equal RMS
  * fused_mul_mat_gguf error on the real activation
  * fused_mul_mat_gguf error on the Gaussian control

If the real-activation error is far worse than the control, activation quantization is
where the model is losing its signal -- which would explain a per-layer MoE cosine of
0.998 (10x worse than bf16 alone) compounding into garbage over 40 layers.

    FREETOKEN_DUMP_LAYER=all ft shell --model MODEL.gguf     # type 'hi', then /exit
    .venv/bin/python scripts/diag_ornith_act_quant.py MODEL.gguf --dump ornith_layer_dump.pt
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def stats(x: torch.Tensor) -> str:
    """Outlier profile of an activation matrix [tokens, features]."""
    xf = x.float()
    rms = xf.pow(2).mean(-1, keepdim=True).sqrt()
    peak = (xf.abs().max(-1).values / rms.squeeze(-1)).max()
    kurt = ((xf - xf.mean()) ** 4).mean() / ((xf - xf.mean()) ** 2).mean() ** 2
    # worst single q8_1 block: how much dynamic range one block has to span
    blocks = xf.reshape(xf.shape[0], -1, 32)
    ratio = (blocks.abs().max(-1).values / blocks.abs().mean(-1).clamp_min(1e-9)).max()
    return f"max/rms={float(peak):7.2f}  kurtosis={float(kurt):9.2f}  worst-block max/mean={float(ratio):7.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--dump", default="ornith_layer_dump.pt")
    ap.add_argument("--layers", default="0,3,12,24,39")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required (these are the borrowed ggml kernels)")
        return 2

    from freetoken.layers.gguf import fused_mul_mat_gguf
    from freetoken.models.gguf.dequant import GGML_Q8_0, dequantize
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config

    cfg = parse_gguf_config(cached_load_hf_config(args.gguf))
    eps = cfg.rms_norm_eps
    dev = torch.device("cuda")
    d = torch.load(args.dump, map_location="cpu")
    if "L0.residual" not in d:
        print("dump has no per-layer residuals; re-run with FREETOKEN_DUMP_LAYER=all")
        return 2

    layers = [int(s) for s in args.layers.split(",")]
    need = set()
    for L in layers:
        need |= {f"blk.{L}.attn_norm.weight",
                 f"blk.{L}.{'attn_qkv' if cfg.is_linear_layer(L) else 'attn_q'}.weight"}
    T_ = {t.name: t for t in iter_gguf_tensors(args.gguf) if t.name in need}

    def deq(t):
        return dequantize(torch.from_numpy(np.ascontiguousarray(t._raw)).reshape(-1),
                          t.ggml_type, torch.float32).reshape(t.shape)

    print("Each row: the same GEMM on a real activation vs a Gaussian of identical RMS.\n"
          "A large gap means q8_1 activation blocks are being wrecked by outliers.\n")
    for L in layers:
        prev = d.get(f"L{L - 1}.residual") if L > 0 else d.get("embed")
        if prev is None:
            print(f"  layer {L}: no dumped input, skipping")
            continue
        w_norm = deq(T_[f"blk.{L}.attn_norm.weight"])
        x = prev.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w_norm  # input_layernorm
        x = x.to(dev).to(torch.bfloat16)

        name = f"blk.{L}.{'attn_qkv' if cfg.is_linear_layer(L) else 'attn_q'}.weight"
        wt = T_[name]
        assert wt.ggml_type == GGML_Q8_0, f"{name} is not Q8_0"
        qw = wt.packed().to(dev)
        w_ref = deq(wt).to(dev)

        ctrl = torch.randn_like(x.float()) * x.float().pow(2).mean().sqrt()
        ctrl = ctrl.to(torch.bfloat16)

        def err(a):
            got = fused_mul_mat_gguf(a, qw, GGML_Q8_0).float()
            ref = a.float() @ w_ref.T
            return float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-9))

        e_real, e_ctrl = err(x), err(ctrl)
        print(f"  layer {L:>2} {name.split('.')[-2]}")
        print(f"     real      {stats(x.cpu())}   ggml_rel_err={e_real:.3e}")
        print(f"     gaussian  {stats(ctrl.cpu())}   ggml_rel_err={e_ctrl:.3e}"
              f"   -> real is {e_real / max(e_ctrl, 1e-12):.1f}x worse")
    return 0


if __name__ == "__main__":
    sys.exit(main())
