"""Recompute one decoder layer from the GGUF in fp32 and diff it against what the
engine actually produced.

The layer probe showed a numerically healthy trunk (smooth residual growth, no collapse
or NaN) with garbage output, and the GDN kernels match the HF reference exactly. That
combination means something computes the *wrong thing at the right magnitude* -- which
RMS cannot see. This does the only thing that can: rebuild the layer from the checkpoint
bytes with plain torch and compare tensor to tensor.

    # 1. capture what the engine computed for layer N on a real prefill
    FREETOKEN_DUMP_LAYER=0 ft shell --model /path/Ornith-1.5-35B-Q8_0.gguf
    > hi
    > /exit

    # 2. recompute the same layer from the GGUF and diff
    .venv/bin/python scripts/diag_ornith_layer_ref.py /path/Ornith-1.5-35B-Q8_0.gguf \
        --layer 0 --dump ornith_layer_dump.pt

Layer 0 is a GDN layer, so it covers the embedding, the input norm, the GDN mixer, the
post-attention norm, the router, the shared expert and the routed experts. Point --layer
at 3 (or any 4k+3) to cover gated full attention instead.

Reported per stage:

  norm_in     embedding + input_layernorm
  mixer_out   GDN (via the HF-derived Qwen3_5GatedDeltaNetReference) or gated attention
  mlp_in      residual add + post_attention_layernorm
  mlp_out     router topk-8 + shared expert + routed experts

The first stage that diverges by O(1) is the bug. Everything downstream of it will also
diverge, so read the FIRST failure, not the loudest.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F


def deq(raw: np.ndarray, ggml_type: int, shape) -> torch.Tensor:
    """Dequantize a GgufTensor's packed bytes to fp32 with the repo's own routine."""
    from freetoken.models.gguf.dequant import dequantize

    flat = dequantize(torch.from_numpy(raw).reshape(-1), ggml_type, torch.float32)
    return flat.reshape(shape)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """GemmaRMSNorm: scale by the raw stored weight (the +1 is baked into the GGUF)."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def report(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float) -> bool:
    got, ref = got.float(), ref.float()
    denom = ref.abs().max().clamp_min(1e-9)
    err = float((got - ref).abs().max() / denom)
    cos = float(
        F.cosine_similarity(got.reshape(1, -1), ref.reshape(1, -1)).clamp(-1, 1)
    )
    ok = err <= tol
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name:<10s} max_rel_err={err:9.3e}  cos={cos:+.6f}  "
        f"rms(engine)={float(got.pow(2).mean().sqrt()):.4f} rms(ref)={float(ref.pow(2).mean().sqrt()):.4f}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--dump", default="ornith_layer_dump.pt")
    ap.add_argument("--tol", type=float, default=5e-2)
    args = ap.parse_args()

    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config

    cfg = parse_gguf_config(cached_load_hf_config(args.gguf))
    L = args.layer
    is_linear = cfg.is_linear_layer(L)
    eps = cfg.rms_norm_eps
    H, I_sh = cfg.hidden_size, cfg.shared_expert_intermediate_size
    print(f"layer {L}: {'GDN (linear attention)' if is_linear else 'gated full attention'}")

    d = torch.load(args.dump, map_location="cpu")
    ids = d["input_ids"].long()
    T = ids.numel()
    print(f"dump: {T} tokens, ids={ids.tolist()}")

    want_prefix = f"blk.{L}."
    T_ = {}
    for t in iter_gguf_tensors(args.gguf):
        if t.name.startswith(want_prefix) or t.name == "token_embd.weight":
            T_[t.name] = t

    def w(suffix: str) -> torch.Tensor:
        t = T_[f"blk.{L}.{suffix}"]
        return deq(t._raw, t.ggml_type, t.shape)

    # ---- embedding (gather only the prompt's rows, then dequantize) -------------
    emb_t = T_["token_embd.weight"]
    rows = np.ascontiguousarray(emb_t._raw[ids.numpy()])
    x = deq(rows, emb_t.ggml_type, (T, H))
    ok_all = report("embed", d["embed"], x, args.tol)

    # ---- input_layernorm --------------------------------------------------------
    h = rms_norm(x, w("attn_norm.weight"), eps)
    ok_all &= report("norm_in", d["norm_in"], h, args.tol)

    # ---- mixer ------------------------------------------------------------------
    if is_linear:
        from freetoken.models.qwen3_5_moe.gdn_reference import Qwen3_5GatedDeltaNetReference

        g = cfg.linear_attention_group()
        ref = Qwen3_5GatedDeltaNetReference(
            hidden_size=H, num_k_heads=g.num_key_heads, num_v_heads=g.num_value_heads,
            head_k_dim=g.key_head_dim, head_v_dim=g.value_head_dim,
            conv_kernel_size=g.conv_kernel_dim, rms_norm_eps=eps,
        ).float()
        with torch.no_grad():
            ref.in_proj_qkv.weight.copy_(w("attn_qkv.weight"))
            ref.in_proj_z.weight.copy_(w("attn_gate.weight"))
            ref.in_proj_b.weight.copy_(w("ssm_beta.weight"))
            ref.in_proj_a.weight.copy_(w("ssm_alpha.weight"))
            conv = w("ssm_conv1d.weight")
            ref.conv1d.weight.copy_(conv.reshape(ref.conv1d.weight.shape))
            ref.dt_bias.copy_(w("ssm_dt.bias"))
            # GGUF stores -exp(A_log); the module wants A_log.
            ref.A_log.copy_(torch.log(-w("ssm_a")))
            ref.norm.weight.copy_(w("ssm_norm.weight"))
            ref.out_proj.weight.copy_(w("ssm_out.weight"))
            mixer = ref(h.unsqueeze(0))[0]
    else:
        n_q, n_kv, hd = cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim
        rot = cfg.rotary_config
        qg = (h @ w("attn_q.weight").T).view(T, n_q, hd * 2)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = (h @ w("attn_k.weight").T).view(T, n_kv, hd)
        v = (h @ w("attn_v.weight").T).view(T, n_kv, hd)
        q = rms_norm(q, w("attn_q_norm.weight"), eps)
        k = rms_norm(k, w("attn_k_norm.weight"), eps)

        # partial NeoX rope over the first rotary_dim dims, frequencies spaced over
        # rotary_dim (HF default; ggml agrees when rope.dimension_count is the partial dim)
        rd = rot.rotary_dim
        pos = d["positions"].float() if "positions" in d else torch.arange(T).float()
        inv = 1.0 / (rot.base ** (torch.arange(0, rd, 2).float() / rd))
        ang = pos[:, None] * inv[None, :]
        cos, sin = ang.cos(), ang.sin()

        def rope(t):  # t: [T, heads, hd]
            r, keep = t[..., :rd], t[..., rd:]
            a, b = r[..., : rd // 2], r[..., rd // 2 :]
            c, s = cos[:, None, :], sin[:, None, :]
            return torch.cat([a * c - b * s, b * c + a * s, keep], dim=-1)

        q, k = rope(q), rope(k)
        rep = n_q // n_kv
        kk = k.repeat_interleave(rep, dim=1)
        vv = v.repeat_interleave(rep, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, kk) / math.sqrt(hd)
        mask = torch.full((T, T), float("-inf")).triu(1)
        attn = (scores + mask).softmax(-1)
        o = torch.einsum("hqk,khd->qhd", attn, vv).reshape(T, n_q * hd)
        mixer = (o * torch.sigmoid(gate.reshape(T, n_q * hd))) @ w("attn_output.weight").T
    ok_all &= report("mixer_out", d["mixer_out"], mixer, args.tol)

    # ---- residual add + post_attention_layernorm --------------------------------
    residual = x + mixer
    h2 = rms_norm(residual, w("post_attention_norm.weight"), eps)
    ok_all &= report("mlp_in", d["mlp_in"], h2, args.tol)

    # ---- MoE: router topk-8 (softmax over all experts) + shared + routed ---------
    logits = h2 @ w("ffn_gate_inp.weight").T
    probs = logits.softmax(-1)
    tw, tid = probs.topk(cfg.num_experts_per_tok, dim=-1)
    tw = tw / tw.sum(-1, keepdim=True)  # norm_topk_prob

    gate_sh, up_sh, down_sh = (
        w("ffn_gate_shexp.weight"), w("ffn_up_shexp.weight"), w("ffn_down_shexp.weight")
    )
    shared = (F.silu(h2 @ gate_sh.T) * (h2 @ up_sh.T)) @ down_sh.T
    shared = shared * torch.sigmoid(h2 @ w("ffn_gate_inp_shexp.weight").reshape(1, H).T)

    # Dequantize only the experts this prompt actually routes to.
    E, I = cfg.num_experts, cfg.moe_intermediate_size
    ge, ue, de = (T_[f"blk.{L}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
    used = sorted(set(tid.reshape(-1).tolist()))
    print(f"  router selected {len(used)} distinct experts of {E} for {T} tokens")
    gw = {e: deq(np.ascontiguousarray(ge._raw[e * I:(e + 1) * I]), ge.ggml_type, (I, H)) for e in used}
    uw = {e: deq(np.ascontiguousarray(ue._raw[e * I:(e + 1) * I]), ue.ggml_type, (I, H)) for e in used}
    dw = {e: deq(np.ascontiguousarray(de._raw[e * H:(e + 1) * H]), de.ggml_type, (H, I)) for e in used}

    routed = torch.zeros(T, H)
    for i in range(T):
        for j in range(cfg.num_experts_per_tok):
            e = int(tid[i, j])
            xi = h2[i]
            routed[i] += float(tw[i, j]) * ((F.silu(xi @ gw[e].T) * (xi @ uw[e].T)) @ dw[e].T)
    ok_all &= report("mlp_out", d["mlp_out"], routed + shared, args.tol)

    print()
    print("all stages match the fp32 reference" if ok_all else
          "FIRST failing stage above is where the engine diverges from the checkpoint")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
