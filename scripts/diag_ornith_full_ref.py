"""Run the whole model on CPU from the GGUF, with switchable checkpoint conventions.

Why this exists: the per-layer reference (diag_ornith_layer_ref.py) shows the engine
reproducing an independent fp32 rebuild at every stage of layers 0, 3 and 39 -- and the
resulting trunk still predicts punctuation salad. Two implementations agreeing while both
produce garbage means the fault is not in either implementation: it is in an assumption
they *share*, namely which GGUF tensor plays which role. The per-layer diff can never see
that, because the reference was built on the adapter's own assumptions.

Several of those assumptions are pure naming conventions that no shape constrains:

  gdn-qk      attn_qkv packs [q|k|v] -- q and k are both num_k_heads*head_k_dim wide, so
              [k|q|v] fits identically. Swapping them is magnitude-preserving and makes
              the delta rule attend to the wrong history.
  moe-gate-up SwiGLU is silu(gate(x)) * up(x); ffn_gate_exps and ffn_up_exps have the
              same shape, so a converter that split HF's fused gate_up_proj the other way
              round is undetectable from the tensors.
  attn-gate   attn_q is [q|gate] per head (both head_dim wide).
  gdn-ba      ssm_beta -> sigmoid (write strength), ssm_alpha -> softplus+decay. Both are
              [num_v_heads, hidden].
  rope-gptj   rope pairs dims (i, i+rot/2) (NeoX) vs (2i, 2i+1) (GPT-J interleaved).
  rope-full   rope frequencies spaced over rotary_dim (HF) vs over the full head_dim with
              the tail left unrotated (what ggml does for a rope_freqs-style partial rope).

Usage -- baseline first, which should reproduce the engine's own garbage and so confirm
the reference is faithful, then one flip at a time:

    .venv/bin/python scripts/diag_ornith_full_ref.py MODEL.gguf
    .venv/bin/python scripts/diag_ornith_full_ref.py MODEL.gguf --flip gdn-qk
    .venv/bin/python scripts/diag_ornith_full_ref.py MODEL.gguf --flip moe-gate-up

The prompt is the model's own chat template around "hi", so a correct configuration
should put ordinary reasoning-opener tokens on top ("Okay", "The", "User", "\n"), not
'###' and '['. Runs on CPU in fp32; a few minutes per pass, one layer's weights resident
at a time.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

FLIPS = ("gdn-qk", "moe-gate-up", "attn-gate", "gdn-ba", "rope-gptj", "rope-full")


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--flip", default="", help=f"comma-separated: {', '.join(FLIPS)}")
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    flips = {f.strip() for f in args.flip.split(",") if f.strip()}
    bad = flips - set(FLIPS)
    if bad:
        print(f"unknown flip(s) {sorted(bad)}; valid: {', '.join(FLIPS)}")
        return 2

    from freetoken.models.gguf.dequant import dequantize
    from freetoken.models.gguf.reader import iter_gguf_tensors, load_gguf_metadata
    from freetoken.models.gguf.tokenizer import load_gguf_tokenizer
    from freetoken.models.qwen3_5_moe.gdn_reference import recurrent_gated_delta_rule
    from freetoken.models.qwen3_5_moe.gguf import parse_gguf_config
    from freetoken.utils import cached_load_hf_config

    cfg = parse_gguf_config(cached_load_hf_config(args.gguf))
    eps, H = cfg.rms_norm_eps, cfg.hidden_size
    g = cfg.linear_attention_group()
    n_k, n_v = g.num_key_heads, g.num_value_heads
    dk, dv = g.key_head_dim, g.value_head_dim
    key_dim, val_dim = n_k * dk, n_v * dv
    n_q, n_kv, hd = cfg.num_qo_heads, cfg.num_kv_heads, cfg.head_dim
    rot = cfg.rotary_config
    print(f"flips: {sorted(flips) or ['(none - baseline)']}")

    T_ = {t.name: t for t in iter_gguf_tensors(args.gguf)}

    def deq(t, shape=None):
        out = dequantize(torch.from_numpy(t._raw).reshape(-1), t.ggml_type, torch.float32)
        return out.reshape(shape if shape is not None else t.shape)

    def W(layer, suffix):
        return deq(T_[f"blk.{layer}.{suffix}"])

    tok = load_gguf_tokenizer(args.gguf)
    ids = torch.tensor(
        tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True, add_generation_prompt=True,
        )
    ).long()
    Tn = ids.numel()
    print(f"prompt -> {Tn} tokens: {ids.tolist()}")

    emb = T_["token_embd.weight"]
    x = deq(
        type("t", (), {"_raw": np.ascontiguousarray(emb._raw[ids.numpy()]),
                       "ggml_type": emb.ggml_type, "shape": (Tn, H)})()
    )
    pos = torch.arange(Tn).float()

    # rope tables -------------------------------------------------------------
    rd = rot.rotary_dim
    if "rope-full" in flips:
        # ggml partial-rope style: space the ladder over the full head_dim and leave the
        # tail unrotated, instead of spacing it over rotary_dim.
        inv = 1.0 / (rot.base ** (torch.arange(0, hd, 2).float() / hd))[: rd // 2]
    else:
        inv = 1.0 / (rot.base ** (torch.arange(0, rd, 2).float() / rd))
    ang = pos[:, None] * inv[None, :]
    cos_t, sin_t = ang.cos(), ang.sin()

    def rope(t):  # [T, heads, hd]
        r, keep = t[..., :rd], t[..., rd:]
        c, s = cos_t[:, None, :], sin_t[:, None, :]
        if "rope-gptj" in flips:
            a, b = r[..., 0::2], r[..., 1::2]
            out = torch.stack([a * c - b * s, b * c + a * s], dim=-1).flatten(-2)
        else:
            a, b = r[..., : rd // 2], r[..., rd // 2 :]
            out = torch.cat([a * c - b * s, b * c + a * s], dim=-1)
        return torch.cat([out, keep], dim=-1)

    t0 = time.time()
    residual = x
    for L in range(cfg.num_layers):
        h = rms_norm(residual, W(L, "attn_norm.weight"), eps)

        if cfg.is_linear_layer(L):
            qkv = h @ W(L, "attn_qkv.weight").T
            z = h @ W(L, "attn_gate.weight").T
            b_raw = h @ W(L, "ssm_beta.weight").T
            a_raw = h @ W(L, "ssm_alpha.weight").T
            if "gdn-ba" in flips:
                b_raw, a_raw = a_raw, b_raw

            conv_w = W(L, "ssm_conv1d.weight").reshape(2 * key_dim + val_dim, 1, -1)
            cd = conv_w.shape[0]
            mixed = F.conv1d(
                qkv.T.unsqueeze(0), conv_w, groups=cd, padding=conv_w.shape[-1] - 1
            )[0, :, :Tn].T
            mixed = F.silu(mixed)
            q, k, v = torch.split(mixed, [key_dim, key_dim, val_dim], dim=-1)
            if "gdn-qk" in flips:
                q, k = k, q
            q = q.reshape(1, Tn, n_k, dk).repeat_interleave(n_v // n_k, dim=2)
            k = k.reshape(1, Tn, n_k, dk).repeat_interleave(n_v // n_k, dim=2)
            v = v.reshape(1, Tn, n_v, dv)

            A_log = torch.log(-W(L, "ssm_a"))
            dt_bias = W(L, "ssm_dt.bias")
            beta = b_raw.sigmoid().reshape(1, Tn, n_v)
            gg = (-A_log.exp() * F.softplus(a_raw + dt_bias)).reshape(1, Tn, n_v)
            core, _ = recurrent_gated_delta_rule(q, k, v, gg, beta, use_qk_l2norm=True)

            core = core[0].reshape(-1, dv)
            zz = z.reshape(-1, dv)
            nw = W(L, "ssm_norm.weight")
            core = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + eps) * nw
            core = core * F.silu(zz)
            mixer = core.reshape(Tn, -1) @ W(L, "ssm_out.weight").T
        else:
            qg = (h @ W(L, "attn_q.weight").T).view(Tn, n_q, hd * 2)
            if "attn-gate" in flips:
                gate, q = qg[..., :hd], qg[..., hd:]
            else:
                q, gate = qg[..., :hd], qg[..., hd:]
            k = (h @ W(L, "attn_k.weight").T).view(Tn, n_kv, hd)
            v = (h @ W(L, "attn_v.weight").T).view(Tn, n_kv, hd)
            q = rope(rms_norm(q, W(L, "attn_q_norm.weight"), eps))
            k = rope(rms_norm(k, W(L, "attn_k_norm.weight"), eps))
            rep = n_q // n_kv
            kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            sc = torch.einsum("qhd,khd->hqk", q, kk) / math.sqrt(hd)
            sc = sc + torch.full((Tn, Tn), float("-inf")).triu(1)
            o = torch.einsum("hqk,khd->qhd", sc.softmax(-1), vv).reshape(Tn, n_q * hd)
            mixer = (o * torch.sigmoid(gate.reshape(Tn, n_q * hd))) @ W(L, "attn_output.weight").T

        residual = residual + mixer
        h2 = rms_norm(residual, W(L, "post_attention_norm.weight"), eps)

        # MoE: shared expert + routed top-k
        gs, us, ds = (W(L, f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
        if "moe-gate-up" in flips:
            gs, us = us, gs
        shared = (F.silu(h2 @ gs.T) * (h2 @ us.T)) @ ds.T
        shared = shared * torch.sigmoid(h2 @ W(L, "ffn_gate_inp_shexp.weight").reshape(1, H).T)

        probs = (h2 @ W(L, "ffn_gate_inp.weight").T).softmax(-1)
        tw, tid = probs.topk(cfg.num_experts_per_tok, dim=-1)
        tw = tw / tw.sum(-1, keepdim=True)
        E, I = cfg.num_experts, cfg.moe_intermediate_size
        ge, ue, de = (T_[f"blk.{L}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
        if "moe-gate-up" in flips:
            ge, ue = ue, ge

        def slab(t, e, rows, shape):
            return deq(type("t", (), {
                "_raw": np.ascontiguousarray(t._raw[e * rows:(e + 1) * rows]),
                "ggml_type": t.ggml_type, "shape": shape}()))

        used = sorted(set(tid.reshape(-1).tolist()))
        gw = {e: slab(ge, e, I, (I, H)) for e in used}
        uw = {e: slab(ue, e, I, (I, H)) for e in used}
        dw = {e: slab(de, e, H, (H, I)) for e in used}
        routed = torch.zeros(Tn, H)
        for i in range(Tn):
            for j in range(cfg.num_experts_per_tok):
                e = int(tid[i, j])
                xi = h2[i]
                routed[i] += float(tw[i, j]) * ((F.silu(xi @ gw[e].T) * (xi @ uw[e].T)) @ dw[e].T)

        residual = residual + routed + shared
        if L % 8 == 0 or L == cfg.num_layers - 1:
            print(f"  layer {L:>2}/{cfg.num_layers - 1}  residual_rms={float(residual.pow(2).mean().sqrt()):.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    final = rms_norm(residual, deq(T_["output_norm.weight"]), eps)
    head = T_["output.weight"]
    xl = final[-1]
    V, blk = head.shape[0], 16384
    logits = torch.empty(V)
    for s in range(0, V, blk):
        e = min(s + blk, V)
        wb = deq(type("t", (), {"_raw": np.ascontiguousarray(head._raw[s:e]),
                                "ggml_type": head.ggml_type, "shape": (e - s, H)})())
        logits[s:e] = wb @ xl
    p = logits.softmax(-1)
    toks = load_gguf_metadata(args.gguf)["tokenizer.ggml.tokens"]
    top = p.topk(args.topk)
    print(f"\ntop-{args.topk} next tokens  (flips: {sorted(flips) or 'none'}):")
    for pv, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {pv*100:6.2f}%  id={i:<7d} {toks[i]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
