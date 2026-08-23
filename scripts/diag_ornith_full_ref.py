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

FLIPS = ("gdn-qk", "moe-gate-up", "attn-gate", "gdn-ba", "rope-gptj", "rope-full",
         "gdn-conv-rev", "gdn-gqa-tile", "attn-gqa-tile")


def rms_norm(x, w, eps):
    # Norms accumulate in fp32 and return in the working dtype, as the fused kernels do.
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return out.to(x.dtype)


def _r(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--flip", default="", help=f"comma-separated: {', '.join(FLIPS)}")
    ap.add_argument("--prompt", default="hi")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--generate", type=int, default=0,
                    help="greedily continue N tokens; a real test of coherence, "
                         "unlike top-k at one high-entropy position")
    ap.add_argument("--engine-dump", default=None,
                    help="a FREETOKEN_DUMP_LAYER=all .pt; adds per-layer cosine vs the engine")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32",
                    help="working precision. bf16 matches what the engine stores between "
                         "layers, which tests whether the drift is precision rather than a bug")
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
    print(f"flips: {sorted(flips) or ['(none - baseline)']}  dtype: {args.dtype}")

    T_ = {t.name: t for t in iter_gguf_tensors(args.gguf)}

    def deq_raw(raw: np.ndarray, ggml_type: int, shape) -> torch.Tensor:
        """Dequantize packed ggml bytes to a dense fp32 tensor of ``shape``."""
        flat = dequantize(torch.from_numpy(np.ascontiguousarray(raw)).reshape(-1),
                          ggml_type, torch.float32)
        return flat.reshape(shape)

    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    def deq(t) -> torch.Tensor:
        return deq_raw(t._raw, t.ggml_type, t.shape).to(dt)

    def deq_rows(t, lo: int, hi: int, shape) -> torch.Tensor:
        """Dequantize a row slice -- one expert, or a block of the LM head."""
        return deq_raw(t._raw[lo:hi], t.ggml_type, shape)

    def W(layer, suffix):
        return deq(T_[f"blk.{layer}.{suffix}"])

    def Wf(layer, suffix):
        """fp32 regardless of --dtype: the scalars the engine keeps out of the downcast."""
        t = T_[f"blk.{layer}.{suffix}"]
        return deq_raw(t._raw, t.ggml_type, t.shape)

    tok = load_gguf_tokenizer(args.gguf)
    text = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    # Go through the backend tokenizers.Tokenizer: the transformers wrapper's return type
    # varies by version (this one hands back a tokenizers.Encoding), while the backend
    # always yields .ids and still applies the added-token vocabulary.
    ids = torch.tensor(tok.backend_tokenizer.encode(text, add_special_tokens=False).ids).long()
    Tn = ids.numel()
    print(f"prompt -> {Tn} tokens: {ids.tolist()}")

    emb = T_["token_embd.weight"]

    # rope tables -------------------------------------------------------------
    rd = rot.rotary_dim
    if "rope-full" in flips:
        # ggml partial-rope style: space the ladder over the full head_dim and leave the
        # tail unrotated, instead of spacing it over rotary_dim.
        inv = 1.0 / (rot.base ** (torch.arange(0, hd, 2).float() / hd))[: rd // 2]
    else:
        inv = 1.0 / (rot.base ** (torch.arange(0, rd, 2).float() / rd))

    # Optional engine trace to compare against. Cosine is the point: the RMS traces agree
    # to ~5% through layer 23 and then break, but equal magnitude does not mean equal
    # vector, so only the angle says where the two streams actually part.
    eng = None
    if args.engine_dump:
        eng = torch.load(args.engine_dump, map_location="cpu")
        got = eng["input_ids"].long().tolist()
        assert got == ids.tolist(), (
            f"engine dump tokenized differently: {got} vs {ids.tolist()}"
        )
        print(f"comparing against engine dump {args.engine_dump}")

    def forward_logits(ids_now, trace: bool):
        """Full forward over ids_now; returns logits for the last position."""
        Tn = ids_now.numel()
        x = deq_raw(emb._raw[ids_now.numpy()], emb.ggml_type, (Tn, H)).to(dt)
        pos = torch.arange(Tn).float()
        ang = pos[:, None] * inv[None, :]
        cos_t, sin_t = ang.cos(), ang.sin()

        def rope(t):  # [T, heads, hd]
            r, keep = t[..., :rd].float(), t[..., rd:].float()
            c, s = cos_t[:, None, :], sin_t[:, None, :]
            if "rope-gptj" in flips:
                a, b = r[..., 0::2], r[..., 1::2]
                out = torch.stack([a * c - b * s, b * c + a * s], dim=-1).flatten(-2)
            else:
                a, b = r[..., : rd // 2], r[..., rd // 2 :]
                out = torch.cat([a * c - b * s, b * c + a * s], dim=-1)
            return torch.cat([out, keep], dim=-1).to(t.dtype)
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
                if "gdn-conv-rev" in flips:
                    # ggml_ssm_conv and torch's conv1d disagree on tap order for some
                    # conversions; a reversed kernel is causal either way and same-magnitude.
                    conv_w = conv_w.flip(-1)
                cd = conv_w.shape[0]
                mixed = F.conv1d(
                    qkv.T.unsqueeze(0), conv_w, groups=cd, padding=conv_w.shape[-1] - 1
                )[0, :, :Tn].T
                mixed = F.silu(mixed)
                q, k, v = torch.split(mixed, [key_dim, key_dim, val_dim], dim=-1)
                if "gdn-qk" in flips:
                    q, k = k, q
                rep_g = n_v // n_k
                if "gdn-gqa-tile" in flips:
                    q = q.reshape(1, Tn, n_k, dk).repeat(1, 1, rep_g, 1)
                    k = k.reshape(1, Tn, n_k, dk).repeat(1, 1, rep_g, 1)
                else:
                    q = q.reshape(1, Tn, n_k, dk).repeat_interleave(rep_g, dim=2)
                    k = k.reshape(1, Tn, n_k, dk).repeat_interleave(rep_g, dim=2)
                v = v.reshape(1, Tn, n_v, dv)

                # A_log / dt_bias are exempt from the model-dtype downcast in the engine and
                # the gating is evaluated in fp32; keep that here or bf16 noise enters the
                # decay before the recurrence ever runs.
                A_log = torch.log(-Wf(L, "ssm_a"))
                dt_bias = Wf(L, "ssm_dt.bias")
                beta = b_raw.float().sigmoid().reshape(1, Tn, n_v)
                gg = (-A_log.exp() * F.softplus(a_raw.float() + dt_bias)).reshape(1, Tn, n_v)
                core, _ = recurrent_gated_delta_rule(q, k, v, gg, beta, use_qk_l2norm=True)

                # Gated RMSNorm in fp32, like the fused rms_norm_gated kernel.
                core = core[0].reshape(-1, dv).float()
                zz = z.reshape(-1, dv).float()
                nw = Wf(L, "ssm_norm.weight")
                core = core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + eps) * nw
                core = (core * F.silu(zz)).to(dt)
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
                if "attn-gqa-tile" in flips:
                    kk, vv = k.repeat(1, rep, 1), v.repeat(1, rep, 1)
                else:
                    kk, vv = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
                # Scores and softmax accumulate in fp32 (as every attention kernel does), then
                # the result returns to the working dtype.
                sc = torch.einsum("qhd,khd->hqk", q.float(), kk.float()) / math.sqrt(hd)
                sc = sc + torch.full((Tn, Tn), float("-inf")).triu(1)
                o = torch.einsum("hqk,khd->qhd", sc.softmax(-1), vv.float())
                o = o.reshape(Tn, n_q * hd).to(dt)
                mixer = (o * torch.sigmoid(gate.reshape(Tn, n_q * hd))) @ W(L, "attn_output.weight").T

            residual = residual + mixer
            resid_after_mixer = residual  # the probe prints the residual at this point
            h2 = rms_norm(residual, W(L, "post_attention_norm.weight"), eps)

            # MoE: shared expert + routed top-k
            gs, us, ds = (W(L, f"ffn_{n}_shexp.weight") for n in ("gate", "up", "down"))
            if "moe-gate-up" in flips:
                gs, us = us, gs
            shared = (F.silu(h2 @ gs.T) * (h2 @ us.T)) @ ds.T
            shared = shared * torch.sigmoid(h2 @ W(L, "ffn_gate_inp_shexp.weight").reshape(1, H).T)

            # Match fused_topk: logits from the working-dtype linear, softmax in fp32. The
            # router is the discrete amplifier in this model, so its precision is not a detail.
            probs = (h2 @ W(L, "ffn_gate_inp.weight").T).float().softmax(-1)
            tw, tid = probs.topk(cfg.num_experts_per_tok, dim=-1)
            tw = (tw / tw.sum(-1, keepdim=True)).to(dt)
            E, I = cfg.num_experts, cfg.moe_intermediate_size
            ge, ue, de = (T_[f"blk.{L}.ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
            if "moe-gate-up" in flips:
                ge, ue = ue, ge

            used = sorted(set(tid.reshape(-1).tolist()))
            gw = {e: deq_rows(ge, e * I, (e + 1) * I, (I, H)).to(dt) for e in used}
            uw = {e: deq_rows(ue, e * I, (e + 1) * I, (I, H)).to(dt) for e in used}
            dw = {e: deq_rows(de, e * H, (e + 1) * H, (H, I)).to(dt) for e in used}
            routed = torch.zeros(Tn, H, dtype=dt)
            for i in range(Tn):
                for j in range(cfg.num_experts_per_tok):
                    e = int(tid[i, j])
                    xi = h2[i]
                    routed[i] += float(tw[i, j]) * ((F.silu(xi @ gw[e].T) * (xi @ uw[e].T)) @ dw[e].T)

            mlp_out = routed + shared
            residual = residual + mlp_out
            # Same fields, same order, same semantics as FREETOKEN_LAYER_PROBE, so an engine
            # trace and this one can be diffed line by line to find the first layer where the
            # two implementations part company.
            line = (
                f"[ref] layer {L:>2} {'gdn ' if cfg.is_linear_layer(L) else 'attn'}"
                f"  norm_in={_r(h):9.4f}  mixer_out={_r(mixer):9.4f}"
                f"  mlp_in={_r(h2):9.4f}  mlp_out={_r(mlp_out):9.4f}"
                f"  residual={_r(resid_after_mixer):9.4f}"
            )
            if eng is not None:
                def cos(a, key):
                    b = eng.get(f"L{L}.{key}")
                    return float("nan") if b is None else float(
                        F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float())
                    )
                cm, cp, cr = cos(mixer, "mixer"), cos(mlp_out, "mlp_out"), cos(residual, "residual")
                line += f"  | cos mixer={cm:+.5f} mlp={cp:+.5f} resid={cr:+.5f}"
                if min(cm, cp, cr) < 0.99:
                    line += "  <<<"
        if trace:
            print(line, flush=True)

        final = rms_norm(residual, deq(T_['output_norm.weight']), eps)
        head_t = T_['output.weight']
        xl = final[-1]
        V, blk = head_t.shape[0], 16384
        logits = torch.empty(V)
        for s in range(0, V, blk):
            e = min(s + blk, V)
            logits[s:e] = deq_rows(head_t, s, e, (e - s, H)) @ xl.float()
        return logits

    toks = load_gguf_metadata(args.gguf)['tokenizer.ggml.tokens']
    logits = forward_logits(ids, trace=True)
    p = logits.softmax(-1)
    top = p.topk(args.topk)
    print(f'\ntop-{args.topk} next tokens  (flips: {sorted(flips) or "none"}):')
    for pv, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f'  {pv*100:6.2f}%  id={i:<7d} {toks[i]!r}')

    if args.generate:
        # Greedy continuation. One token at a time, recomputing the whole prefill --
        # slow, but it needs no cache and so cannot inherit a cache bug.
        cur = ids.clone()
        out = []
        for _ in range(args.generate):
            nxt = int(forward_logits(cur, trace=False).argmax())
            out.append(nxt)
            cur = torch.cat([cur, torch.tensor([nxt])])
        text = ''.join(toks[i] for i in out).replace('\u0120', ' ').replace('\u010a', chr(10))
        print(f'\ngreedy {args.generate} tokens (flips: {sorted(flips) or "none"}):')
        print(f'  ids  {out}')
        print(f'  text {text!r}')
    return 0