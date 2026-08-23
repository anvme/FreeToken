"""qwen35moe (Ornith-1.5-35B / Qwen3.5 hybrid MoE) GGUF adapter.

Builds the FreeToken ``ModelConfig`` from GGUF ``general.architecture == "qwen35moe"``
metadata and reads the weights straight from the packed GGUF: quantized projections stay
in their native block layout (ggml Q8_0 / K-quants, dequantized inside the borrowed
llama.cpp kernels) and the ~34 GB of routed experts stream into per-layer host banks for
the offload cache. The geometry is identical to the HF qwen3_5_moe model -- GDN linear
layers interleaved with gated full attention every ``full_attention_interval``-th layer,
256 routed experts (top-8) plus a gated shared expert -- so this produces the same
ModelConfig as ``qwen3_5_moe.config.parse_config``.

Differences from the gemma4 GGUF adapter worth remembering:

* Hybrid layers: GDN (linear) vs full attention per the interval, not a per-layer SWA
  flag; the full layers here are *gated* (q tensor is 2x wide: ``[q | gate]`` per head).
* GDN input projection fuses FOUR tensors into one GEMM
  (``attn_qkv | attn_gate | ssm_beta | ssm_alpha``) and full attention fuses three
  (``attn_q | attn_k | attn_v``); the shared expert fuses ``gate | up`` like gemma4.
  A group fuses packed when every part shares one quantized ggml type; if any part is a
  different type (or F32) the whole group dequantizes to dense bf16 at load.
* ``ssm_a`` stores the gate coefficient ``-exp(A_log)`` directly (llama.cpp
  ``src/models/qwen35moe.cpp``: ``ssm_a = -A_log.exp()``), so the loader computes
  ``A_log = log(-ssm_a)`` -- FreeToken's GDN gate is ``-exp(A_log) * softplus(dt)``.
* llama.cpp's converter bakes ``+1`` into every RMSNorm weight *except*
  ``linear_attn.norm`` (``conversion/qwen.py``), and this runtime's norms are Gemma-style
  (scale by the raw weight) -- so GGUF norm values are already ``(1 + w)`` and are used
  as-is, no +1 added at load.
* Text-only, no MTP: ``block_count`` includes one extra MTP block (``blk.L`` with
  ``nextn.*`` tensors under it); the trunk is the first ``num_layers`` blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch

from freetoken.layers import BaseOP
from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
)
from freetoken.models.gguf.dequant import (
    GGML_NAME,
    GGML_TO_GGUF_FORMAT,
    GGUF_FORMAT_TO_GGML,
    dequantize,
    row_bytes,
)

if TYPE_CHECKING:
    from freetoken.models.gguf.config import GgufConfigShim

ARCH = "qwen35moe"

# ggml types kept native-packed at load; every other (F32/F16/BF16 or unknown) type
# dequantizes to dense bf16.
_QUANT = set(GGUF_FORMAT_TO_GGML.values())


def _require_tp1(what: str) -> None:
    """GGUF quant layers / expert banks are not sharded; reject TP>1 with a clear
    error instead of failing later on a confusing shape mismatch (mirrors the HF
    loader's TP=1 restriction)."""
    from freetoken.distributed import get_tp_info

    if get_tp_info().size > 1:
        raise NotImplementedError(
            f"qwen35moe GGUF {what} currently supports TP=1 only "
            "(GGUF quant layers and expert banks are not tensor-parallel sharded)."
        )


# --------------------------------------------------------------------------------------
# ModelConfig from GGUF metadata (+ tensor table for per-slot quant types).
# --------------------------------------------------------------------------------------


def parse_gguf_config(shim: "GgufConfigShim") -> ModelConfig:
    from freetoken.models.gguf.reader import gguf_tensor_geom

    m = shim.metadata

    def g(key: str):
        val = m.get(f"{ARCH}.{key}")
        if val is None:
            raise KeyError(f"missing GGUF metadata key {ARCH}.{key}")
        return val

    block_count = int(g("block_count"))
    nextn = int(m.get(f"{ARCH}.nextn_predict_layers") or 0)
    num_layers = block_count - nextn
    assert num_layers > 0, f"block_count {block_count} - nextn {nextn} <= 0"

    hidden = int(g("embedding_length"))
    vocab = int(shim.vocab_size)
    num_qo_heads = int(g("attention.head_count"))
    kv = g("attention.head_count_kv")  # scalar or per-layer list
    num_kv_heads = int(kv[0]) if isinstance(kv, (list, tuple)) else int(kv)
    head_dim = int(g("attention.key_length"))
    max_pos = int(g("context_length"))
    eps = float(g("attention.layer_norm_rms_epsilon"))

    rope_dim = int(m.get(f"{ARCH}.rope.dimension_count", head_dim))
    rope_base = float(g("rope.freq_base"))

    n_experts = int(g("expert_count"))
    top_k = int(g("expert_used_count"))
    moe_inter = int(g("expert_feed_forward_length"))
    shared_inter = int(g("expert_shared_feed_forward_length"))

    conv_k = int(g("ssm.conv_kernel"))
    key_head_dim = int(g("ssm.state_size"))
    n_k_heads = int(g("ssm.group_count"))
    n_v_heads = int(g("ssm.time_step_rank"))
    d_inner = int(g("ssm.inner_size"))
    value_head_dim = d_inner // n_v_heads

    interval = int(g("full_attention_interval"))
    full_ids = tuple(i for i in range(num_layers) if (i + 1) % interval == 0)
    linear_ids = tuple(i for i in range(num_layers) if (i + 1) % interval != 0)
    assert full_ids and linear_ids, "qwen35moe needs both full and linear layers"
    assert d_inner % n_v_heads == 0, "ssm.inner_size not divisible by num value heads"

    # --- per-slot ggml quant types from the tensor table (geometry, no data reads) ---
    geom = gguf_tensor_geom(shim.model_path)

    def ttype(name: str):
        e = geom.get(name)
        return e[1] if e is not None else None

    def tshape(name: str):
        e = geom.get(name)
        if e is None:
            raise KeyError(f"missing GGUF tensor {name!r}")
        return e[0]

    # Cheap layout guards (catch a different-file surprise early, with a clear name).
    assert tshape("token_embd.weight") == (vocab, hidden), "token_embd shape"
    assert tshape(f"blk.0.ffn_gate_exps.weight") == (n_experts, moe_inter, hidden)
    assert tshape(f"blk.0.ffn_down_exps.weight") == (n_experts, hidden, moe_inter)
    assert tshape(f"blk.0.ssm_conv1d.weight") == (2 * n_k_heads * key_head_dim + d_inner, conv_k)
    assert tshape(f"blk.0.ssm_a") == (n_v_heads,)
    assert tshape(f"blk.0.attn_norm.weight") == (hidden,)

    def group_type(ctx: str, layer_ids: tuple[int, ...], suffixes: tuple[str, ...]):
        """The group's ggml type: the shared quant type when every part of every layer
        is the same quantized type (packed fused path); ``None`` (dense bf16 fallback)
        when any part is F32/F16/BF16 (a dequantized part can't join a packed concat);
        raises on a mixed-quant or incomplete group."""
        types = {ttype(f"blk.{i}.{sfx}") for i in layer_ids for sfx in suffixes}
        if None in types:
            missing = [f"blk.{i}.{sfx}" for i in layer_ids for sfx in suffixes
                       if ttype(f"blk.{i}.{sfx}") is None]
            raise KeyError(f"{ctx}: tensors missing from the GGUF table: {missing[:3]}...")
        quants = sorted(t for t in types if t in _QUANT)
        if len(quants) > 1:
            raise ValueError(
                f"{ctx}: mixed quant types {[GGML_NAME.get(t, t) for t in quants]}; "
                "a fused group must share one quant per file"
            )
        if types - _QUANT:
            return None
        return quants[0]

    embed_t = ttype("token_embd.weight")
    if embed_t is None:
        raise KeyError("missing token_embd.weight")
    if embed_t not in _QUANT:
        raise ValueError(
            f"token_embd.weight is {GGML_NAME.get(embed_t, embed_t)}: the GGUF embedding "
            "must be block-quantized (Q4_0/Q8_0/K-quants)"
        )
    lmh_t = embed_t if shim.tie_word_embeddings else ttype("output.weight")
    if lmh_t is None:
        raise KeyError("missing output.weight (and embeddings are untied)")

    all_ids = tuple(range(num_layers))
    types: dict[str, int | None] = {
        "embed": embed_t,
        "lm_head": lmh_t,
        "gdn_in": group_type(
            "GDN in_proj", linear_ids,
            ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"),
        ),
        "gdn_out": group_type("GDN ssm_out", linear_ids, ("ssm_out.weight",)),
        "attn_qkv": group_type(
            "attn qkv", full_ids, ("attn_q.weight", "attn_k.weight", "attn_v.weight")
        ),
        "attn_o": group_type("attn o_proj", full_ids, ("attn_output.weight",)),
        "shared_gate_up": group_type(
            "shared gate_up", all_ids, ("ffn_gate_shexp.weight", "ffn_up_shexp.weight")
        ),
        "shared_down": group_type("shared down", all_ids, ("ffn_down_shexp.weight",)),
    }

    # Routed experts: all three tensors must share one quant type (single bank format).
    exp_types = {
        ttype(f"blk.0.{sfx}")
        for sfx in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    }
    if len(exp_types) != 1 or not (exp_types & _QUANT):
        raise ValueError(
            f"routed-expert tensors must share one quant ggml type, got "
            f"{sorted(GGML_NAME.get(t, t) for t in exp_types)}"
        )
    expert_fmt = GGML_TO_GGUF_FORMAT[next(iter(exp_types))]
    assert expert_fmt in GGUF_FORMAT_TO_GGML

    rotary = RotaryConfig(
        head_dim=head_dim,
        rotary_dim=rope_dim,
        max_position=max_pos,
        base=rope_base,
        # Text-only default rope (mRoPE reduces to standard partial rope for text),
        # mirroring the HF parse_config.
        scaling=None,
    )
    groups = (
        FullAttentionGroupConfig(
            name="full",
            layer_ids=full_ids,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_config=rotary,
        ),
        LinearGatedDeltaGroupConfig(
            name="linear",
            layer_ids=linear_ids,
            num_key_heads=n_k_heads,
            num_value_heads=n_v_heads,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            conv_kernel_dim=conv_k,
            output_gate=True,
        ),
    )
    groups = tuple(sorted(groups, key=lambda gr: gr.layer_ids[0]))

    return ModelConfig(
        num_layers=num_layers,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=0,
        hidden_act="silu",
        rms_norm_eps=eps,
        rotary_config=rotary,
        tie_word_embeddings=bool(shim.tie_word_embeddings),
        num_experts=n_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=moe_inter,
        shared_expert_intermediate_size=shared_inter,
        norm_topk_prob=True,
        model_type="qwen3_5_moe",
        architectures=list(shim.architectures),
        moe_enabled=True,
        expert_quant=expert_fmt,
        moe_weight_format=expert_fmt,
        use_qk_norm=True,
        attention_groups=groups,
        gguf_types=types,
    )


# --------------------------------------------------------------------------------------
# Weight loading: GGUF tensor names -> FreeToken qwen3_5_moe module params.
# --------------------------------------------------------------------------------------


def _dense_of(t, dtype) -> torch.Tensor:
    """Dequantize a GgufTensor (F32/F16/BF16/Q*) to a dense tensor of its torch shape."""
    flat = dequantize(t.packed().reshape(-1), t.ggml_type, dtype)
    return flat.reshape(t.shape)


def iter_gguf_weights(
    model_path: str,
    device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (param_name, tensor) for every non-expert qwen35moe param.

    Quantized projections (attention qkv/o, GDN in_proj/out, shared-expert gate/up/down)
    and the embedding stay in their native packed block layout (``.qweight``, uint8);
    norms / router / GDN scalars dequantize to bf16 (A_log/dt_bias/conv1d stay fp32).
    Fused groups are emitted by concatenating packed rows along the output dim -- the
    same ``row_bytes`` because all parts share the input dim. Routed experts are served
    from the offload cache (``load_gguf_expert_sources``) and the MTP block is skipped
    (text-only, no lookahead).
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.utils import cached_load_hf_config

    assert not include_moe_experts, (
        "qwen35moe GGUF serves the routed experts from the offload cache; "
        "they are loaded into it via load_gguf_expert_sources() and not yielded here."
    )
    assert include_non_moe
    _require_tp1("weight loading")

    config = parse_gguf_config(cached_load_hf_config(model_path))
    T = config.gguf_types
    assert T is not None
    L = config.num_layers
    I_sh = config.shared_expert_intermediate_size
    g = config.linear_attention_group()
    conv_dim = 2 * g.num_key_heads * g.key_head_dim + g.num_value_heads * g.value_head_dim
    assert I_sh > 0, "qwen35moe expects a shared expert"

    bf16, f32 = torch.bfloat16, torch.float32

    # Per-layer fusion buffers: layer -> {slot: (GgufTensor, packed_rows)}.
    in_proj_buf: dict[int, dict[str, tuple]] = {}
    qkv_buf: dict[int, dict[str, tuple]] = {}
    shared_gu_buf: dict[int, dict[str, tuple]] = {}
    seen_shared: set[int] = set()
    seen_experts: set[int] = set()

    def emit_fused(
        buf: dict[int, dict[str, tuple]],
        layer: int,
        out_key: str,
        quant_key: str,
        order: tuple[str, ...],
    ):
        """Pop one completed fusion group and yield (param, tensor): packed rows
        concatenated along the output dim when the checkpoint group is uniformly
        quantized (``T[quant_key]`` != None), else dequantized dense bf16."""
        parts = buf.pop(layer)
        if T[quant_key] is not None:
            for slot, (t, _packed) in parts.items():
                assert t.ggml_type == T[quant_key], (
                    f"{out_key} blk.{layer}.{slot}: ggml type {t.ggml_type} "
                    f"!= group type {T[quant_key]}"
                )
            yield (
                out_key,
                torch.cat([parts[s][1] for s in order], dim=0),
            )
        else:
            yield (out_key, torch.cat([_dense_of(parts[s][0], bf16) for s in order], dim=0))

    for t in iter_gguf_tensors(model_path):
        name = t.name
        if name == "token_embd.weight":
            yield "model.embed_tokens.qweight", t.packed()
            continue
        if name == "output.weight":
            assert not config.tie_word_embeddings, "output.weight present but embeddings tied"
            yield "lm_head.qweight", t.packed()
            continue
        if name == "output_norm.weight":
            yield "model.norm.weight", _dense_of(t, bf16)
            continue
        if not name.startswith("blk."):
            continue  # no other top-level tensors in this arch (rope computed in-engine)
        layer = int(name.split(".")[1])
        if layer >= L:
            continue  # MTP block(s) blk.L..blk.L+nextn-1 (incl. nextn.*) -- not served
        suffix = name.split(".", 2)[2]  # after "blk.N."
        base = f"model.layers.{layer}"
        is_linear = config.is_linear_layer(layer)

        # ---- norms / router / scalars (all layers; GGUF norms already carry (1+w)) ----
        if suffix == "attn_norm.weight":
            yield f"{base}.input_layernorm.weight", _dense_of(t, bf16)
        elif suffix == "post_attention_norm.weight":
            yield f"{base}.post_attention_layernorm.weight", _dense_of(t, bf16)
        elif suffix == "ffn_gate_inp.weight":
            yield f"{base}.mlp.gate.weight", _dense_of(t, bf16)
        elif suffix == "ffn_gate_inp_shexp.weight":
            yield f"{base}.mlp.shared_expert_gate.weight", _dense_of(t, bf16).reshape(1, -1)
        # ---- GDN only ----
        elif is_linear and suffix in (
            "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"
        ):
            in_proj_buf.setdefault(layer, {})[suffix] = (t, t.packed())
            if len(in_proj_buf[layer]) == 4:
                yield from emit_fused(
                    in_proj_buf, layer,
                    f"{base}.linear_attn.in_proj.qweight"
                    if T["gdn_in"] is not None
                    else f"{base}.linear_attn.in_proj.weight",
                    "gdn_in",
                    ("attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight", "ssm_alpha.weight"),
                )
        elif is_linear and suffix == "ssm_out.weight":
            if T["gdn_out"] is not None:
                assert t.ggml_type == T["gdn_out"]
                yield f"{base}.linear_attn.out_proj.qweight", t.packed()
            else:
                yield f"{base}.linear_attn.out_proj.weight", _dense_of(t, bf16)
        elif is_linear and suffix == "ssm_conv1d.weight":
            yield f"{base}.linear_attn.conv1d.weight", _dense_of(t, f32).reshape(conv_dim, 1, -1)
        elif is_linear and suffix == "ssm_a":
            a = _dense_of(t, f32)
            assert (a < 0).all(), "ssm_a must be negative (stores -exp(A_log))"
            yield f"{base}.linear_attn.A_log", torch.log(-a)
        elif is_linear and suffix == "ssm_dt.bias":
            yield f"{base}.linear_attn.dt_bias", _dense_of(t, f32)
        elif is_linear and suffix == "ssm_norm.weight":
            yield f"{base}.linear_attn.norm.weight", _dense_of(t, bf16)
        # ---- full attention only ----
        elif not is_linear and suffix in ("attn_q.weight", "attn_k.weight", "attn_v.weight"):
            qkv_buf.setdefault(layer, {})[suffix] = (t, t.packed())
            if len(qkv_buf[layer]) == 3:
                yield from emit_fused(
                    qkv_buf, layer,
                    f"{base}.self_attn.qkv_proj.qweight"
                    if T["attn_qkv"] is not None
                    else f"{base}.self_attn.qkv_proj.weight",
                    "attn_qkv",
                    ("attn_q.weight", "attn_k.weight", "attn_v.weight"),
                )
        elif not is_linear and suffix == "attn_output.weight":
            if T["attn_o"] is not None:
                assert t.ggml_type == T["attn_o"]
                yield f"{base}.self_attn.o_proj.qweight", t.packed()
            else:
                yield f"{base}.self_attn.o_proj.weight", _dense_of(t, bf16)
        elif not is_linear and suffix == "attn_q_norm.weight":
            yield f"{base}.self_attn.q_norm.weight", _dense_of(t, bf16)
        elif not is_linear and suffix == "attn_k_norm.weight":
            yield f"{base}.self_attn.k_norm.weight", _dense_of(t, bf16)
        # ---- shared expert (all layers) ----
        elif suffix in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            shared_gu_buf.setdefault(layer, {})[
                "gate" if suffix.startswith("ffn_gate") else "up"
            ] = (t, t.packed())
            if "gate" in shared_gu_buf[layer] and "up" in shared_gu_buf[layer]:
                seen_shared.add(layer)
                yield from emit_fused(
                    shared_gu_buf, layer,
                    f"{base}.mlp.shared_expert.gate_up_proj.qweight"
                    if T["shared_gate_up"] is not None
                    else f"{base}.mlp.shared_expert.gate_up_proj.weight",
                    "shared_gate_up",
                    ("gate", "up"),
                )
        elif suffix == "ffn_down_shexp.weight":
            if T["shared_down"] is not None:
                assert t.ggml_type == T["shared_down"]
                yield f"{base}.mlp.shared_expert.down_proj.qweight", t.packed()
            else:
                yield f"{base}.mlp.shared_expert.down_proj.weight", _dense_of(t, bf16)
        # ---- routed experts -> offload banks (see load_gguf_expert_sources) ----
        elif suffix in (
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
        ):
            seen_experts.add(layer)
        else:
            raise ValueError(f"unmapped qwen35moe GGUF tensor: {name}")

    assert not in_proj_buf, f"incomplete GDN in_proj groups: {sorted(in_proj_buf)}"
    assert not qkv_buf, f"incomplete qkv groups: {sorted(qkv_buf)}"
    assert not shared_gu_buf, f"incomplete shared gate_up groups: {sorted(shared_gu_buf)}"
    want = set(range(L))
    assert seen_shared == want, f"missing shared-expert layers: {sorted(want - seen_shared)}"
    assert seen_experts == want, f"missing routed-expert layers: {sorted(want - seen_experts)}"


# --------------------------------------------------------------------------------------
# Model layer swap: dense bf16 Linear/Embedding -> native GGUF-quant ops.
# --------------------------------------------------------------------------------------


def is_gguf_model(config: ModelConfig) -> bool:
    """True when the model was parsed from a GGUF checkpoint (native-quant path)."""
    return config.gguf_types is not None


class GGUFParallelLMHead(BaseOP):
    """Untied GGUF LM head: owns its native-packed ``qweight`` and computes logits via
    ggml matmul (weights never dequantized to bf16).

    A ``BaseOP`` so the state-dict traversal collects ``lm_head.qweight``; the loader
    yields exactly that key. (Not a ``ParallelLMHead``: its ``weight``/``tied_embedding``
    contract doesn't fit a packed block-quant table.) TP=1 only (GGUF quants are not
    sharded).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, quant_type: int):
        self._quant_type = quant_type
        self.qweight = torch.empty(
            num_embeddings, row_bytes(embedding_dim, quant_type), dtype=torch.uint8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.core import get_global_ctx
        from freetoken.layers.gguf import fused_mul_mat_gguf

        batch = get_global_ctx().batch
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(batch.size)
            x = x[indices].contiguous()
        return fused_mul_mat_gguf(x, self.qweight, self._quant_type)


def convert_qwen35moe_to_gguf(model, config: ModelConfig) -> None:
    """In place: replace the projections the checkpoint stores quantized (per the
    ``gguf_types`` map) with native GGUF ops; groups left dense in the checkpoint keep
    their bf16 linears (the loader yields dequantized ``.weight`` for them).

    Quant slots: token embedding + lm head (always), attention qkv/o, GDN in_proj/out,
    shared-expert gate_up/down (each ``None`` -> stays dense). Routed experts never live
    in the model -- they are served from the offload cache.
    """
    from freetoken.layers.gguf import GGUFEmbedding, GGUFLinear

    assert is_gguf_model(config), "convert_qwen35moe_to_gguf on a non-GGUF config"
    T = config.gguf_types
    H = config.hidden_size
    I_sh = config.shared_expert_intermediate_size
    inner = model.model

    inner.embed_tokens = GGUFEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=H,
        quant_type=T["embed"],
    )

    n_q, hd = config.num_qo_heads, config.head_dim
    g = next(
        (gr for gr in config.attention_groups if isinstance(gr, LinearGatedDeltaGroupConfig)),
        None,
    )
    assert g is not None
    conv_dim = 2 * g.num_key_heads * g.key_head_dim + g.num_value_heads * g.value_head_dim
    value_dim = g.num_value_heads * g.value_head_dim
    in_proj_dim = conv_dim + value_dim + 2 * g.num_value_heads

    for layer in inner.layers.op_list:
        if layer._is_linear:
            if T["gdn_in"] is not None:
                layer.linear_attn.in_proj = GGUFLinear(H, in_proj_dim, T["gdn_in"])
            if T["gdn_out"] is not None:
                layer.linear_attn.out_proj = GGUFLinear(value_dim, H, T["gdn_out"])
        else:
            if T["attn_qkv"] is not None:
                qkv_dim = n_q * hd * 2 + 2 * config.num_kv_heads * hd
                layer.self_attn.qkv_proj = GGUFLinear(H, qkv_dim, T["attn_qkv"])
            if T["attn_o"] is not None:
                layer.self_attn.o_proj = GGUFLinear(n_q * hd, H, T["attn_o"])
        if T["shared_gate_up"] is not None:
            layer.mlp.shared_expert.gate_up_proj = GGUFLinear(H, 2 * I_sh, T["shared_gate_up"])
        if T["shared_down"] is not None:
            layer.mlp.shared_expert.down_proj = GGUFLinear(I_sh, H, T["shared_down"])

    if config.tie_word_embeddings:
        from freetoken.models.gemma4.gguf import GGUFTiedLMHead

        model.lm_head = GGUFTiedLMHead(inner.embed_tokens, T["lm_head"])
    else:
        model.lm_head = GGUFParallelLMHead(config.vocab_size, H, T["lm_head"])


# --------------------------------------------------------------------------------------
# Routed-expert host banks (native GGUF quants) for the offload cache.
# --------------------------------------------------------------------------------------


def _expert_bank_specs(config: ModelConfig, quant: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    E, H, I = config.num_experts, config.hidden_size, config.moe_intermediate_size
    return {
        "gate_up": ((E, 2 * I, row_bytes(H, quant)), torch.uint8),
        "down": ((E, H, row_bytes(I, quant)), torch.uint8),
    }


def load_gguf_expert_sources(
    model_path: str,
    config: ModelConfig,
    *,
    quant: int,
    layer_sink=None,
) -> dict[str, list[torch.Tensor]]:
    """Per-layer host banks of the routed experts' native GGUF block bytes.

    ``gate_up`` is one ``[E, 2I, rb(H)]`` tensor per layer (``ffn_gate_exps`` rows in
    ``[:, :I]``, ``ffn_up_exps`` rows in ``[:, I:]``) and ``down`` one ``[E, H, rb(I)]``
    per layer -- independent :class:`HostBank` allocations. Each expert's packed rows go
    in verbatim (no dequant); a layer is complete after 3 writes (gate + up + down), so
    the offload cache streams whole experts into the ggml MoE kernels.

    ``layer_sink=None`` (serving): pin each layer's banks as they complete via an
    internally-owned :class:`PinPipeline` (or, on a CUDA-less host, allocate the mmap
    banks but never pin -- the CPU executor reads them pageable). ``layer_sink`` given
    (converter): the completion tracker fires into it instead -- nothing is pinned and
    the returned tensors are only valid until the sink has consumed them.
    """
    from freetoken.models.gguf.reader import iter_gguf_tensors
    from freetoken.moe.host_banks import LayerCompletionTracker, PinPipeline, alloc_layer_banks

    _require_tp1("expert banks")
    L, E = config.num_layers, config.num_experts
    H, I = config.hidden_size, config.moe_intermediate_size
    h_bytes, i_bytes = row_bytes(H, quant), row_bytes(I, quant)
    hb = alloc_layer_banks(_expert_bank_specs(config, quant), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    seen: dict[str, set[int]] = {"gate": set(), "up": set(), "down": set()}

    def _load(sink) -> None:
        tracker = LayerCompletionTracker(3, hb, sink) if sink is not None else None
        for t in iter_gguf_tensors(model_path):
            if not t.name.startswith("blk."):
                continue
            layer = int(t.name.split(".")[1])
            if layer >= L:
                continue
            if t.name.endswith("ffn_gate_exps.weight"):
                assert t.ggml_type == quant, "ffn_gate_exps ggml type != requested quant"
                banks["gate_up"][layer][:, :I].copy_(t.packed().reshape(E, I, h_bytes))
                seen["gate"].add(layer)
            elif t.name.endswith("ffn_up_exps.weight"):
                assert t.ggml_type == quant, "ffn_up_exps ggml type != requested quant"
                banks["gate_up"][layer][:, I:].copy_(t.packed().reshape(E, I, h_bytes))
                seen["up"].add(layer)
            elif t.name.endswith("ffn_down_exps.weight"):
                assert t.ggml_type == quant, "ffn_down_exps ggml type != requested quant"
                banks["down"][layer].copy_(t.packed().reshape(E, H, i_bytes))
                seen["down"].add(layer)
            else:
                continue
            if tracker is not None:
                tracker.note(layer)

    if layer_sink is not None:
        _load(layer_sink)
    elif torch.cuda.is_available():
        with PinPipeline() as pins:
            _load(pins)
    else:
        _load(None)  # CUDA-less: mmap banks stay pageable, never pinned

    want = set(range(L))
    assert all(s == want for s in seen.values()), (
        f"missing expert tensors: "
        f"{[[k, sorted(want - s)] for k, s in seen.items() if s != want]}"
    )
    return banks


def dummy_gguf_expert_sources(config: ModelConfig, quant: int) -> dict[str, list[torch.Tensor]]:
    """Random native-GGUF-quant expert banks shaped like ``load_gguf_expert_sources``
    output (for kernel/executor tests that skip the actual file)."""
    from freetoken.moe.host_banks import alloc_layer_banks, pin_banks

    L = config.num_layers
    hb = alloc_layer_banks(_expert_bank_specs(config, quant), L)
    banks = {name: [b.tensor for b in hb[name]] for name in hb}
    for t in banks["gate_up"] + banks["down"]:
        t.random_(0, 256)
    if torch.cuda.is_available():
        pin_banks(hb)  # match the other dummies: pin-after-fill (no-op mmap fill on CPU-only)
    return banks


__all__ = [
    "ARCH",
    "parse_gguf_config",
    "iter_gguf_weights",
    "is_gguf_model",
    "convert_qwen35moe_to_gguf",
    "load_gguf_expert_sources",
    "dummy_gguf_expert_sources",
    "GGUFParallelLMHead",
]
