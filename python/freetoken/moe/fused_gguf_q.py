"""Grouped expert GEMM over native GGUF quant banks (borrowed ggml MoE kernels).

Ports vLLM/sglang's ``_fused_moe_gguf`` MMVQ path onto FreeToken's offload-cache
interface: the experts are streamed to the GPU as their native packed block bytes
and dequantized *inside* ``ggml_moe_a8_vec`` -- no bf16 expert copy is materialized.
We use the MMVQ (vector) kernel for both prefill and decode: it consumes
``topk_ids`` directly (no ``moe_align_block_size`` needed) and on small batches it
is the right choice anyway. ``topk_ids`` already index the streamed cache slots
(decode) or the materialized layer positions (prefill).

One function per ggml type would repeat the same call sequence, so the quant type
is a parameter; the per-format seams are ``_BANK_SCHEMAS``, the expert-bank
providers and ``OffloadMoELayer._expert_gemm``.
"""

from __future__ import annotations

import torch

from freetoken.layers.activation import gelu_and_mul, gelu_tanh_and_mul, silu_and_mul
from freetoken.models.gguf.dequant import GGUF_FORMAT_TO_GGML

_ACT = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}


def fused_experts_gguf(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,  # [num_slots, 2I, row_bytes(H, qt)] uint8
    down_q: torch.Tensor,  # [num_slots, H, row_bytes(I, qt)] uint8
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    quant_format: str,
) -> torch.Tensor:
    act_fn = _ACT.get(activation)
    if act_fn is None:
        raise ValueError(f"unsupported MoE activation {activation!r}")
    qt = GGUF_FORMAT_TO_GGML.get(quant_format)
    if qt is None:
        raise ValueError(f"no native GGUF ggml type for expert format {quant_format!r}")

    from freetoken.kernel.gguf import ggml_moe_a8_vec

    num_tokens = hidden_states.shape[0]
    n2 = gate_up_q.shape[1]  # 2 * intermediate
    h = down_q.shape[1]  # hidden
    top_k = topk_ids.shape[1]

    # gate_up: [num_tokens*top_k, 2I] -> activation -> [num_tokens*top_k, I]
    gate_up = ggml_moe_a8_vec(hidden_states, gate_up_q, topk_ids, top_k, qt, n2, num_tokens)
    inter = act_fn(gate_up)
    # down: each of the num_tokens*top_k intermediate rows uses its own expert id.
    out = ggml_moe_a8_vec(inter, down_q, topk_ids, 1, qt, h, num_tokens * top_k)
    out = out.reshape(num_tokens, top_k, h) * topk_weights.reshape(num_tokens, top_k, 1).to(
        out.dtype
    )
    return out.sum(dim=1)


__all__ = ["fused_experts_gguf"]
