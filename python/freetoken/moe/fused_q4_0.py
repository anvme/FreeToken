"""Native GGUF Q4_0 expert GEMM -- now a thin alias of the quant-parameter kernel."""

from __future__ import annotations

import torch

from freetoken.moe.fused_gguf_q import fused_experts_gguf


def fused_experts_gguf_q4_0(
    hidden_states: torch.Tensor,
    gate_up_q: torch.Tensor,
    down_q: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
) -> torch.Tensor:
    return fused_experts_gguf(
        hidden_states, gate_up_q, down_q, topk_weights, topk_ids, activation, "q4_0"
    )


__all__ = ["fused_experts_gguf_q4_0"]
