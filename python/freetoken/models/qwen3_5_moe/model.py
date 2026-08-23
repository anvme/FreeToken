from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# FREETOKEN_LAYER_PROBE=N logs the RMS of each sublayer's output for the first N
# forwards. A correct model keeps the residual stream growing smoothly and every
# sublayer contributing a comparable share; a mis-mapped weight shows up as one
# sublayer whose RMS collapses to ~0, explodes, or goes NaN, which localizes a
# garbage-output bug to a single op without bisecting the checkpoint.
_PROBE = int(os.environ.get("FREETOKEN_LAYER_PROBE", "0"))
_probe_forwards = 0


def _probing() -> bool:
    # The probe reads tensors back to host, which is illegal mid-capture, so it stays
    # off while a CUDA graph is being recorded (prefill, the interesting case, is never
    # captured anyway).
    return (
        _probe_forwards < _PROBE
        and not torch.cuda.is_current_stream_capturing()
    )


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                expert_quant=config.expert_quant,
                attn_quant=config.attn_quant,
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id)
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = Qwen3_5MoE(config, layer_id) if config.moe_enabled else Qwen3_5DenseMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        probe = _probing()
        norm_in = _rms(hidden) if probe else 0.0
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        mixer_out = _rms(hidden) if probe else 0.0
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        mlp_in = _rms(hidden) if probe else 0.0
        hidden = self.mlp.forward(hidden)
        if probe:
            print(
                f"[probe] layer {self._layer_id:>2} {'gdn ' if self._is_linear else 'attn'}"
                f"  norm_in={norm_in:9.4f}  mixer_out={mixer_out:9.4f}"
                f"  mlp_in={mlp_in:9.4f}  mlp_out={_rms(hidden):9.4f}"
                f"  residual={_rms(residual):9.4f}",
                flush=True,
            )
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3_5DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        if _probing():
            print(
                f"[probe] forward #{_probe_forwards} tokens={input_ids.shape[0]} "
                f"ids={input_ids[:16].tolist()} embed_rms={_rms(x):.4f}",
                flush=True,
            )
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        probe = _probing()
        x, _ = self.norm.forward_add_residual(x, residual)
        if probe:
            global _probe_forwards
            print(f"[probe] final_norm_rms={_rms(x):.4f}", flush=True)
            _probe_forwards += 1
        return x


class Qwen3_5MoEForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        if getattr(config, "lm_head_quant", "none") == "nvfp4":
            # checkpoint stores the (untied) lm_head as NVFP4: keep it native (W4A16) -- the
            # bf16 dequant of this ~1 GB matrix was the single largest decode kernel.
            from freetoken.kernel.triton.nvfp4_linear import Nvfp4LMHead

            assert not config.tie_word_embeddings, "NVFP4 lm_head assumes untied embeddings"
            self.lm_head = Nvfp4LMHead(
                num_embeddings=config.vocab_size, embedding_dim=config.hidden_size
            )
        else:
            self.lm_head = ParallelLMHead(
                num_embeddings=config.vocab_size,
                embedding_dim=config.hidden_size,
                tie_word_embeddings=config.tie_word_embeddings,
                tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            )
        super().__init__()
        from .gguf import convert_qwen35moe_to_gguf, is_gguf_model

        if is_gguf_model(config):
            # GGUF checkpoint: swap the quantized projections/embedding for native ggml
            # ops (must run after super().__init__ so state-dict collection sees the
            # swapped modules, cf. the gemma4 hook).
            convert_qwen35moe_to_gguf(self, config)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["Qwen3_5MoEForCausalLM"]
