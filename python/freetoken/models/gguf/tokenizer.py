"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from freetoken.utils import init_logger

from .reader import gguf_architecture, load_gguf_metadata

logger = init_logger(__name__)

# GGUF architecture -> transformers GGUF tokenizer-converter key.
_TOKENIZER_ARCH = {"gemma4": "gemma4_text"}

# tokenizer.ggml.token_type values (llama.cpp llama_token_attr).
_GGML_TOKEN_CONTROL = 3
_GGML_TOKEN_USER_DEFINED = 4

# Arch names newer than the installed transformers (whose GGUF_TO_FAST_CONVERTERS
# predates them) -> converter keys whose construction is compatible (GPT2 vocab +
# pre-tokenizer/normalizer read from the GGUF metadata), tried in order. freetoken
# re-sets bos/eos/unk/pad from tokenizer.ggml.* below, so the Qwen-family converters
# are interchangeable for serving.
_CONVERTER_FALLBACKS: dict[str, list[str]] = {
    "qwen35moe": ["qwen35moe", "qwen3moe", "qwen3", "qwen2"],
}


def _converter_key(arch: str) -> str:
    from transformers.integrations.ggml import GGUF_TO_FAST_CONVERTERS

    chain = _CONVERTER_FALLBACKS.get(arch, [arch])
    for key in chain:
        if key in GGUF_TO_FAST_CONVERTERS:
            return key
    raise ValueError(
        f"no GGUF tokenizer converter in the installed transformers for {arch!r} "
        f"(tried {chain}; available: {sorted(GGUF_TO_FAST_CONVERTERS)})"
    )


def _register_ggml_added_tokens(fast, tok_dict: dict[str, Any]) -> tuple[int, int]:
    """Teach the fast tokenizer about every marker token the GGUF declares.

    transformers' Qwen GGUF converters hardcode a three-token special list
    (``<|endoftext|>``, ``<|im_start|>``, ``<|im_end|>``). Every *other* marker the
    chat template emits is therefore unknown to the tokenizer and gets BPE-split into
    ordinary pieces -- on Ornith, ``<think>`` became ``[13314, 741, 29]`` instead of
    ``[248068]``, so the assistant turn began with three tokens the model has never
    seen in that position (its training data always starts the turn with ``<think>``).

    The GGUF already carries the right classification per token, so use it instead of
    a hardcoded list:

    * ``CONTROL`` -> special, i.e. stripped by ``skip_special_tokens`` (``<|im_start|>``,
      the vision/audio markers).
    * ``USER_DEFINED`` -> added but *not* special, matching Qwen's own
      ``tokenizer_config.json``: ``</think>`` and ``</tool_call>`` must survive
      detokenization or the reasoning / tool-call parsers never see them.

    Returns ``(n_control, n_user_defined)``. Every one of these strings is already in
    ``tokenizer.ggml.tokens``, so registering them maps to existing ids rather than
    minting new ones -- asserted here, since a grown vocab would silently shift the
    embedding table out of alignment with the checkpoint.
    """
    from tokenizers import AddedToken

    tokens = tok_dict["tokens"]
    types = tok_dict.get("token_type") or []
    by_attr: dict[int, list[str]] = {_GGML_TOKEN_CONTROL: [], _GGML_TOKEN_USER_DEFINED: []}
    for tok, attr in zip(tokens, types):
        bucket = by_attr.get(int(attr))
        if bucket is not None:
            bucket.append(tok)

    before = fast.get_vocab_size(with_added_tokens=True)
    control = by_attr[_GGML_TOKEN_CONTROL]
    user = by_attr[_GGML_TOKEN_USER_DEFINED]
    if control:
        fast.add_special_tokens([AddedToken(t, normalized=False, special=True) for t in control])
    if user:
        fast.add_tokens([AddedToken(t, normalized=False, special=False) for t in user])
    after = fast.get_vocab_size(with_added_tokens=True)
    assert after == before, (
        f"registering the GGUF's control/user-defined tokens grew the vocab "
        f"{before} -> {after}: some are absent from tokenizer.ggml.tokens"
    )
    return len(control), len(user)


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(_converter_key(conv_arch), tok_dict)
    n_control, n_user = _register_ggml_added_tokens(fast, tok_dict)
    logger.info_rank0(
        f"GGUF tokenizer: registered {n_control} control + {n_user} user-defined tokens "
        "from tokenizer.ggml.token_type"
    )

    tokens = tok_dict["tokens"]

    def tok_for(id_key: str, default: str) -> str:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        return tokens[int(tid)] if tid is not None and int(tid) < len(tokens) else default

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal <eos> plus the chat turn end <turn|>."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id).
    for name in ("<eos>", "<turn|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
