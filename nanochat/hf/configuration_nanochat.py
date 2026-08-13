"""Hugging Face configuration classes for nanochat models."""

from __future__ import annotations

from typing import Optional, Sequence

from transformers import PretrainedConfig


class GPTConfig(PretrainedConfig):
    """Configuration for the Hugging Face nanochat GPT implementation."""

    model_type = "nanochat_gpt"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        sequence_len: int = 2048,
        vocab_size: int = 32768,
        n_layer: int = 12,
        n_head: int = 6,
        n_kv_head: int = 6,
        n_embd: int = 768,
        window_pattern: str = "SSSL",
        padded_vocab_size: Optional[int] = None,
        compute_dtype: str = "float32",
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.sequence_len = int(sequence_len)
        self.vocab_size = int(vocab_size)
        self.n_layer = int(n_layer)
        self.n_head = int(n_head)
        self.n_kv_head = int(n_kv_head)
        self.n_embd = int(n_embd)
        self.window_pattern = str(window_pattern)
        self.padded_vocab_size = int(padded_vocab_size or vocab_size)
        self.compute_dtype = str(compute_dtype).removeprefix("torch.")

        if self.sequence_len <= 0 or self.vocab_size <= 0 or self.n_layer <= 0:
            raise ValueError("sequence_len, vocab_size, and n_layer must be positive")
        if self.n_embd <= 0 or self.n_head <= 0 or self.n_embd % self.n_head:
            raise ValueError("n_embd must be positive and divisible by n_head")
        if self.n_kv_head <= 0 or self.n_kv_head > self.n_head or self.n_head % self.n_kv_head:
            raise ValueError("n_kv_head must divide n_head and cannot exceed it")
        if self.padded_vocab_size < self.vocab_size:
            raise ValueError("padded_vocab_size cannot be smaller than vocab_size")
        if not self.window_pattern or any(char not in "SLsl" for char in self.window_pattern):
            raise ValueError("window_pattern must be a non-empty string containing only S and L")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")

        # Standard Transformers aliases used by tooling and model introspection.
        self.max_position_embeddings = self.sequence_len
        self.hidden_size = self.n_embd
        self.num_hidden_layers = self.n_layer
        self.num_attention_heads = self.n_head
        self.num_key_value_heads = self.n_kv_head
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("use_cache", False)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


class ARMTGPTConfig(GPTConfig):
    """Configuration for the Hugging Face nanochat ARMTGPT implementation."""

    model_type = "nanochat_armt_gpt"

    def __init__(
        self,
        num_mem_tokens: int = 16,
        d_mem: int = 64,
        segment_size: int = 512,
        segment_position_mode: str = "absolute",
        correction: bool = True,
        n_heads: int = 1,
        use_denom: bool = True,
        gating: bool = False,
        freeze_mem: bool = False,
        use_sink: bool = False,
        memory_dtype: str = "float32",
        wrap_layers: Optional[Sequence[int]] = None,
        dpfp_nu: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mem_tokens = int(num_mem_tokens)
        self.d_mem = int(d_mem)
        self.segment_size = int(segment_size)
        self.segment_position_mode = str(segment_position_mode)
        self.correction = bool(correction)
        self.n_heads = int(n_heads)
        self.use_denom = bool(use_denom)
        self.gating = bool(gating)
        self.freeze_mem = bool(freeze_mem)
        self.use_sink = bool(use_sink)
        self.memory_dtype = str(memory_dtype).removeprefix("torch.")
        self.wrap_layers = None if wrap_layers is None else tuple(int(value) for value in wrap_layers)
        self.dpfp_nu = int(dpfp_nu)

        if self.num_mem_tokens <= 0 or self.segment_size <= 0:
            raise ValueError("num_mem_tokens and segment_size must be positive")
        if self.d_mem <= 0 or self.dpfp_nu <= 0:
            raise ValueError("d_mem and dpfp_nu must be positive")
        if self.segment_position_mode not in {"absolute", "reset"}:
            raise ValueError("segment_position_mode must be 'absolute' or 'reset'")
        if self.n_heads <= 0 or self.d_mem % self.n_heads or self.n_embd % self.n_heads:
            raise ValueError("d_mem and n_embd must both be divisible by ARMT n_heads")
        if self.wrap_layers is not None and len(self.wrap_layers) != self.n_layer:
            raise ValueError("wrap_layers must contain exactly n_layer entries")


__all__ = ["GPTConfig", "ARMTGPTConfig"]
