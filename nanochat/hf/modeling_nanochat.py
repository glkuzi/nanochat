"""Self-contained Hugging Face models for nanochat GPT and ARMTGPT.

This module intentionally depends only on PyTorch and Transformers so it can be
stored in a Hub model repository and loaded with ``trust_remote_code=True``.
Parameter names match the native nanochat implementations, allowing a strict
state-dict transfer during conversion.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_nanochat import ARMTGPTConfig, GPTConfig


_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    key = str(value).lower().removeprefix("torch.")
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {value!r}")
    return _DTYPES[key]


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Bias-free linear layer with nanochat's activation-dtype weight cast."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias)


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


def _causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    window: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=device)
    mask = positions.view(-1, 1) >= positions.view(1, -1)
    if window >= 0:
        mask &= (positions.view(-1, 1) - positions.view(1, -1)) <= window
    mask = mask.view(1, 1, seq_len, seq_len).expand(batch_size, -1, -1, -1)
    if attention_mask is not None:
        valid = attention_mask.to(device=device, dtype=torch.bool)
        mask = mask & valid[:, None, :, None] & valid[:, None, None, :]
    return mask


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).view(batch_size, seq_len, self.n_head, self.head_dim)
        k = self.c_k(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(batch_size, seq_len, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(batch_size, seq_len, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = 1.2 * norm(apply_rotary_emb(q, cos, sin))
        k = 1.2 * norm(apply_rotary_emb(k, cos, sin))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.n_head != self.n_kv_head:
            repeats = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)
        if attention_mask is not None and attention_mask.ndim == 4:
            mask = attention_mask.to(device=x.device, dtype=torch.bool)
        else:
            mask = _causal_attention_mask(
                attention_mask, batch_size, seq_len, window_size[0], x.device
            )
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos_sin: tuple[torch.Tensor, torch.Tensor],
        window_size: tuple[int, int],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), ve, cos_sin, window_size, attention_mask)
        return x + self.mlp(norm(x))


class NanoChatPreTrainedModel(PreTrainedModel):
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.8)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.transformer["wte"]

    def set_input_embeddings(self, value):
        self.transformer["wte"] = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # No KV cache is used: recomputing the full prefix is slower but exactly
        # preserves smear, sliding-window, and ARMT recurrence semantics.
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    @staticmethod
    def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.shape != logits.shape[:2]:
            raise ValueError(
                f"labels must have shape {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
            )
        return F.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )

    def _format_output(
        self,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor],
        hidden_states: Optional[torch.Tensor],
        return_dict: bool,
        output_hidden_states: bool,
    ):
        loss = None if labels is None else self._causal_lm_loss(logits, labels)
        all_hidden = (hidden_states,) if output_hidden_states and hidden_states is not None else None
        if not return_dict:
            values = (logits,)
            if all_hidden is not None:
                values += (all_hidden,)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=all_hidden,
            attentions=None,
        )


class GPT(NanoChatPreTrainedModel, GenerationMixin):
    """Hugging Face causal language model equivalent of ``nanochat.gpt.GPT``."""

    config_class = GPTConfig

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        compute_dtype = _resolve_dtype(config.compute_dtype)
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.padded_vocab_size, config.n_embd, dtype=compute_dtype),
                "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            }
        )
        self.lm_head = Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(config.padded_vocab_size, kv_dim, dtype=compute_dtype)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        self.post_init()

    @staticmethod
    def _compute_window_sizes(config: GPTConfig) -> list[tuple[int, int]]:
        short_window = -(-config.sequence_len // 4 // 128) * 128
        mapping = {"L": (config.sequence_len, 0), "S": (short_window, 0)}
        pattern = config.window_pattern.upper()
        windows = [mapping[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        windows[-1] = (config.sequence_len, 0)
        return windows

    def _cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = self.config.n_embd // self.config.n_head
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device)
        inv_freq = 1.0 / (100000 ** (channel_range / head_dim))
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.view(1, 1, -1)
        return freqs.cos().to(dtype).unsqueeze(2), freqs.sin().to(dtype).unsqueeze(2)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (batch, time), got {tuple(input_ids.shape)}")
        batch_size, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("nanochat GPT cannot process an empty sequence")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if position_ids is None:
            if attention_mask is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).view(1, -1)
                position_ids = position_ids.expand(batch_size, -1)
            else:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.clamp_min(0)
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")
        if int(position_ids.max().item()) >= self.config.sequence_len * 10:
            raise ValueError("position_ids exceed nanochat's 10x rotary-position budget")

        x = norm(self.transformer["wte"](input_ids))
        valid = None if attention_mask is None else attention_mask.to(x.dtype).unsqueeze(-1)
        if valid is not None:
            x = x * valid
        if seq_len > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            smeared = x[:, 1:] + gate * x[:, :-1]
            x = torch.cat((x[:, :1], smeared), dim=1)
            if valid is not None:
                x = x * valid
        cos_sin = self._cos_sin(position_ids, x.dtype)

        x0 = x
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None
            if ve is not None and valid is not None:
                ve = ve * valid
            x = block(x, ve, cos_sin, self.window_sizes[i], attention_mask)
            if valid is not None:
                x = x * valid
            if i == self.config.n_layer // 2:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return norm(x)

    def _logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)[..., : self.config.vocab_size].float()
        return 15 * torch.tanh(logits / 15)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del past_key_values, use_cache, kwargs
        if inputs_embeds is not None:
            raise ValueError("nanochat value embeddings require input_ids; inputs_embeds is unsupported")
        if input_ids is None:
            raise ValueError("input_ids is required")
        if output_attentions:
            raise NotImplementedError("output_attentions is not implemented")
        output_hidden_states = bool(output_hidden_states)
        return_dict = self.config.return_dict if return_dict is None else return_dict
        hidden = self._forward_hidden(input_ids, attention_mask, position_ids)
        return self._format_output(
            self._logits(hidden), labels, hidden, return_dict, output_hidden_states
        )


def _dpfp(x: torch.Tensor, nu: int) -> torch.Tensor:
    x = torch.cat((F.relu(x), F.relu(-x)), dim=-1)
    rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)
    return torch.cat([x] * nu, dim=-1) * rolled


class AssociativeMemoryController(nn.Module):
    def __init__(self, config: ARMTGPTConfig) -> None:
        super().__init__()
        self.d_model = config.n_embd
        self.num_mem_tokens = config.num_mem_tokens
        self.d_mem = config.d_mem
        self.segment_size = config.segment_size
        self.n_heads = config.n_heads
        self.correction = config.correction
        self.use_denom = config.use_denom
        self.gating = config.gating
        self.use_sink = config.use_sink
        self.memory_dtype = _resolve_dtype(config.memory_dtype)
        self.dpfp_nu = config.dpfp_nu
        self.d_key = 2 * config.dpfp_nu * config.d_mem
        self.W_mq = Linear(config.n_embd, config.d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mk = Linear(config.n_embd, config.d_mem, bias=False, dtype=self.memory_dtype)
        self.W_mv = Linear(config.n_embd, config.n_embd, bias=False, dtype=self.memory_dtype)
        self.W_mb = Linear(
            config.n_embd,
            config.n_embd if config.gating else config.n_heads,
            bias=False,
            dtype=self.memory_dtype,
        )

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        return x.view(batch_size, seq_len, self.n_heads, width // self.n_heads).permute(0, 2, 1, 3)

    @staticmethod
    def _from_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, width = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, n_heads * width)

    def _initial_memory(self, batch_size: int, device: torch.device):
        memory = torch.zeros(
            batch_size,
            self.n_heads,
            self.d_key // self.n_heads,
            self.d_model // self.n_heads,
            device=device,
            dtype=self.memory_dtype,
        )
        denominator = None
        if self.use_denom:
            denominator = torch.zeros(
                batch_size,
                self.n_heads,
                self.d_key // self.n_heads,
                device=device,
                dtype=self.memory_dtype,
            )
        return memory, denominator

    def _associate(self, hidden_states, memory, denominator):
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.memory_dtype)
        mq = F.normalize(_dpfp(self._to_heads(self.W_mq(hidden_states)), self.dpfp_nu), dim=-1)
        result = torch.einsum("bhsk,bhkd->bhsd", mq, memory)
        if denominator is not None:
            result = result / (torch.einsum("bhk,bhsk->bhs", denominator, mq).unsqueeze(-1) + 1e-5)
        return self._from_heads(result).to(original_dtype)

    def _update(self, mem_tokens, memory, denominator, first_segment: bool):
        mem_tokens = mem_tokens.to(self.memory_dtype)
        mk = F.normalize(_dpfp(self._to_heads(self.W_mk(mem_tokens)), self.dpfp_nu), dim=-1)
        new_mv = self._to_heads(self.W_mv(mem_tokens))
        if first_segment:
            previous_mv = torch.zeros_like(new_mv)
            new_info_coef: torch.Tensor | float = 1.0
        else:
            previous_mv = torch.einsum("bhmk,bhkd->bhmd", mk, memory)
            if denominator is not None:
                divisor = torch.einsum("bhk,bhmk->bhm", denominator, mk).unsqueeze(-1) + 1e-5
                previous_mv = previous_mv / divisor
                if self.correction:
                    feature_norm_sq = torch.linalg.vector_norm(mk, dim=-1).square().unsqueeze(-1)
                    new_info_coef = (1 - divisor / feature_norm_sq.clamp_min(1e-12)).clamp(0, 1).detach()
                else:
                    new_info_coef = 1.0
            else:
                new_info_coef = 1.0
        delta = new_mv - previous_mv
        gate = self._to_heads(torch.sigmoid(self.W_mb(mem_tokens)))
        if self.gating:
            associations = torch.einsum("bhmk,bhmd,bhmd->bhkd", mk, delta, gate)
        else:
            associations = torch.einsum("bhmk,bhmd,bhmx->bhkd", mk, delta, gate)
        memory = memory + associations
        if denominator is not None:
            denominator = denominator + (new_info_coef * mk).sum(dim=-2)
        return memory, denominator

    def forward(self, block, x, ve, cos_sin, window_size):
        extra_tokens = self.num_mem_tokens + int(self.use_sink)
        chunk_size = self.segment_size + extra_tokens
        x_chunks = x.split(chunk_size, dim=1)
        ve_chunks = [None] * len(x_chunks) if ve is None else ve.split(chunk_size, dim=1)
        cos_chunks = cos_sin[0].split(chunk_size, dim=1)
        sin_chunks = cos_sin[1].split(chunk_size, dim=1)
        memory, denominator = self._initial_memory(x.size(0), x.device)
        outputs = []
        for segment_index, (x_seg, ve_seg, cos_seg, sin_seg) in enumerate(
            zip(x_chunks, ve_chunks, cos_chunks, sin_chunks)
        ):
            if segment_index:
                x_seg = x_seg + self._associate(x_seg, memory, denominator)
            seg_out = block(x_seg, ve_seg, (cos_seg, sin_seg), window_size)
            mem_out = seg_out[:, -self.num_mem_tokens :, :]
            memory, denominator = self._update(
                mem_out, memory, denominator, first_segment=segment_index == 0
            )
            outputs.append(seg_out)
        return torch.cat(outputs, dim=1)

    def forward_ragged(
        self, block, x, ve, cos_sin, window_size, segment_lengths, valid_mask
    ):
        extra_tokens = self.num_mem_tokens + int(self.use_sink)
        chunk_size = self.segment_size + extra_tokens
        x_chunks = x.split(chunk_size, dim=1)
        ve_chunks = [None] * len(x_chunks) if ve is None else ve.split(chunk_size, dim=1)
        cos_chunks = cos_sin[0].split(chunk_size, dim=1)
        sin_chunks = cos_sin[1].split(chunk_size, dim=1)
        mask_chunks = valid_mask.split(chunk_size, dim=1)
        memory, denominator = self._initial_memory(x.size(0), x.device)
        outputs = []
        positions = torch.arange(chunk_size, device=x.device)
        causal = positions.view(-1, 1) >= positions.view(1, -1)
        if window_size[0] >= 0:
            causal &= (positions.view(-1, 1) - positions.view(1, -1)) <= window_size[0]
        for segment_index, (x_seg, ve_seg, cos_seg, sin_seg, valid_seg) in enumerate(
            zip(x_chunks, ve_chunks, cos_chunks, sin_chunks, mask_chunks)
        ):
            active = segment_lengths[:, segment_index] > 0
            if segment_index:
                x_seg = x_seg + self._associate(x_seg, memory, denominator) * active[:, None, None]
            segment_mask = (
                causal[None, None, : x_seg.size(1), : x_seg.size(1)]
                & valid_seg[:, None, :, None]
                & valid_seg[:, None, None, :]
            )
            seg_out = block(x_seg, ve_seg, (cos_seg, sin_seg), window_size, segment_mask)
            memory_start = int(self.use_sink) + segment_lengths[:, segment_index]
            offsets = torch.arange(self.num_mem_tokens, device=x.device)
            indices = memory_start[:, None] + offsets[None, :]
            mem_out = seg_out.gather(1, indices[:, :, None].expand(-1, -1, seg_out.size(-1)))
            new_memory, new_denominator = self._update(
                mem_out, memory, denominator, first_segment=segment_index == 0
            )
            memory = torch.where(active[:, None, None, None], new_memory, memory)
            if denominator is not None:
                denominator = torch.where(active[:, None, None], new_denominator, denominator)
            outputs.append(seg_out)
        return torch.cat(outputs, dim=1)


class ARMTGPT(GPT, GenerationMixin):
    """Hugging Face causal language model equivalent of ``nanochat.armt.ARMTGPT``."""

    config_class = ARMTGPTConfig

    def __init__(self, config: ARMTGPTConfig) -> None:
        super().__init__(config)
        self.memory_dtype = _resolve_dtype(config.memory_dtype)
        self.wrap_layers = tuple(config.wrap_layers or (1,) * config.n_layer)
        self.memory = nn.Parameter(
            torch.empty(config.num_mem_tokens, config.n_embd, dtype=self.memory_dtype)
        )
        if config.use_sink:
            self.sink = nn.Parameter(torch.empty(1, config.n_embd, dtype=self.memory_dtype))
        else:
            self.register_parameter("sink", None)
        self.armt_layers = nn.ModuleList(
            [AssociativeMemoryController(config) if wrapped else nn.Identity() for wrapped in self.wrap_layers]
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        if self.sink is not None:
            nn.init.normal_(self.sink, mean=0.0, std=0.02)
        for wrapped, controller in zip(self.wrap_layers, self.armt_layers):
            if wrapped:
                nn.init.zeros_(controller.W_mv.weight)
        if config.freeze_mem:
            for wrapped, controller in zip(self.wrap_layers, self.armt_layers):
                if wrapped:
                    controller.requires_grad_(False)

    def _augment_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        memory = self.memory.to(hidden_states.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        sink = None
        if self.sink is not None:
            sink = self.sink.to(hidden_states.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        augmented = []
        for segment in hidden_states.split(self.config.segment_size, dim=1):
            pieces = ([sink] if sink is not None else []) + [segment, memory]
            augmented.append(torch.cat(pieces, dim=1))
        return torch.cat(augmented, dim=1)

    def _augment_values(self, values: torch.Tensor) -> torch.Tensor:
        batch_size, _, width = values.shape
        memory = values.new_zeros(batch_size, self.config.num_mem_tokens, width)
        sink = values.new_zeros(batch_size, 1, width) if self.config.use_sink else None
        augmented = []
        for segment in values.split(self.config.segment_size, dim=1):
            pieces = ([sink] if sink is not None else []) + [segment, memory]
            augmented.append(torch.cat(pieces, dim=1))
        return torch.cat(augmented, dim=1)

    def _clean_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra
        cleaned = [
            segment[:, int(self.config.use_sink) : -self.config.num_mem_tokens, :]
            for segment in hidden_states.split(chunk_size, dim=1)
        ]
        return torch.cat(cleaned, dim=1)

    def _armt_cos_sin(self, real_seq_len: int, device, dtype):
        real_lengths = [
            min(self.config.segment_size, real_seq_len - start)
            for start in range(0, real_seq_len, self.config.segment_size)
        ]
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        positions = []
        absolute_start = 0
        for real_length in real_lengths:
            length = real_length + extra
            start = 0 if self.config.segment_position_mode == "reset" else absolute_start
            positions.append(torch.arange(start, start + length, device=device))
            absolute_start += length
        position_ids = torch.cat(positions).long().unsqueeze(0)
        if int(position_ids.max().item()) >= self.config.sequence_len * 10:
            raise ValueError("ARMT augmented positions exceed nanochat's 10x rotary-position budget")
        return self._cos_sin(position_ids, dtype)

    def _forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if seq_len == 0:
            raise ValueError("ARMTGPT cannot process an empty sequence")
        x = norm(self.transformer["wte"](input_ids))
        if seq_len > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x = self._augment_hidden(x)
        cos_sin = self._armt_cos_sin(seq_len, input_ids.device, x.dtype)
        x0 = x
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None
            if str(i) in self.value_embeds:
                ve = self._augment_values(self.value_embeds[str(i)](input_ids).to(x.dtype))
            if self.wrap_layers[i]:
                x = self.armt_layers[i](block, x, ve, cos_sin, self.window_sizes[i])
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i])
            if i == self.config.n_layer // 2:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return self._clean_sequence(norm(x))

    def _padded_layout(self, lengths: torch.Tensor):
        batch_size = lengths.numel()
        num_segments = math.ceil(int(lengths.max().item()) / self.config.segment_size)
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra
        starts = torch.arange(num_segments, device=lengths.device) * self.config.segment_size
        segment_lengths = (lengths[:, None] - starts[None, :]).clamp(0, self.config.segment_size)
        valid_mask = torch.zeros(batch_size, num_segments * chunk_size, dtype=torch.bool, device=lengths.device)
        position_ids = torch.zeros(batch_size, num_segments * chunk_size, dtype=torch.long, device=lengths.device)
        for segment_index in range(num_segments):
            chunk_start = segment_index * chunk_size
            for row in range(batch_size):
                real_length = int(segment_lengths[row, segment_index].item())
                if not real_length:
                    continue
                valid_length = real_length + extra
                valid_mask[row, chunk_start : chunk_start + valid_length] = True
                start = 0 if self.config.segment_position_mode == "reset" else chunk_start
                position_ids[row, chunk_start : chunk_start + valid_length] = torch.arange(
                    start, start + valid_length, device=lengths.device
                )
        return segment_lengths, valid_mask, position_ids

    def _augment_padded_hidden(self, hidden, segment_lengths):
        batch_size, _, width = hidden.shape
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra
        output = hidden.new_zeros(batch_size, segment_lengths.size(1) * chunk_size, width)
        memory = self.memory.to(hidden.dtype)
        sink = None if self.sink is None else self.sink.to(hidden.dtype)
        for segment_index in range(segment_lengths.size(1)):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size
            for row in range(batch_size):
                length = int(segment_lengths[row, segment_index].item())
                if not length:
                    continue
                cursor = chunk_start
                if sink is not None:
                    output[row, cursor : cursor + 1] = sink
                    cursor += 1
                output[row, cursor : cursor + length] = hidden[row, real_start : real_start + length]
                cursor += length
                output[row, cursor : cursor + self.config.num_mem_tokens] = memory
        return output

    def _augment_padded_values(self, values, segment_lengths):
        batch_size, _, width = values.shape
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra
        output = values.new_zeros(batch_size, segment_lengths.size(1) * chunk_size, width)
        for segment_index in range(segment_lengths.size(1)):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size + int(self.config.use_sink)
            for row in range(batch_size):
                length = int(segment_lengths[row, segment_index].item())
                if length:
                    output[row, chunk_start : chunk_start + length] = values[
                        row, real_start : real_start + length
                    ]
        return output

    def _clean_padded(self, hidden, segment_lengths, output_width):
        batch_size, _, width = hidden.shape
        extra = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra
        output = hidden.new_zeros(batch_size, output_width, width)
        for segment_index in range(segment_lengths.size(1)):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size + int(self.config.use_sink)
            for row in range(batch_size):
                length = int(segment_lengths[row, segment_index].item())
                if length:
                    output[row, real_start : real_start + length] = hidden[
                        row, chunk_start : chunk_start + length
                    ]
        return output

    def _forward_padded(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        segment_lengths, valid_mask, position_ids = self._padded_layout(lengths)
        x = norm(self.transformer["wte"](input_ids))
        if input_ids.size(1) > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x = self._augment_padded_hidden(x, segment_lengths)
        cos_sin = self._cos_sin(position_ids, x.dtype)
        x0 = x
        x_backout = None
        for i, block in enumerate(self.transformer["h"]):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None
            if str(i) in self.value_embeds:
                values = self.value_embeds[str(i)](input_ids).to(x.dtype)
                ve = self._augment_padded_values(values, segment_lengths)
            if self.wrap_layers[i]:
                x = self.armt_layers[i].forward_ragged(
                    block, x, ve, cos_sin, self.window_sizes[i], segment_lengths, valid_mask
                )
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i], valid_mask)
            if i == self.config.n_layer // 2:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return self._clean_padded(norm(x), segment_lengths, input_ids.size(1))

    def _forward_masked(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fallback for left-padded or otherwise noncontiguous valid tokens."""
        output = self.transformer["wte"].weight.new_zeros(
            input_ids.size(0), input_ids.size(1), self.config.n_embd
        )
        for row in range(input_ids.size(0)):
            row_hidden = self._forward_full(input_ids[row, mask[row]].unsqueeze(0))
            output[row, mask[row]] = row_hidden[0]
        return output

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        del position_ids, past_key_values, use_cache, kwargs
        if inputs_embeds is not None or input_ids is None:
            raise ValueError("ARMTGPT requires input_ids and does not support inputs_embeds")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, time)")
        if output_attentions:
            raise NotImplementedError("output_attentions is not implemented")
        if attention_mask is None:
            hidden = self._forward_full(input_ids)
        else:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            mask = attention_mask.to(dtype=torch.bool, device=input_ids.device)
            lengths = mask.sum(-1)
            if bool(torch.any(lengths == 0)):
                raise ValueError("Every ARMTGPT batch row must contain at least one valid token")
            right_padded = (
                torch.arange(input_ids.size(1), device=input_ids.device)[None, :]
                < lengths[:, None]
            )
            if bool(torch.all(lengths == input_ids.size(1))):
                hidden = self._forward_full(input_ids)
            elif torch.equal(mask, right_padded):
                hidden = self._forward_padded(input_ids, lengths)
            else:
                hidden = self._forward_masked(input_ids, mask)
        output_hidden_states = bool(output_hidden_states)
        return_dict = self.config.return_dict if return_dict is None else return_dict
        return self._format_output(
            self._logits(hidden), labels, hidden, return_dict, output_hidden_states
        )


__all__ = [
    "GPTConfig",
    "ARMTGPTConfig",
    "GPT",
    "ARMTGPT",
    "AssociativeMemoryController",
]
