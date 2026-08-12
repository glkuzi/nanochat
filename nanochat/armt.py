"""Associative Recurrent Memory Transformer (ARMT) for nanochat's GPT.

This module adapts the inner-loop ARMT implementation from
``armt-instruct-tune/modeling_amt/inner_loop.py`` to nanochat's native GPT API.
The base transformer blocks remain in ``transformer.h`` (and therefore retain
their original state-dict names); separate controllers in ``armt_layers``
segment hidden states and perform the associative read/write around selected
blocks.

Typical construction follows nanochat's meta-device initialization pattern::

    config = ARMTGPTConfig(..., segment_size=512, num_mem_tokens=16)
    with torch.device("meta"):
        model = ARMTGPT(config)
    model.to_empty(device=device)
    model.init_weights()

``ARMTGPT.from_gpt`` can instead initialize the wrapper from an existing GPT.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import COMPUTE_DTYPE, print0
from nanochat.gpt import GPT, GPTConfig, Linear, norm
from nanochat.optim import MuonAdamW


def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    key = value.lower().removeprefix("torch.")
    dtypes = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if key not in dtypes:
        raise ValueError(f"Unsupported ARMT memory dtype: {value!r}")
    return dtypes[key]


@dataclass
class ARMTGPTConfig(GPTConfig):
    """GPT architecture settings plus nanochat-specific ARMT settings.

    ``sequence_len`` is the maximum number of real tokens; inserted memory and
    sink tokens do not consume that budget. ``n_heads`` refers to associative
    memory heads and is independent of GPT's attention ``n_head``.
    """

    num_mem_tokens: int = 16
    d_mem: int = 64
    segment_size: int = 512
    segment_position_mode: str = "absolute"
    correction: bool = True
    n_heads: int = 1
    use_denom: bool = True
    gating: bool = False
    freeze_mem: bool = False
    use_sink: bool = False
    memory_dtype: str = "float32"
    wrap_layers: Optional[tuple[int, ...]] = None
    dpfp_nu: int = 3


def _dpfp(x: torch.Tensor, nu: int) -> torch.Tensor:
    """Deterministic Parameter-Free Projection used by the ARMT memory."""
    x = torch.cat((F.relu(x), F.relu(-x)), dim=-1)
    rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu + 1)], dim=-1)
    return torch.cat([x] * nu, dim=-1) * rolled


class AssociativeMemoryController(nn.Module):
    """Run one nanochat block segment-by-segment with associative memory."""

    def __init__(
        self,
        d_model: int,
        num_mem_tokens: int,
        d_mem: int,
        segment_size: int,
        n_heads: int,
        correction: bool,
        use_denom: bool,
        gating: bool,
        use_sink: bool,
        memory_dtype: torch.dtype,
        dpfp_nu: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.segment_size = segment_size
        self.n_heads = n_heads
        self.correction = correction
        self.use_denom = use_denom
        self.gating = gating
        self.use_sink = use_sink
        self.memory_dtype = memory_dtype
        self.dpfp_nu = dpfp_nu
        self.d_key = 2 * dpfp_nu * d_mem

        # Bias-free projections fit nanochat's architecture and keep every
        # controller parameter two-dimensional (and therefore Muon-compatible).
        self.W_mq = Linear(d_model, d_mem, bias=False, dtype=memory_dtype)
        self.W_mk = Linear(d_model, d_mem, bias=False, dtype=memory_dtype)
        self.W_mv = Linear(d_model, d_model, bias=False, dtype=memory_dtype)
        mb_dim = d_model if gating else n_heads
        self.W_mb = Linear(d_model, mb_dim, bias=False, dtype=memory_dtype)

    @torch.no_grad()
    def init_weights(self) -> None:
        # nn.Linear.reset_parameters uses this initialization for weights. It
        # must be done explicitly because nanochat constructs on the meta device.
        for projection in (self.W_mq, self.W_mk, self.W_mb):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
        # A zero value projection makes insertion of ARMT initially neutral,
        # matching the reference implementation.
        nn.init.zeros_(self.W_mv.weight)

    def freeze_mem(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        if width % self.n_heads != 0:
            raise ValueError(f"Projection width {width} is not divisible by {self.n_heads} heads")
        return x.view(batch_size, seq_len, self.n_heads, width // self.n_heads).permute(0, 2, 1, 3)

    @staticmethod
    def _from_heads(x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_width = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, n_heads * head_width)

    def _initial_memory(self, batch_size: int, device: torch.device):
        W_mem = torch.zeros(
            batch_size,
            self.n_heads,
            self.d_key // self.n_heads,
            self.d_model // self.n_heads,
            device=device,
            dtype=self.memory_dtype,
        )
        z = None
        if self.use_denom:
            z = torch.zeros(
                batch_size,
                self.n_heads,
                self.d_key // self.n_heads,
                device=device,
                dtype=self.memory_dtype,
            )
        return W_mem, z

    def _associate(
        self,
        hidden_states: torch.Tensor,
        W_mem: torch.Tensor,
        z: Optional[torch.Tensor],
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.memory_dtype)
        mq = _dpfp(self._to_heads(self.W_mq(hidden_states)), self.dpfp_nu)
        mq = F.normalize(mq, dim=-1, p=2.0)
        numerator = torch.einsum("bhsk,bhkd->bhsd", mq, W_mem)
        if z is not None:
            denominator = torch.einsum("bhk,bhsk->bhs", z, mq).unsqueeze(-1) + 1e-5
            numerator = numerator / denominator
        return self._from_heads(numerator).to(original_dtype)

    def _update(
        self,
        mem_tokens: torch.Tensor,
        W_mem: torch.Tensor,
        z: Optional[torch.Tensor],
        first_segment: bool,
    ):
        mem_tokens = mem_tokens.to(self.memory_dtype)
        mk = _dpfp(self._to_heads(self.W_mk(mem_tokens)), self.dpfp_nu)
        mk = F.normalize(mk, dim=-1, p=2.0)
        new_mv = self._to_heads(self.W_mv(mem_tokens))

        if first_segment:
            previous_mv = torch.zeros_like(new_mv)
            new_info_coef: torch.Tensor | float = 1.0
        else:
            previous_mv = torch.einsum("bhmk,bhkd->bhmd", mk, W_mem)
            if z is not None:
                denominator = torch.einsum("bhk,bhmk->bhm", z, mk).unsqueeze(-1) + 1e-5
                previous_mv = previous_mv / denominator
                if self.correction:
                    feature_norm_sq = torch.linalg.vector_norm(mk, dim=-1).square().unsqueeze(-1)
                    new_info_coef = (1 - denominator / feature_norm_sq.clamp_min(1e-12)).clamp(0, 1).detach()
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
        W_mem = W_mem + associations
        if z is not None:
            z = z + (new_info_coef * mk).sum(dim=-2)
        return W_mem, z

    def forward(self, block, x, ve, cos_sin, window_size):
        extra_tokens = self.num_mem_tokens + int(self.use_sink)
        chunk_size = self.segment_size + extra_tokens
        x_chunks = x.split(chunk_size, dim=1)
        ve_chunks = [None] * len(x_chunks) if ve is None else ve.split(chunk_size, dim=1)
        cos_chunks = cos_sin[0].split(chunk_size, dim=1)
        sin_chunks = cos_sin[1].split(chunk_size, dim=1)

        W_mem, z = self._initial_memory(x.size(0), x.device)
        outputs = []
        for segment_index, (x_seg, ve_seg, cos_seg, sin_seg) in enumerate(
            zip(x_chunks, ve_chunks, cos_chunks, sin_chunks)
        ):
            if x_seg.size(1) <= extra_tokens:
                raise ValueError("Every ARMT segment must contain at least one real token")
            if segment_index > 0:
                x_seg = x_seg + self._associate(x_seg, W_mem, z)
            seg_out = block(x_seg, ve_seg, (cos_seg, sin_seg), window_size, kv_cache=None)
            mem_out = seg_out[:, -self.num_mem_tokens :, :]
            W_mem, z = self._update(mem_out, W_mem, z, first_segment=segment_index == 0)
            outputs.append(seg_out)
        output = torch.cat(outputs, dim=1)
        if len(outputs) == 1:
            # Muon assumes every grouped matrix has a gradient. On a one-segment
            # batch the recurrent state is not consumed, so anchor the otherwise
            # inactive controller parameters without changing the output.
            anchor = sum((parameter.sum() for parameter in self.parameters()), output.new_zeros(()))
            output = output + anchor.to(output.dtype) * 0
        return output

    def forward_ragged(
        self,
        block,
        x,
        ve,
        cos_sin,
        window_size,
        segment_lengths,
        valid_mask,
    ):
        """Run fixed-width padded segments while preserving per-row ARMT state."""
        extra_tokens = self.num_mem_tokens + int(self.use_sink)
        chunk_size = self.segment_size + extra_tokens
        x_chunks = x.split(chunk_size, dim=1)
        ve_chunks = [None] * len(x_chunks) if ve is None else ve.split(chunk_size, dim=1)
        cos_chunks = cos_sin[0].split(chunk_size, dim=1)
        sin_chunks = cos_sin[1].split(chunk_size, dim=1)
        mask_chunks = valid_mask.split(chunk_size, dim=1)

        W_mem, z = self._initial_memory(x.size(0), x.device)
        outputs = []
        local_positions = torch.arange(chunk_size, device=x.device)
        causal = local_positions.view(1, -1, 1) >= local_positions.view(1, 1, -1)
        window = window_size[0]
        if window >= 0:
            causal &= (
                local_positions.view(1, -1, 1) - local_positions.view(1, 1, -1)
            ) <= window

        for segment_index, (x_seg, ve_seg, cos_seg, sin_seg, valid_seg) in enumerate(
            zip(x_chunks, ve_chunks, cos_chunks, sin_chunks, mask_chunks)
        ):
            active = segment_lengths[:, segment_index] > 0
            if segment_index > 0:
                association = self._associate(x_seg, W_mem, z)
                x_seg = x_seg + association * active[:, None, None]
            attention_mask = (
                causal[:, : x_seg.size(1), : x_seg.size(1)]
                & valid_seg[:, None, :, None]
                & valid_seg[:, None, None, :]
            )
            seg_out = block(
                x_seg,
                ve_seg,
                (cos_seg, sin_seg),
                window_size,
                kv_cache=None,
                attention_mask=attention_mask,
            )

            memory_start = int(self.use_sink) + segment_lengths[:, segment_index]
            memory_offsets = torch.arange(self.num_mem_tokens, device=x.device)
            memory_indices = memory_start[:, None] + memory_offsets[None, :]
            mem_out = seg_out.gather(
                1, memory_indices[:, :, None].expand(-1, -1, seg_out.size(-1))
            )
            new_W_mem, new_z = self._update(
                mem_out, W_mem, z, first_segment=segment_index == 0
            )
            W_mem = torch.where(active[:, None, None, None], new_W_mem, W_mem)
            if z is not None:
                z = torch.where(active[:, None, None], new_z, z)
            outputs.append(seg_out)

        output = torch.cat(outputs, dim=1)
        if len(outputs) == 1:
            anchor = sum(
                (parameter.sum() for parameter in self.parameters()),
                output.new_zeros(()),
            )
            output = output + anchor.to(output.dtype) * 0
        return output


class ARMTGPT(GPT):
    """nanochat GPT with per-layer inner-loop associative memory.

    The forward signature and return values are identical to :class:`GPT`, so
    base pretraining, SFT, RL loss calls, and nanochat evaluators can call this
    model without a special batch format.
    """

    config: ARMTGPTConfig

    def __init__(self, config: ARMTGPTConfig, pad_vocab_size_to: int = 64):
        self._validate_config(config)
        super().__init__(config, pad_vocab_size_to=pad_vocab_size_to)
        self.config = config
        self.memory_dtype = _resolve_dtype(config.memory_dtype)
        self.wrap_layers = tuple(config.wrap_layers or (1,) * config.n_layer)

        self.memory = nn.Parameter(
            torch.empty(config.num_mem_tokens, config.n_embd, dtype=self.memory_dtype)
        )
        if config.use_sink:
            self.sink = nn.Parameter(torch.empty(1, config.n_embd, dtype=self.memory_dtype))
        else:
            self.register_parameter("sink", None)

        controllers = []
        for should_wrap in self.wrap_layers:
            if should_wrap:
                controllers.append(
                    AssociativeMemoryController(
                        d_model=config.n_embd,
                        num_mem_tokens=config.num_mem_tokens,
                        d_mem=config.d_mem,
                        segment_size=config.segment_size,
                        n_heads=config.n_heads,
                        correction=config.correction,
                        use_denom=config.use_denom,
                        gating=config.gating,
                        use_sink=config.use_sink,
                        memory_dtype=self.memory_dtype,
                        dpfp_nu=config.dpfp_nu,
                    )
                )
            else:
                controllers.append(nn.Identity())
        self.armt_layers = nn.ModuleList(controllers)

        #print("#################### ARMT DEBUG ######################")
        #print(self.wrap_layers)
        #print(self.armt_layers)

        if config.freeze_mem:
            for layer_index, controller in enumerate(self.armt_layers):
                if self.wrap_layers[layer_index]:
                    controller.freeze_mem()

    @staticmethod
    def _validate_config(config: ARMTGPTConfig) -> None:
        if config.num_mem_tokens <= 0:
            raise ValueError("num_mem_tokens must be positive")
        if config.segment_size <= 0:
            raise ValueError("segment_size must be positive")
        if config.d_mem <= 0 or config.dpfp_nu <= 0:
            raise ValueError("d_mem and dpfp_nu must be positive")
        if config.segment_position_mode not in {"absolute", "reset"}:
            raise ValueError("segment_position_mode must be 'absolute' or 'reset'")
        if config.n_heads <= 0 or config.d_mem % config.n_heads or config.n_embd % config.n_heads:
            raise ValueError("d_mem and n_embd must both be divisible by ARMT n_heads")
        if config.wrap_layers is not None and len(config.wrap_layers) != config.n_layer:
            raise ValueError("wrap_layers must contain exactly n_layer entries")

    @torch.no_grad()
    def init_weights(self) -> None:
        super().init_weights()
        nn.init.normal_(self.memory, mean=0.0, std=0.02)
        if self.sink is not None:
            nn.init.normal_(self.sink, mean=0.0, std=0.02)
        for layer_index, controller in enumerate(self.armt_layers):
            if self.wrap_layers[layer_index]:
                controller.init_weights()

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

    def _augment_value_embeddings(self, value_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, _, width = value_embeddings.shape
        zeros_mem = value_embeddings.new_zeros(batch_size, self.config.num_mem_tokens, width)
        zeros_sink = value_embeddings.new_zeros(batch_size, 1, width) if self.config.use_sink else None
        augmented = []
        for segment in value_embeddings.split(self.config.segment_size, dim=1):
            pieces = ([zeros_sink] if zeros_sink is not None else []) + [segment, zeros_mem]
            augmented.append(torch.cat(pieces, dim=1))
        return torch.cat(augmented, dim=1)

    def _clean_sequence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra_tokens
        cleaned = []
        for segment in hidden_states.split(chunk_size, dim=1):
            cleaned.append(segment[:, int(self.config.use_sink) : -self.config.num_mem_tokens, :])
        return torch.cat(cleaned, dim=1)

    def _padded_layout(self, sequence_lengths: torch.Tensor):
        """Build the physical layout and RoPE positions for a ragged ARMT batch."""
        batch_size = sequence_lengths.numel()
        max_real_tokens = int(sequence_lengths.max().item())
        num_segments = math.ceil(max_real_tokens / self.config.segment_size)
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra_tokens
        total_augmented = num_segments * chunk_size
        starts = torch.arange(num_segments, device=sequence_lengths.device)
        starts = starts * self.config.segment_size
        segment_lengths = (
            sequence_lengths[:, None] - starts[None, :]
        ).clamp(min=0, max=self.config.segment_size)
        valid_mask = torch.zeros(
            batch_size,
            total_augmented,
            dtype=torch.bool,
            device=sequence_lengths.device,
        )
        position_ids = torch.zeros(
            batch_size,
            total_augmented,
            dtype=torch.long,
            device=sequence_lengths.device,
        )
        for segment_index in range(num_segments):
            chunk_start = segment_index * chunk_size
            for row in range(batch_size):
                real_length = int(segment_lengths[row, segment_index].item())
                if real_length == 0:
                    continue
                valid_length = real_length + extra_tokens
                valid_mask[row, chunk_start : chunk_start + valid_length] = True
                position_start = (
                    0
                    if self.config.segment_position_mode == "reset"
                    else chunk_start
                )
                position_ids[row, chunk_start : chunk_start + valid_length] = torch.arange(
                    position_start,
                    position_start + valid_length,
                    device=sequence_lengths.device,
                )
        return segment_lengths, valid_mask, position_ids

    def _augment_padded_hidden(
        self,
        hidden_states: torch.Tensor,
        segment_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, width = hidden_states.shape
        num_segments = segment_lengths.size(1)
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra_tokens
        output = hidden_states.new_zeros(batch_size, num_segments * chunk_size, width)
        memory = self.memory.to(hidden_states.dtype)
        sink = None if self.sink is None else self.sink.to(hidden_states.dtype)
        for segment_index in range(num_segments):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size
            for row in range(batch_size):
                real_length = int(segment_lengths[row, segment_index].item())
                if real_length == 0:
                    continue
                cursor = chunk_start
                if sink is not None:
                    output[row, cursor : cursor + 1] = sink
                    cursor += 1
                output[row, cursor : cursor + real_length] = hidden_states[
                    row, real_start : real_start + real_length
                ]
                cursor += real_length
                output[row, cursor : cursor + self.config.num_mem_tokens] = memory
        return output

    def _augment_padded_values(
        self,
        value_embeddings: torch.Tensor,
        segment_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, width = value_embeddings.shape
        num_segments = segment_lengths.size(1)
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra_tokens
        output = value_embeddings.new_zeros(batch_size, num_segments * chunk_size, width)
        for segment_index in range(num_segments):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size + int(self.config.use_sink)
            for row in range(batch_size):
                real_length = int(segment_lengths[row, segment_index].item())
                if real_length:
                    output[row, chunk_start : chunk_start + real_length] = value_embeddings[
                        row, real_start : real_start + real_length
                    ]
        return output

    def _clean_padded_hidden(
        self,
        hidden_states: torch.Tensor,
        segment_lengths: torch.Tensor,
        max_real_tokens: int,
    ) -> torch.Tensor:
        batch_size, _, width = hidden_states.shape
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        chunk_size = self.config.segment_size + extra_tokens
        output = hidden_states.new_zeros(batch_size, max_real_tokens, width)
        for segment_index in range(segment_lengths.size(1)):
            real_start = segment_index * self.config.segment_size
            chunk_start = segment_index * chunk_size + int(self.config.use_sink)
            for row in range(batch_size):
                real_length = int(segment_lengths[row, segment_index].item())
                if real_length:
                    output[row, real_start : real_start + real_length] = hidden_states[
                        row, chunk_start : chunk_start + real_length
                    ]
        return output

    @staticmethod
    def _padded_causal_mask(valid_mask: torch.Tensor, window_size):
        total_length = valid_mask.size(1)
        positions = torch.arange(total_length, device=valid_mask.device)
        mask = positions.view(1, -1, 1) >= positions.view(1, 1, -1)
        window = window_size[0]
        if window >= 0:
            mask &= (
                positions.view(1, -1, 1) - positions.view(1, 1, -1)
            ) <= window
        return (
            mask[:, None]
            & valid_mask[:, None, :, None]
            & valid_mask[:, None, None, :]
        )

    def forward_padded(
        self,
        idx: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Inference forward for right-padded, mixed-length ARMT batches."""
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape (batch, time), got {tuple(idx.shape)}")
        if sequence_lengths.shape != (idx.size(0),):
            raise ValueError("sequence_lengths must contain one value per batch row")
        sequence_lengths = sequence_lengths.to(device=idx.device, dtype=torch.long)
        if torch.any(sequence_lengths <= 0) or torch.any(sequence_lengths > idx.size(1)):
            raise ValueError("sequence_lengths must be between 1 and the padded width")
        if idx.device != self.cos.device:
            raise ValueError("idx and rotary embeddings must be on the same device")

        segment_lengths, valid_mask, position_ids = self._padded_layout(sequence_lengths)
        max_position = int(position_ids.max().item()) + 1
        if max_position > self.cos.size(1):
            raise ValueError(
                f"ARMT augmented position {max_position} exceeds RoPE cache size {self.cos.size(1)}"
            )
        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if idx.size(1) > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24])
            )
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x = self._augment_padded_hidden(x, segment_lengths)
        rotary_shape = (*position_ids.shape, 1, self.cos.size(-1))
        cos_sin = (
            self.cos[0].index_select(0, position_ids.flatten()).view(rotary_shape),
            self.sin[0].index_select(0, position_ids.flatten()).view(rotary_shape),
        )

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for layer_index, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_index] * x + self.x0_lambdas[layer_index] * x0
            value_embeddings = None
            if str(layer_index) in self.value_embeds:
                values = self.value_embeds[str(layer_index)](idx).to(x.dtype)
                value_embeddings = self._augment_padded_values(values, segment_lengths)
            if self.wrap_layers[layer_index]:
                x = self.armt_layers[layer_index].forward_ragged(
                    block,
                    x,
                    value_embeddings,
                    cos_sin,
                    self.window_sizes[layer_index],
                    segment_lengths,
                    valid_mask,
                )
            else:
                attention_mask = self._padded_causal_mask(
                    valid_mask, self.window_sizes[layer_index]
                )
                x = block(
                    x,
                    value_embeddings,
                    cos_sin,
                    self.window_sizes[layer_index],
                    kv_cache=None,
                    attention_mask=attention_mask,
                )
            if layer_index == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = self._clean_padded_hidden(
            norm(x), segment_lengths, max_real_tokens=idx.size(1)
        )
        memory_anchor = self.memory.sum()
        if self.sink is not None:
            memory_anchor = memory_anchor + self.sink.sum()
        x = x + memory_anchor.to(x.dtype) * 0
        softcap = 15
        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        return softcap * torch.tanh(logits / softcap)

    def _armt_cos_sin(self, real_seq_len: int, device: torch.device):
        real_lengths = [
            min(self.config.segment_size, real_seq_len - start)
            for start in range(0, real_seq_len, self.config.segment_size)
        ]
        extra_tokens = self.config.num_mem_tokens + int(self.config.use_sink)
        positions = []
        absolute_start = 0
        for real_length in real_lengths:
            augmented_length = real_length + extra_tokens
            start = 0 if self.config.segment_position_mode == "reset" else absolute_start
            positions.append(torch.arange(start, start + augmented_length, device=device))
            absolute_start += augmented_length
        position_ids = torch.cat(positions).long()
        num_segments = len(real_lengths)
        if self.config.segment_position_mode == "reset":
            max_position = max(real_lengths) + extra_tokens
        else:
            max_position = real_seq_len + num_segments * extra_tokens
        if max_position > self.cos.size(1):
            raise ValueError(
                f"ARMT augmented position {max_position} exceeds RoPE cache size {self.cos.size(1)}; "
                "increase GPTConfig.sequence_len or reduce inserted memory tokens"
            )
        return self.cos.index_select(1, position_ids), self.sin.index_select(1, position_ids)

    def _forward_full(self, idx: torch.Tensor, targets=None, loss_reduction: str = "mean"):
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape (batch, time), got {tuple(idx.shape)}")
        batch_size, seq_len = idx.shape
        if seq_len == 0:
            raise ValueError("ARMTGPT cannot process an empty sequence")
        #if seq_len > self.config.sequence_len:
        #    raise ValueError(f"Sequence length {seq_len} exceeds configured maximum {self.config.sequence_len}")
        if idx.device != self.cos.device:
            raise ValueError(f"Rotary embeddings and idx are on different devices: {self.cos.device} and {idx.device}")
        if self.cos.dtype != COMPUTE_DTYPE:
            raise ValueError(f"Rotary embeddings must be {COMPUTE_DTYPE}, got {self.cos.dtype}")

        x = norm(self.transformer.wte(idx).to(COMPUTE_DTYPE))
        if seq_len > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat((x[:, :1], x[:, 1:] + gate * x[:, :-1]), dim=1)
        x = self._augment_hidden(x)
        cos_sin = self._armt_cos_sin(seq_len, idx.device)

        x0 = x
        backout_layer = self.config.n_layer // 2
        x_backout = None
        for layer_index, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[layer_index] * x + self.x0_lambdas[layer_index] * x0
            value_embeddings = None
            if str(layer_index) in self.value_embeds:
                value_embeddings = self.value_embeds[str(layer_index)](idx).to(x.dtype)
                value_embeddings = self._augment_value_embeddings(value_embeddings)
            if self.wrap_layers[layer_index]:
                x = self.armt_layers[layer_index](
                    block, x, value_embeddings, cos_sin, self.window_sizes[layer_index]
                )
            else:
                x = block(x, value_embeddings, cos_sin, self.window_sizes[layer_index], kv_cache=None)
            if layer_index == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = self._clean_sequence(norm(x))
        # A single segment cannot consume its appended memory tokens. Keep the
        # shared embedding in the graph so nanochat optimizer steps still see a
        # zero gradient instead of None.
        memory_anchor = self.memory.sum()
        if self.sink is not None:
            memory_anchor = memory_anchor + self.sink.sum()
        x = x + memory_anchor.to(x.dtype) * 0

        softcap = 15
        logits = self.lm_head(x)[..., : self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        if targets is None:
            return logits
        if targets.shape != idx.shape:
            raise ValueError(f"targets must match idx shape {tuple(idx.shape)}, got {tuple(targets.shape)}")
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        if kv_cache is None:
            return self._forward_full(idx, targets=targets, loss_reduction=loss_reduction)
        if targets is not None:
            raise ValueError("targets are not supported together with kv_cache")

        # Exact compatibility fallback for nanochat's Engine. ARMT recurrence is
        # not representable by the base GPT KV cache, so token history is carried
        # in its existing prev_embedding slot and the prefix is recomputed. This
        # is correct but intentionally less efficient than base GPT decoding.
        history = kv_cache.prev_embedding
        if history is not None:
            history = history.squeeze(-1).to(device=idx.device, dtype=idx.dtype)
            if history.size(0) == 1 and idx.size(0) != 1:
                history = history.expand(idx.size(0), -1)
            if history.size(0) != idx.size(0):
                raise ValueError("KV cache history batch size does not match idx")
            full_idx = torch.cat((history, idx), dim=1)
        else:
            full_idx = idx
        logits = self._forward_full(full_idx)
        kv_cache.prev_embedding = full_idx.unsqueeze(-1)
        if hasattr(kv_cache, "cache_seqlens"):
            kv_cache.cache_seqlens.fill_(full_idx.size(1))
        return logits[:, -idx.size(1) :, :]

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        armt_matrices = sum(p.numel() for p in self.armt_layers.parameters())
        memory = self.memory.numel() + (0 if self.sink is None else self.sink.numel())
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = wte + value_embeds + lm_head + transformer_matrices + armt_matrices + memory + scalars
        if total != sum(p.numel() for p in self.parameters()):
            raise RuntimeError("ARMT parameter count mismatch")
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "armt_matrices": armt_matrices,
            "memory": memory,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
    ):
        trainable = lambda params: [p for p in params if p.requires_grad]
        matrix_params = trainable(list(self.transformer.h.parameters()) + list(self.armt_layers.parameters()))
        value_embeds_params = trainable(self.value_embeds.parameters())
        embedding_params = trainable(
            list(self.transformer.wte.parameters())
            + [self.memory]
            + ([] if self.sink is None else [self.sink])
        )
        lm_head_params = trainable(self.lm_head.parameters())
        resid_params = trainable([self.resid_lambdas])
        x0_params = trainable([self.x0_lambdas])
        smear_params = trainable([self.smear_gate.weight, self.smear_lambda, self.backout_lambda])

        grouped = matrix_params + value_embeds_params + embedding_params + lm_head_params + resid_params + x0_params + smear_params
        all_trainable = [p for p in self.parameters() if p.requires_grad]
        if len(grouped) != len(all_trainable) or {id(p) for p in grouped} != {id(p) for p in all_trainable}:
            raise RuntimeError("Some trainable ARMT parameters were not assigned to an optimizer group")

        dmodel_lr_scale = (self.config.n_embd / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({self.config.n_embd}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def load_base_state_dict(self, state_dict, strict: bool = True):
        """Load a plain GPT state dict while retaining initialized ARMT weights."""
        state_dict = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        result = self.load_state_dict(state_dict, strict=False)
        allowed_missing = ("memory", "sink", "armt_layers.")
        bad_missing = [key for key in result.missing_keys if not key.startswith(allowed_missing)]
        if strict and (bad_missing or result.unexpected_keys):
            raise RuntimeError(
                f"Base GPT state dict mismatch; missing={bad_missing}, unexpected={result.unexpected_keys}"
            )
        return result

    @classmethod
    def from_gpt(cls, model: GPT, **armt_kwargs):
        """Create an initialized ARMTGPT from an existing nanochat GPT."""
        if isinstance(model, ARMTGPT):
            raise TypeError("from_gpt expects a plain GPT, not an ARMTGPT")
        config_kwargs = asdict(model.config)
        config_kwargs.update(armt_kwargs)
        config = ARMTGPTConfig(**config_kwargs)
        with torch.device("meta"):
            wrapped = cls(config)
        wrapped.to_empty(device=model.get_device())
        wrapped.init_weights()
        wrapped.load_base_state_dict(model.state_dict(), strict=True)
        wrapped.train(model.training)
        return wrapped


__all__ = ["ARMTGPTConfig", "AssociativeMemoryController", "ARMTGPT"]
