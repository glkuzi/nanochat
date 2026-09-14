"""Focused CPU tests for nanochat's ARMT wrapper."""

import torch

from nanochat.armt import ARMTGPT, ARMTGPTConfig
from nanochat.babilong_eval import _ensure_rotary_capacity as _ensure_babilong_rotary_capacity
from nanochat.gpt import GPT, GPTConfig


def _build_armt(**overrides):
    kwargs = dict(
        sequence_len=8,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        num_mem_tokens=2,
        d_mem=8,
        segment_size=4,
        n_heads=2,
        memory_dtype="float32",
    )
    kwargs.update(overrides)
    config = ARMTGPTConfig(**kwargs)
    with torch.device("meta"):
        model = ARMTGPT(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_forward_loss_and_backward_match_nanochat_api():
    torch.manual_seed(1234)
    model = _build_armt()
    idx = torch.randint(0, model.config.vocab_size, (2, 3))
    targets = torch.randint(0, model.config.vocab_size, (2, 3))

    logits = model(idx)
    loss = model(idx, targets)
    token_losses = model(idx, targets, loss_reduction="none")

    assert logits.shape == (2, 3, model.config.vocab_size)
    assert loss.ndim == 0
    assert token_losses.shape == (6,)
    loss.backward()
    assert model.memory.grad is not None
    assert model.armt_layers[0].W_mv.weight.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert model.num_scaling_params()["total"] == sum(p.numel() for p in model.parameters())


def test_plain_gpt_weights_and_cache_fallback():
    base_config = GPTConfig(
        sequence_len=8,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    with torch.device("meta"):
        base = GPT(base_config, pad_vocab_size_to=1)
    base.to_empty(device="cpu")
    base.init_weights()
    model = ARMTGPT.from_gpt(
        base,
        num_mem_tokens=2,
        d_mem=8,
        segment_size=4,
        n_heads=2,
        wrap_layers=(1, 1),
    )
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])

    class Cache:
        def __init__(self):
            self.prev_embedding = None
            self.cache_seqlens = torch.zeros(1, dtype=torch.int32)

    prompt = torch.randint(0, base_config.vocab_size, (1, 5))
    next_token = torch.randint(0, base_config.vocab_size, (1, 1))
    cache = Cache()
    assert model(prompt, kv_cache=cache).shape == (1, 5, base_config.vocab_size)
    decode_logits = model(next_token, kv_cache=cache)
    full_logits = model(torch.cat((prompt, next_token), dim=1))
    torch.testing.assert_close(decode_logits[:, -1], full_logits[:, -1])
    assert cache.cache_seqlens.item() == 6


def test_padded_mixed_length_forward_matches_independent_rows():
    torch.manual_seed(4321)
    model = _build_armt(sequence_len=16, use_sink=True)
    lengths = torch.tensor([3, 7])
    idx = torch.randint(0, model.config.vocab_size, (2, 7))

    with torch.inference_mode():
        batched_logits = model.forward_padded(idx, lengths)
        for row, length in enumerate(lengths.tolist()):
            independent_logits = model(idx[row : row + 1, :length])
            torch.testing.assert_close(
                batched_logits[row, :length],
                independent_logits[0],
                atol=1e-5,
                rtol=1e-4,
            )


def test_loaded_context_length_can_shrink_or_grow_without_nan():
    torch.manual_seed(9876)
    model = _build_armt(sequence_len=16, segment_size=4, num_mem_tokens=2)

    model.set_sequence_len(8)
    assert model.config.sequence_len == 8
    assert model.window_sizes[-1] == (8, 0)
    x = torch.randint(0, model.config.vocab_size, (2, 8), dtype=torch.int32)
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    assert torch.isfinite(model(x, targets))

    # Absolute ARMT positions at this length need 201 positions, well beyond
    # the initial planned-context cache, so this exercises dynamic growth.
    model.set_sequence_len(133)
    assert model.config.sequence_len == 133
    assert model.cos.size(1) >= model._required_rotary_positions(133)
    x = torch.randint(0, model.config.vocab_size, (1, 17), dtype=torch.int32)
    targets = torch.randint(0, model.config.vocab_size, (1, 17))
    assert torch.isfinite(model(x, targets))


def test_forward_dynamically_grows_beyond_checkpoint_context_and_rope_cache():
    torch.manual_seed(2468)
    model = _build_armt(sequence_len=4, segment_size=4, num_mem_tokens=2)
    original_cache_size = model.cos.size(1)
    sequence_len = 33
    assert model._required_rotary_positions(sequence_len) > original_cache_size

    x = torch.randint(0, model.config.vocab_size, (1, sequence_len))
    targets = torch.randint(0, model.config.vocab_size, (1, sequence_len))
    loss = model(x, targets)

    assert torch.isfinite(loss)
    assert model.config.sequence_len == 4
    assert model.cos.size(1) >= model._required_rotary_positions(sequence_len)

    padded_model = _build_armt(sequence_len=4, segment_size=4, num_mem_tokens=2)
    padded_model.set_sequence_len(9)
    train_x = torch.randint(0, padded_model.config.vocab_size, (1, 9))
    train_targets = torch.randint(0, padded_model.config.vocab_size, (1, 9))
    compiled_model = torch.compile(padded_model, backend="aot_eager", dynamic=False)
    initial_loss = compiled_model(train_x, train_targets)
    initial_loss.backward()
    padded_model.zero_grad(set_to_none=True)

    lengths = torch.tensor([sequence_len, 17])
    padded_x = torch.randint(
        0, padded_model.config.vocab_size, (2, sequence_len)
    )
    padded_model.eval()
    with torch.inference_mode():
        _ensure_babilong_rotary_capacity(padded_model, sequence_len)
        logits = padded_model.forward_padded(padded_x, lengths)

    assert torch.isfinite(logits).all()
    assert not torch.is_inference(padded_model.cos)
    assert not torch.is_inference(padded_model.sin)
    assert padded_model.cos.size(1) >= padded_model._required_rotary_positions(
        sequence_len
    )

    # Match babilong_train: resume through the already-compiled training graph
    # and reuse the cache that BABILong enlarged during inference evaluation.
    padded_model.train()
    resumed_loss = compiled_model(train_x, train_targets)
    resumed_loss.backward()

    assert torch.isfinite(resumed_loss)
    assert padded_model.transformer.h[0].attn.c_q.weight.grad is not None
