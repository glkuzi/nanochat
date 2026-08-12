"""Focused CPU tests for nanochat's ARMT wrapper."""

import torch

from nanochat.armt import ARMTGPT, ARMTGPTConfig
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
