"""Packed SFT sample-isolation tests for GPT and ARMT."""

import torch

import nanochat.flash_attention as attention_backend
from nanochat.armt import ARMTGPT, ARMTGPTConfig
from nanochat.gpt import GPT, GPTConfig, _packed_causal_attention_mask
from nanochat.loss_eval import evaluate_bpb


def _initialize_model(model_cls, config):
    with torch.device("meta"):
        model = model_cls(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def _build_gpt():
    config = GPTConfig(
        sequence_len=12,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    return _initialize_model(GPT, config)


def _build_armt():
    config = ARMTGPTConfig(
        sequence_len=12,
        vocab_size=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
        num_mem_tokens=2,
        d_mem=8,
        segment_size=3,
        n_heads=2,
        memory_dtype="float32",
    )
    return _initialize_model(ARMTGPT, config)


def test_packed_attention_mask_is_block_diagonal_and_causal():
    sample_ids = torch.tensor([[0, 0, 1, 1, -1]], dtype=torch.int32)
    mask = _packed_causal_attention_mask(sample_ids, (8, 0))[0, 0]

    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_gpt_packed_logits_match_independent_samples(monkeypatch):
    monkeypatch.setattr(attention_backend, "_override_impl", "sdpa")
    torch.manual_seed(123)
    model = _build_gpt().eval()
    idx = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.int32)
    sample_ids = torch.tensor([[0] * 5 + [1] * 5], dtype=torch.int32)

    with torch.no_grad():
        packed = model(idx, attention_mask=sample_ids)
        first = model(idx[:, :5])
        second = model(idx[:, 5:])

    torch.testing.assert_close(packed[:, :5], first, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(packed[:, 5:], second, atol=1e-5, rtol=1e-4)


def test_compiled_gpt_accepts_packed_attention_mask(monkeypatch):
    monkeypatch.setattr(attention_backend, "_override_impl", "sdpa")
    torch.manual_seed(321)
    model = _build_gpt()
    compiled = torch.compile(model, backend="aot_eager", dynamic=False)
    idx = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.int32)
    targets = torch.randint(0, model.config.vocab_size, (1, 10))
    sample_ids = torch.tensor([[0] * 5 + [1] * 5], dtype=torch.int32)

    loss = compiled(idx, targets, attention_mask=sample_ids)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.transformer.h[0].attn.c_q.weight.grad is not None


def test_armt_packed_logits_and_backward_match_independent_samples(monkeypatch):
    monkeypatch.setattr(attention_backend, "_override_impl", "sdpa")
    torch.manual_seed(456)
    model = _build_armt()
    idx = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.int32)
    targets = torch.randint(0, model.config.vocab_size, (1, 10))
    sample_ids = torch.tensor([[0] * 5 + [1] * 5], dtype=torch.int32)

    with torch.no_grad():
        packed = model(idx, attention_mask=sample_ids)
        first = model(idx[:, :5])
        second = model(idx[:, 5:])
        packed_losses = model(
            idx, targets, attention_mask=sample_ids, loss_reduction="none"
        ).view_as(targets)
        first_losses = model(
            idx[:, :5], targets[:, :5], loss_reduction="none"
        ).view(1, 5)
        second_losses = model(
            idx[:, 5:], targets[:, 5:], loss_reduction="none"
        ).view(1, 5)
    torch.testing.assert_close(packed[:, :5], first)
    torch.testing.assert_close(packed[:, 5:], second)
    torch.testing.assert_close(packed_losses[:, :5], first_losses)
    torch.testing.assert_close(packed_losses[:, 5:], second_losses)

    loss = model(idx, targets, attention_mask=sample_ids)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.memory.grad is not None


def test_evaluate_bpb_passes_optional_attention_mask():
    class Model:
        def __init__(self):
            self.received_attention_mask = None

        def get_device(self):
            return torch.device("cpu")

        def __call__(self, x, y, loss_reduction, attention_mask=None):
            self.received_attention_mask = attention_mask
            return torch.zeros_like(y, dtype=torch.float32).reshape(-1)

    model = Model()
    x = torch.tensor([[1, 2]], dtype=torch.int32)
    y = torch.tensor([[2, 3]])
    sample_ids = torch.tensor([[0, 0]], dtype=torch.int32)
    token_bytes = torch.ones(64, dtype=torch.int64)

    assert evaluate_bpb(model, [(x, y, sample_ids)], 1, token_bytes) == 0.0
    assert model.received_attention_mask is sample_ids
