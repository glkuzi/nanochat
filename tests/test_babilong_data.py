"""Tests for deterministic, answer-only BABILong training batches."""

import pytest
import torch

import nanochat.babilong_data as babilong_data

from nanochat.armt import ARMTGPT, ARMTGPTConfig
from nanochat.babilong_data import (
    BABILongBatchStream,
    encode_babilong_training_example,
    split_babilong_indices,
)
from nanochat.gpt import GPT, GPTConfig


class CharacterTokenizer:
    def __init__(self):
        self.bos = 255

    def get_bos_token_id(self):
        return self.bos

    def encode(self, text, prepend=None):
        tokens = [1 + (ord(char) % 200) for char in text]
        if prepend is not None:
            tokens.insert(0, prepend)
        return tokens

    __call__ = encode


def sample(index=0):
    return {
        "input": f"Mary moved to room {index}.",
        "question": "Where is Mary?",
        "target": f"room {index}",
    }


def test_training_encoding_masks_prompt_and_padding():
    tokenizer = CharacterTokenizer()
    encoded = encode_babilong_training_example(
        sample(), "qa1", tokenizer, sequence_len=256, oversize_policy="error"
    )
    assert encoded is not None
    inputs, targets = encoded
    assert len(inputs) == len(targets) == 256

    supervised = [index for index, target in enumerate(targets) if target >= 0]
    answer_ids = tokenizer(" " + sample()["target"])
    assert len(supervised) == len(answer_ids) + 1
    assert [targets[index] for index in supervised[:-1]] == answer_ids
    assert targets[supervised[-1]] == tokenizer.get_bos_token_id()
    assert all(target == -1 for target in targets[: supervised[0]])
    assert all(target == -1 for target in targets[supervised[-1] + 1 :])


def test_oversize_policy_can_skip_or_explain():
    tokenizer = CharacterTokenizer()
    assert (
        encode_babilong_training_example(
            sample(), "qa1", tokenizer, sequence_len=8, oversize_policy="skip"
        )
        is None
    )
    with pytest.raises(ValueError, match="needs .* tokens"):
        encode_babilong_training_example(
            sample(), "qa1", tokenizer, sequence_len=8, oversize_policy="error"
        )


def test_train_validation_split_is_deterministic_and_disjoint():
    first = split_babilong_indices(50, 7, 42, "qa1", "2k")
    second = split_babilong_indices(50, 7, 42, "qa1", "2k")
    different_seed = split_babilong_indices(50, 7, 43, "qa1", "2k")
    assert first == second
    assert first != different_seed
    train, validation = first
    assert len(train) == 43
    assert len(validation) == 7
    assert set(train).isdisjoint(validation)
    assert set(train) | set(validation) == set(range(50))


def test_batch_stream_uses_datasets_cache_by_default(tmp_path, monkeypatch):
    rows = [sample(index) for index in range(20)]
    calls = []

    def fake_load_dataset(
        dataset_name,
        *,
        name,
        split,
        cache_dir,
        keep_in_memory,
    ):
        calls.append((dataset_name, name, split, cache_dir, keep_in_memory))
        return rows

    monkeypatch.setattr(babilong_data, "load_dataset", fake_load_dataset)
    cache_dir = tmp_path / "hf-cache"
    stream = BABILongBatchStream(
        CharacterTokenizer(),
        batch_size=2,
        sequence_len=256,
        device="cpu",
        tasks="qa1",
        lengths="0k",
        dataset_name="test/babilong",
        split="train",
        validation_examples=4,
        seed=42,
        cell_batches=1,
        oversize_policy="error",
        cache_dir=str(cache_dir),
        pin_memory=False,
    )

    next(stream)
    assert calls == [("test/babilong", "0k", "qa1", str(cache_dir), False)]


def test_batches_mix_tasks_but_not_lengths(monkeypatch):
    observed = []
    loads = []

    def loader(repo_id, subset, split):
        loads.append((subset, split))
        return [
            {
                "input": f"{subset}-{split}-{index}",
                "question": "question",
                "target": "answer",
                "length": subset,
            }
            for index in range(20)
        ]

    def fake_encode(sample, task, tokenizer, sequence_len, oversize_policy):
        observed.append((task, sample["length"]))
        return [1] * sequence_len, [2] * sequence_len

    monkeypatch.setattr(
        babilong_data, "encode_babilong_training_example", fake_encode
    )
    stream = BABILongBatchStream(
        CharacterTokenizer(),
        batch_size=8,
        sequence_len=8,
        device="cpu",
        tasks="qa1-qa5",
        lengths="0k,2k",
        dataset_name="test/babilong",
        split="train",
        validation_examples=4,
        seed=42,
        cell_batches=1,
        oversize_policy="error",
        dataset_loader=loader,
        pin_memory=False,
    )

    batch_summaries = []
    for _ in range(2):
        start = len(observed)
        next(stream)
        records = observed[start:]
        batch_summaries.append(
            ({task for task, _ in records}, {length for _, length in records})
        )

    assert {next(iter(lengths)) for _, lengths in batch_summaries} == {"0k", "2k"}
    for tasks, lengths in batch_summaries:
        assert tasks == {"qa1", "qa2", "qa3", "qa4", "qa5"}
        assert len(lengths) == 1
    assert set(loads) == {
        (length, task)
        for length in ("0k", "2k")
        for task in ("qa1", "qa2", "qa3", "qa4", "qa5")
    }


def test_batch_stream_repeats_exactly_for_same_seed():
    tokenizer = CharacterTokenizer()
    rows = [sample(index) for index in range(20)]

    def loader(repo_id, subset, split):
        assert repo_id == "test/babilong"
        assert subset == "0k"
        assert split == "qa1"
        return rows

    def build_stream():
        return BABILongBatchStream(
            tokenizer,
            batch_size=2,
            sequence_len=256,
            device="cpu",
            tasks="qa1",
            lengths="0k",
            dataset_name="test/babilong",
            split="train",
            validation_examples=4,
            seed=123,
            cell_batches=2,
            oversize_policy="error",
            dataset_loader=loader,
            pin_memory=False,
        )

    stream_a = build_stream()
    stream_b = build_stream()
    for _ in range(4):
        inputs_a, targets_a = next(stream_a)
        inputs_b, targets_b = next(stream_b)
        torch.testing.assert_close(inputs_a, inputs_b, rtol=0, atol=0)
        torch.testing.assert_close(targets_a, targets_b, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", ["gpt", "armt"])
def test_masked_batch_trains_both_nanochat_models(model_type, monkeypatch):
    # The test is intentionally CPU-only even when pytest runs on a GPU host.
    import nanochat.flash_attention as flash_attention

    monkeypatch.setattr(flash_attention, "USE_FA3", False)
    tokenizer = CharacterTokenizer()
    rows = [sample(index) for index in range(20)]

    def loader(repo_id, subset, split):
        return rows

    stream = BABILongBatchStream(
        tokenizer,
        batch_size=2,
        sequence_len=256,
        device="cpu",
        tasks="qa1",
        lengths="0k",
        dataset_name="test/babilong",
        split="train",
        validation_examples=4,
        seed=42,
        cell_batches=1,
        oversize_policy="error",
        dataset_loader=loader,
        pin_memory=False,
    )
    common = dict(
        sequence_len=256,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    if model_type == "gpt":
        config = GPTConfig(**common)
        model_cls = GPT
    else:
        config = ARMTGPTConfig(
            **common, num_mem_tokens=4, d_mem=8, segment_size=64
        )
        model_cls = ARMTGPT
    with torch.device("meta"):
        model = model_cls(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()

    inputs, targets = next(stream)
    loss = model(inputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
