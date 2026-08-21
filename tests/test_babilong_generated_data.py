"""Tests for online bAbI + PG-19 BABILong train/validation data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nanochat.armt import ARMTGPT, ARMTGPTConfig
from nanochat.gpt import GPT, GPTConfig
from nanochat.babilong_generated_data import (
    BabiSample,
    GeneratedBABILongBatchStream,
    SentenceSampler,
    encode_generated_babilong_example,
    generated_sample_sizes,
    load_babi_task_file,
)


class WordTokenizer:
    def __init__(self):
        self.bos = 255
        self.vocab = {}

    def get_bos_token_id(self):
        return self.bos

    def encode(self, text):
        if isinstance(text, list):
            return [self.encode(value) for value in text]
        result = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1
            result.append(self.vocab[word])
        return result

    __call__ = encode


def _write_task_files(tmp_path, task="qa1_single-supporting-fact"):
    content = (
        "1 Mary moved to the kitchen.\n"
        "2 John went to the hallway.\n"
        "3 Where is Mary?\tkitchen\t1\n"
        "4 Daniel travelled to the garden.\n"
        "5 Where is John?\thallway\t2\n"
        "1 Sandra journeyed to the office.\n"
        "2 Where is Sandra?\toffice\t1\n"
    )
    for split in ("train", "test"):
        (tmp_path / f"{task}_{split}.txt").write_text(content, encoding="utf-8")


def _noise_rows():
    sentence = "one two three four five six seven eight nine ten eleven twelve."
    return [{"text": "boundary. " + " ".join([sentence] * 30) + " ending."}]


def _split_sentences(text):
    return [part.strip() for part in text.split(".") if part.strip()]


def test_babi_parser_splits_multi_question_stories(tmp_path):
    _write_task_files(tmp_path)
    samples = load_babi_task_file(
        str(tmp_path / "qa1_single-supporting-fact_train.txt")
    )

    assert len(samples) == 3
    assert samples[0].answer == "kitchen"
    assert samples[0].references == ("Mary moved to the kitchen.",)
    assert samples[1].facts == (
        "Mary moved to the kitchen.",
        "John went to the hallway.",
        "Daniel travelled to the garden.",
    )
    assert all("Where" not in fact for fact in samples[1].facts)


def test_segment_multiple_length_budgets_match_reference_strategy():
    assert generated_sample_sizes(128, 32, vary_lengths=True) == (32, 64, 96, 128)
    assert generated_sample_sizes(130, 32, vary_lengths=True) == (32, 64, 96, 128, 130)
    assert generated_sample_sizes(128, 32, vary_lengths=False) == (128,)


def test_generated_encoding_masks_everything_except_answer_and_terminator():
    tokenizer = WordTokenizer()
    noise_sampler = SentenceSampler(
        _noise_rows(),
        tokenizer,
        min_sentence_len=1,
        shuffle=True,
        random_seed=7,
        sentence_splitter=_split_sentences,
    )
    sample = BabiSample(
        facts=("Mary moved to the kitchen.",),
        question="Where is Mary?",
        answer="kitchen",
        references=("Mary moved to the kitchen.",),
    )
    encoded = encode_generated_babilong_example(
        sample,
        tokenizer,
        noise_sampler,
        desired_sequence_len=64,
        rng=np.random.default_rng(9),
        qa_margin=12,
        oversize_policy="error",
    )
    assert encoded is not None
    inputs, targets = encoded
    supervised = [value for value in targets if value >= 0]
    assert supervised == tokenizer.encode("kitchen") + [tokenizer.get_bos_token_id()]
    assert len(inputs) <= 64


def test_generated_stream_is_reproducible_and_validation_is_fixed_length(tmp_path):
    _write_task_files(tmp_path)
    tokenizer = WordTokenizer()

    def build(split):
        return GeneratedBABILongBatchStream(
            tokenizer,
            batch_size=2,
            max_sequence_length=64,
            segment_size=32,
            device="cpu",
            tasks=("qa1",),
            babi_path=str(tmp_path),
            split=split,
            vary_lengths=split == "train",
            mixed_length_ratio=0.0,
            qa_margin=12,
            min_sentence_len=1,
            seed=123,
            noise_dataset=_noise_rows(),
            sentence_splitter=_split_sentences,
            oversize_policy="error",
            pin_memory=False,
        )

    first = build("train")
    second = build("train")
    for _ in range(4):
        x_first, y_first = next(first)
        x_second, y_second = next(second)
        torch.testing.assert_close(x_first, x_second, rtol=0, atol=0)
        torch.testing.assert_close(y_first, y_second, rtol=0, atol=0)
        assert x_first.shape[1] in {32, 64}

    validation = build("validation")
    assert validation.sample_sizes == (64,)
    x_validation, _ = next(validation)
    assert x_validation.shape == (2, 64)


@pytest.mark.parametrize("model_type", ("gpt", "armt"))
def test_generated_batch_trains_gpt_and_armt(model_type, tmp_path, monkeypatch):
    import nanochat.flash_attention as flash_attention

    monkeypatch.setattr(flash_attention, "USE_FA3", False)
    _write_task_files(tmp_path)
    tokenizer = WordTokenizer()
    stream = GeneratedBABILongBatchStream(
        tokenizer,
        batch_size=2,
        max_sequence_length=64,
        segment_size=32,
        device="cpu",
        tasks=("qa1",),
        babi_path=str(tmp_path),
        split="train",
        vary_lengths=False,
        qa_margin=12,
        min_sentence_len=1,
        seed=42,
        noise_dataset=_noise_rows(),
        sentence_splitter=_split_sentences,
        oversize_policy="error",
        pin_memory=False,
    )
    common = dict(
        sequence_len=64,
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
            **common,
            num_mem_tokens=4,
            d_mem=8,
            segment_size=32,
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
