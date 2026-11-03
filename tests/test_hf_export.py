"""Regression tests for native-to-Hugging Face checkpoint conversion."""

from dataclasses import asdict
import json

import pytest
import torch

transformers = pytest.importorskip("transformers")

from nanochat.armt import ARMTGPT as NativeARMTGPT
from nanochat.armt import ARMTGPTConfig as NativeARMTGPTConfig
from nanochat.gpt import GPT as NativeGPT
from nanochat.gpt import GPTConfig as NativeGPTConfig
from scripts.convert_to_hf import build_hf_model


def _native_model(model_cls, config):
    with torch.device("meta"):
        model = model_cls(config, pad_vocab_size_to=1)
    model.to_empty(device="cpu")
    model.init_weights()
    return model.eval()


def _activate_armt_memory(model):
    # Native initialization makes ARMT insertion neutral. Simulate a trained
    # checkpoint so recurrence parity exercises actual associative reads/writes.
    with torch.no_grad():
        for wrapped, controller in zip(model.wrap_layers, model.armt_layers):
            if wrapped:
                controller.W_mv.weight.normal_(mean=0.0, std=0.02)
    return model


@pytest.mark.parametrize("architecture", ["gpt", "armtgpt"])
def test_native_parity_and_remote_code_roundtrip(tmp_path, architecture):
    common = dict(
        sequence_len=16,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    if architecture == "gpt":
        native_config = NativeGPTConfig(**common)
        native = _native_model(NativeGPT, native_config)
    else:
        native_config = NativeARMTGPTConfig(
            **common,
            num_mem_tokens=2,
            d_mem=8,
            segment_size=4,
            n_heads=2,
            use_sink=True,
            memory_dtype="float32",
        )
        native = _activate_armt_memory(_native_model(NativeARMTGPT, native_config))

    state_dict = native.state_dict()
    model, detected = build_hf_model(state_dict, {"model_config": asdict(native_config)})
    assert detected == architecture
    input_ids = torch.randint(0, native_config.vocab_size, (2, 7))
    with torch.inference_mode():
        expected = native(input_ids)
        actual = model(input_ids).logits
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    model.save_pretrained(tmp_path)
    config_data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert "AutoConfig" in config_data["auto_map"]
    assert "AutoModelForCausalLM" in config_data["auto_map"]
    assert (tmp_path / "configuration_nanochat.py").is_file()
    assert (tmp_path / "modeling_nanochat.py").is_file()

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True
    )
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(input_ids).logits, actual)
    generated = reloaded.generate(
        input_ids[:1], max_new_tokens=3, do_sample=False, use_cache=True
    )
    assert generated.shape == (1, 10)


def test_armt_right_padded_batch_matches_independent_rows():
    config = NativeARMTGPTConfig(
        sequence_len=16,
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
        use_sink=True,
        memory_dtype="float32",
    )
    native = _activate_armt_memory(_native_model(NativeARMTGPT, config))
    model, _ = build_hf_model(native.state_dict(), {"model_config": asdict(config)})
    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]]
    )
    with torch.inference_mode():
        batched = model(input_ids, attention_mask=attention_mask).logits
        independent = model(input_ids[:1, :3]).logits
    torch.testing.assert_close(batched[0, :3], independent[0], atol=2e-5, rtol=2e-5)

    left_padded_ids = torch.cat((torch.zeros(1, 4, dtype=torch.long), input_ids[:1, :3]), dim=1)
    left_padded_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1]])
    with torch.inference_mode():
        left_padded = model(left_padded_ids, attention_mask=left_padded_mask).logits
    torch.testing.assert_close(left_padded[0, -3:], independent[0], atol=2e-5, rtol=2e-5)

    generated = model.generate(
        left_padded_ids,
        attention_mask=left_padded_mask,
        max_new_tokens=2,
        do_sample=False,
        use_cache=False,
        pad_token_id=0,
    )
    assert generated.shape == (1, 9)


@pytest.mark.parametrize("architecture", ["gpt", "armtgpt"])
@pytest.mark.parametrize("num_beams", [1, 2])
def test_cached_generation_matches_uncached_scores_and_tokens(architecture, num_beams):
    common = dict(
        sequence_len=32,
        vocab_size=128,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )
    if architecture == "gpt":
        native_config = NativeGPTConfig(**common)
        native = _native_model(NativeGPT, native_config)
    else:
        native_config = NativeARMTGPTConfig(
            **common,
            num_mem_tokens=2,
            d_mem=8,
            segment_size=4,
            n_heads=2,
            use_sink=True,
            memory_dtype="float32",
        )
        native = _activate_armt_memory(_native_model(NativeARMTGPT, native_config))

    model, _ = build_hf_model(native.state_dict(), {"model_config": asdict(native_config)})
    input_ids = torch.randint(0, native_config.vocab_size, (1, 10))
    generation_kwargs = dict(
        max_new_tokens=6,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        num_beams=num_beams,
    )
    with torch.inference_mode():
        uncached = model.generate(input_ids, use_cache=False, **generation_kwargs)
        cached = model.generate(input_ids, use_cache=True, **generation_kwargs)

    torch.testing.assert_close(cached.sequences, uncached.sequences)
    assert len(cached.scores) == len(uncached.scores)
    for cached_score, uncached_score in zip(cached.scores, uncached.scores):
        torch.testing.assert_close(cached_score, uncached_score, atol=3e-5, rtol=3e-5)


def test_armt_cache_avoids_reprocessing_completed_segments():
    config = NativeARMTGPTConfig(
        sequence_len=32,
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
        use_sink=True,
        memory_dtype="float32",
    )
    native = _activate_armt_memory(_native_model(NativeARMTGPT, config))
    model, _ = build_hf_model(native.state_dict(), {"model_config": asdict(config)})
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    calls = 0

    def count_block_calls(module, args, output):
        del module, args, output
        nonlocal calls
        calls += 1

    handles = [block.register_forward_hook(count_block_calls) for block in model.transformer["h"]]
    try:
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=False)
        uncached_calls = calls
        calls = 0
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=True)
        cached_calls = calls
    finally:
        for handle in handles:
            handle.remove()

    assert cached_calls < uncached_calls
