"""Regression tests for native-to-Hugging Face tokenizer conversion."""

import json
import pickle

import pytest

tiktoken = pytest.importorskip("tiktoken")
transformers = pytest.importorskip("transformers")

from scripts.convert_tokenizer_to_hf import (
    build_hf_tokenizer,
    load_native_encoding,
    update_model_token_ids,
    verify_tokenizer,
)


SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _native_encoding():
    ranks = {bytes([value]): value for value in range(256)}
    for token in (b"he", b"hel", b"hell", b"hello", b" w", b" wo", b" wor", b" worl", b" world"):
        ranks[token] = len(ranks)
    specials = {token: len(ranks) + index for index, token in enumerate(SPECIAL_TOKENS)}
    return tiktoken.Encoding(
        name="nanochat-hf-export-test",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=ranks,
        special_tokens=specials,
    )


def test_fast_tokenizer_roundtrip_and_model_config_update(tmp_path):
    encoding = _native_encoding()
    native_dir = tmp_path / "native"
    output_dir = tmp_path / "hf"
    native_dir.mkdir()
    output_dir.mkdir()
    with (native_dir / "tokenizer.pkl").open("wb") as handle:
        pickle.dump(encoding, handle)
    (output_dir / "config.json").write_text(
        json.dumps({"vocab_size": encoding.n_vocab, "sequence_len": 4096}),
        encoding="utf-8",
    )

    loaded_encoding = load_native_encoding(native_dir)
    tokenizer = build_hf_tokenizer(loaded_encoding, output_dir)
    verify_tokenizer(loaded_encoding, tokenizer)
    tokenizer.save_pretrained(output_dir)
    assert update_model_token_ids(output_dir, tokenizer) == [output_dir / "config.json"]

    reloaded = transformers.AutoTokenizer.from_pretrained(
        output_dir, local_files_only=True, use_fast=True
    )
    verify_tokenizer(loaded_encoding, reloaded)
    assert reloaded.encode("hello world", add_special_tokens=False) == encoding.encode_ordinary(
        "hello world"
    )
    assert reloaded.bos_token_id == encoding._special_tokens["<|bos|>"]
    assert reloaded.eos_token_id == encoding._special_tokens["<|assistant_end|>"]
    assert reloaded.pad_token_id == reloaded.bos_token_id
    assert reloaded.model_max_length == 4096

    model_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert model_config["bos_token_id"] == reloaded.bos_token_id
    assert model_config["eos_token_id"] == reloaded.eos_token_id
    assert model_config["pad_token_id"] == reloaded.pad_token_id
    assert (output_dir / "tokenizer.json").is_file()
    assert (output_dir / "tokenizer_config.json").is_file()
