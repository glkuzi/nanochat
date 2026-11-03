"""Convert nanochat's tokenizer.pkl to a standard Hugging Face fast tokenizer.

Write it into the directory produced by ``scripts.convert_to_hf`` so model and
tokenizer can be uploaded together::

    python -m scripts.convert_tokenizer_to_hf \
        --tokenizer-dir "$NANOCHAT_BASE_DIR/tokenizer" \
        --output-dir /path/to/hf-checkpoint

Only unpickle tokenizer files that you trust. Pickle can execute arbitrary code.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

try:
    import tiktoken
    from huggingface_hub import HfApi
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode
except ImportError as exc:  # pragma: no cover - CLI dependency error
    raise SystemExit(
        "Tokenizer conversion requires transformers, huggingface-hub, tokenizers, and tiktoken. "
        "Install them with: pip install 'transformers>=4.48,<6' tiktoken"
    ) from exc


DEFAULT_BOS_TOKEN = "<|bos|>"
DEFAULT_EOS_TOKEN = "<|assistant_end|>"
DEFAULT_PAD_TOKEN = "<|bos|>"


def resolve_pickle_path(tokenizer_path: Path) -> Path:
    tokenizer_path = tokenizer_path.expanduser().resolve()
    path = tokenizer_path / "tokenizer.pkl" if tokenizer_path.is_dir() else tokenizer_path
    if not path.is_file():
        raise FileNotFoundError(f"Native tokenizer pickle not found: {path}")
    return path


def _validate_native_ids(encoding: tiktoken.Encoding, source: Path) -> None:
    merge_ids = list(encoding._mergeable_ranks.values())
    special_ids = list(encoding._special_tokens.values())
    expected_merge_ids = set(range(len(merge_ids)))
    expected_special_ids = set(range(len(merge_ids), encoding.n_vocab))
    if set(merge_ids) != expected_merge_ids or len(set(merge_ids)) != len(merge_ids):
        raise ValueError(f"{source} mergeable-token IDs are not contiguous from zero")
    if set(special_ids) != expected_special_ids or len(set(special_ids)) != len(special_ids):
        raise ValueError(
            f"{source} special-token IDs must immediately follow mergeable-token IDs"
        )


def load_native_encoding(tokenizer_path: Path) -> tiktoken.Encoding:
    path = resolve_pickle_path(tokenizer_path)
    with path.open("rb") as handle:
        encoding = pickle.load(handle)
    if not isinstance(encoding, tiktoken.Encoding):
        raise TypeError(f"{path} contains {type(encoding).__name__}, expected tiktoken.Encoding")
    _validate_native_ids(encoding, path)
    return encoding


def _require_special(specials: dict[str, int], token: str, role: str) -> None:
    if token not in specials:
        available = ", ".join(sorted(specials)) or "none"
        raise ValueError(
            f"Requested {role} token {token!r} is not native. Available: {available}"
        )


def detect_model_max_length(output_dir: Path, requested: int | None) -> int:
    if requested is not None:
        if requested <= 0:
            raise ValueError("model_max_length must be positive")
        return requested
    config_path = output_dir / "config.json"
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        value = config.get("sequence_len", config.get("max_position_embeddings"))
        if value is not None:
            return int(value)
    return 2048


def _token_bytes_to_string(token: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[value] for value in token)


def build_backend_tokenizer(encoding: tiktoken.Encoding) -> Tokenizer:
    """Reproduce Transformers' tiktoken converter without its blobfile dependency."""
    ranks = dict(encoding._mergeable_ranks)
    byte_encoder = bytes_to_unicode()
    vocab = {
        _token_bytes_to_string(token, byte_encoder): rank for token, rank in ranks.items()
    }
    ranked_merges: list[tuple[bytes, bytes, int]] = []
    for token, rank in ranks.items():
        if len(token) == 1:
            continue
        candidates = []
        for index in range(1, len(token)):
            left, right = token[:index], token[index:]
            if left in ranks and right in ranks:
                candidates.append((left, right, rank))
        candidates.sort(key=lambda value: (ranks[value[0]], ranks[value[1]]))
        ranked_merges.extend(candidates)
    ranked_merges.sort(key=lambda value: value[2])
    merges = [
        (
            _token_bytes_to_string(left, byte_encoder),
            _token_bytes_to_string(right, byte_encoder),
        )
        for left, right, _ in ranked_merges
    ]

    backend = Tokenizer(models.BPE(vocab, merges, fuse_unk=False))
    if hasattr(backend.model, "ignore_merges"):
        backend.model.ignore_merges = True
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(encoding._pat_str), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    backend.decoder = decoders.ByteLevel()
    ordered_specials = [
        token for token, _ in sorted(encoding._special_tokens.items(), key=lambda item: item[1])
    ]
    backend.add_special_tokens(
        [AddedToken(token, normalized=False, special=True) for token in ordered_specials]
    )
    backend.post_processor = processors.ByteLevel(trim_offsets=False)
    return backend


def build_hf_tokenizer(
    encoding: tiktoken.Encoding,
    output_dir: Path,
    *,
    bos_token: str = DEFAULT_BOS_TOKEN,
    eos_token: str = DEFAULT_EOS_TOKEN,
    pad_token: str = DEFAULT_PAD_TOKEN,
    model_max_length: int | None = None,
) -> PreTrainedTokenizerFast:
    specials = dict(encoding._special_tokens)
    for role, token in (("BOS", bos_token), ("EOS", eos_token), ("PAD", pad_token)):
        _require_special(specials, token, role)
    ordered_specials = [token for token, _ in sorted(specials.items(), key=lambda item: item[1])]
    roles = {bos_token, eos_token, pad_token}
    return PreTrainedTokenizerFast(
        tokenizer_object=build_backend_tokenizer(encoding),
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        additional_special_tokens=[token for token in ordered_specials if token not in roles],
        model_max_length=detect_model_max_length(output_dir, model_max_length),
        clean_up_tokenization_spaces=False,
        padding_side="right",
        truncation_side="right",
    )


def verify_tokenizer(encoding: tiktoken.Encoding, tokenizer: PreTrainedTokenizerFast) -> None:
    if len(tokenizer) != encoding.n_vocab:
        raise RuntimeError(f"Vocabulary mismatch: native={encoding.n_vocab}, HF={len(tokenizer)}")
    for token, expected_id in encoding._special_tokens.items():
        actual_id = tokenizer.convert_tokens_to_ids(token)
        if actual_id != expected_id:
            raise RuntimeError(
                f"Special ID mismatch for {token!r}: native={expected_id}, HF={actual_id}"
            )
    probes = (
        "", "hello world", " Numbers 0123456789 and punctuation: !?.,\n",
        "naïve café — русский текст — 你好 🙂", "def f(x):\n    return x + 1\n",
        "tabs\tspaces   and CRLF\r\n",
    )
    for text in probes:
        native_ids = encoding.encode_ordinary(text)
        hf_ids = tokenizer.encode(text, add_special_tokens=False)
        if hf_ids != native_ids:
            raise RuntimeError(f"Encoding mismatch for {text!r}: native={native_ids}, HF={hf_ids}")
        native_text = encoding.decode(native_ids)
        hf_text = tokenizer.decode(
            hf_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if hf_text != native_text:
            raise RuntimeError(
                f"Decoding mismatch for {text!r}: native={native_text!r}, HF={hf_text!r}"
            )


def update_model_token_ids(output_dir: Path, tokenizer: PreTrainedTokenizerFast) -> list[Path]:
    updated = []
    for filename in ("config.json", "generation_config.json"):
        path = output_dir / filename
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            config: dict[str, Any] = json.load(handle)
        if filename == "config.json" and "vocab_size" in config:
            if int(config["vocab_size"]) != len(tokenizer):
                raise ValueError(
                    f"Model vocabulary ({config['vocab_size']}) does not match tokenizer "
                    f"vocabulary ({len(tokenizer)}) in {path}"
                )
        config.update(
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        with path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        updated.append(path)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-dir", "--tokenizer-path", dest="tokenizer_path",
        type=Path, required=True, help="Native tokenizer directory or tokenizer.pkl path",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="HF output directory; may be an existing converted model directory",
    )
    parser.add_argument("--bos-token", default=DEFAULT_BOS_TOKEN)
    parser.add_argument("--eos-token", default=DEFAULT_EOS_TOKEN)
    parser.add_argument("--pad-token", default=DEFAULT_PAD_TOKEN)
    parser.add_argument(
        "--model-max-length", type=int,
        help="Maximum real-token length (default: config.json, otherwise 2048)",
    )
    parser.add_argument(
        "--no-update-model-config", action="store_true",
        help="Do not add BOS/EOS/PAD IDs to existing model/generation configs",
    )
    parser.add_argument(
        "--push-to-hub", metavar="OWNER/REPOSITORY",
        help="Create/update this Hub model repository with tokenizer files",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--hub-token", help="Defaults to cached login or HF_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoding = load_native_encoding(args.tokenizer_path)
    tokenizer = build_hf_tokenizer(
        encoding, output_dir, bos_token=args.bos_token, eos_token=args.eos_token,
        pad_token=args.pad_token, model_max_length=args.model_max_length,
    )
    verify_tokenizer(encoding, tokenizer)
    saved_files = [Path(path) for path in tokenizer.save_pretrained(output_dir)]
    updated_configs = [] if args.no_update_model_config else update_model_token_ids(output_dir, tokenizer)

    # Validate the exact on-disk artifacts through the public AutoTokenizer API.
    reloaded = AutoTokenizer.from_pretrained(output_dir, local_files_only=True, use_fast=True)
    verify_tokenizer(encoding, reloaded)
    print(
        f"Converted nanochat tokenizer ({encoding.n_vocab:,} tokens) to {output_dir}; "
        f"BOS={reloaded.bos_token_id}, EOS={reloaded.eos_token_id}, "
        f"PAD={reloaded.pad_token_id}, max_length={reloaded.model_max_length}"
    )

    if args.push_to_hub:
        api = HfApi(token=args.hub_token)
        api.create_repo(
            repo_id=args.push_to_hub, repo_type="model", private=args.private, exist_ok=True
        )
        upload_names = {path.name for path in saved_files + updated_configs}
        api.upload_folder(
            repo_id=args.push_to_hub,
            repo_type="model",
            folder_path=str(output_dir),
            allow_patterns=sorted(upload_names),
            commit_message="Upload converted nanochat tokenizer",
        )
        print(f"Uploaded tokenizer to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
