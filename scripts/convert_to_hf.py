"""Convert a native nanochat GPT/ARMTGPT checkpoint to Hugging Face format.

The output is self-contained remote code: upload the entire output directory,
then load it with ``AutoModelForCausalLM.from_pretrained(...,
trust_remote_code=True)``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

try:
    from huggingface_hub import HfApi
    from nanochat.hf import ARMTGPT, ARMTGPTConfig, GPT, GPTConfig
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "Hugging Face conversion requires transformers and huggingface-hub. "
        "Install them with: pip install 'transformers>=4.48,<6' safetensors"
    ) from exc


MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def find_step(checkpoint_dir: Path) -> int:
    steps = []
    for path in checkpoint_dir.iterdir():
        match = MODEL_RE.fullmatch(path.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No model_<step>.pt files found in {checkpoint_dir}")
    return max(steps)


def load_native_checkpoint(checkpoint_dir: Path, step: int):
    model_path = checkpoint_dir / f"model_{step:06d}.pt"
    meta_path = checkpoint_dir / f"meta_{step:06d}.json"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only
        state_dict = torch.load(model_path, map_location="cpu")
    if not isinstance(state_dict, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise TypeError(f"{model_path} does not contain a tensor state dict")
    state_dict = {
        key.removeprefix("_orig_mod."): value.detach().cpu()
        for key, value in state_dict.items()
    }
    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if "model_config" not in metadata:
        raise KeyError(f"{meta_path} has no model_config entry")
    return state_dict, metadata


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def build_hf_model(state_dict: dict[str, torch.Tensor], metadata: dict[str, Any]):
    config_kwargs = dict(metadata["model_config"])
    # Match nanochat.checkpoint_manager's legacy compatibility behavior.
    config_kwargs.setdefault("window_pattern", "L")
    n_layer = int(config_kwargs["n_layer"])
    state_dict.setdefault("resid_lambdas", torch.ones(n_layer))
    state_dict.setdefault("x0_lambdas", torch.zeros(n_layer))

    embedding_key = "transformer.wte.weight"
    lm_head_key = "lm_head.weight"
    if embedding_key not in state_dict or lm_head_key not in state_dict:
        raise KeyError("Checkpoint is missing transformer.wte.weight or lm_head.weight")
    padded_vocab_size = state_dict[embedding_key].shape[0]
    if state_dict[lm_head_key].shape[0] != padded_vocab_size:
        raise ValueError("Token embedding and lm_head padded vocabulary sizes do not match")
    if padded_vocab_size < int(config_kwargs["vocab_size"]):
        raise ValueError("Checkpoint padded vocabulary is smaller than configured vocab_size")
    config_kwargs["padded_vocab_size"] = padded_vocab_size
    config_kwargs["compute_dtype"] = _dtype_name(state_dict[embedding_key].dtype)

    is_armt = "num_mem_tokens" in config_kwargs
    config_cls = ARMTGPTConfig if is_armt else GPTConfig
    model_cls = ARMTGPT if is_armt else GPT
    config = config_cls(**config_kwargs)
    config.architectures = [model_cls.__name__]
    config_cls.register_for_auto_class()
    model_cls.register_for_auto_class("AutoModelForCausalLM")
    model = model_cls(config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, "armtgpt" if is_armt else "gpt"


def write_model_card(output_dir: Path, architecture: str, step: int) -> None:
    title = "nanochat ARMTGPT" if architecture == "armtgpt" else "nanochat GPT"
    text = f"""---
library_name: transformers
tags:
- nanochat
- text-generation
---

# {title}

Converted from native nanochat checkpoint step `{step}`. This repository uses
custom Transformers code, so loading it requires explicitly trusting the code:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "OWNER/REPOSITORY",
    trust_remote_code=True,
)
output = model.generate(input_ids, max_new_tokens=32, use_cache=False)
```

Review remote model code before enabling `trust_remote_code` for repositories
you do not control. The nanochat tokenizer is not included by this converter.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing model_<step>.pt and meta_<step>.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, help="Checkpoint step (default: latest)")
    parser.add_argument(
        "--max-shard-size",
        default="5GB",
        help="Maximum Hugging Face weight shard size (default: 5GB)",
    )
    parser.add_argument(
        "--no-safe-serialization",
        action="store_true",
        help="Write pytorch_model.bin instead of safetensors",
    )
    parser.add_argument(
        "--push-to-hub",
        metavar="OWNER/REPOSITORY",
        help="Create/update this Hub model repository after conversion",
    )
    parser.add_argument("--private", action="store_true", help="Create a private Hub repository")
    parser.add_argument("--hub-token", help="Hub token (defaults to the cached login or HF_TOKEN)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    step = find_step(checkpoint_dir) if args.step is None else args.step
    state_dict, metadata = load_native_checkpoint(checkpoint_dir, step)
    model, architecture = build_hf_model(state_dict, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
    )
    with (output_dir / "nanochat_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    write_model_card(output_dir, architecture, step)

    required_code = {"configuration_nanochat.py", "modeling_nanochat.py"}
    missing_code = sorted(name for name in required_code if not (output_dir / name).is_file())
    if missing_code:
        raise RuntimeError(
            "save_pretrained did not copy the remote model implementation: "
            + ", ".join(missing_code)
        )
    print(f"Converted {architecture.upper()} checkpoint step {step} to {output_dir}")

    if args.push_to_hub:
        api = HfApi(token=args.hub_token)
        api.create_repo(
            repo_id=args.push_to_hub,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.push_to_hub,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message=f"Upload converted nanochat {architecture} checkpoint step {step}",
        )
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
