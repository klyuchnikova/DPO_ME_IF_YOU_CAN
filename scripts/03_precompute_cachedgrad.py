#!/usr/bin/env python3
"""
Precompute cached-gradient token weights for CachedGrad-DPO.

Run after you have processed train/val JSONL files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from scripts._common import add_common_args
from src.attribution import precompute_cached_grad_weights, save_cached_grad_weights
from src.dataset import preference_collate_fn
from src.utils import ensure_dir, load_yaml, resolve_device, resolve_dtype, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()
    device = resolve_device(args.device)
    dtype = resolve_dtype(device, cfg.get("precision", "auto"))

    # Load base model WITHOUT LoRA for reference attribution
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    ref_model.config.use_cache = False
    ref_model.to(device)
    ref_model.eval()

    split_paths = {
        "train": args.train_path or cfg["train_path"],
        "val": args.val_path or cfg.get("val_path"),
        "test": args.test_path or cfg.get("test_path"),
    }
    path = split_paths[args.split]
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    from src.dataset import PreferenceDataset, load_preference_jsonl

    examples = load_preference_jsonl(path)
    dataset = PreferenceDataset(
        examples,
        tokenizer,
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: preference_collate_fn(b, tokenizer.pad_token_id),
    )

    logger.info("Precomputing cached grad weights for %d examples (%s)", len(dataset), args.split)
    weights = precompute_cached_grad_weights(ref_model, loader, device)
    out = args.output or f"cache/{Path(model_name).name}/{args.split}_cachedgrad_weights.pt"
    ensure_dir(Path(out).parent)
    save_cached_grad_weights(weights, out)
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
