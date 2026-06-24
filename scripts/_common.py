#!/usr/bin/env python3
"""Shared CLI helpers for training scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attribution import load_cached_grad_weights
from src.dataset import PreferenceDataset, load_preference_jsonl
from src.dpo import WeightMethod
from src.model import load_policy_model
from src.utils import load_yaml, resolve_device, resolve_dtype, set_seed, setup_logging


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | cuda:0")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--val-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)


def load_bundle_and_data(args: argparse.Namespace, need_cached: bool = False):
    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, cfg.get("precision", "auto"))
    logger.info("device=%s dtype=%s", device, dtype)

    bundle = load_policy_model(
        cfg["model_name"],
        device=device,
        dtype=dtype,
        lora_config=cfg.get("lora"),
        gradient_checkpointing=True,
    )

    train_path = args.train_path or cfg["train_path"]
    val_path = args.val_path or cfg.get("val_path")
    test_path = args.test_path or cfg.get("test_path")

    cached = None
    cached_path = cfg.get("cached_weights_path")
    if need_cached and cached_path:
        cached = load_cached_grad_weights(cached_path)

    train_examples = load_preference_jsonl(train_path)
    train_ds = PreferenceDataset(
        train_examples,
        bundle.tokenizer,
        max_length=cfg["max_length"],
        max_prompt_length=cfg["max_prompt_length"],
        cached_weights=cached,
    )

    val_ds = test_ds = None
    if val_path and Path(val_path).exists():
        val_ds = PreferenceDataset(
            load_preference_jsonl(val_path),
            bundle.tokenizer,
            cfg["max_length"],
            cfg["max_prompt_length"],
            cached_weights=load_cached_grad_weights(cfg["val_cached_weights_path"])
            if cfg.get("val_cached_weights_path")
            else None,
        )
    if test_path and Path(test_path).exists():
        test_ds = PreferenceDataset(
            load_preference_jsonl(test_path),
            bundle.tokenizer,
            cfg["max_length"],
            cfg["max_prompt_length"],
        )

    weight_method = WeightMethod(cfg["weight_method"])
    return cfg, bundle, train_ds, val_ds, test_ds, weight_method, logger
