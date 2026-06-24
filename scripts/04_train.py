#!/usr/bin/env python3
"""Train DPO or weighted-DPO from a YAML config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._common import add_common_args, load_bundle_and_data
from src.dpo import WeightMethod
from src.trainer import TrainConfig, train_dpo
from src.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    need_cached = cfg.get("weight_method") == WeightMethod.CACHED_GRAD.value
    cfg, bundle, train_ds, val_ds, _, weight_method, logger = load_bundle_and_data(args, need_cached=need_cached)

    train_cfg = TrainConfig(
        output_dir=cfg["output_dir"],
        weight_method=weight_method,
        beta=cfg["beta"],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_epochs=cfg["num_epochs"],
        surprisal_w_min=cfg.get("surprisal_w_min", 0.2),
        surprisal_w_max=cfg.get("surprisal_w_max", 3.0),
        use_amp=bundle.device.type == "cuda",
        lambda_blend=cfg.get("lambda_blend", 0.7),
        gaussian_sigma_scale=cfg.get("gaussian_sigma_scale", 4.0),
        importance_update_freq=cfg.get("importance_update_freq", 10),
        importance_ema_decay=cfg.get("importance_ema_decay", 0.9),
    )
    logger.info("Training method=%s -> %s", weight_method.value, train_cfg.output_dir)
    stats = train_dpo(bundle, train_ds, val_ds, train_cfg)
    mean_loss = sum(stats.losses) / max(len(stats.losses), 1)
    logger.info("Done. steps=%d mean_loss=%.4f", stats.global_step, mean_loss)


if __name__ == "__main__":
    main()
