"""Shared utilities: config, device, seeding, I/O."""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("cached_token_dpo")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None = None) -> torch.device:
    if requested in (None, "", "auto"):
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cpu":
        return torch.device("cpu")

    if not torch.cuda.is_available():
        logging.getLogger("geognn").warning(
            "Device '%s' requested but CUDA is not available. Using CPU.",
            requested,
        )
        return torch.device("cpu")

    try:
        # Critical fix for V100 + CUDA 12.1 + cuDNN 9.x
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        x = torch.randn(2, 3, device="cuda")
        gru = torch.nn.GRU(3, 4, batch_first=True).cuda()
        gru(x)
        torch.cuda.synchronize()

        # Re-enable for convolutions if needed
        # torch.backends.cudnn.enabled = True
        return torch.device(requested)
    except Exception as exc:
        logging.getLogger("geognn").warning(
            "Device '%s' is not usable (%s). Falling back to CPU.",
            requested,
            exc,
        )
        return torch.device("cpu")


def resolve_dtype(device: torch.device, precision: str = "auto") -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    # auto: fp16 on CUDA (V100-friendly), fp32 on CPU
    return torch.float16 if device.type == "cuda" else torch.float32


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
