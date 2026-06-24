#!/usr/bin/env python3
"""Compute dataset statistics checklist (run on processed JSONL or use as Jupyter reference)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.dataset import load_preference_jsonl
from src.utils import ensure_dir, save_json, setup_logging


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="JSONL split to analyze")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--output", type=str, default="data/stats/dataset_stats.json")
    args = parser.parse_args()

    logger = setup_logging()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    examples = load_preference_jsonl(args.path)

    prompt_lens, chosen_lens, rejected_lens = [], [], []
    chosen_longer = 0
    truncated = 0
    length_ratios = []

    for ex in examples:
        pl = token_len(tokenizer, ex.prompt)
        cl = token_len(tokenizer, ex.chosen)
        rl = token_len(tokenizer, ex.rejected)
        prompt_lens.append(pl)
        chosen_lens.append(cl)
        rejected_lens.append(rl)
        if cl > rl:
            chosen_longer += 1
        if rl > 0:
            length_ratios.append(cl / rl)
        if pl > args.max_prompt_length or pl + cl > args.max_length or pl + rl > args.max_length:
            truncated += 1

    def pct(xs, p):
        xs = sorted(xs)
        if not xs:
            return 0
        k = int(round((p / 100) * (len(xs) - 1)))
        return xs[k]

    stats = {
        "num_samples": len(examples),
        "median_prompt_len": pct(prompt_lens, 50),
        "p90_prompt_len": pct(prompt_lens, 90),
        "median_chosen_len": pct(chosen_lens, 50),
        "median_rejected_len": pct(rejected_lens, 50),
        "p90_chosen_len": pct(chosen_lens, 90),
        "p90_rejected_len": pct(rejected_lens, 90),
        "chosen_longer_pct": 100.0 * chosen_longer / max(len(examples), 1),
        "mean_length_ratio": sum(length_ratios) / max(len(length_ratios), 1),
        "truncation_rate_pct": 100.0 * truncated / max(len(examples), 1),
        "duplicate_prompt_rate_pct": 100.0
        * (1 - len(set(ex.prompt for ex in examples)) / max(len(examples), 1)),
    }

    ensure_dir(Path(args.output).parent)
    save_json(args.output, stats)
    logger.info("Stats written to %s", args.output)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
