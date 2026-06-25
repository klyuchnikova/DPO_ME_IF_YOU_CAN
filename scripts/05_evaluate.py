#!/usr/bin/env python3
"""Run all evaluation routines for one experiment.

This script combines:

1. Held-out preference accuracy / margins
2. Runtime training-step benchmark
3. Token-weight visualizations
4. Summary JSON / CSV / Markdown tables

Outputs are saved to:

    outputs/results/<experiment_name>/

Example:

    python scripts/evaluate_all.py \
        --config configs/qwen_0.5b_surprisal.yaml \
        --checkpoint outputs/checkpoints/qwen_0.5b_surprisal \
        --exp-name qwen_0.5b_surprisal \
        --split test \
        --num-runtime-steps 20 \
        --num-visualize 5

Base model eval:

    python scripts/evaluate_all.py \
        --config configs/qwen_0.5b_dpo.yaml \
        --exp-name qwen_0.5b_base \
        --split test \
        --skip-runtime

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch

from scripts._common import add_common_args, load_bundle_and_data, resolve_split_cached_weights_path
from src.attribution import compute_cached_grad_importance, load_cached_grad_weights
from src.dataset import PreferenceDataset, load_preference_jsonl, tokenize_preference_pair
from src.dpo import (
    WeightMethod,
    build_gaussian_weights,
    build_surprisal_weights,
    get_per_token_logps,
)
from src.eval import benchmark_step_time, evaluate_preference, evaluate_raw_preference
from src.model import load_base_model_bundle, load_policy_from_checkpoint, load_policy_model
from src.utils import (
    ensure_dir,
    load_json,
    load_train_stats,
    load_yaml,
    resolve_device,
    resolve_dtype,
    save_json,
    set_seed,
    setup_logging,
)


# ---------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_name(s: str) -> str:
    """Make a filesystem-safe experiment name."""
    return (
        s.replace("/", "__")
        .replace("\\", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    # Stable field order: keys from first row, then any extras.
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for k in row.keys():
            if k not in keys:
                keys.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_markdown_table(path: str | Path, rows: List[Dict[str, Any]], title: Optional[str] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")

        if not rows:
            f.write("_No rows._\n")
            return

        keys = list(rows[0].keys())
        for row in rows[1:]:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)

        f.write("| " + " | ".join(keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")

        for row in rows:
            values = []
            for k in keys:
                v = row.get(k, "")
                if isinstance(v, float):
                    values.append(f"{v:.6f}")
                else:
                    values.append(str(v))
            f.write("| " + " | ".join(values) + " |\n")


def _flatten_for_table(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one-level nested dict for compact table output."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{prefix}{k}.{kk}"] = vv
        else:
            out[f"{prefix}{k}"] = v
    return out


def _get_split_path(args: argparse.Namespace, cfg: Dict[str, Any]) -> str:
    """Resolve val/test path from CLI override or config."""
    if args.split == "val":
        return args.val_path or cfg["val_path"]
    return args.test_path or cfg["test_path"]


def _load_eval_bundle(args: argparse.Namespace, cfg: Dict[str, Any], device: str, dtype: torch.dtype):
    """Load checkpoint policy, or frozen base model when no checkpoint is given."""
    if args.checkpoint:
        return load_policy_from_checkpoint(cfg["model_name"], args.checkpoint, device, dtype)
    return load_base_model_bundle(cfg["model_name"], device, dtype)


def _is_base_eval(args: argparse.Namespace) -> bool:
    return not args.checkpoint


def _resolve_exp_name(args: argparse.Namespace, cfg: Dict[str, Any], weight_method: WeightMethod) -> str:
    if args.exp_name:
        return _safe_name(args.exp_name)

    model_part = _safe_name(cfg["model_name"].split("/")[-1])
    method_part = _safe_name(weight_method.value)

    if args.checkpoint:
        ckpt_part = _safe_name(Path(args.checkpoint).name)
        return f"{model_part}__{method_part}__{ckpt_part}"

    return f"{model_part}__{method_part}__base"


# ---------------------------------------------------------------------
# Preference evaluation
# ---------------------------------------------------------------------


def run_preference_eval(
    *,
    bundle,
    dataset: PreferenceDataset,
    weight_method: WeightMethod,
    cfg: Dict[str, Any],
    batch_size: int,
    is_base_model: bool = False,
) -> Dict[str, Any]:
    if is_base_model:
        result = evaluate_raw_preference(
            bundle,
            dataset,
            batch_size=max(1, batch_size),
        )
        return {
            "preference_accuracy": result.preference_accuracy,
            "weighted_preference_accuracy": None,
            "mean_margin": result.mean_margin,
            "median_margin": result.median_margin,
            "mean_loss": result.mean_loss,
            "num_examples": result.num_examples,
            "evaluation_method": "raw_logprob",
            "training_method": "base",
            "metric_note": "sum of response log-probs; not ref-normalized",
        }

    eval_method_name = cfg.get("evaluation_method", "uniform")
    eval_weight_method = WeightMethod(eval_method_name)

    result = evaluate_preference(
        bundle,
        dataset,
        weight_method=weight_method,
        eval_weight_method=eval_weight_method,
        beta=cfg["beta"],
        batch_size=max(1, batch_size),
        surprisal_w_min=cfg.get("surprisal_w_min", 0.2),
        surprisal_w_max=cfg.get("surprisal_w_max", 3.0),
    )

    return {
        "preference_accuracy": result.preference_accuracy,
        "weighted_preference_accuracy": result.weighted_preference_accuracy,
        "mean_margin": result.mean_margin,
        "median_margin": result.median_margin,
        "mean_loss": result.mean_loss,
        "num_examples": result.num_examples,
        "evaluation_method": eval_method_name,
        "training_method": weight_method.value,
    }


# ---------------------------------------------------------------------
# Runtime evaluation
# ---------------------------------------------------------------------


def run_runtime_eval(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    weight_method: WeightMethod,
) -> Dict[str, Any]:
    """Benchmark training-step time on train split (matches train script setup)."""
    need_cached = weight_method == WeightMethod.CACHED_GRAD
    cfg2, bundle, train_ds, _, _, wm2, logger = load_bundle_and_data(args, need_cached=need_cached)

    result = benchmark_step_time(
        bundle,
        train_ds,
        wm2,
        beta=cfg2["beta"],
        num_steps=args.num_runtime_steps,
        surprisal_w_min=cfg2.get("surprisal_w_min", 0.2),
        surprisal_w_max=cfg2.get("surprisal_w_max", 3.0),
        lambda_blend=cfg2.get("lambda_blend", 0.7),
        gaussian_sigma_scale=cfg2.get("gaussian_sigma_scale", 4.0),
        importance_update_freq=cfg2.get("importance_update_freq", 10),
        importance_ema_decay=cfg2.get("importance_ema_decay", 0.9),
    )

    result["weight_method"] = wm2.value
    result["model_name"] = cfg2["model_name"]
    result["num_runtime_steps"] = args.num_runtime_steps
    if need_cached:
        result["cached_weights_path"] = resolve_split_cached_weights_path(cfg2, "train")

    return result


# ---------------------------------------------------------------------
# Token weight visualization
# ---------------------------------------------------------------------


def _tensorize_one(tokenized: Dict[str, Any], key: str, device: str) -> torch.Tensor:
    return torch.tensor([tokenized[key]], device=device)


def _normalize_cached_importance(imp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Normalize attribution weights to mean 1 over active response tokens."""
    active = mask.bool()
    if active.sum() == 0:
        return imp * 0.0
    return imp / imp[active].mean().clamp_min(1e-8) * mask


def _plot_weights(
    *,
    tokens: List[str],
    series: Dict[str, torch.Tensor],
    title: str,
    output_png: Path,
    max_tokens: int = 160,
) -> None:
    n = min(len(tokens), max_tokens)
    for v in series.values():
        n = min(n, int(v.numel()))

    fig_width = max(12, min(30, n * 0.18))
    fig, ax = plt.subplots(figsize=(fig_width, 4))

    x = list(range(n))
    for name, values in series.items():
        values_cpu = values[:n].detach().float().cpu().numpy()
        ax.plot(x, values_cpu, label=name, alpha=0.85, linewidth=1.7)

    ax.set_title(title)
    ax.set_xlabel("response token index")
    ax.set_ylabel("normalized weight")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def _write_weight_text(
    *,
    tokens: List[str],
    series: Dict[str, torch.Tensor],
    output_txt: Path,
    max_tokens: int = 300,
) -> None:
    n = min(len(tokens), max_tokens)
    for v in series.values():
        n = min(n, int(v.numel()))

    names = list(series.keys())

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("token_index\ttoken\t" + "\t".join(names) + "\n")

        for idx in range(n):
            vals = []
            for name in names:
                vals.append(f"{float(series[name][idx].detach().float().cpu()):.6f}")
            tok = tokens[idx].replace("\n", "\\n").replace("\t", "\\t")
            f.write(f"{idx}\t{tok}\t" + "\t".join(vals) + "\n")


def run_weight_visualizations(
    *,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    bundle,
    examples,
    device: str,
    out_dir: Path,
    logger,
) -> Dict[str, Any]:
    """Create token weight plots and text files.

    Produces:
        outputs/results/<exp>/figures/weights_example_*.png
        outputs/results/<exp>/figures/weights_example_*.txt
        outputs/results/<exp>/tables/weight_examples.csv
    """
    fig_dir = _ensure_dir(out_dir / "figures")
    table_dir = _ensure_dir(out_dir / "tables")

    tokenizer = bundle.tokenizer

    cached = None
    if args.cached_weights:
        cached = load_cached_grad_weights(args.cached_weights)
    elif resolve_split_cached_weights_path(cfg, args.split):
        split_cached = resolve_split_cached_weights_path(cfg, args.split)
        if split_cached and Path(split_cached).exists():
            cached = load_cached_grad_weights(split_cached)
        else:
            raise ValueError(f"No cached grad weights found at {split_cached}")

    rows_for_table: List[Dict[str, Any]] = []

    n_examples = min(args.num_visualize, len(examples))
    selected = examples[:n_examples]

    for i, ex in enumerate(selected):
        tok_pair = tokenize_preference_pair(
            tokenizer,
            ex.prompt,
            ex.chosen,
            ex.rejected,
            cfg["max_length"],
            cfg["max_prompt_length"],
        )

        c_ids = _tensorize_one(tok_pair, "chosen_input_ids", device)
        c_attn = _tensorize_one(tok_pair, "chosen_attention_mask", device)
        c_labels = _tensorize_one(tok_pair, "chosen_labels", device)

        # get_per_token_logps usually returns length T-1 because logits are shifted.
        # mask follows the same shifted convention.
        mask = (c_labels[:, 1:] != -100).float()

        with torch.no_grad():
            ref_logps = get_per_token_logps(bundle.policy, c_ids, c_attn, c_labels, use_ref=True)

        gauss = build_gaussian_weights(mask)[0].detach().float().cpu()
        surpr = build_surprisal_weights(
            ref_logps,
            mask,
            w_min=cfg.get("surprisal_w_min", 0.2),
            w_max=cfg.get("surprisal_w_max", 3.0),
        )[0].detach().float().cpu()

        if cached is not None:
            # The old script assumes cached[i]["chosen_weights"].
            # Keep same convention.
            cgrad = torch.as_tensor(cached[i]["chosen_weights"]).detach().float().cpu()
        else:
            imp = compute_cached_grad_importance(bundle.policy, c_ids, c_attn, c_labels)[0]
            cgrad = _normalize_cached_importance(imp, mask[0]).detach().float().cpu()

        # ------------------------------------------------------------------
        # Align tokens with shifted causal-LM logprobs.
        #
        # get_per_token_logps(...) returns values for labels[:, 1:].
        # Therefore weights/mask correspond to input_ids[1:], not input_ids.
        # We select only active response positions where shifted labels != -100.
        # ------------------------------------------------------------------

        input_ids = tok_pair["chosen_input_ids"]
        labels = tok_pair["chosen_labels"]

        shifted_input_ids = input_ids[1:]
        shifted_labels = labels[1:]

        active_mask = torch.tensor(
            [lab != -100 for lab in shifted_labels],
            dtype=torch.bool,
        )

        # Move mask to CPU because gauss/surpr/cgrad are CPU tensors here.
        active_mask = active_mask[: min(active_mask.numel(), gauss.numel(), surpr.numel(), cgrad.numel())]

        aligned_input_ids = shifted_input_ids[: active_mask.numel()]
        resp_ids = [
            tid
            for tid, active in zip(aligned_input_ids, active_mask.tolist())
            if active
        ]

        tokens = tokenizer.convert_ids_to_tokens(resp_ids)

        gauss_active = gauss[: active_mask.numel()][active_mask]
        surpr_active = surpr[: active_mask.numel()][active_mask]
        cgrad_active = cgrad[: active_mask.numel()][active_mask]

        n = min(len(tokens), gauss_active.numel(), surpr_active.numel(), cgrad_active.numel())

        tokens = tokens[:n]
        series = {
            "gaussian": gauss_active[:n],
            "surprisal": surpr_active[:n],
            "cached_grad": cgrad_active[:n],
        }

        example_id = getattr(ex, "id", i)
        title = f"Example {example_id}: chosen response token weights"

        png_path = fig_dir / f"weights_example_{i}.png"
        txt_path = fig_dir / f"weights_example_{i}.txt"

        _plot_weights(
            tokens=tokens,
            series=series,
            title=title,
            output_png=png_path,
            max_tokens=args.max_plot_tokens,
        )
        _write_weight_text(
            tokens=tokens,
            series=series,
            output_txt=txt_path,
            max_tokens=args.max_text_tokens,
        )

        # Summary row for table.
        row = {
            "example_index": i,
            "example_id": example_id,
            "num_response_tokens": n,
            "prompt_preview": ex.prompt[:160].replace("\n", " "),
            "chosen_preview": ex.chosen[:160].replace("\n", " "),
            "png": str(png_path.relative_to(out_dir)),
            "txt": str(txt_path.relative_to(out_dir)),
        }

        for name, values in series.items():
            values = values[:n]
            if n > 0:
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_max"] = float(values.max())
                row[f"{name}_std"] = float(values.std(unbiased=False))
            else:
                row[f"{name}_mean"] = 0.0
                row[f"{name}_max"] = 0.0
                row[f"{name}_std"] = 0.0

        rows_for_table.append(row)

        logger.info("Saved weight visualization %s", png_path)

    _write_csv(table_dir / "weight_examples.csv", rows_for_table)
    _write_markdown_table(
        table_dir / "weight_examples.md",
        rows_for_table,
        title="Token Weight Visualization Examples",
    )

    return {
        "num_visualized": n_examples,
        "figure_dir": str(fig_dir),
        "table_csv": str(table_dir / "weight_examples.csv"),
    }


# ---------------------------------------------------------------------
# Combined summary outputs
# ---------------------------------------------------------------------


def _load_precompute_stats(cfg: Dict[str, Any], split: str = "train") -> Optional[Dict[str, Any]]:
    weights_path = resolve_split_cached_weights_path(cfg, split)
    if not weights_path:
        return None
    stats_path = Path(weights_path).with_suffix(".stats.json")
    if not stats_path.is_file():
        return None
    stats = load_json(stats_path)
    stats["stats_path"] = str(stats_path)
    return stats


def write_combined_outputs(
    *,
    out_dir: Path,
    metadata: Dict[str, Any],
    preference_result: Optional[Dict[str, Any]],
    runtime_result: Optional[Dict[str, Any]],
    training_result: Optional[Dict[str, Any]],
    precompute_result: Optional[Dict[str, Any]],
    visualization_result: Optional[Dict[str, Any]],
) -> None:
    tables_dir = _ensure_dir(out_dir / "tables")

    full = {
        "metadata": metadata,
        "preference": preference_result,
        "runtime": runtime_result,
        "training": training_result,
        "precompute": precompute_result,
        "visualization": visualization_result,
    }

    _write_json(out_dir / "summary.json", full)

    # Compact machine-readable table.
    row = {}
    row.update(_flatten_for_table("", metadata))

    if preference_result:
        row.update(_flatten_for_table("preference.", preference_result))

    if runtime_result:
        # Avoid nested/unexpected types.
        for k, v in runtime_result.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"runtime.{k}"] = v

    if training_result:
        for k, v in training_result.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"training.{k}"] = v

    if precompute_result:
        for k, v in precompute_result.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[f"precompute.{k}"] = v

    if visualization_result:
        row.update(_flatten_for_table("visualization.", visualization_result))

    _write_csv(tables_dir / "summary.csv", [row])
    _write_markdown_table(tables_dir / "summary.md", [row], title="Experiment Summary")

    # Separate pretty preference/runtime tables.
    if preference_result:
        pref_row = {"metric": "value"}
        pref_rows = [{"metric": k, "value": v} for k, v in preference_result.items()]
        _write_csv(tables_dir / "preference_metrics.csv", pref_rows)
        _write_markdown_table(tables_dir / "preference_metrics.md", pref_rows, title="Preference Metrics")

    if runtime_result:
        runtime_rows = [
            {"metric": k, "value": v}
            for k, v in runtime_result.items()
            if isinstance(v, (str, int, float, bool)) or v is None
        ]
        _write_csv(tables_dir / "runtime_metrics.csv", runtime_rows)
        _write_markdown_table(tables_dir / "runtime_metrics.md", runtime_rows, title="Runtime Metrics")

    if training_result or precompute_result:
        timing_rows = []
        if training_result:
            timing_rows.extend(
                {"metric": f"training.{k}", "value": v}
                for k, v in training_result.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            )
        if precompute_result:
            timing_rows.extend(
                {"metric": f"precompute.{k}", "value": v}
                for k, v in precompute_result.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            )
        if training_result and precompute_result and "total_train_sec" in training_result:
            total = float(training_result["total_train_sec"]) + float(
                precompute_result.get("precompute_sec", 0.0)
            )
            timing_rows.append({"metric": "total_wall_sec", "value": total})
        _write_csv(tables_dir / "timing_metrics.csv", timing_rows)
        _write_markdown_table(tables_dir / "timing_metrics.md", timing_rows, title="Training & Precompute Timing")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    # Common config/data/model args.
    add_common_args(parser)

    # Evaluation-specific args.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="LoRA checkpoint dir. If omitted, evaluate base model.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Preference split to evaluate/visualize.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name. Output goes to outputs/results/<exp-name>/.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="outputs/results",
        help="Root folder for combined evaluation outputs.",
    )

    # Runtime.
    parser.add_argument("--num-runtime-steps", type=int, default=20)
    parser.add_argument("--skip-runtime", action="store_true")

    # Preference.
    parser.add_argument("--skip-preference", action="store_true")

    # Visualizations.
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--num-visualize", type=int, default=3)
    parser.add_argument("--cached-weights", type=str, default=None)
    parser.add_argument("--max-plot-tokens", type=int, default=160)
    parser.add_argument("--max-text-tokens", type=int, default=300)

    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg.get("seed", 42))
    logger = setup_logging()

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, cfg.get("precision", "auto"))

    weight_method = WeightMethod(cfg["weight_method"])
    exp_name = _resolve_exp_name(args, cfg, weight_method)

    out_dir = _ensure_dir(Path(args.results_root) / exp_name)
    _ensure_dir(out_dir / "tables")
    _ensure_dir(out_dir / "figures")
    _ensure_dir(out_dir / "raw")

    logger.info("Writing combined evaluation to %s", out_dir)

    # Save exact config and command metadata.
    _write_json(out_dir / "raw" / "config.json", cfg)
    _write_json(
        out_dir / "raw" / "args.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    # Load model for preference eval and visualizations.
    is_base = _is_base_eval(args)
    bundle = _load_eval_bundle(args, cfg, device, dtype)

    split_path = _get_split_path(args, cfg)
    examples = load_preference_jsonl(split_path)

    cached_weights = None
    cached_weights_path = args.cached_weights or resolve_split_cached_weights_path(cfg, args.split)
    if cached_weights_path and Path(cached_weights_path).exists():
        cached_weights = load_cached_grad_weights(cached_weights_path)
        logger.info("Loaded cached grad weights from %s", cached_weights_path)
    else:
        logger.info("No cached grad weights found")

    dataset = PreferenceDataset(
        examples,
        bundle.tokenizer,
        cfg["max_length"],
        cfg["max_prompt_length"],
        cached_weights=cached_weights,
    )

    metadata = {
        "exp_name": exp_name,
        "model_name": cfg["model_name"],
        "checkpoint": args.checkpoint or "base",
        "is_base_model": is_base,
        "weight_method": weight_method.value,
        "evaluation_method": cfg.get("evaluation_method", "uniform"),
        "cached_weights_path": cached_weights_path,
        "split": args.split,
        "split_path": split_path,
        "num_split_examples": len(examples),
        "max_length": cfg["max_length"],
        "max_prompt_length": cfg["max_prompt_length"],
        "beta": cfg["beta"],
        "precision": cfg.get("precision", "auto"),
        "device": str(device),
        "timestamp_unix": time.time(),
    }

    preference_result = None
    runtime_result = None
    training_result = load_train_stats(args.checkpoint, cfg)
    precompute_result = None
    visualization_result = None

    if training_result:
        logger.info(
            "Training stats: total_sec=%.1f mean_step_sec=%.3f steps=%s",
            float(training_result.get("total_train_sec", 0.0)),
            float(training_result.get("mean_step_sec", 0.0)),
            training_result.get("global_step"),
        )
        _write_json(out_dir / "raw" / "training.json", training_result)

    if weight_method == WeightMethod.CACHED_GRAD:
        precompute_result = _load_precompute_stats(cfg, "train")
        if precompute_result:
            logger.info(
                "Precompute stats: %.1f sec (%.3f sec/example)",
                float(precompute_result.get("precompute_sec", 0.0)),
                float(precompute_result.get("sec_per_example", 0.0)),
            )
            _write_json(out_dir / "raw" / "precompute.json", precompute_result)

    # 1. Preference eval.
    if not args.skip_preference:
        logger.info("Running preference evaluation (%s)...", "base raw logprob" if is_base else "dpo ref-normalized")
        preference_result = run_preference_eval(
            bundle=bundle,
            dataset=dataset,
            weight_method=weight_method,
            cfg=cfg,
            batch_size=max(1, cfg.get("batch_size", 1)),
            is_base_model=is_base,
        )
        _write_json(out_dir / "raw" / "preference.json", preference_result)
        logger.info(
            "Preference: acc=%.4f weighted_acc=%s mean_margin=%.4f",
            preference_result["preference_accuracy"],
            f"{preference_result['weighted_preference_accuracy']:.4f}"
            if preference_result.get("weighted_preference_accuracy") is not None
            else "n/a",
            preference_result["mean_margin"],
        )

    # 2. Runtime eval.
    if not args.skip_runtime:
        logger.info("Running runtime benchmark...")
        runtime_result = run_runtime_eval(
            args=args,
            cfg=cfg,
            weight_method=weight_method,
        )
        _write_json(out_dir / "raw" / "runtime.json", runtime_result)
        if "mean_step_sec" in runtime_result:
            logger.info("Runtime: mean_step_sec=%.4f", runtime_result["mean_step_sec"])

    # 3. Visualizations.
    if not args.skip_visualize:
        logger.info("Running token weight visualizations...")
        visualization_result = run_weight_visualizations(
            args=args,
            cfg=cfg,
            bundle=bundle,
            examples=examples,
            device=device,
            out_dir=out_dir,
            logger=logger,
        )
        _write_json(out_dir / "raw" / "visualization.json", visualization_result)

    # 4. Combined summary.
    write_combined_outputs(
        out_dir=out_dir,
        metadata=metadata,
        preference_result=preference_result,
        runtime_result=runtime_result,
        training_result=training_result,
        precompute_result=precompute_result,
        visualization_result=visualization_result,
    )

    logger.info("Done. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
