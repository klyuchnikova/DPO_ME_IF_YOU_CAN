#!/usr/bin/env python3
"""Aggregate experiment results into one comparison table (includes base model row)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHOD_ORDER = ["base", "uniform", "dpo", "gaussian", "surprisal", "cached_grad", "online_hybrid"]


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def _method_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    method = str(row.get("method", "")).lower()
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), row.get("exp_name", ""))
    if method == "base":
        return (0, row.get("exp_name", ""))
    return (99, method)


def _resolve_method(exp_name: str, pref: dict[str, Any], meta: dict[str, Any]) -> str:
    if pref.get("training_method") == "base" or meta.get("is_base_model") or meta.get("checkpoint") == "base":
        return "base"
    if pref.get("training_method"):
        return str(pref["training_method"])
    wm = meta.get("weight_method", "")
    if wm == "uniform":
        return "dpo"
    return str(wm or exp_name)


def collect_rows(results_root: Path, model_filter: str | None, split_filter: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not results_root.is_dir():
        return rows

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        pref = _load_json(exp_dir / "raw" / "preference.json")
        if not pref:
            continue

        summary = _load_json(exp_dir / "summary.json") or {}
        meta = summary.get("metadata", {})
        training = _load_json(exp_dir / "raw" / "training.json") or summary.get("training") or {}
        precompute = _load_json(exp_dir / "raw" / "precompute.json") or summary.get("precompute") or {}
        runtime = _load_json(exp_dir / "raw" / "runtime.json") or summary.get("runtime") or {}

        model_name = meta.get("model_name", "")
        split = meta.get("split", pref.get("split", "test"))

        if model_filter and model_filter not in model_name:
            continue
        if split_filter and split != split_filter:
            continue

        method = _resolve_method(exp_dir.name, pref, meta)
        total_train = training.get("total_train_sec")
        precompute_sec = precompute.get("precompute_sec")
        total_wall = None
        if total_train is not None:
            total_wall = float(total_train) + float(precompute_sec or 0.0)

        rows.append(
            {
                "exp_name": exp_dir.name,
                "method": method,
                "model_name": model_name,
                "split": split,
                "checkpoint": meta.get("checkpoint", ""),
                "preference_accuracy": pref.get("preference_accuracy"),
                "mean_margin": pref.get("mean_margin"),
                "median_margin": pref.get("median_margin"),
                "mean_loss": pref.get("mean_loss"),
                "num_examples": pref.get("num_examples"),
                "eval_metric": pref.get("evaluation_method", meta.get("evaluation_method", "")),
                "mean_step_sec": training.get("mean_step_sec"),
                "total_train_sec": total_train,
                "precompute_sec": precompute_sec,
                "total_wall_sec": total_wall,
                "runtime_mean_step_sec": runtime.get("mean_step_sec"),
            }
        )

    rows.sort(key=_method_sort_key)
    return rows


def write_table(path: Path, rows: list[dict[str, Any]], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    columns = [
        "method",
        "preference_accuracy",
        "mean_margin",
        "median_margin",
        "mean_loss",
        "total_train_sec",
        "precompute_sec",
        "total_wall_sec",
        "mean_step_sec",
        "runtime_mean_step_sec",
        "num_examples",
        "eval_metric",
        "exp_name",
        "checkpoint",
    ]

    if fmt == "csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return

    with path.open("w", encoding="utf-8") as f:
        f.write("# Preference comparison\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in rows:
            vals = []
            for col in columns:
                v = row.get(col, "")
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                elif v is None:
                    vals.append("")
                else:
                    vals.append(str(v))
            f.write("| " + " | ".join(vals) + " |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/results")
    parser.add_argument("--model-filter", type=str, default=None, help="Substring match on model_name")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    rows = collect_rows(results_root, args.model_filter, args.split)

    out_dir = Path(args.output_dir)
    write_table(out_dir / "comparison_table.csv", rows, "csv")
    write_table(out_dir / "comparison_table.md", rows, "md")
    _ = out_dir / "comparison_table.json"
    with (out_dir / "comparison_table.json").open("w") as f:
        json.dump({"split": args.split, "rows": rows}, f, indent=2)

    print(f"Wrote {len(rows)} rows to {out_dir / 'comparison_table.md'}")
    for row in rows:
        acc = row.get("preference_accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else "n/a"
        print(f"  {row['method']:14s}  PrefAcc={acc_s}  margin={row.get('mean_margin', '')}")


if __name__ == "__main__":
    main()
