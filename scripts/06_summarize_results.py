#!/usr/bin/env python3
"""Aggregate experiment summaries into cumulative JSON/CSV/Markdown reports.

Reads:
    outputs/results/<exp_name>/summary.json

Writes:
    outputs/results/comparison/
        comparison_table.json
        comparison_table.csv
        comparison_table.md
        grouped_by_model.json
        best_by_model.json
        report.md

Example:
    python scripts/aggregate_results.py \
        --results-root outputs/results \
        --output-dir outputs/results/comparison \
        --split test

Filter one model:
    python scripts/aggregate_results.py \
        --model-filter Qwen2.5-0.5B \
        --split test

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


METHOD_ORDER = [
    "base",
    "dpo",
    "uniform",
    "gaussian",
    "surprisal",
    "cached_grad",
    "topk_cached_grad",
    "online_hybrid",
]


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        return f"{v:.{digits}f}"
    return str(v)


def _pct(v: Any, digits: int = 2) -> str:
    if v is None:
        return ""
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{100.0 * v:.{digits}f}%"


def _sec(v: Any) -> str:
    if v is None:
        return ""
    if not isinstance(v, (int, float)):
        return str(v)
    if v < 60:
        return f"{v:.2f}s"
    if v < 3600:
        return f"{v / 60:.2f}m"
    return f"{v / 3600:.2f}h"


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


# ---------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------


def _resolve_method(exp_name: str, meta: Dict[str, Any], pref: Dict[str, Any]) -> str:
    """Resolve training method.

    Prefer explicit fields:
        preference.training_method
        metadata.is_base_model
        metadata.weight_method

    Normalize uniform -> dpo unless it is truly base.
    """
    if (
        pref.get("training_method") == "base"
        or meta.get("is_base_model")
        or meta.get("checkpoint") == "base"
    ):
        return "base"

    tm = pref.get("training_method")
    if tm:
        tm = str(tm)
        if tm == "uniform":
            return "dpo"
        return tm

    wm = meta.get("weight_method")
    if wm:
        wm = str(wm)
        if wm == "uniform":
            return "dpo"
        return wm

    # Fallback from experiment name.
    name = exp_name.lower()
    if "cached" in name:
        return "cached_grad"
    if "surprisal" in name:
        return "surprisal"
    if "gaussian" in name:
        return "gaussian"
    if "dpo" in name:
        return "dpo"
    if "base" in name:
        return "base"

    return exp_name


def _method_sort_key(row: Dict[str, Any]) -> Tuple[int, str, str]:
    method = str(row.get("method", "")).lower()
    model = str(row.get("model_short", row.get("model_name", "")))
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), model, row.get("exp_name", ""))
    return (99, model, method)


def _model_short(model_name: str) -> str:
    if not model_name:
        return ""
    return model_name.split("/")[-1]


def _extract_train_summary(summary: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    """Training info may be in summary.training or raw/training.json."""
    training = summary.get("training")
    if not training:
        training = _load_json(exp_dir / "raw" / "training.json")
    return training or {}


def _extract_precompute_summary(summary: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    precompute = summary.get("precompute")
    if not precompute:
        precompute = _load_json(exp_dir / "raw" / "precompute.json")
    return precompute or {}


def _extract_runtime_summary(summary: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    runtime = summary.get("runtime")
    if not runtime:
        runtime = _load_json(exp_dir / "raw" / "runtime.json")
    return runtime or {}


def _extract_pref_summary(summary: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    pref = summary.get("preference")
    if not pref:
        pref = _load_json(exp_dir / "raw" / "preference.json")
    return pref or {}


def _approx_acc_ci95(acc: Optional[float], n: Optional[int]) -> Optional[float]:
    """Approx 95% confidence half-width for binomial accuracy."""
    if acc is None or n is None or n <= 0:
        return None
    return 1.96 * math.sqrt(acc * (1.0 - acc) / n)


def _extract_row(exp_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = exp_dir / "summary.json"
    summary = _load_json(summary_path)
    if not summary:
        return None

    meta = summary.get("metadata", {}) or {}
    pref = _extract_pref_summary(summary, exp_dir)
    training = _extract_train_summary(summary, exp_dir)
    precompute = _extract_precompute_summary(summary, exp_dir)
    runtime = _extract_runtime_summary(summary, exp_dir)

    if not pref:
        return None

    exp_name = meta.get("exp_name") or exp_dir.name
    model_name = meta.get("model_name", "")
    method = _resolve_method(exp_name, meta, pref)

    n = pref.get("num_examples")
    try:
        n_int = int(n) if n is not None else None
    except Exception:
        n_int = None

    acc = _safe_float(pref.get("preference_accuracy"))
    ci95 = _approx_acc_ci95(acc, n_int)

    weighted_acc = _safe_float(pref.get("weighted_preference_accuracy"))
    mean_margin = _safe_float(pref.get("mean_margin"))
    median_margin = _safe_float(pref.get("median_margin"))
    mean_loss = _safe_float(pref.get("mean_loss"))

    total_train_sec = _safe_float(training.get("total_train_sec"))
    train_mean_step_sec = _safe_float(training.get("mean_step_sec"))
    global_step = training.get("global_step")

    precompute_sec = _safe_float(
        precompute.get("precompute_sec")
        or precompute.get("total_precompute_sec")
        or precompute.get("time_sec")
    )

    runtime_mean_step_sec = _safe_float(runtime.get("mean_step_sec"))

    total_wall_sec = None
    if total_train_sec is not None or precompute_sec is not None:
        total_wall_sec = float(total_train_sec or 0.0) + float(precompute_sec or 0.0)

    eval_method = (
        pref.get("evaluation_method")
        or meta.get("evaluation_method")
        or ""
    )

    training_method = pref.get("training_method") or method

    row = {
        # identity
        "exp_name": exp_name,
        "exp_dir": str(exp_dir),
        "model_name": model_name,
        "model_short": _model_short(model_name),
        "method": method,
        "training_method": training_method,
        "weight_method": meta.get("weight_method", ""),
        "evaluation_method": eval_method,
        "checkpoint": meta.get("checkpoint", ""),
        "is_base_model": bool(meta.get("is_base_model", False) or method == "base"),

        # data/eval config
        "split": meta.get("split", ""),
        "split_path": meta.get("split_path", ""),
        "num_examples": n_int,
        "max_length": meta.get("max_length"),
        "max_prompt_length": meta.get("max_prompt_length"),
        "beta": meta.get("beta"),
        "precision": meta.get("precision"),
        "device": meta.get("device"),
        "cached_weights_path": meta.get("cached_weights_path"),

        # metrics
        "preference_accuracy": acc,
        "preference_accuracy_ci95": ci95,
        "weighted_preference_accuracy": weighted_acc,
        "mean_margin": mean_margin,
        "median_margin": median_margin,
        "mean_loss": mean_loss,
        "metric_note": pref.get("metric_note", ""),

        # training
        "global_step": global_step,
        "train_mean_loss": _safe_float(training.get("mean_loss")),
        "train_mean_step_sec": train_mean_step_sec,
        "total_train_sec": total_train_sec,
        "train_stats_path": training.get("train_stats_path", ""),

        # precompute
        "precompute_sec": precompute_sec,
        "precompute_path": precompute.get("path") or precompute.get("output") or "",

        # runtime
        "runtime_mean_step_sec": runtime_mean_step_sec,
        "runtime_num_steps": runtime.get("num_steps") or runtime.get("num_runtime_steps"),

        # combined
        "total_wall_sec": total_wall_sec,

        # metadata
        "timestamp_unix": meta.get("timestamp_unix"),
        "summary_path": str(summary_path),
    }

    return row


# ---------------------------------------------------------------------
# Collection and postprocessing
# ---------------------------------------------------------------------


def collect_rows(
    results_root: Path,
    model_filter: Optional[str],
    split_filter: Optional[str],
    include_dirs: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    rows = []

    if not results_root.is_dir():
        print(f"[WARN] results root does not exist: {results_root}")
        return rows

    include_set = set(include_dirs or [])

    for exp_dir in sorted(results_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        # Skip aggregate output directory itself.
        if exp_dir.name in {"comparison", "aggregate", "tables"}:
            continue

        if include_set and exp_dir.name not in include_set:
            continue

        row = _extract_row(exp_dir)
        if row is None:
            continue

        model_name = str(row.get("model_name", ""))
        split = str(row.get("split", ""))

        if model_filter and model_filter not in model_name and model_filter not in row.get("model_short", ""):
            continue
        if split_filter and split and split != split_filter:
            continue

        rows.append(row)

    rows.sort(key=lambda r: (r.get("model_short", ""), _method_sort_key(r)))
    return rows


def add_improvements_over_base(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add deltas relative to base for each model.

    Fields added:
        base_preference_accuracy
        delta_vs_base
        relative_delta_vs_base
        margin_delta_vs_base
    """
    base_by_model: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        if row.get("method") == "base" or row.get("is_base_model"):
            model = row.get("model_name", "")
            if model:
                base_by_model[model] = row

    for row in rows:
        model = row.get("model_name", "")
        base = base_by_model.get(model)

        if not base:
            row["base_preference_accuracy"] = None
            row["delta_vs_base"] = None
            row["relative_delta_vs_base"] = None
            row["margin_delta_vs_base"] = None
            continue

        base_acc = _safe_float(base.get("preference_accuracy"))
        acc = _safe_float(row.get("preference_accuracy"))

        base_margin = _safe_float(base.get("mean_margin"))
        margin = _safe_float(row.get("mean_margin"))

        row["base_preference_accuracy"] = base_acc

        if acc is not None and base_acc is not None:
            row["delta_vs_base"] = acc - base_acc
            if abs(base_acc) > 1e-12:
                row["relative_delta_vs_base"] = (acc - base_acc) / base_acc
            else:
                row["relative_delta_vs_base"] = None
        else:
            row["delta_vs_base"] = None
            row["relative_delta_vs_base"] = None

        if margin is not None and base_margin is not None:
            row["margin_delta_vs_base"] = margin - base_margin
        else:
            row["margin_delta_vs_base"] = None

    return rows


def grouped_by_model(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = row.get("model_name", "") or "unknown"
        out.setdefault(key, []).append(row)
    return out


def best_by_model(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped = grouped_by_model(rows)
    out = {}

    for model, model_rows in grouped.items():
        candidates = [
            r for r in model_rows
            if r.get("preference_accuracy") is not None and r.get("method") != "base"
        ]

        if not candidates:
            continue

        best = max(candidates, key=lambda r: float(r["preference_accuracy"]))

        out[model] = {
            "best_exp_name": best.get("exp_name"),
            "best_method": best.get("method"),
            "best_preference_accuracy": best.get("preference_accuracy"),
            "best_delta_vs_base": best.get("delta_vs_base"),
            "best_mean_margin": best.get("mean_margin"),
            "best_total_wall_sec": best.get("total_wall_sec"),
            "best_runtime_mean_step_sec": best.get("runtime_mean_step_sec"),
        }

    return out


def compute_global_notes(rows: List[Dict[str, Any]]) -> List[str]:
    notes = []

    if not rows:
        notes.append("No experiment rows found.")
        return notes

    eval_methods = sorted(set(str(r.get("evaluation_method", "")) for r in rows))
    if len(eval_methods) > 1:
        notes.append(
            "Warning: multiple evaluation methods appear in the table: "
            + ", ".join(eval_methods)
            + ". Direct accuracy comparisons are safest only when evaluation methods match."
        )

    base_raw = [
        r for r in rows
        if r.get("method") == "base" and str(r.get("evaluation_method", "")).lower() == "raw_logprob"
    ]
    nonbase_ref = [
        r for r in rows
        if r.get("method") != "base" and "raw" not in str(r.get("evaluation_method", "")).lower()
    ]

    if base_raw and nonbase_ref:
        notes.append(
            "Base rows use raw response log-probability, while trained DPO rows may use "
            "reference-normalized margins. Preference accuracy can still be useful, but margins/losses "
            "are not directly comparable between base and DPO-style rows."
        )

    ns = sorted(set(r.get("num_examples") for r in rows if r.get("num_examples") is not None))
    if len(ns) > 1:
        notes.append(
            "Warning: experiments use different test-set sizes: "
            + ", ".join(map(str, ns))
            + ". Accuracy confidence intervals differ."
        )

    return notes


# ---------------------------------------------------------------------
# Markdown writers
# ---------------------------------------------------------------------


COMPACT_COLUMNS = [
    "model_short",
    "method",
    "preference_accuracy",
    "preference_accuracy_ci95",
    "delta_vs_base",
    "mean_margin",
    "median_margin",
    "mean_loss",
    "global_step",
    "train_mean_step_sec",
    "runtime_mean_step_sec",
    "precompute_sec",
    "total_train_sec",
    "total_wall_sec",
    "num_examples",
    "evaluation_method",
    "exp_name",
]


FULL_COLUMNS = [
    "exp_name",
    "model_name",
    "method",
    "training_method",
    "weight_method",
    "evaluation_method",
    "checkpoint",
    "split",
    "num_examples",
    "preference_accuracy",
    "preference_accuracy_ci95",
    "weighted_preference_accuracy",
    "base_preference_accuracy",
    "delta_vs_base",
    "relative_delta_vs_base",
    "mean_margin",
    "margin_delta_vs_base",
    "median_margin",
    "mean_loss",
    "global_step",
    "train_mean_loss",
    "train_mean_step_sec",
    "runtime_mean_step_sec",
    "precompute_sec",
    "total_train_sec",
    "total_wall_sec",
    "metric_note",
    "summary_path",
]


def _md_table(rows: List[Dict[str, Any]], columns: List[str]) -> str:
    if not rows:
        return "_No rows._\n"

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col)

            if col in {
                "preference_accuracy",
                "preference_accuracy_ci95",
                "weighted_preference_accuracy",
                "base_preference_accuracy",
                "delta_vs_base",
                "relative_delta_vs_base",
            }:
                vals.append(_fmt(v, 4))
            elif col.endswith("_sec") or col in {
                "train_mean_step_sec",
                "runtime_mean_step_sec",
                "precompute_sec",
                "total_train_sec",
                "total_wall_sec",
            }:
                vals.append(_sec(v))
            elif isinstance(v, float):
                vals.append(_fmt(v, 4))
            elif v is None:
                vals.append("")
            else:
                # avoid breaking markdown tables
                vals.append(str(v).replace("|", "\\|").replace("\n", " "))

        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines) + "\n"


def write_comparison_md(path: Path, rows: List[Dict[str, Any]], title: str = "Preference comparison") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(_md_table(rows, COMPACT_COLUMNS))


def write_report_md(
    path: Path,
    rows: List[Dict[str, Any]],
    best: Dict[str, Dict[str, Any]],
    notes: List[str],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    grouped = grouped_by_model(rows)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Cumulative experiment report\n\n")

        f.write("## Filters\n\n")
        f.write(f"- Results root: `{args.results_root}`\n")
        f.write(f"- Split: `{args.split}`\n")
        f.write(f"- Model filter: `{args.model_filter}`\n")
        f.write(f"- Number of rows: `{len(rows)}`\n\n")

        f.write("## Notes\n\n")
        if notes:
            for note in notes:
                f.write(f"- {note}\n")
        else:
            f.write("- No major warnings detected.\n")
        f.write("\n")

        f.write("## Best method by model\n\n")
        if best:
            best_rows = []
            for model, b in best.items():
                best_rows.append({
                    "model": model,
                    "best_method": b.get("best_method"),
                    "best_pref_acc": b.get("best_preference_accuracy"),
                    "delta_vs_base": b.get("best_delta_vs_base"),
                    "mean_margin": b.get("best_mean_margin"),
                    "total_wall_sec": b.get("best_total_wall_sec"),
                    "runtime_step_sec": b.get("best_runtime_mean_step_sec"),
                    "exp_name": b.get("best_exp_name"),
                })
            f.write(_md_table(
                best_rows,
                [
                    "model",
                    "best_method",
                    "best_pref_acc",
                    "delta_vs_base",
                    "mean_margin",
                    "total_wall_sec",
                    "runtime_step_sec",
                    "exp_name",
                ],
            ))
        else:
            f.write("_No non-base candidates found._\n")
        f.write("\n")

        f.write("## Compact comparison table\n\n")
        f.write(_md_table(rows, COMPACT_COLUMNS))
        f.write("\n")

        f.write("## Per-model tables\n\n")
        for model, model_rows in grouped.items():
            f.write(f"### {model}\n\n")
            f.write(_md_table(model_rows, COMPACT_COLUMNS))
            f.write("\n")

        f.write("## Full table\n\n")
        f.write(_md_table(rows, FULL_COLUMNS))
        f.write("\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=str, default="outputs/results")
    parser.add_argument("--output-dir", type=str, default="outputs/results/comparison")
    parser.add_argument("--model-filter", type=str, default=None, help="Substring match on model_name/model_short")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--include-dir",
        action="append",
        default=None,
        help="Only include a specific experiment directory name. Can be passed multiple times.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(
        results_root=results_root,
        model_filter=args.model_filter,
        split_filter=args.split,
        include_dirs=args.include_dir,
    )
    rows = add_improvements_over_base(rows)

    grouped = grouped_by_model(rows)
    best = best_by_model(rows)
    notes = compute_global_notes(rows)

    # Outputs.
    _write_json(out_dir / "comparison_table.json", {
        "filters": {
            "results_root": str(results_root),
            "model_filter": args.model_filter,
            "split": args.split,
            "include_dir": args.include_dir,
        },
        "notes": notes,
        "rows": rows,
    })

    _write_json(out_dir / "grouped_by_model.json", grouped)
    _write_json(out_dir / "best_by_model.json", best)

    _write_csv(out_dir / "comparison_table.csv", rows, FULL_COLUMNS)
    write_comparison_md(out_dir / "comparison_table.md", rows, title="Preference comparison")
    write_report_md(out_dir / "report.md", rows, best, notes, args)

    print(f"Wrote {len(rows)} rows to {out_dir}")
    print(f"Main report: {out_dir / 'report.md'}")

    for row in rows:
        acc = row.get("preference_accuracy")
        delta = row.get("delta_vs_base")
        acc_s = _fmt(acc, 4)
        delta_s = _fmt(delta, 4)
        print(
            f"  {row.get('model_short', ''):28s} "
            f"{row.get('method', ''):16s} "
            f"PrefAcc={acc_s:>8s} "
            f"Δbase={delta_s:>8s} "
            f"exp={row.get('exp_name', '')}"
        )


if __name__ == "__main__":
    main()
