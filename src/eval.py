"""Preference accuracy, margins, runtime, and weight visualization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import PreferenceDataset, preference_collate_fn
from src.dpo import (
    WeightMethod,
    build_gaussian_weights,
    build_surprisal_weights,
    build_uniform_weights,
    compute_dpo_loss_from_logps,
    get_per_token_logps,
    weighted_sequence_score,
)
from src.model import ModelBundle


@dataclass
class EvalResult:
    preference_accuracy: float
    weighted_preference_accuracy: float | None
    mean_margin: float
    median_margin: float
    mean_loss: float
    num_examples: int


def _resolve_eval_weights(
    method: WeightMethod,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    chosen_ref_logps: torch.Tensor,
    rejected_ref_logps: torch.Tensor,
    chosen_ext: torch.Tensor | None,
    rejected_ext: torch.Tensor | None,
    surprisal_w_min: float,
    surprisal_w_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if method == WeightMethod.UNIFORM:
        return build_uniform_weights(chosen_mask), build_uniform_weights(rejected_mask)
    if method == WeightMethod.GAUSSIAN:
        return build_gaussian_weights(chosen_mask), build_gaussian_weights(rejected_mask)
    if method == WeightMethod.SURPRISAL:
        return (
            build_surprisal_weights(chosen_ref_logps, chosen_mask, surprisal_w_min, surprisal_w_max),
            build_surprisal_weights(rejected_ref_logps, rejected_mask, surprisal_w_min, surprisal_w_max),
        )
    if method == WeightMethod.CACHED_GRAD:
        if chosen_ext is None or rejected_ext is None:
            raise ValueError(
                "cached_grad evaluation requires precomputed chosen/rejected weights. "
                "Run scripts/03_precompute_cachedgrad.py for this split, or set "
                "evaluation_method: uniform in the config."
            )
        return chosen_ext * chosen_mask, rejected_ext * rejected_mask
    if method == WeightMethod.ONLINE_HYBRID:
        raise ValueError(
            "online_hybrid is training-only; set evaluation_method: uniform in config."
        )
    raise ValueError(method)


@torch.no_grad()
def evaluate_preference(
    bundle: ModelBundle,
    dataset: PreferenceDataset,
    weight_method: WeightMethod = WeightMethod.UNIFORM,
    eval_weight_method: WeightMethod | None = None,
    beta: float = 0.1,
    batch_size: int = 2,
    eval_weighted_accuracy: bool = True,
    surprisal_w_min: float = 0.2,
    surprisal_w_max: float = 3.0,
) -> EvalResult:
    """Evaluate preference accuracy.

    ``preference_accuracy`` is always unweighted (fair across methods).
    ``eval_weight_method`` controls weighted accuracy and reported loss;
    defaults to ``weight_method``. For CachedGrad checkpoints, use
    ``evaluation_method: uniform`` in config unless test weights exist.
    """
    eval_wm = eval_weight_method or weight_method
    has_cached_weights = dataset.cached_weights is not None
    if eval_wm == WeightMethod.CACHED_GRAD and not has_cached_weights:
        raise ValueError(
            "eval_weight_method=cached_grad but dataset has no cached weights. "
            "Precompute with scripts/03_precompute_cachedgrad.py or set evaluation_method: uniform."
        )
    if eval_wm == WeightMethod.ONLINE_HYBRID:
        eval_wm = WeightMethod.UNIFORM
    bundle.policy.eval()
    device = bundle.device
    collate = lambda batch: preference_collate_fn(batch, bundle.tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    correct = 0
    weighted_correct = 0
    margins: list[float] = []
    losses: list[float] = []
    total = 0

    for batch in tqdm(loader, desc="eval"):
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_attn = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_attn = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        chosen_mask = (chosen_labels[:, 1:] != -100).float()
        rejected_mask = (rejected_labels[:, 1:] != -100).float()

        chosen_pi = get_per_token_logps(bundle.policy, chosen_ids, chosen_attn, chosen_labels)
        rejected_pi = get_per_token_logps(bundle.policy, rejected_ids, rejected_attn, rejected_labels)
        chosen_ref = get_per_token_logps(bundle.policy, chosen_ids, chosen_attn, chosen_labels, use_ref=True)
        rejected_ref = get_per_token_logps(bundle.policy, rejected_ids, rejected_attn, rejected_labels, use_ref=True)

        ext_cw = batch.get("chosen_weights")
        ext_rw = batch.get("rejected_weights")
        if ext_cw is not None:
            ext_cw = ext_cw.to(device)
            ext_rw = ext_rw.to(device)

        unweighted_chosen = weighted_sequence_score(chosen_pi, chosen_ref, chosen_mask, build_uniform_weights(chosen_mask))
        unweighted_rejected = weighted_sequence_score(
            rejected_pi, rejected_ref, rejected_mask, build_uniform_weights(rejected_mask)
        )
        correct += (unweighted_chosen > unweighted_rejected).sum().item()

        if eval_weighted_accuracy:
            cw, rw = _resolve_eval_weights(
                eval_wm,
                chosen_mask,
                rejected_mask,
                chosen_ref,
                rejected_ref,
                ext_cw,
                ext_rw,
                surprisal_w_min,
                surprisal_w_max,
            )
            w_chosen = weighted_sequence_score(chosen_pi, chosen_ref, chosen_mask, cw)
            w_rejected = weighted_sequence_score(rejected_pi, rejected_ref, rejected_mask, rw)
            weighted_correct += (w_chosen > w_rejected).sum().item()

        margin = (unweighted_chosen - unweighted_rejected).cpu().tolist()
        margins.extend(margin)

        loss, _ = compute_dpo_loss_from_logps(
            chosen_pi,
            rejected_pi,
            chosen_ref,
            rejected_ref,
            chosen_mask,
            rejected_mask,
            eval_wm,
            beta,
            chosen_external_weights=ext_cw,
            rejected_external_weights=ext_rw,
            surprisal_w_min=surprisal_w_min,
            surprisal_w_max=surprisal_w_max,
        )
        losses.append(loss.item())
        total += chosen_ids.size(0)

    sorted_margins = sorted(margins)
    mid = len(sorted_margins) // 2
    median_margin = (
        sorted_margins[mid]
        if len(sorted_margins) % 2 == 1
        else (sorted_margins[mid - 1] + sorted_margins[mid]) / 2
    )

    return EvalResult(
        preference_accuracy=correct / max(total, 1),
        weighted_preference_accuracy=weighted_correct / max(total, 1) if eval_weighted_accuracy else None,
        mean_margin=sum(margins) / max(len(margins), 1),
        median_margin=median_margin,
        mean_loss=sum(losses) / max(len(losses), 1),
        num_examples=total,
    )


def benchmark_step_time(
    bundle: ModelBundle,
    dataset: PreferenceDataset,
    weight_method: WeightMethod,
    beta: float = 0.1,
    num_steps: int = 20,
    surprisal_w_min: float = 0.2,
    surprisal_w_max: float = 3.0,
    lambda_blend: float = 0.7,
    gaussian_sigma_scale: float = 4.0,
    importance_update_freq: int = 10,
    importance_ema_decay: float = 0.9,
) -> dict[str, float]:
    """Measure forward+backward time per optimizer step on a few batches."""
    bundle.policy.train()
    device = bundle.device
    collate = lambda batch: preference_collate_fn(batch, bundle.tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW((p for p in bundle.policy.parameters() if p.requires_grad), lr=1e-5)

    hybrid_computer = None
    if weight_method == WeightMethod.ONLINE_HYBRID:
        from src.ti_dpo_importance import OnlineHybridWeightComputer

        hybrid_computer = OnlineHybridWeightComputer(
            bundle.policy,
            lambda_blend=lambda_blend,
            sigma_scale=gaussian_sigma_scale,
            update_freq=importance_update_freq,
            ema_decay=importance_ema_decay,
        )

    times: list[float] = []
    for i, batch in enumerate(loader):
        if i >= num_steps:
            break
        t0 = time.perf_counter()
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_attn = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_attn = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)
        chosen_mask = (chosen_labels[:, 1:] != -100).float()
        rejected_mask = (rejected_labels[:, 1:] != -100).float()
        ext_cw = batch.get("chosen_weights")
        ext_rw = batch.get("rejected_weights")
        if weight_method == WeightMethod.ONLINE_HYBRID:
            assert hybrid_computer is not None
            ext_cw, ext_rw = hybrid_computer.maybe_recompute(
                batch["ids"],
                chosen_ids,
                chosen_attn,
                chosen_labels,
                rejected_ids,
                rejected_attn,
                rejected_labels,
                force=(i == 0),
            )
        elif ext_cw is not None:
            ext_cw = ext_cw.to(device)
            ext_rw = ext_rw.to(device)

        chosen_pi = get_per_token_logps(bundle.policy, chosen_ids, chosen_attn, chosen_labels)
        rejected_pi = get_per_token_logps(bundle.policy, rejected_ids, rejected_attn, rejected_labels)
        with torch.no_grad():
            chosen_ref = get_per_token_logps(bundle.policy, chosen_ids, chosen_attn, chosen_labels, use_ref=True)
            rejected_ref = get_per_token_logps(
                bundle.policy, rejected_ids, rejected_attn, rejected_labels, use_ref=True
            )
        loss, _ = compute_dpo_loss_from_logps(
            chosen_pi,
            rejected_pi,
            chosen_ref,
            rejected_ref,
            chosen_mask,
            rejected_mask,
            weight_method,
            beta,
            chosen_external_weights=ext_cw,
            rejected_external_weights=ext_rw,
            surprisal_w_min=surprisal_w_min,
            surprisal_w_max=surprisal_w_max,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if hybrid_computer is not None:
            hybrid_computer.on_optimizer_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return {
        "mean_step_sec": sum(times) / max(len(times), 1),
        "num_steps": len(times),
    }
