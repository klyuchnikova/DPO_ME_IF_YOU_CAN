"""Weighted DPO loss, per-token logprobs, and token-importance weight schemes."""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel


class WeightMethod(str, Enum):
    UNIFORM = "uniform"  # vanilla DPO
    GAUSSIAN = "gaussian"
    SURPRISAL = "surprisal"
    CACHED_GRAD = "cached_grad"


def get_response_mask(labels: Tensor, label_pad_token_id: int) -> Tensor:
    """True for response tokens (non-padding, non-ignored label positions)."""
    return labels != label_pad_token_id


def gather_log_probs_for_labels(
    log_probs: Tensor,
    labels: Tensor,
    label_pad_token_id: int = -100,
) -> tuple[Tensor, Tensor]:
    """
    Gather per-token log-probs for shifted labels.
    Returns (log_probs, mask). Masked positions are zeroed — safe for gather on CUDA.
    """
    target_ids = labels[:, 1:].clone()
    loss_mask = target_ids != label_pad_token_id
    # gather does not accept -100; substitute a dummy index then mask out
    safe_ids = target_ids.masked_fill(~loss_mask, 0)
    per_token = torch.gather(log_probs, dim=2, index=safe_ids.unsqueeze(-1)).squeeze(-1)
    return per_token * loss_mask.float(), loss_mask


def get_per_token_logps(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    use_ref: bool = False,
    label_pad_token_id: int = -100,
) -> Tensor:
    """
    Per-token log-probabilities aligned with `labels`.
    Prompt / padding positions are masked to 0 in the returned tensor.
    """
    if use_ref and hasattr(model, "disable_adapter"):
        context = model.disable_adapter()
    else:
        from contextlib import nullcontext

        context = nullcontext()

    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token, _ = gather_log_probs_for_labels(log_probs, labels, label_pad_token_id)
    return per_token


def normalize_weights(weights: Tensor, mask: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize so mean weight over active tokens is 1."""
    active = weights * mask
    denom = active.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return active / denom.clamp_min(eps)


def clamp_weights(weights: Tensor, w_min: float, w_max: float) -> Tensor:
    return weights.clamp(min=w_min, max=w_max)


def gaussian_weights(seq_len: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Positional Gaussian prior over T response tokens, shape (T,)."""
    if seq_len <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    mu = (seq_len - 1) / 2.0
    sigma = max(seq_len / 4.0, 1.0)
    w = torch.exp(-((t - mu) ** 2) / (2.0 * sigma**2))
    w = w / w.mean().clamp_min(1e-8)
    return w


def build_gaussian_weights(mask: Tensor) -> Tensor:
    """Build per-batch Gaussian weights from response mask. Shape (B, L-1)."""
    batch, length = mask.shape
    out = torch.zeros_like(mask, dtype=torch.float32)
    for b in range(batch):
        idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        # mask is on shifted label positions; use count of active tokens
        gw = gaussian_weights(idx.numel(), mask.device, mask.dtype)
        out[b, idx] = gw
    return out


def build_surprisal_weights(
    ref_logps: Tensor,
    mask: Tensor,
    w_min: float = 0.2,
    w_max: float = 3.0,
) -> Tensor:
    """Surprisal = -ref_logp, normalized then clamped."""
    raw = (-ref_logps).clamp_min(0.0) * mask
    weights = normalize_weights(raw, mask)
    return clamp_weights(weights, w_min, w_max) * mask


def build_uniform_weights(mask: Tensor) -> Tensor:
    return mask.float()


def weighted_sequence_score(
    policy_logps: Tensor,
    ref_logps: Tensor,
    mask: Tensor,
    weights: Tensor,
) -> Tensor:
    """Scalar preference score per example: sum_t w_t * (log pi - log pi_ref)."""
    adv = (policy_logps - ref_logps) * mask
    return (adv * weights).sum(dim=-1)


def dpo_loss(
    chosen_scores: Tensor,
    rejected_scores: Tensor,
    beta: float = 0.1,
) -> Tensor:
    logits = beta * (chosen_scores - rejected_scores)
    return -F.logsigmoid(logits).mean()


def compute_dpo_loss_from_logps(
    chosen_pi_logps: Tensor,
    rejected_pi_logps: Tensor,
    chosen_ref_logps: Tensor,
    rejected_ref_logps: Tensor,
    chosen_mask: Tensor,
    rejected_mask: Tensor,
    weight_method: WeightMethod,
    beta: float,
    chosen_external_weights: Tensor | None = None,
    rejected_external_weights: Tensor | None = None,
    surprisal_w_min: float = 0.2,
    surprisal_w_max: float = 3.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    if weight_method == WeightMethod.UNIFORM:
        cw = build_uniform_weights(chosen_mask)
        rw = build_uniform_weights(rejected_mask)
    elif weight_method == WeightMethod.GAUSSIAN:
        cw = build_gaussian_weights(chosen_mask)
        rw = build_gaussian_weights(rejected_mask)
    elif weight_method == WeightMethod.SURPRISAL:
        cw = build_surprisal_weights(chosen_ref_logps, chosen_mask, surprisal_w_min, surprisal_w_max)
        rw = build_surprisal_weights(rejected_ref_logps, rejected_mask, surprisal_w_min, surprisal_w_max)
    elif weight_method == WeightMethod.CACHED_GRAD:
        if chosen_external_weights is None or rejected_external_weights is None:
            raise ValueError("cached_grad requires precomputed chosen/rejected weights")
        cw = chosen_external_weights * chosen_mask
        rw = rejected_external_weights * rejected_mask
    else:
        raise ValueError(f"Unknown weight method: {weight_method}")

    chosen_scores = weighted_sequence_score(chosen_pi_logps, chosen_ref_logps, chosen_mask, cw)
    rejected_scores = weighted_sequence_score(rejected_pi_logps, rejected_ref_logps, rejected_mask, rw)
    loss = dpo_loss(chosen_scores, rejected_scores, beta=beta)
    metrics = {
        "loss": loss.detach(),
        "chosen_score": chosen_scores.detach().mean(),
        "rejected_score": rejected_scores.detach().mean(),
        "margin": (chosen_scores - rejected_scores).detach().mean(),
    }
    return loss, metrics
