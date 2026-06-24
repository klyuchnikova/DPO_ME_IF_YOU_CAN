"""Token-importance precomputation: cached gradient attribution and surprisal."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.dpo import gather_log_probs_for_labels, normalize_weights


def _response_token_indices(labels_row: Tensor, label_pad_token_id: int = -100) -> Tensor:
    return (labels_row[1:] != label_pad_token_id).nonzero(as_tuple=False).squeeze(-1)


@torch.enable_grad()
def compute_cached_grad_importance(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    label_pad_token_id: int = -100,
) -> Tensor:
    """
    Gradient L1 norm w.r.t. input embeddings for response NLL under the reference model.
    Returns importance aligned with per-token logprob positions (length L-1).
    """
    model.eval()
    embed = model.get_input_embeddings()
    input_embeds = embed(input_ids)
    input_embeds = input_embeds.detach().requires_grad_(True)

    if hasattr(model, "disable_adapter"):
        ctx = model.disable_adapter()
    else:
        from contextlib import nullcontext

        ctx = nullcontext()

    with ctx:
        outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, use_cache=False)

    logits = outputs.logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    token_logps, loss_mask = gather_log_probs_for_labels(log_probs, labels, label_pad_token_id)
    mask = loss_mask.float()
    nll = -(token_logps * mask).sum()
    nll.backward()

    grad = input_embeds.grad.abs().sum(dim=-1)  # (B, L)
    # align to token_logps positions (shift by 1)
    grad_shifted = grad[:, 1:]
    importance = grad_shifted * mask
    return importance


def importance_to_weights(importance: Tensor, mask: Tensor) -> Tensor:
    weights = normalize_weights(importance, mask)
    return weights * mask


@torch.no_grad()
def compute_surprisal_importance(
    model: PreTrainedModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    label_pad_token_id: int = -100,
) -> Tensor:
    from src.dpo import get_per_token_logps

    logps = get_per_token_logps(model, input_ids, attention_mask, labels, use_ref=True)
    mask = labels[:, 1:] != label_pad_token_id
    return (-logps).clamp_min(0.0) * mask.float()


def precompute_cached_grad_weights(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: torch.device,
    label_pad_token_id: int = -100,
) -> list[dict[str, Tensor]]:
    """
    Iterate dataloader and return per-example dicts:
      {"chosen_weights": Tensor(Lc-1,), "rejected_weights": Tensor(Lr-1,)}
    Weights are normalized (mean=1 over response tokens).
    """
    results: list[dict[str, Tensor]] = []
    model.to(device)

    for batch in tqdm(dataloader, desc="cached_grad precompute"):
        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_attn = batch["chosen_attention_mask"].to(device)
        chosen_labels = batch["chosen_labels"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_attn = batch["rejected_attention_mask"].to(device)
        rejected_labels = batch["rejected_labels"].to(device)

        for i in range(chosen_ids.size(0)):
            c_imp = compute_cached_grad_importance(
                model,
                chosen_ids[i : i + 1],
                chosen_attn[i : i + 1],
                chosen_labels[i : i + 1],
                label_pad_token_id=label_pad_token_id,
            )[0]
            r_imp = compute_cached_grad_importance(
                model,
                rejected_ids[i : i + 1],
                rejected_attn[i : i + 1],
                rejected_labels[i : i + 1],
                label_pad_token_id=label_pad_token_id,
            )[0]
            c_mask = (chosen_labels[i, 1:] != label_pad_token_id).float()
            r_mask = (rejected_labels[i, 1:] != label_pad_token_id).float()
            results.append(
                {
                    "chosen_weights": importance_to_weights(c_imp, c_mask).cpu(),
                    "rejected_weights": importance_to_weights(r_imp, r_mask).cpu(),
                }
            )
            model.zero_grad(set_to_none=True)

    return results


def save_cached_grad_weights(weights: list[dict[str, Tensor]], path: str) -> None:
    torch.save(weights, path)


def load_cached_grad_weights(path: str) -> list[dict[str, Tensor]]:
    return torch.load(path, map_location="cpu", weights_only=True)


def pad_weights_to_batch(
    weight_list: list[Tensor],
    max_len: int,
    device: torch.device,
) -> Tensor:
    """Pad variable-length per-example weight vectors to (B, max_len)."""
    batch = []
    for w in weight_list:
        if w.numel() < max_len:
            pad = torch.zeros(max_len - w.numel(), dtype=w.dtype)
            w = torch.cat([w, pad], dim=0)
        else:
            w = w[:max_len]
        batch.append(w)
    return torch.stack(batch, dim=0).to(device)
