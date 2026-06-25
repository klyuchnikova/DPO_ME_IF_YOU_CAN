"""Online TI-DPO hybrid importance: DPO-NLL gradient + Gaussian (matches CachedGrad grad path)."""

from __future__ import annotations

import torch
from torch import Tensor

from src.attribution import compute_cached_grad_importance, importance_to_weights
from src.dpo import build_gaussian_weights, normalize_weights


class OnlineHybridWeightComputer:
    """Recompute λ·grad + (1-λ)·gaussian weights every N optimizer steps with EMA."""

    def __init__(
        self,
        model,
        lambda_blend: float = 0.7,
        sigma_scale: float = 4.0,  # kept for config compat; gaussian uses seq_len/4 in dpo.py
        update_freq: int = 10,
        ema_decay: float = 0.9,
    ):
        self.model = model
        self.lambda_blend = lambda_blend
        self.update_freq = max(1, update_freq)
        self.ema_decay = ema_decay
        self.optimizer_step = 0
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def _hybrid_weights(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Per-token weights aligned with logprob positions (T-1), mean-normalized."""
        mask = (labels[:, 1:] != -100).float()
        grad_imp = compute_cached_grad_importance(
            self.model, input_ids, attention_mask, labels
        )
        grad_w = importance_to_weights(grad_imp, mask)
        gauss_w = build_gaussian_weights(mask)
        blended = self.lambda_blend * grad_w + (1.0 - self.lambda_blend) * gauss_w
        return normalize_weights(blended, mask)

    def _ema_update(self, ex_id: int, chosen_w: Tensor, rejected_w: Tensor) -> tuple[Tensor, Tensor]:
        if ex_id in self._cache:
            prev_c, prev_r = self._cache[ex_id]
            if prev_c.shape == chosen_w.shape and prev_r.shape == rejected_w.shape:
                a = self.ema_decay
                chosen_w = a * prev_c.to(chosen_w.device) + (1 - a) * chosen_w
                rejected_w = a * prev_r.to(rejected_w.device) + (1 - a) * rejected_w
        self._cache[ex_id] = (chosen_w.detach().cpu(), rejected_w.detach().cpu())
        return chosen_w, rejected_w

    def maybe_recompute(
        self,
        example_ids: list,
        chosen_ids: Tensor,
        chosen_attn: Tensor,
        chosen_labels: Tensor,
        rejected_ids: Tensor,
        rejected_attn: Tensor,
        rejected_labels: Tensor,
        force: bool = False,
    ) -> tuple[Tensor, Tensor]:
        should_update = force or self.optimizer_step % self.update_freq == 0
        if not should_update and all(int(i) in self._cache for i in example_ids):
            chosen_list, rejected_list = [], []
            for ex_id in example_ids:
                cw, rw = self._cache[int(ex_id)]
                chosen_list.append(cw.to(chosen_ids.device))
                rejected_list.append(rw.to(rejected_ids.device))
            return self._pad_batch(chosen_list, chosen_labels), self._pad_batch(rejected_list, rejected_labels)

        chosen_ws, rejected_ws = [], []
        for i in range(chosen_ids.size(0)):
            ex_id = int(example_ids[i])
            cw = self._hybrid_weights(
                chosen_ids[i : i + 1], chosen_attn[i : i + 1], chosen_labels[i : i + 1]
            ).squeeze(0)
            rw = self._hybrid_weights(
                rejected_ids[i : i + 1], rejected_attn[i : i + 1], rejected_labels[i : i + 1]
            ).squeeze(0)
            cw, rw = self._ema_update(ex_id, cw, rw)
            chosen_ws.append(cw)
            rejected_ws.append(rw)

        return self._pad_batch(chosen_ws, chosen_labels), self._pad_batch(rejected_ws, rejected_labels)

    @staticmethod
    def _pad_batch(weight_list: list[Tensor], labels: Tensor) -> Tensor:
        width = labels.size(1) - 1
        out = torch.zeros(len(weight_list), width, device=labels.device, dtype=torch.float32)
        for i, w in enumerate(weight_list):
            n = min(w.numel(), width)
            out[i, :n] = w[:n]
        return out

    def on_optimizer_step(self) -> None:
        self.optimizer_step += 1
