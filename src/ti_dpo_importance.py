"""Online TI-DPO-style hybrid importance (gradient + Gaussian) via src.ti_ppo."""

from __future__ import annotations

import torch
from torch import Tensor

from src.dpo import normalize_weights
from src.ti_ppo.token_importance import HybridImportance


class OnlineHybridWeightComputer:
    """Recompute hybrid token weights every N steps with EMA smoothing (TI-PPO style)."""

    def __init__(
        self,
        model,
        lambda_blend: float = 0.7,
        sigma_scale: float = 4.0,
        update_freq: int = 10,
        ema_decay: float = 0.9,
    ):
        self.model = model
        self.scorer = HybridImportance(lambda_blend=lambda_blend, sigma_scale=sigma_scale)
        self.update_freq = max(1, update_freq)
        self.ema_decay = ema_decay
        self.optimizer_step = 0
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def _slice_to_logprob_positions(
        self,
        full_importance: Tensor,
        labels: Tensor,
        label_pad_token_id: int = -100,
    ) -> Tensor:
        """Map (B, T) importance to (B, T-1) response-aligned weights, mean-normalized."""
        mask = (labels[:, 1:] != label_pad_token_id).float()
        weights = full_importance[:, 1:].float() * mask
        return normalize_weights(weights, mask)

    @torch.no_grad()
    def _ema_update(self, ex_id: int, chosen_w: Tensor, rejected_w: Tensor) -> tuple[Tensor, Tensor]:
        if ex_id in self._cache:
            prev_c, prev_r = self._cache[ex_id]
            if prev_c.shape == chosen_w.shape and prev_r.shape == rejected_w.shape:
                a = self.ema_decay
                chosen_w = a * prev_c + (1 - a) * chosen_w
                rejected_w = a * prev_r + (1 - a) * rejected_w
        self._cache[ex_id] = (chosen_w.cpu(), rejected_w.cpu())
        return chosen_w, rejected_w

    def _compute_sequence_weights(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
    ) -> Tensor:
        importance = self.scorer.score(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        importance = torch.nan_to_num(importance, nan=1.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        return self._slice_to_logprob_positions(importance, labels)

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
            for i, ex_id in enumerate(example_ids):
                cw, rw = self._cache[int(ex_id)]
                chosen_list.append(cw.to(chosen_ids.device))
                rejected_list.append(rw.to(rejected_ids.device))
            return self._pad_batch(chosen_list, chosen_labels), self._pad_batch(rejected_list, rejected_labels)

        chosen_ws, rejected_ws = [], []
        for i in range(chosen_ids.size(0)):
            ex_id = int(example_ids[i])
            cw = self._compute_sequence_weights(
                chosen_ids[i : i + 1], chosen_attn[i : i + 1], chosen_labels[i : i + 1]
            ).squeeze(0)
            rw = self._compute_sequence_weights(
                rejected_ids[i : i + 1], rejected_attn[i : i + 1], rejected_labels[i : i + 1]
            ).squeeze(0)
            cw, rw = self._ema_update(ex_id, cw, rw)
            chosen_ws.append(cw.to(chosen_ids.device))
            rejected_ws.append(rw.to(rejected_ids.device))

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
