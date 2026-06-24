"""DPO / weighted-DPO training loop with 1–2 GPU and CPU support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from src.dataset import PreferenceDataset, preference_collate_fn
from src.dpo import WeightMethod, compute_dpo_loss_from_logps, get_per_token_logps
from src.model import ModelBundle, save_lora_checkpoint
from src.utils import ensure_dir, save_json


@dataclass
class TrainConfig:
    output_dir: str
    weight_method: WeightMethod = WeightMethod.UNIFORM
    beta: float = 0.1
    learning_rate: float = 5e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    surprisal_w_min: float = 0.2
    surprisal_w_max: float = 3.0
    use_amp: bool = True


@dataclass
class TrainStats:
    global_step: int = 0
    epoch: int = 0
    losses: list[float] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)


def _autocast_ctx(device: torch.device, dtype: torch.dtype, enabled: bool):
    if not enabled or device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_dpo(
    bundle: ModelBundle,
    train_dataset: PreferenceDataset,
    val_dataset: PreferenceDataset | None,
    config: TrainConfig,
) -> TrainStats:
    device = bundle.device
    dtype = bundle.dtype
    bundle.policy.train()

    collate = lambda batch: preference_collate_fn(batch, bundle.tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(
        (p for p in bundle.policy.parameters() if p.requires_grad),
        lr=config.learning_rate,
    )
    total_steps = max(1, len(train_loader) * config.num_epochs // config.gradient_accumulation_steps)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    stats = TrainStats()
    ensure_dir(config.output_dir)
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(config.num_epochs):
        stats.epoch = epoch
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.num_epochs}")
        for step, batch in enumerate(pbar):
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
            if ext_cw is not None:
                ext_cw = ext_cw.to(device)
                ext_rw = ext_rw.to(device)

            with _autocast_ctx(device, dtype, config.use_amp):
                chosen_pi = get_per_token_logps(bundle.policy, chosen_ids, chosen_attn, chosen_labels)
                rejected_pi = get_per_token_logps(bundle.policy, rejected_ids, rejected_attn, rejected_labels)
                with torch.no_grad():
                    chosen_ref = get_per_token_logps(
                        bundle.policy, chosen_ids, chosen_attn, chosen_labels, use_ref=True
                    )
                    rejected_ref = get_per_token_logps(
                        bundle.policy, rejected_ids, rejected_attn, rejected_labels, use_ref=True
                    )

                loss, metrics = compute_dpo_loss_from_logps(
                    chosen_pi,
                    rejected_pi,
                    chosen_ref,
                    rejected_ref,
                    chosen_mask,
                    rejected_mask,
                    config.weight_method,
                    config.beta,
                    chosen_external_weights=ext_cw,
                    rejected_external_weights=ext_rw,
                    surprisal_w_min=config.surprisal_w_min,
                    surprisal_w_max=config.surprisal_w_max,
                )
                loss = loss / config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(bundle.policy.parameters(), config.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                stats.global_step += 1

                if stats.global_step % config.logging_steps == 0:
                    pbar.set_postfix(
                        loss=f"{metrics['loss'].item():.4f}",
                        margin=f"{metrics['margin'].item():.4f}",
                    )

                if stats.global_step % config.save_steps == 0:
                    ckpt_dir = Path(config.output_dir) / f"checkpoint-{stats.global_step}"
                    save_lora_checkpoint(bundle, str(ckpt_dir))

            stats.losses.append(metrics["loss"].item())
            stats.step_times.append(time.perf_counter() - t0)

    final_dir = Path(config.output_dir) / "final"
    save_lora_checkpoint(bundle, str(final_dir))
    save_json(
        Path(config.output_dir) / "train_stats.json",
        {
            "global_step": stats.global_step,
            "mean_loss": sum(stats.losses) / max(len(stats.losses), 1),
            "mean_step_sec": sum(stats.step_times) / max(len(stats.step_times), 1),
            "weight_method": config.weight_method.value,
        },
    )
    return stats
