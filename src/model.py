"""Model loading with LoRA policy and shared frozen reference via adapter disable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class ModelBundle:
    policy: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    dtype: torch.dtype

    @property
    def ref(self) -> PreTrainedModel:
        """Reference is the same weights with LoRA adapters disabled."""
        return self.policy


def default_lora_config(
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_policy_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    lora_config: LoraConfig | dict[str, Any] | None = None,
    gradient_checkpointing: bool = True,
    trust_remote_code: bool = False,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if lora_config is None:
        lora_config = default_lora_config()
    elif isinstance(lora_config, dict):
        lora_config = default_lora_config(**lora_config)

    model = get_peft_model(model, lora_config)
    model.to(device)

    return ModelBundle(policy=model, tokenizer=tokenizer, device=device, dtype=dtype)


def load_policy_from_checkpoint(
    base_model_name: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
) -> ModelBundle:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = PeftModel.from_pretrained(base, checkpoint_path)
    model.to(device)
    return ModelBundle(policy=model, tokenizer=tokenizer, device=device, dtype=dtype)


def save_lora_checkpoint(bundle: ModelBundle, output_dir: str) -> None:
    bundle.policy.save_pretrained(output_dir)
    bundle.tokenizer.save_pretrained(output_dir)
