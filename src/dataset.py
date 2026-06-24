"""Dataset loading and DPO tokenization for Qwen-style chat models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


@dataclass
class PreferenceExample:
    id: str | int
    prompt: str
    chosen: str
    rejected: str


def load_preference_jsonl(path: str | Path) -> list[PreferenceExample]:
    examples: list[PreferenceExample] = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            for field in REQUIRED_FIELDS:
                if field not in row or not str(row[field]).strip():
                    raise ValueError(f"Line {i}: missing or empty '{field}'")
            ex_id = row.get("id", i)
            examples.append(
                PreferenceExample(
                    id=ex_id,
                    prompt=str(row["prompt"]),
                    chosen=str(row["chosen"]),
                    rejected=str(row["rejected"]),
                )
            )
    return examples


def format_prompt_response(tokenizer: PreTrainedTokenizerBase, prompt: str, response: str) -> str:
    """Use chat template when available; otherwise simple concatenation."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"User: {prompt}\nAssistant: {response}"


def format_prompt_only(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {prompt}\nAssistant:"


def tokenize_preference_pair(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    chosen: str,
    rejected: str,
    max_length: int,
    max_prompt_length: int,
    label_pad_token_id: int = -100,
) -> dict[str, Any]:
    prompt_text = format_prompt_only(tokenizer, prompt)
    chosen_text = format_prompt_response(tokenizer, prompt, chosen)
    rejected_text = format_prompt_response(tokenizer, prompt, rejected)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    def encode_response(full_text: str) -> tuple[list[int], list[int], list[int]]:
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        # Re-tokenize prefix to find prompt boundary (chat templates can merge tokens)
        prefix_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        if prefix_len > len(full_ids):
            prefix_len = len(full_ids)
        input_ids = full_ids[:max_length]
        labels = [label_pad_token_id] * len(input_ids)
        for j in range(prefix_len, len(input_ids)):
            labels[j] = input_ids[j]
        attn = [1] * len(input_ids)
        return input_ids, attn, labels

    c_ids, c_attn, c_labels = encode_response(chosen_text)
    r_ids, r_attn, r_labels = encode_response(rejected_text)

    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_attn,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_attn,
        "rejected_labels": r_labels,
    }


class PreferenceDataset(Dataset):
    def __init__(
        self,
        examples: list[PreferenceExample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        max_prompt_length: int = 256,
        cached_weights: list[dict[str, torch.Tensor]] | None = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.cached_weights = cached_weights

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        item = tokenize_preference_pair(
            self.tokenizer,
            ex.prompt,
            ex.chosen,
            ex.rejected,
            self.max_length,
            self.max_prompt_length,
        )
        item["id"] = ex.id
        if self.cached_weights is not None:
            item["chosen_weights"] = self.cached_weights[idx]["chosen_weights"]
            item["rejected_weights"] = self.cached_weights[idx]["rejected_weights"]
        return item


def preference_collate_fn(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    def pad_1d(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def pad_weights(seqs: list[torch.Tensor] | None, length_minus_1: int) -> torch.Tensor | None:
        if seqs is None:
            return None
        out = torch.zeros(len(seqs), length_minus_1, dtype=torch.float32)
        for i, w in enumerate(seqs):
            n = min(w.numel(), length_minus_1)
            out[i, :n] = w[:n].float()
        return out

    chosen_len = max(len(b["chosen_input_ids"]) for b in batch)
    rejected_len = max(len(b["rejected_input_ids"]) for b in batch)

    collated = {
        "ids": [b["id"] for b in batch],
        "chosen_input_ids": pad_1d([b["chosen_input_ids"] for b in batch], pad_token_id),
        "chosen_attention_mask": pad_1d([b["chosen_attention_mask"] for b in batch], 0),
        "chosen_labels": pad_1d([b["chosen_labels"] for b in batch], -100),
        "rejected_input_ids": pad_1d([b["rejected_input_ids"] for b in batch], pad_token_id),
        "rejected_attention_mask": pad_1d([b["rejected_attention_mask"] for b in batch], 0),
        "rejected_labels": pad_1d([b["rejected_labels"] for b in batch], -100),
    }

    if "chosen_weights" in batch[0]:
        collated["chosen_weights"] = pad_weights(
            [b["chosen_weights"] for b in batch], chosen_len - 1
        )
        collated["rejected_weights"] = pad_weights(
            [b["rejected_weights"] for b in batch], rejected_len - 1
        )

    return collated
