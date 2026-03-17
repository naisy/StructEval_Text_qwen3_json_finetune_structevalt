from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def build_sft_feature(
    *,
    tokenizer: Any,
    prompt_text: str,
    target_text: str,
    eos_text: str,
    max_length: int,
    assistant_only_loss: bool,
) -> dict[str, list[int]]:
    """Encode one SFT example with optional assistant-only label masking.

    The repo already formats chat prompts explicitly via `build_prompt(...)`, so we
    tokenize with `add_special_tokens=False` to avoid injecting extra BOS/EOS tokens.

    Labels:
    - assistant_only_loss=False: standard causal LM labels over the full sequence.
    - assistant_only_loss=True : prompt/user tokens are masked to -100 and only the
      assistant completion (target + EOS) contributes to the loss.
    """
    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be a non-empty string")
    if not isinstance(target_text, str) or not target_text:
        raise ValueError("target_text must be a non-empty string")

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(f"{target_text}{eos_text}", add_special_tokens=False)
    input_ids = (prompt_ids + target_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    if assistant_only_loss:
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = ([-100] * prompt_len) + input_ids[prompt_len:]
    else:
        labels = list(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


@dataclass
class SFTDataCollator:
    """Pad causal-LM features with explicit labels.

    This collator assumes each feature already contains `input_ids`,
    `attention_mask`, and `labels`.
    """

    tokenizer: Any

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("features must not be empty")

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("tokenizer.pad_token_id must be set for SFTDataCollator")

        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feat in features:
            seq_len = len(feat["input_ids"])
            pad_len = max_len - seq_len
            batch_input_ids.append(feat["input_ids"] + [pad_id] * pad_len)
            batch_attention_mask.append(feat["attention_mask"] + [0] * pad_len)
            batch_labels.append(feat["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }
