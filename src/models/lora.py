from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model


def guess_target_modules(model) -> list[str]:
    """Heuristic list for common transformer blocks.

    You should tailor this for the exact model architecture if needed.
    """
    # A pragmatic default; refine after inspecting model.named_modules()
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_lora_config(cfg: dict[str, Any], model=None) -> LoraConfig | None:
    if not cfg.get("enabled", True):
        return None
    target = cfg.get("target_modules", "auto")
    if target == "auto":
        if model is None:
            # fallback list; better to pass model
            target = ["q_proj", "k_proj", "v_proj", "o_proj"]
        else:
            target = guess_target_modules(model)

    return LoraConfig(
        r=int(cfg.get("r", 16)),
        lora_alpha=int(cfg.get("alpha", 32)),
        lora_dropout=float(cfg.get("dropout", 0.0)),
        target_modules=target,
        bias="none",
        task_type="CAUSAL_LM",
    )


def apply_lora(model, lora_cfg: LoraConfig | None):
    if lora_cfg is None:
        return model
    return get_peft_model(model, lora_cfg)
