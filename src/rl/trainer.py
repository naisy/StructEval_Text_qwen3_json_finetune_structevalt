from __future__ import annotations

"""TRL GRPOTrainer compatibility wrapper.

TRL and Transformers APIs change frequently. This project keeps all GRPOTrainer
construction quirks in one place so upgrades only touch this file.
"""

import inspect
from types import SimpleNamespace
from typing import Any


def build_grpo_trainer(*args: Any, **kwargs: Any):
    """Create a GRPOTrainer instance, adapting to TRL version differences."""

    from trl import GRPOTrainer  # type: ignore

    sig = inspect.signature(GRPOTrainer.__init__)
    params = sig.parameters
    accepts_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def has(name: str) -> bool:
        return name in params

    # ---------------------------------------------------------------------
    # Common arg renames across Transformers/TRL
    # ---------------------------------------------------------------------
    # tokenizer -> processing_class (Transformers v5+ alignment)
    if "tokenizer" in kwargs and not has("tokenizer"):
        if has("processing_class"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            kwargs.pop("tokenizer", None)

    # reward_funcs -> reward_func
    if "reward_funcs" in kwargs and not has("reward_funcs") and has("reward_func"):
        kwargs["reward_func"] = kwargs.pop("reward_funcs")

    # ---------------------------------------------------------------------
    # args/config handling (dict -> GRPOConfig/SimpleNamespace)
    # ---------------------------------------------------------------------
    args_dict = None
    if isinstance(kwargs.get("args"), dict):
        args_dict = dict(kwargs.get("args") or {})

    # If max length controls are passed as top-level kwargs but the TRL version
    # expects them inside args (GRPOConfig), inject them.
    if args_dict is not None:
        for src_name, dst_name in (
            ("max_prompt_length", "max_prompt_length"),
            ("max_prompt_len", "max_prompt_length"),
            ("max_completion_length", "max_completion_length"),
            ("max_completion_len", "max_completion_length"),
        ):
            if src_name in kwargs:
                args_dict.setdefault(dst_name, kwargs[src_name])

        # Prefer GRPOConfig when available.
        converted = None
        try:
            from trl import GRPOConfig  # type: ignore

            grpo_sig = inspect.signature(GRPOConfig.__init__)
            grpo_params = grpo_sig.parameters
            grpo_accepts_varkw = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in grpo_params.values()
            )
            if grpo_accepts_varkw:
                converted = GRPOConfig(**args_dict)
            else:
                filtered = {k: v for k, v in args_dict.items() if k in grpo_params}
                converted = GRPOConfig(**filtered)
        except Exception:
            # Fallback: attribute-access object.
            args_dict.setdefault("model_init_kwargs", None)
            converted = SimpleNamespace(**args_dict)

        # TRL versions differ in parameter naming: args= vs grpo_config=/config=
        if has("args"):
            kwargs["args"] = converted
        else:
            # Mirror into a supported param name
            for cand in ("grpo_config", "config"):
                if has(cand):
                    kwargs[cand] = converted
                    kwargs.pop("args", None)
                    break
            else:
                # As a last resort, keep it as "args" and rely on **kwargs.
                kwargs["args"] = converted

    # Some TRL versions take max_* lengths from args only; drop top-level kwargs
    # if the init doesn't accept them.
    for k in (
        "max_prompt_length",
        "max_completion_length",
        "max_prompt_len",
        "max_completion_len",
    ):
        if k in kwargs and not has(k):
            kwargs.pop(k, None)

    # Finally, filter unknown kwargs if the init does not accept **kwargs.
    if not accepts_varkw:
        kwargs = {k: v for k, v in kwargs.items() if k in params}

    return GRPOTrainer(*args, **kwargs)
