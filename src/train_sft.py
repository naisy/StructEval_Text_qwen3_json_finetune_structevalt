from __future__ import annotations

import inspect
import os
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore

from src.data.dataset import load_dataset_any, build_prompt
from src.models.load import load_model, load_tokenizer
from src.models.lora import build_lora_config, apply_lora
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed
from src.utils.logging import info, warn
from src.experiments.run_manager import start_run, save_json




def _try_force_checkpoint(trainer: Trainer) -> bool:
    """Try to force a full Trainer checkpoint (model+optimizer+state).

    Hugging Face Trainer has an internal `_save_checkpoint` API that saves a
    resumable checkpoint directory (checkpoint-<global_step>).
    We call it in a version-tolerant way when available.
    """
    if not hasattr(trainer, "_save_checkpoint"):
        return False
    try:
        fn = getattr(trainer, "_save_checkpoint")
        sig = inspect.signature(fn)
        kwargs = {}
        if "model" in sig.parameters:
            kwargs["model"] = trainer.model
        if "trial" in sig.parameters:
            kwargs["trial"] = None
        if "metrics" in sig.parameters:
            kwargs["metrics"] = None
        fn(**kwargs)
        return True
    except Exception as e:  # pragma: no cover
        warn(f"Failed to force-save Trainer checkpoint: {e}")
        return False

def run_sft(
    config_path: str,
    *,
    run_name: str | None = None,
    train_path_override: str | None = None,
    valid_path_override: str | None = None,
    eval_tasks_path_override: str | None = None,
) -> None:
    cfg = load_yaml(config_path)
    rc = start_run(stage="sft", run_name=run_name)
    # Use per-run output dir for easy comparison
    cfg["training"]["output_dir"] = str(rc.run_dir / "checkpoints")
    save_json(rc.run_dir / "config.json", cfg)
    set_seed(int(cfg["training"].get("seed", 42)))
    out_dir = ensure_dir(cfg["training"]["output_dir"])

    # Load dataset config
    dcfg = load_yaml(cfg["data"]["dataset_config"])
    dcfg.setdefault("dataset", {})
    if train_path_override:
        dcfg["dataset"]["train_path"] = str(train_path_override)
    if valid_path_override:
        dcfg["dataset"]["valid_path"] = str(valid_path_override)
    train_items = load_dataset_any(dcfg["dataset"]["train_path"], dcfg["dataset"]["format"])
    valid_items = load_dataset_any(dcfg["dataset"]["valid_path"], dcfg["dataset"]["format"])

    def _target(ex: dict) -> str:
        out = (ex.get("output") or "").strip()
        if not out:
            raise ValueError(
                "SFT datasets must provide an 'output' field (target text). "
                "StructEval-T JSON tasks do not include gold outputs, so they are not suitable for SFT."
            )
        return out

    # Build prompts (labels = full prompt + output)
    train_texts = [build_prompt(ex, dcfg) + _target(ex) for ex in train_items]
    valid_texts = [build_prompt(ex, dcfg) + _target(ex) for ex in valid_items]

    ds_train = Dataset.from_dict({"text": train_texts})
    ds_valid = Dataset.from_dict({"text": valid_texts})

    model_name = cfg["model"]["base_model"]
    tok = load_tokenizer(model_name, trust_remote_code=cfg["model"].get("trust_remote_code", True))

    dtype = torch.bfloat16 if cfg["training"].get("bf16", False) else torch.float16
    model = load_model(model_name, dtype=dtype, trust_remote_code=cfg["model"].get("trust_remote_code", True))

    lora_cfg = build_lora_config(cfg.get("lora", {}), model=model)
    model = apply_lora(model, lora_cfg)

    # Training-time memory controls
    # - disable_cache: avoid keeping past_key_values during training
    # - gradient_checkpointing: reduce activation memory (slower)
    if bool(cfg["training"].get("disable_cache", True)):
        if getattr(model, "config", None) is not None:
            try:
                model.config.use_cache = False
            except Exception:
                pass

    if bool(cfg["training"].get("gradient_checkpointing", False)):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            warn("gradient_checkpointing requested but not supported by this model.")
        if getattr(model, "config", None) is not None:
            try:
                model.config.use_cache = False
            except Exception:
                pass

    # Tokenize
    max_len = int(cfg["training"].get("max_seq_len", 2048))
    pad_to_max = bool(cfg["training"].get("pad_to_max_length", False))

    def tok_fn(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length" if pad_to_max else False,
        )

    ds_train = ds_train.map(tok_fn, batched=True, remove_columns=["text"])
    ds_valid = ds_valid.map(tok_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Transformers has renamed/changed some TrainingArguments fields across versions.
    # For example, newer versions may use `eval_strategy` instead of `evaluation_strategy`.
    # Decide logging backend. Default to "none" to avoid interactive prompts.
    # If the environment disables W&B (WANDB_DISABLED), we must not request report_to=wandb.
    report_to_cfg = str(cfg["training"].get("report_to", "none")).strip().lower()
    wandb_disabled = str(os.getenv("WANDB_DISABLED", "")).strip().lower() in {"1", "true", "yes"}
    if wandb_disabled:
        report_to = []
    else:
        report_to = ["wandb"] if report_to_cfg in {"wandb", "w&b"} else []

    empty_cache_steps = int(cfg["training"].get("torch_empty_cache_steps", 0) or 0)
    # Some transformers versions require this to be >0 when provided.
    # Treat 0/None as "disabled" by omitting the argument.

    ta_kwargs = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(cfg["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["training"]["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        num_train_epochs=float(cfg["training"]["num_train_epochs"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
        logging_steps=int(cfg["training"].get("logging_steps", 10)),
        eval_steps=int(cfg["training"].get("eval_steps", 200)),
        save_steps=int(cfg["training"].get("save_steps", 200)),
        save_total_limit=int(cfg["training"].get("save_total_limit", 3)),
        save_first_step=bool(cfg["training"].get("save_first_step", True)),
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=bool(cfg["training"].get("bf16", False)),
        fp16=bool(cfg["training"].get("fp16", False)),
        report_to=report_to,
        # Optional knobs (filtered out automatically if your transformers version lacks them)
        group_by_length=bool(cfg["training"].get("group_by_length", False)),
        dataloader_num_workers=int(cfg["training"].get("dataloader_num_workers", 0)),
    )

    if empty_cache_steps > 0:
        ta_kwargs["torch_empty_cache_steps"] = empty_cache_steps

    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())

    # Rename `evaluation_strategy` -> `eval_strategy` when needed.
    if "evaluation_strategy" in ta_kwargs and "evaluation_strategy" not in params:
        if "eval_strategy" in params:
            ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
        else:
            ta_kwargs.pop("evaluation_strategy", None)

    # Drop any unknown kwargs to be robust across versions.
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in params}

    args = TrainingArguments(**ta_kwargs)

    info("Starting SFT training...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        tokenizer=tok,
    )
    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        warn("SFT training interrupted (Ctrl+C). Saving a checkpoint/model before exiting...")
        # 1) Prefer a full resumable checkpoint if possible.
        _try_force_checkpoint(trainer)
        # 2) Also save the current model weights for inspection/inference.
        interrupt_dir = ensure_dir(out_dir / "interrupted")
        trainer.save_model(str(interrupt_dir))
        tok.save_pretrained(str(interrupt_dir))
        save_json(rc.run_dir / "train_done.json", {"status": "interrupted"})
        return
    save_json(rc.run_dir / "train_done.json", {"status": "ok"})
    final_dir = out_dir / "final"
    # If we used LoRA, save a merged model for downstream stages (e.g., GRPO)
    # to avoid stacking multiple PEFT adapters by accident.
    merge_before_save = bool((cfg.get('training') or {}).get('merge_lora_before_save', (cfg.get('lora') or {}).get('merge_before_save', True)))
    if PeftModel is not None and isinstance(trainer.model, PeftModel) and merge_before_save:
        info("Merging LoRA adapter into base model before saving final checkpoint...")
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(final_dir), safe_serialization=True)
        tok.save_pretrained(str(final_dir))
        # Also keep the adapter itself for inspection/debugging.
        adapter_dir = ensure_dir(out_dir / "adapter")
        trainer.model.save_pretrained(str(adapter_dir))
    else:
        trainer.save_model(str(final_dir))
        tok.save_pretrained(str(final_dir))

    info("SFT done.")
    if cfg.get("eval", {}).get("run_eval_after_train", False):
        import subprocess
        import sys
        import yaml
        from pathlib import Path

        from src.utils.torch_memory import cleanup_after_train

        # Resolve eval tasks path. Priority:
        # 1) EVAL_TASKS_PATH env var
        # 2) configs/dataset_sft.yaml: eval_dataset.path
        # 3) common downloaded defaults (StructEval multi-format)
        # 4) local mock tasks
        sft_dcfg = load_yaml(cfg["data"]["dataset_config"])
        env_override = os.getenv("EVAL_TASKS_PATH", "").strip()
        arg_override = str(eval_tasks_path_override or "").strip()
        cfg_path = str((sft_dcfg.get("eval_dataset") or {}).get("path") or "").strip()
        fallback_path = str((sft_dcfg.get("eval_dataset") or {}).get("fallback_path") or "data/valid_structeval_t.json").strip()

        candidates: list[str] = []
        if arg_override:
            candidates.append(arg_override)
        if env_override:
            candidates.append(env_override)
        if cfg_path:
            candidates.append(cfg_path)
        candidates += [
            "data/structeval_text_all.json",
            "data/structeval_json_eval.json",
            fallback_path,
            "data/valid_structeval_t.json",
        ]

        selected = None
        for c in candidates:
            if c and os.path.isfile(c):
                selected = c
                break

        if selected is None:
            warn("No eval dataset found. Generating mock tasks and using them for post-eval.")
            subprocess.check_call([
                "python", "-m", "src.data.make_mock_structeval_t",
                # Default post-eval expects 10 items per output type.
                # Keep the fallback aligned with configs/eval.yaml defaults.
                "--n-train", "250", "--n-valid", "50",
                "--output-types", "JSON,YAML,TOML,XML,CSV",
            ])
            selected = "data/valid_structeval_t.json"

        eval_cfg_path = cfg["eval"]["eval_config"]
        base_eval_cfg = load_yaml(eval_cfg_path)
        base_dataset_cfg_path = (base_eval_cfg.get("data") or {}).get("dataset_config", "configs/dataset_eval.yaml")
        base_dataset_cfg = load_yaml(base_dataset_cfg_path)
        base_dataset_cfg.setdefault("dataset", {})
        base_dataset_cfg["dataset"]["format"] = "structeval_json"
        base_dataset_cfg["dataset"]["train_path"] = selected
        base_dataset_cfg["dataset"]["valid_path"] = selected

        tmp_dir = ensure_dir("outputs/tmp")
        tmp_dataset_cfg = Path(tmp_dir) / "dataset.posteval.yaml"
        tmp_dataset_cfg.write_text(
            yaml.safe_dump(base_dataset_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        base_eval_cfg.setdefault("data", {})
        base_eval_cfg["data"]["dataset_config"] = str(tmp_dataset_cfg)
        tmp_eval_cfg = Path(tmp_dir) / "eval.posteval.yaml"
        tmp_eval_cfg.write_text(
            yaml.safe_dump(base_eval_cfg, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        # Post-train eval can be *much* slower if we keep training-time CUDA
        # allocations alive (optimizer states, grad buffers, etc.). To avoid VRAM
        # spill to shared memory, we release trainer/model tensors first.
        cleanup_after_train(trainer=trainer, model=model)
        # Drop local references so GC can collect aggressively in in-process mode.
        trainer = None
        model = None

        mode = str((cfg.get("eval", {}) or {}).get("post_train_eval_mode") or "exec").strip().lower()
        final_path = str(out_dir / "final")

        # Build an eval command that reloads the saved model in a clean context.
        cmd = [
            sys.executable,
            "-m",
            "src.cli",
            "eval",
            "--config",
            str(tmp_eval_cfg),
            "--model-path",
            final_path,
            "--eval-tasks-path",
            str(selected),
        ]

        if mode == "inprocess":
            # In-process eval: model will be reloaded by run_eval after cleanup.
            from src.eval.run_eval import run_eval
            run_eval(str(tmp_eval_cfg), override_model_path=final_path)
        elif mode == "subprocess":
            subprocess.check_call(cmd)
        else:
            # Recommended: replace the current process so all training memory is gone.
            os.execv(sys.executable, cmd)