from __future__ import annotations

import json
import os
from pathlib import Path

import torch

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None  # type: ignore
from datasets import Dataset

from src.data.dataset import load_dataset_any, build_prompt
from src.data.validators import build_schema_validator
from src.models.load import load_model, load_tokenizer
from src.models.lora import build_lora_config, apply_lora
from src.rl.rewards import compute_reward_components, combine_reward
from src.data import validators as V
from src.structeval_t.scorer import eval_structeval_t
from src.rl.trainer import build_grpo_trainer
from src.utils.io import load_yaml, ensure_dir
from src.utils.seed import set_seed
from src.utils.logging import info, warn
from src.experiments.run_manager import start_run, save_json


def _find_latest_sft_final() -> str | None:
    """Return path to the latest SFT `checkpoints/final` directory, if any."""
    base = Path("runs") / "sft"
    if not base.exists():
        return None
    candidates = sorted(base.glob("**/checkpoints/final"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.is_dir():
            return str(p)
    return None


def run_grpo(
    config_path: str,
    *,
    run_name: str | None = None,
    train_path_override: str | None = None,
    valid_path_override: str | None = None,
    eval_tasks_path_override: str | None = None,
) -> None:
    cfg = load_yaml(config_path)
    rc = start_run(stage="grpo", run_name=run_name)
    cfg["training"]["output_dir"] = str(rc.run_dir / "checkpoints")
    save_json(rc.run_dir / "config.json", cfg)
    set_seed(int(cfg["training"].get("seed", 42)))
    out_dir = ensure_dir(cfg["training"]["output_dir"])

    # Dataset config
    dcfg = load_yaml(cfg["data"]["dataset_config"])
    dcfg.setdefault("dataset", {})
    if train_path_override:
        dcfg["dataset"]["train_path"] = str(train_path_override)
    if valid_path_override:
        dcfg["dataset"]["valid_path"] = str(valid_path_override)
    train_items = load_dataset_any(dcfg["dataset"]["train_path"], dcfg["dataset"]["format"])

    prompts = [build_prompt(ex, dcfg) for ex in train_items]
    ds_train = Dataset.from_dict(
        {
            "prompt": prompts,
            "task_id": [ex.get("task_id") for ex in train_items],
            "query": [ex.get("query") for ex in train_items],
            "raw_output_metric": [ex.get("raw_output_metric") for ex in train_items],
            "output_type": [ex.get("output_type") for ex in train_items],
        }
    )

    # Load schema if enabled
    schema_validator = None
    allowed_top = None
    if dcfg.get("schema", {}).get("enabled", False):
        schema_path = Path(dcfg["schema"]["schema_path"])
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema_validator = build_schema_validator(schema)
            if isinstance(schema.get("properties"), dict):
                allowed_top = set(schema["properties"].keys())
        else:
            warn(f"Schema enabled but not found: {schema_path}")

    # Model
    model_name = cfg["model"]["base_model"]
    if str(model_name).upper() == "AUTO_SFT":
        latest = _find_latest_sft_final()
        if latest is not None:
            info(f"GRPO base_model=AUTO_SFT -> {latest}")
            model_name = latest
        else:
            fallback = "Qwen/Qwen3-4B-Instruct-2507"
            warn(
                "GRPO base_model=AUTO_SFT but no runs/sft/**/checkpoints/final was found. "
                f"Falling back to {fallback}."
            )
            model_name = fallback
    tok = load_tokenizer(model_name, trust_remote_code=cfg["model"].get("trust_remote_code", True))
    dtype = torch.bfloat16 if cfg["training"].get("bf16", False) else torch.float16
    model = load_model(model_name, dtype=dtype, trust_remote_code=cfg["model"].get("trust_remote_code", True))

    # If the SFT checkpoint was saved as a PEFT adapter, loading it may yield a PeftModel.
    # Applying LoRA again would stack adapters (PEFT2重). That's almost never what we want here.
    # We default to merging/unloading the existing adapter, then applying the GRPO LoRA config.
    if PeftModel is not None and isinstance(model, PeftModel):
        if cfg.get('grpo', {}).get('merge_sft_lora_before_grpo', True):
            info('Merging existing PEFT adapter (from SFT) into the base model before GRPO...')
            model = model.merge_and_unload()
        else:
            warn('Loaded model is already a PEFT model; GRPO will continue without adding a second adapter.')



    lora_cfg = build_lora_config(cfg.get("lora", {}), model=model)
    if PeftModel is not None and isinstance(model, PeftModel):
        # Avoid stacking adapters; either merged above or user opted out.
        pass
    else:
        model = apply_lora(model, lora_cfg)

    # Reward function: TRL expects a callable that maps (prompts, completions) to rewards.
    def reward_fn(prompts: list[str], completions: list[str], **kwargs):
        """Composite reward.

        If the dataset example provides `raw_output_metric` (StructEval-T style),
        we score using StructEval-T formula:
            final = 0.2*syntax + 0.8*key_validation
        Otherwise we fallback to component-based reward.
        """
        rewards = []
        # TRL sometimes provides `kwargs['batch']` or similar; keep robust.
        batch = kwargs.get("batch")
        raw_metrics = None
        if isinstance(batch, dict) and "raw_output_metric" in batch:
            raw_metrics = batch["raw_output_metric"]

        # If raw_metrics is provided per-example, it may be a list-of-lists aligned with completions.
        for i, c in enumerate(completions):
            # Derive output_type for this example if provided by the dataset.
            otype = None
            if isinstance(batch, dict) and "output_type" in batch:
                ots = batch.get("output_type")
                if isinstance(ots, list) and i < len(ots):
                    otype = ots[i]
                else:
                    otype = ots
            otype = str(otype or "JSON").strip().upper()

            # Extract payload and flag extraneous wrapper text. We always score the payload,
            # but can optionally penalize wrapper text.
            payload, has_extraneous = V.extract_payload_and_extraneous(c, otype)

            # Prefer StructEval-T reward ONLY when we have a non-empty raw_output_metric.
            # (Many HF datasets don't include ATTRIBUTES blocks, so raw_output_metric becomes empty.)
            if raw_metrics is not None:
                m = None
                if isinstance(raw_metrics, list) and i < len(raw_metrics):
                    m = raw_metrics[i]
                else:
                    m = raw_metrics
                metric_list = [str(x) for x in (m or [])] if isinstance(m, (list, tuple)) else []
                if metric_list:
                    res = eval_structeval_t(payload, metric_list, output_type=otype)
                    r = float(res.final_eval_score)
                    if has_extraneous:
                        # format-specific override supported: p_extraneous_xml, etc.
                        pen = cfg.get("reward", {}).get(f"p_extraneous_{otype.lower()}")
                        if pen is None:
                            pen = cfg.get("reward", {}).get("p_extraneous", 0.0)
                        r += float(pen)
                    rewards.append(r)
                    continue

            # Otherwise use component-based deterministic reward shaping (+ optional gold matching).
            ref_out = None
            if isinstance(batch, dict) and "reference_output" in batch:
                ro = batch.get("reference_output")
                if isinstance(ro, list) and i < len(ro):
                    ref_out = ro[i]
                else:
                    ref_out = ro

            comps = compute_reward_components(
                completion=c,
                output_type=otype,
                schema_validator=schema_validator,
                allowed_top_level=allowed_top,
                reference_output=str(ref_out) if ref_out is not None else None,
            )
            rewards.append(combine_reward(comps, cfg, output_type=otype))
        return rewards


    # NOTE: TRL GRPO API details vary by version.
    # This template keeps the call minimal; adjust kwargs to match your installed TRL.
    info("Starting GRPO training...")
    trainer = build_grpo_trainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds_train,
        reward_funcs=reward_fn,
        args=dict(
            output_dir=str(out_dir),
            per_device_train_batch_size=int(cfg["training"]["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
            # TRL expects this flag on args for some versions (even when False).
            gradient_checkpointing=bool(cfg["training"].get("gradient_checkpointing", False)),
            learning_rate=float(cfg["training"]["learning_rate"]),
            warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.0)),
            logging_steps=int(cfg["training"].get("logging_steps", 10)),
            save_strategy="steps",
            save_steps=int(cfg["training"].get("save_steps", 200)),
            save_total_limit=int(cfg["training"].get("save_total_limit", 3)),
            save_first_step=bool(cfg["training"].get("save_first_step", True)),
            bf16=bool(cfg['training'].get('bf16', False)),
            max_prompt_length=int(cfg['training'].get('max_prompt_len', 1536)),
            max_completion_length=int(cfg['training'].get('max_completion_len', 512)),
        ),
        num_generations=int(cfg["grpo"].get("num_generations", 4)),
        temperature=float(cfg["grpo"].get("temperature", 0.9)),
        top_p=float(cfg["grpo"].get("top_p", 0.95)),
        beta=float(cfg["grpo"].get("beta", 0.05)),
        max_prompt_length=int(cfg["training"].get("max_prompt_len", 1536)),
        max_completion_length=int(cfg["training"].get("max_completion_len", 512)),
    )

    interrupted = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        interrupted = True
        warn("GRPO training interrupted (Ctrl+C). Saving a checkpoint/model before exiting...")
        interrupt_dir = ensure_dir(out_dir / "interrupted")
        try:
            trainer.save_model(str(interrupt_dir))
        except Exception as e:  # pragma: no cover
            warn(f"Failed to save interrupted GRPO model: {e}")
        tok.save_pretrained(str(interrupt_dir))
        save_json(rc.run_dir / "train_done.json", {"status": "interrupted"})
        return
    save_json(rc.run_dir / "train_done.json", {"status": "ok"})
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    info("GRPO done.")
    if cfg.get("eval", {}).get("run_eval_after_train", False):
        import subprocess
        import sys
        import yaml
        from pathlib import Path

        from src.utils.torch_memory import cleanup_after_train

        # Resolve eval tasks path. Priority:
        # 1) EVAL_TASKS_PATH env var
        # 2) dataset config: eval_dataset.path
        # 3) common downloaded defaults (StructEval multi-format)
        # 4) local mock tasks
        grpo_dcfg = load_yaml(cfg["data"]["dataset_config"])
        env_override = os.getenv("EVAL_TASKS_PATH", "").strip()
        arg_override = str(eval_tasks_path_override or "").strip()
        cfg_path = str((grpo_dcfg.get("eval_dataset") or {}).get("path") or "").strip()
        fallback_path = str((grpo_dcfg.get("eval_dataset") or {}).get("fallback_path") or "data/valid_structeval_t.json").strip()

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

        # Release training-time CUDA allocations before post-eval.
        cleanup_after_train(trainer=trainer, model=model)
        trainer = None
        model = None

        mode = str((cfg.get("eval", {}) or {}).get("post_train_eval_mode") or "exec").strip().lower()
        final_path = str(out_dir / "final")

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
            from src.eval.run_eval import run_eval
            run_eval(str(tmp_eval_cfg), override_model_path=final_path)
        elif mode == "subprocess":
            subprocess.check_call(cmd)
        else:
            os.execv(sys.executable, cmd)
