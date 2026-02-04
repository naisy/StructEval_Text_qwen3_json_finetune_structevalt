"""Utilities to release CUDA memory between training and evaluation.

Background
----------
After training, the current Python process often still holds references to
CUDA tensors (model parameters, optimizer states, gradients, accelerator
buffers, etc.). If post-train evaluation runs in the same process, VRAM can
remain inflated and spill into shared/system GPU memory, severely slowing
inference.

We support two safe patterns:
- In-process cleanup: delete Trainer/model objects and flush CUDA cache.
- Process replacement: `os.execv` into a fresh eval process (recommended).
"""

from __future__ import annotations

import gc
from typing import Any, Optional


def cleanup_cuda() -> None:
    """Run GC and clear CUDA caching allocator if torch is available."""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # Best-effort: these are no-ops on some builds.
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        # Torch not installed (e.g., CPU-only utilities)
        return


def teardown_trainer(trainer: Any) -> None:
    """Best-effort release of Trainer/Accelerate references."""
    if trainer is None:
        return

    # Free accelerator buffers if present.
    for attr in ("accelerator", "_accelerator"):
        acc = getattr(trainer, attr, None)
        if acc is not None:
            free_mem = getattr(acc, "free_memory", None)
            if callable(free_mem):
                try:
                    free_mem()
                except Exception:
                    pass

    # Drop common large refs.
    for k in (
        "model",
        "optimizer",
        "lr_scheduler",
        "train_dataset",
        "eval_dataset",
        "_train_batch_size",
        "state",
    ):
        if hasattr(trainer, k):
            try:
                setattr(trainer, k, None)
            except Exception:
                pass


def teardown_model(model: Any) -> None:
    """Move model to CPU (if possible) and drop references."""
    if model is None:
        return
    try:
        # Moving to CPU can release large parameter storages on some setups.
        cpu = getattr(model, "cpu", None)
        if callable(cpu):
            cpu()
    except Exception:
        pass


def cleanup_after_train(*, trainer: Any | None = None, model: Any | None = None) -> None:
    """Release as much training-time GPU memory as possible."""
    teardown_trainer(trainer)
    teardown_model(model)
    # Delete remaining references from our side.
    try:
        del trainer
    except Exception:
        pass
    try:
        del model
    except Exception:
        pass
    cleanup_cuda()
