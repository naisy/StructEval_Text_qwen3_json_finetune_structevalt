from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

def cache_key(provider: str, model: str, payload: Dict[str, Any]) -> str:
    """Stable cache key for a judge request.

    NOTE: This key is intentionally independent from task_id; it hashes the full request payload.
    """
    h = hashlib.sha256()
    h.update(provider.encode("utf-8")); h.update(b"\0")
    h.update(model.encode("utf-8")); h.update(b"\0")
    h.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()

def cache_path(cache_dir: Path, provider: str, model: str, payload: Dict[str, Any]) -> Tuple[str, Path]:
    k = cache_key(provider, model, payload)
    p = cache_dir / provider / model / f"{k}.json"
    return k, p

def load_cache(cache_dir: Path, provider: str, model: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _k, p = cache_path(cache_dir, provider, model, payload)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def _merge_task_ids(existing: Dict[str, Any], new_task_id: Optional[str]) -> Dict[str, Any]:
    if not new_task_id:
        return existing

    ids: List[str] = []
    if isinstance(existing.get("task_ids"), list):
        ids = [str(x) for x in existing["task_ids"] if x is not None]
    elif existing.get("task_id") is not None:
        ids = [str(existing["task_id"])]

    ids.append(str(new_task_id))
    # Preserve order but dedupe
    seen=set()
    dedup=[]
    for x in ids:
        if x not in seen:
            seen.add(x)
            dedup.append(x)

    # Store both for convenience
    existing["task_id"] = dedup[0]
    existing["task_ids"] = dedup
    return existing

def save_cache(cache_dir: Path, provider: str, model: str, payload: Dict[str, Any], value: Dict[str, Any]) -> None:
    k, p = cache_path(cache_dir, provider, model, payload)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Merge task_id/task_ids if present (helps debugging which task produced this cache entry).
    existing: Dict[str, Any] = {}
    if p.exists():
        try:
            existing = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    merged = dict(value)
    merged = _merge_task_ids(merged, merged.get("task_id") if isinstance(merged.get("task_id"), str) else None)
    merged = _merge_task_ids(merged, existing.get("task_id") if isinstance(existing.get("task_id"), str) else None)
    if isinstance(existing.get("task_ids"), list) and merged.get("task_id"):
        # include any previous ids
        for tid in existing.get("task_ids", []):
            merged = _merge_task_ids(merged, str(tid))

    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
