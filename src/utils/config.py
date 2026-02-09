from __future__ import annotations
from typing import Any, Dict, Optional, Set
import os
import yaml

def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out

def _resolve_base_path(cfg_path: str, base_path: str) -> str:
    # Try as-is (relative to CWD), then relative to current cfg file
    cands = [
        os.path.normpath(base_path),
        os.path.normpath(os.path.join(os.path.dirname(cfg_path), base_path)),
    ]
    for cand in cands:
        if os.path.exists(cand):
            return cand
    # fallback: return path relative to cfg dir
    return os.path.normpath(os.path.join(os.path.dirname(cfg_path), base_path))

def load_config(cfg_path: str, _seen: Optional[Set[str]] = None) -> Dict[str, Any]:
    if _seen is None:
        _seen = set()

    cfg_path = os.path.normpath(cfg_path)
    abs_path = os.path.abspath(cfg_path)

    if abs_path in _seen:
        raise ValueError(f"Circular _base reference detected: {abs_path}")
    _seen.add(abs_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping/dict: {cfg_path}")

    base_path = cfg.get("_base")
    if base_path:
        base_path = _resolve_base_path(cfg_path, str(base_path))
        base_cfg = load_config(base_path, _seen=_seen)   # <-- РЕКУРСИЯ
        cfg = _deep_update(base_cfg, {k: v for k, v in cfg.items() if k != "_base"})

    return cfg
