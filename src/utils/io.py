import os
import json
import numpy as np
from typing import Any, Dict

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_npz(path: str, **arrays) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def load_npz(path: str):
    return np.load(path, allow_pickle=True)
