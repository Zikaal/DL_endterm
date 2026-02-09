from __future__ import annotations
import numpy as np
from typing import List, Dict
from src.data.split import walk_forward_splits

def simple_directional_pnl(yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Toy backtest: take position sign(pred) each step.
    Returns per-step pnl for each horizon dimension (not realistic, but OK for demo).
    """
    pos = np.sign(yhat)
    return pos * y

def walk_forward_plan(dates: np.ndarray, start_train_end: str, step_days: int = 252, val_days: int = 126):
    return walk_forward_splits(dates, start_train_end=start_train_end, step_days=step_days, val_days=val_days)
