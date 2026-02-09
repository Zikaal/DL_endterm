from __future__ import annotations
import numpy as np
from typing import Dict

def time_split(dates: np.ndarray, train_end: str, val_end: str) -> Dict[str, np.ndarray]:
    """
    Split indices by date thresholds (inclusive for train_end/val_end).
    dates: numpy array of datetime64
    train_end, val_end: 'YYYY-MM-DD'
    """
    train_end_dt = np.datetime64(train_end)
    val_end_dt = np.datetime64(val_end)

    idx = np.arange(len(dates))
    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt) & (dates <= val_end_dt)
    test_mask = dates > val_end_dt

    return {
        "train": idx[train_mask],
        "val": idx[val_mask],
        "test": idx[test_mask],
    }

def walk_forward_splits(dates: np.ndarray, start_train_end: str, step_days: int = 252, val_days: int = 126):
    """
    Generate walk-forward splits for backtesting.
    Roughly assumes daily trading data; step_days ~ 1 year (252 trading days).
    Returns list of dict splits with train/val/test indices.
    """
    start_dt = np.datetime64(start_train_end)
    idx = np.arange(len(dates))

    # find starting train_end index
    start_i = int(np.searchsorted(dates, start_dt, side="right") - 1)
    splits = []
    i = start_i
    while i + val_days < len(dates) - 1:
        train_idx = idx[: i + 1]
        val_idx = idx[i + 1 : i + 1 + val_days]
        test_idx = idx[i + 1 + val_days : min(i + 1 + 2 * val_days, len(dates))]
        splits.append({"train": train_idx, "val": val_idx, "test": test_idx})
        i += step_days
    return splits
