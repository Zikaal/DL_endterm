from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class WindowedTSDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, lookback: int, indices: np.ndarray, scaler: Optional[StandardScaler] = None):
        self.X = X
        self.Y = Y
        self.lookback = int(lookback)
        self.indices = indices.astype(int)

        self.scaler = scaler

        # valid center indices must allow [t-lookback+1 ... t] window
        self.valid = self.indices[self.indices >= (self.lookback - 1)]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, i):
        t = int(self.valid[i])
        x = self.X[t - self.lookback + 1 : t + 1]  # [L, D]
        y = self.Y[t]                               # [H]
        return torch.from_numpy(x), torch.from_numpy(y)

def make_scaler(X: np.ndarray, train_indices: np.ndarray, lookback: int) -> StandardScaler:
    # Fit scaler on feature rows used in training windows only (avoid leakage)
    train_indices = train_indices.astype(int)
    train_valid = train_indices[train_indices >= (lookback - 1)]
    rows = []
    for t in train_valid:
        rows.append(X[t - lookback + 1 : t + 1])
    if not rows:
        raise ValueError("Not enough training points to build windows. Decrease lookback or provide more data.")
    stack = np.concatenate(rows, axis=0)  # [(N*L), D]
    sc = StandardScaler()
    sc.fit(stack)
    return sc

def apply_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    # Apply scaler per-row
    X2 = scaler.transform(X)
    return X2.astype(np.float32)

def make_loaders(
    X: np.ndarray,
    Y: np.ndarray,
    splits: Dict[str, np.ndarray],
    lookback: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    scaler = make_scaler(X, splits["train"], lookback)
    Xs = apply_scaler(X, scaler)

    train_ds = WindowedTSDataset(Xs, Y, lookback, splits["train"], scaler)
    val_ds = WindowedTSDataset(Xs, Y, lookback, splits["val"], scaler)
    test_ds = WindowedTSDataset(Xs, Y, lookback, splits["test"], scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, scaler
