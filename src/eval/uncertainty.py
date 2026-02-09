from __future__ import annotations
import numpy as np
import torch
from typing import List, Tuple

@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    yhats = []
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        yhat, _ = model(x)
        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())
    y_np = np.concatenate(ys, axis=0) if ys else np.empty((0,))
    yhat_np = np.concatenate(yhats, axis=0) if yhats else np.empty((0,))
    return y_np, yhat_np

@torch.no_grad()
def ensemble_predict(models: List[torch.nn.Module], loader, device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y, mean_pred, std_pred across models."""
    all_preds = []
    y_true = None
    for m in models:
        y, yhat = predict(m, loader, device)
        if y_true is None:
            y_true = y
        all_preds.append(yhat)
    preds = np.stack(all_preds, axis=0)  # [K, N, H]
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return y_true, mean, std

def interval_from_std(mean: np.ndarray, std: np.ndarray, z: float = 1.645):
    """Approximate two-sided interval using normal z-score.
    z=1.645 -> ~90%, z=1.96 -> ~95%
    """
    lo = mean - z * std
    hi = mean + z * std
    return lo, hi

def enable_dropout(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, loader, device, n: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout: multiple stochastic passes (dropout ON at inference)."""
    model.eval()
    enable_dropout(model)
    preds = []
    y_true = None
    for _ in range(n):
        y, yhat = predict(model, loader, device)
        if y_true is None:
            y_true = y
        preds.append(yhat)
    preds = np.stack(preds, axis=0)  # [n, N, H]
    return y_true, preds.mean(axis=0), preds.std(axis=0)
