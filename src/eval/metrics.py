from __future__ import annotations
import numpy as np
from typing import Dict

def mae(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yhat - y)))

def rmse(yhat: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yhat - y) ** 2)))

def directional_accuracy(yhat: np.ndarray, y: np.ndarray) -> float:
    # compares sign
    return float(np.mean((np.sign(yhat) == np.sign(y)).astype(np.float32)))

def per_horizon_metrics(yhat: np.ndarray, y: np.ndarray, horizons: list[int]) -> Dict[str, float]:
    out = {}
    for i, h in enumerate(horizons):
        out[f"mae_h{h}"] = mae(yhat[:, i], y[:, i])
        out[f"rmse_h{h}"] = rmse(yhat[:, i], y[:, i])
        out[f"diracc_h{h}"] = directional_accuracy(yhat[:, i], y[:, i])
    # overall averages
    out["mae_mean"] = float(np.mean([out[f"mae_h{h}"] for h in horizons]))
    out["rmse_mean"] = float(np.mean([out[f"rmse_h{h}"] for h in horizons]))
    out["diracc_mean"] = float(np.mean([out[f"diracc_h{h}"] for h in horizons]))
    return out
