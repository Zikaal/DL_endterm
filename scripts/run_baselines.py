from __future__ import annotations
import os
import argparse
import numpy as np

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json
from src.data.download import download_yf
from src.data.features import make_features
from src.data.split import time_split
from src.eval.metrics import per_horizon_metrics

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    df = download_yf(
        ticker=cfg["data"]["ticker"],
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        interval=cfg["data"].get("interval", "1d"),
        raw_dir="data/raw",
        force=False,
    )
    X, Y, dates, feature_names = make_features(
        df,
        horizons=cfg["features"]["horizons"],
        drop_na=bool(cfg["features"].get("drop_na", True)),
    )
    splits = time_split(dates, train_end=cfg["split"]["train_end"], val_end=cfg["split"]["val_end"])
    lookback = int(cfg["features"]["lookback"])
    horizons = cfg["features"]["horizons"]

    # align with dataset windowing: need t >= lookback-1
    test_idx = splits["test"]
    test_idx = test_idx[test_idx >= (lookback - 1)]

    y_true = Y[test_idx]
    H = len(horizons)

    # locate log_ret_1 feature
    idx_ret = 0
    if "log_ret_1" in feature_names:
        idx_ret = feature_names.index("log_ret_1")

    # Baseline A: zero return
    pred_zero = np.zeros_like(y_true, dtype=np.float32)

    # Baseline B: last observed daily return, repeated for all horizons
    last_ret = X[test_idx, idx_ret].astype(np.float32).reshape(-1, 1)
    pred_last = np.repeat(last_ret, repeats=H, axis=1)

    # Baseline C: mean return over the lookback window, repeated for all horizons
    pred_mean = np.empty((len(test_idx), H), dtype=np.float32)
    for i, t in enumerate(test_idx):
        mu = float(X[t - lookback + 1 : t + 1, idx_ret].mean())
        pred_mean[i, :] = mu

    results = {
        "zero_return": per_horizon_metrics(pred_zero, y_true, horizons),
        "last_return": per_horizon_metrics(pred_last, y_true, horizons),
        "mean_return": per_horizon_metrics(pred_mean, y_true, horizons),
    }

    out_dir = os.path.join(cfg["output"].get("save_dir", "results"), "logs", "baselines")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "baseline_metrics.json")
    save_json(out_path, results)

    print("Saved:", out_path)
    for name, m in results.items():
        print("\n==", name, "==")
        print(m)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
