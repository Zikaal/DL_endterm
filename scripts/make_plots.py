from __future__ import annotations
import os
import argparse
import numpy as np

from src.utils.io import load_npz, ensure_dir
from src.eval.plots import (
    plot_learning_curve,
    plot_predictions_with_intervals,
    plot_attention_weights,
    plot_interval_coverage,
)

def main(run_dir: str):
    # run_dir: results/logs/<run_name>
    fig_dir = os.path.join("results", "figures", os.path.basename(run_dir))
    ensure_dir(fig_dir)

    log_csv = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(log_csv):
        plot_learning_curve(log_csv, os.path.join(fig_dir, "learning_curve.png"))

    # single model predictions
    pred_path = os.path.join(run_dir, "predictions_test.npz")
    if os.path.exists(pred_path):
        npz = load_npz(pred_path)
        y = npz["y"]
        yhat = npz["yhat"]
        horizons = npz["horizons"].tolist()
        attn = npz["attn"] if "attn" in npz.files else None
        # handle saved None as an object array
        if attn is not None and isinstance(attn, np.ndarray) and attn.dtype == object:
            try:
                if attn.size == 1 and attn.item() is None:
                    attn = None
            except Exception:
                pass
        plot_predictions_with_intervals(y, yhat, None, None, horizons, os.path.join(fig_dir, "pred_vs_true.png"))
        if attn is not None and attn.size > 0:
            plot_attention_weights(attn, os.path.join(fig_dir, "attention_avg.png"))

    # ensemble predictions
    ens_path = os.path.join(run_dir, "ensemble_predictions_test.npz")
    if os.path.exists(ens_path):
        npz = load_npz(ens_path)
        y = npz["y"]
        mean = npz["mean"]
        lo = npz["lo"]
        hi = npz["hi"]
        horizons = npz["horizons"].tolist()
        plot_predictions_with_intervals(y, mean, lo, hi, horizons, os.path.join(fig_dir, "ensemble_pred_interval.png"))
        plot_interval_coverage(y, lo, hi, os.path.join(fig_dir, "coverage90.png"))

    print("Figures saved to:", fig_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to run directory, e.g. results/logs/<run_name>")
    args = ap.parse_args()
    main(args.run)
