from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
from src.utils.io import ensure_dir

def plot_learning_curve(log_csv: str, out_path: str):
    df = pd.read_csv(log_csv)
    ensure_dir(os.path.dirname(out_path))
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_predictions_with_intervals(
    y: np.ndarray,
    mean: np.ndarray,
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    horizons: List[int],
    out_path: str,
    max_points: int = 500,
):
    ensure_dir(os.path.dirname(out_path))
    n = min(len(y), max_points)
    x = np.arange(n)

    for i, h in enumerate(horizons):
        plt.figure()
        plt.plot(x, y[:n, i], label="true")
        plt.plot(x, mean[:n, i], label="pred_mean")
        if lo is not None and hi is not None:
            plt.fill_between(x, lo[:n, i], hi[:n, i], alpha=0.25, label="interval")
        plt.title(f"Horizon {h}")
        plt.xlabel("Time index (subset)")
        plt.ylabel("Log return")
        plt.legend()
        plt.tight_layout()
        base, ext = os.path.splitext(out_path)
        plt.savefig(f"{base}_h{h}{ext}", dpi=160)
        plt.close()

def plot_attention_weights(attn: np.ndarray, out_path: str, max_points: int = 200):
    """attn: [N, L] weights."""
    ensure_dir(os.path.dirname(out_path))
    n = min(attn.shape[0], max_points)
    avg = attn[:n].mean(axis=0)  # [L]
    plt.figure()
    plt.plot(np.arange(len(avg)), avg)
    plt.xlabel("Lag (0=oldest ... L-1=most recent)")
    plt.ylabel("Average attention weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    inside = ((y >= lo) & (y <= hi)).astype(np.float32)
    cov = inside.mean(axis=0)  # per horizon
    plt.figure()
    plt.bar(np.arange(len(cov)), cov)
    plt.ylim(0, 1)
    plt.xlabel("Horizon index")
    plt.ylabel("Coverage")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
