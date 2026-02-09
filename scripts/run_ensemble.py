from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import time

from src.utils.config import load_config
from src.utils.io import ensure_dir, save_json, save_npz
from src.data.download import download_yf
from src.data.features import make_features
from src.data.split import time_split
from src.data.dataset import make_loaders
from src.models.gru_attn import GRUAttn
from src.models.lstm_attn import LSTMAttn
from src.models.transformer import TSTransformer
from src.train.trainer import train_model
from src.eval.uncertainty import ensemble_predict, interval_from_std
from src.eval.metrics import per_horizon_metrics
from scripts.run_train import build_model, _auto_run_name

def main(cfg_path: str, k: int, base_seed: int = 42):
    cfg = load_config(cfg_path)
    cfg["train"]["seed"] = int(base_seed)

    save_dir = cfg["output"].get("save_dir", "results")
    run_name = cfg["output"].get("run_name") or _auto_run_name(cfg) + f"_ens{k}"
    run_dir = os.path.join(save_dir, "logs", run_name)
    ensure_dir(run_dir)
    save_json(os.path.join(run_dir, "config.json"), cfg)

    df = download_yf(
        ticker=cfg["data"]["ticker"],
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        interval=cfg["data"].get("interval", "1d"),
        raw_dir="data/raw",
        force=False,
    )

    X, Y, dates, feature_names = make_features(df, horizons=cfg["features"]["horizons"], drop_na=bool(cfg["features"].get("drop_na", True)))
    splits = time_split(dates, train_end=cfg["split"]["train_end"], val_end=cfg["split"]["val_end"])
    lookback = int(cfg["features"]["lookback"])
    train_loader, val_loader, test_loader, scaler = make_loaders(
        X, Y, splits, lookback=lookback,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        pin_memory=bool(cfg["dataloader"].get("pin_memory", True)),
    )

    # Train K models with different seeds
    ckpts = []
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(k):
        seed = base_seed + i * 100
        cfg_i = dict(cfg)
        cfg_i["train"] = dict(cfg["train"])
        cfg_i["train"]["seed"] = int(seed)
        subdir = os.path.join(run_dir, f"member_{i+1}")
        ensure_dir(subdir)
        from src.utils.seed import set_seed
        set_seed(seed)

        model = build_model(cfg, in_dim=X.shape[1], out_dim=len(cfg["features"]["horizons"]))
        summary, ckpt_path = train_model(model, train_loader, val_loader, cfg_i, run_dir=subdir, device=device)
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])
        model.to(device)
        models.append(model)
        ckpts.append(ckpt_path)

    # Ensemble prediction
    y, mean, std = ensemble_predict(models, test_loader, device=device)
    lo, hi = interval_from_std(mean, std, z=1.645)  # ~90% interval

    metrics = per_horizon_metrics(mean, y, cfg["features"]["horizons"])
    # interval coverage
    inside = ((y >= lo) & (y <= hi)).astype(np.float32)
    cov = inside.mean(axis=0)
    metrics.update({f"cov90_h{h}": float(cov[j]) for j, h in enumerate(cfg["features"]["horizons"])})
    metrics["cov90_mean"] = float(cov.mean())

    save_json(os.path.join(run_dir, "ensemble_metrics.json"), metrics)
    save_npz(os.path.join(run_dir, "ensemble_predictions_test.npz"),
             y=y, mean=mean, std=std, lo=lo, hi=hi, horizons=np.array(cfg["features"]["horizons"], dtype=np.int32))

    print("Ensemble run:", run_name)
    print("Saved to:", run_dir)
    print("Metrics:", metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.config, k=args.k, base_seed=args.seed)
