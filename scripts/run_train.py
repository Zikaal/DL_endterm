from __future__ import annotations
import os
import argparse
import json
import time
import numpy as np
import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.io import ensure_dir, save_json, save_npz
from src.data.download import download_yf
from src.data.features import make_features
from src.data.split import time_split
from src.data.dataset import make_loaders
from src.models.gru_attn import GRUAttn
from src.models.lstm_attn import LSTMAttn
from src.models.transformer import TSTransformer
from src.train.trainer import train_model, evaluate
from src.eval.metrics import per_horizon_metrics

def _auto_run_name(cfg):
    ticker = cfg["data"]["ticker"]
    model = cfg["model"]["name"]
    t = time.strftime("%Y%m%d-%H%M%S")
    return f"{ticker}_{model}_{t}"

def build_model(cfg, in_dim: int, out_dim: int):
    mcfg = cfg["model"]
    name = mcfg["name"].lower()
    drop = float(mcfg.get("dropout", cfg.get("regularization", {}).get("dropout", 0.0)))
    if name == "gru_attn":
        return GRUAttn(
            in_dim=in_dim,
            hidden=int(mcfg.get("hidden", 128)),
            num_layers=int(mcfg.get("num_layers", 1)),
            bidirectional=bool(mcfg.get("bidirectional", False)),
            dropout=drop,
            out_dim=out_dim,
        )
    if name == "lstm_attn":
        return LSTMAttn(
            in_dim=in_dim,
            hidden=int(mcfg.get("hidden", 128)),
            num_layers=int(mcfg.get("num_layers", 1)),
            bidirectional=bool(mcfg.get("bidirectional", False)),
            dropout=drop,
            out_dim=out_dim,
        )
    if name == "transformer":
        return TSTransformer(
            in_dim=in_dim,
            d_model=int(mcfg.get("d_model", 128)),
            nhead=int(mcfg.get("nhead", 4)),
            num_layers=int(mcfg.get("layers", 3)),
            dim_ff=int(mcfg.get("dim_ff", 256)),
            dropout=drop,
            out_dim=out_dim,
        )
    raise ValueError(f"Unknown model name: {name}")

def main(cfg_path: str, override_seed: int | None = None):
    cfg = load_config(cfg_path)
    if override_seed is not None:
        cfg["train"]["seed"] = int(override_seed)
    set_seed(int(cfg["train"]["seed"]))

    # Prepare output dirs
    save_dir = cfg["output"].get("save_dir", "results")
    ensure_dir(save_dir)
    run_name = cfg["output"].get("run_name") or _auto_run_name(cfg)
    run_dir = os.path.join(save_dir, "logs", run_name)
    ensure_dir(run_dir)

    # Save config snapshot
    save_json(os.path.join(run_dir, "config.json"), cfg)

    # Load data
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

    # Build model
    out_dim = len(cfg["features"]["horizons"])
    model = build_model(cfg, in_dim=X.shape[1], out_dim=out_dim)

    # Train
    summary, ckpt_path = train_model(model, train_loader, val_loader, cfg, run_dir=run_dir)

    # Evaluate best
    device = torch.device(summary["device"])
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.to(device)

    loss_fn_name = cfg["train"].get("loss", "mse")
    from src.train.losses import make_loss
    loss_fn = make_loss(loss_fn_name, huber_delta=float(cfg["train"].get("huber_delta", 1.0)))

    test_loss, y_true, y_pred, attn = evaluate(model, test_loader, loss_fn, device)
    metrics = per_horizon_metrics(y_pred, y_true, cfg["features"]["horizons"])
    metrics["test_loss"] = float(test_loss)
    save_json(os.path.join(run_dir, "test_metrics.json"), metrics)

    # Save predictions
    if bool(cfg["output"].get("save_preds", True)):
        pred_dict = {
            'y': y_true,
            'yhat': y_pred,
            'horizons': np.array(cfg['features']['horizons'], dtype=np.int32),
        }
        if attn is not None:
            pred_dict['attn'] = attn
        save_npz(os.path.join(run_dir, 'predictions_test.npz'), **pred_dict)

    # Copy checkpoint to results/checkpoints for convenience
    ckpt_out = os.path.join(save_dir, "checkpoints", f"{run_name}.pt")
    ensure_dir(os.path.dirname(ckpt_out))
    torch.save(payload, ckpt_out)
    save_json(os.path.join(run_dir, "checkpoint_ref.json"), {"checkpoint": ckpt_out})

    print("Run:", run_name)
    print("Saved to:", run_dir)
    print("Checkpoint:", ckpt_out)
    print("Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to model config YAML")
    ap.add_argument("--seed", type=int, default=None, help="Override seed")
    args = ap.parse_args()
    main(args.config, override_seed=args.seed)
