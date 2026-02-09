from __future__ import annotations
import os
import argparse
import numpy as np
import torch

from src.utils.io import save_json, ensure_dir, save_npz
from src.eval.metrics import per_horizon_metrics
from src.train.losses import make_loss
from src.train.trainer import evaluate
from src.utils.config import load_config
from scripts.run_train import build_model
from src.data.download import download_yf
from src.data.features import make_features
from src.data.split import time_split
from src.data.dataset import make_loaders

def main(checkpoint: str, cfg_path: str | None):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = payload.get("cfg")
    if cfg_path:
        cfg = load_config(cfg_path)
    if cfg is None:
        raise ValueError("No config found. Provide --config used for training.")

    df = download_yf(
        ticker=cfg["data"]["ticker"],
        start=cfg["data"]["start"],
        end=cfg["data"]["end"],
        interval=cfg["data"].get("interval", "1d"),
        raw_dir="data/raw",
        force=False,
    )
    X, Y, dates, _ = make_features(df, horizons=cfg["features"]["horizons"], drop_na=bool(cfg["features"].get("drop_na", True)))
    splits = time_split(dates, train_end=cfg["split"]["train_end"], val_end=cfg["split"]["val_end"])
    lookback = int(cfg["features"]["lookback"])
    train_loader, val_loader, test_loader, scaler = make_loaders(
        X, Y, splits, lookback=lookback,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        num_workers=int(cfg["dataloader"].get("num_workers", 0)),
        pin_memory=bool(cfg["dataloader"].get("pin_memory", True)),
    )

    model = build_model(cfg, in_dim=X.shape[1], out_dim=len(cfg["features"]["horizons"]))
    model.load_state_dict(payload["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = make_loss(cfg["train"].get("loss", "mse"), huber_delta=float(cfg["train"].get("huber_delta", 1.0)))
    test_loss, y_true, y_pred, attn = evaluate(model, test_loader, loss_fn, device)

    metrics = per_horizon_metrics(y_pred, y_true, cfg["features"]["horizons"])
    metrics["test_loss"] = float(test_loss)

    out_dir = os.path.join(cfg["output"].get("save_dir", "results"), "logs", "eval_" + os.path.splitext(os.path.basename(checkpoint))[0])
    ensure_dir(out_dir)
    save_json(os.path.join(out_dir, "eval_metrics.json"), metrics)
    pred_dict = {
        'y': y_true,
        'yhat': y_pred,
        'horizons': np.array(cfg['features']['horizons'], dtype=np.int32),
    }
    if attn is not None:
        pred_dict['attn'] = attn
    save_npz(os.path.join(out_dir, 'eval_predictions_test.npz'), **pred_dict)
    print("Saved eval to:", out_dir)
    print(metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="Optional config YAML (if checkpoint doesn't include cfg)")
    args = ap.parse_args()
    main(args.checkpoint, args.config)
