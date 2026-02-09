from __future__ import annotations
import os
import time
from typing import Dict, Any, Tuple
import numpy as np
import torch
from tqdm import tqdm

from src.train.losses import make_loss
from src.train.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.io import ensure_dir, save_json

def _device_from_cfg(cfg: Dict[str, Any]) -> torch.device:
    dev = cfg["train"].get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)

def _make_optimizer(cfg: Dict[str, Any], params):
    opt = (cfg["train"].get("optimizer", "adamw") or "adamw").lower()
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.0))
    if opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if opt == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {opt}")

def _make_scheduler(cfg: Dict[str, Any], optimizer):
    sch = (cfg["train"].get("scheduler", "none") or "none").lower()
    if sch in ["none", "null"]:
        return None
    if sch == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    raise ValueError(f"Unknown scheduler: {sch}")

@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    all_y = []
    all_yhat = []
    all_attn = []
    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()
        yhat, attn = model(x)
        loss = loss_fn(yhat, y).item()
        losses.append(loss)
        all_y.append(y.detach().cpu().numpy())
        all_yhat.append(yhat.detach().cpu().numpy())
        if attn is not None:
            all_attn.append(attn.detach().cpu().numpy())
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    y_np = np.concatenate(all_y, axis=0) if all_y else np.empty((0,))
    yhat_np = np.concatenate(all_yhat, axis=0) if all_yhat else np.empty((0,))
    attn_np = np.concatenate(all_attn, axis=0) if all_attn else None
    return mean_loss, y_np, yhat_np, attn_np

def train_model(model, train_loader, val_loader, cfg: Dict[str, Any], run_dir: str, device=None):
    ensure_dir(run_dir)
    device = device or _device_from_cfg(cfg)
    model.to(device)

    loss_fn = make_loss(cfg["train"].get("loss", "mse"), huber_delta=float(cfg["train"].get("huber_delta", 1.0)))
    optimizer = _make_optimizer(cfg, model.parameters())
    scheduler = _make_scheduler(cfg, optimizer)

    es = EarlyStopping(
        patience=int(cfg["train"].get("patience", 5)),
        min_delta=float(cfg["train"].get("min_delta", 0.0)),
    )
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    ckpt = ModelCheckpoint(ckpt_path)

    log_path = os.path.join(run_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,lr,seconds\n")

    amp = bool(cfg["train"].get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    epochs = int(cfg["train"].get("epochs", 10))
    grad_clip = float(cfg["train"].get("grad_clip", 0.0))

    best_epoch = -1
    for epoch in range(1, epochs + 1):
        model.train()
        start_t = time.time()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for step, (x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                yhat, _ = model(x)
                loss = loss_fn(yhat, y)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.detach().item())
            if step % int(cfg["train"].get("log_every", 50)) == 0:
                pbar.set_postfix({"loss": float(np.mean(train_losses[-10:]))})

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss, _, _, _ = evaluate(model, val_loader, loss_fn, device)

        if scheduler is not None:
            # reduce_on_plateau expects metric
            scheduler.step(val_loss)

        # checkpoint on improvement
        improved = es.step(val_loss)
        if improved:
            best_epoch = epoch
            ckpt.save(model, extra={"cfg": cfg, "best_val_loss": es.best, "best_epoch": best_epoch})

        seconds = time.time() - start_t
        lr = optimizer.param_groups[0]["lr"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.8f},{val_loss:.8f},{lr:.8g},{seconds:.2f}\n")

        if es.stopped:
            break

    # Load best checkpoint for final evaluation consistency
    if os.path.exists(ckpt_path):
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state"])

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": float(es.best),
        "stopped_early": bool(es.stopped),
        "device": str(device),
        "log_path": log_path,
        "ckpt_path": ckpt_path,
    }
    save_json(os.path.join(run_dir, "train_summary.json"), summary)
    return summary, ckpt_path
