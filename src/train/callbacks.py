from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import os
import torch
from src.utils.io import ensure_dir

@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0
    best: float = float("inf")
    bad_epochs: int = 0
    stopped: bool = False

    def step(self, val_loss: float) -> bool:
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
        return improved

class ModelCheckpoint:
    def __init__(self, path: str):
        self.path = path
        ensure_dir(os.path.dirname(path))

    def save(self, model: torch.nn.Module, extra: dict):
        payload = {"model_state": model.state_dict(), **(extra or {})}
        torch.save(payload, self.path)
