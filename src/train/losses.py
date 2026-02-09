from __future__ import annotations
import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = float(delta)

    def forward(self, yhat, y):
        return torch.nn.functional.huber_loss(yhat, y, delta=self.delta)

def make_loss(name: str, huber_delta: float = 1.0) -> nn.Module:
    name = (name or "mse").lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return HuberLoss(delta=huber_delta)
    raise ValueError(f"Unknown loss: {name}")
