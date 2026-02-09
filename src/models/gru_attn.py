from __future__ import annotations
import torch
import torch.nn as nn

class GRUAttn(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        out_dim: int = 3,
        use_attention: bool = True,
        pooling: str = "last",  # when use_attention=False: "last" or "mean"
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_attention = bool(use_attention)
        self.pooling = str(pooling)

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)

        # Attention head (only used if use_attention=True)
        self.attn = nn.Linear(feat_dim, 1)
        self.head = nn.Linear(feat_dim, out_dim)

    def forward(self, x):
        # x: [B, L, D]
        h, _ = self.gru(x)          # [B, L, H*dir]
        h = self.drop(h)

        if self.use_attention:
            w = torch.softmax(self.attn(h).squeeze(-1), dim=1)  # [B, L]
            ctx = (h * w.unsqueeze(-1)).sum(dim=1)              # [B, H*dir]
            y = self.head(ctx)
            return y, w
        else:
            if self.pooling == "mean":
                ctx = h.mean(dim=1)
            else:
                ctx = h[:, -1, :]  # last timestep
            y = self.head(ctx)
            return y, None
