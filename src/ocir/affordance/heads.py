"""
Task heads that turn per-point backbone features into a (N,1) affordance logit.

  * LinearHead : one linear layer. Use for LINEAR PROBING — it tells you whether
                 the frozen Sonata features already separate contact from non-contact.
  * MLPHead    : 2-layer MLP with GELU + dropout. Slightly more capacity; the MVP
                 default. Cheap enough that it won't overfit small data on its own.

Both are backbone-agnostic: give them (M, C) point features, get (M,) logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:   # (M,C) -> (M,out)
        return self.fc(feat)


class MLPHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 128,
                 out_channels: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:   # (M,C) -> (M,out)
        return self.net(feat)


def build_head(kind: str, in_channels: int, hidden: int = 128, out_channels: int = 1):
    if kind == "linear":
        return LinearHead(in_channels, out_channels)
    if kind == "mlp":
        return MLPHead(in_channels, hidden, out_channels)
    raise ValueError(f"unknown head kind: {kind}")
