"""PyTorch MLP for EHR-style binary / small multi-class classification."""

from __future__ import annotations

import torch
from torch import nn


class EHRMLP(nn.Module):
    """Feed-forward network with dropout; logits output (use CrossEntropyLoss)."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout1: float = 0.3,
        dropout2: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
