from __future__ import annotations

import torch
import torch.nn as nn


class TokenProjector(nn.Module):
    """Project token features into a consumer-specific hidden size."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)
