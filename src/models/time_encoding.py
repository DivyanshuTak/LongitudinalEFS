"""Fourier features of days since the first scan, frequencies log-spaced over
periods [min_period_days, max_period_days]."""
import math

import torch
import torch.nn as nn


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int, min_period_days: float = 30.0,
                 max_period_days: float = 2000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        if not (0 < min_period_days < max_period_days):
            raise ValueError(
                f"require 0 < min_period_days < max_period_days, got "
                f"{min_period_days} / {max_period_days}"
            )
        freqs = torch.logspace(
            math.log10(1.0 / max_period_days), math.log10(1.0 / min_period_days),
            steps=dim // 2,
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, days: torch.Tensor) -> torch.Tensor:
        """days: (...,) -> (..., dim)."""
        angles = 2 * math.pi * days.float().unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(days.dtype)
