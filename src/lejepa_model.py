from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ProjectorCfg:
    in_dim: int
    proj_dim: int
    hidden_dim: int
    layers: int


class MLPProjector(nn.Module):
    """
    Simple MLP projector with BN1d between layers (demo-style).
    """
    def __init__(self, cfg: ProjectorCfg):
        super().__init__()
        if cfg.layers not in (2, 3):
            raise ValueError("proj_layers must be 2 or 3")

        dims = [cfg.in_dim]
        if cfg.layers == 3:
            dims += [cfg.hidden_dim, cfg.hidden_dim, cfg.proj_dim]
        else:
            dims += [cfg.hidden_dim, cfg.proj_dim]

        blocks = []
        for i in range(len(dims) - 1):
            blocks.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < len(dims) - 2:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
                blocks.append(nn.ReLU(inplace=True))
            else:
                blocks.append(nn.BatchNorm1d(dims[i + 1]))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def gap_pool(feat: torch.Tensor) -> torch.Tensor:
    """
    feat: (N,C,H,W) -> (N,C)
    """
    return feat.mean(dim=(2, 3))
