# src/objectives/vicreg.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.projectors import ProjectorCfg, MLPProjector, gap_pool  
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from src.objectives.simclr import _GatherLayer

def _get_feat_out(y):
    return y["out"] if isinstance(y, dict) else y


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[0]
    return x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()


def _gather_cat_autograd(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x
    try:
        xs = all_gather(x)
        return torch.cat(xs, dim=0)
    except Exception:
        return _GatherLayer.apply(x)

class VICRegObjective(nn.Module):
    """
    VICReg = invariance + variance + covariance regularization.

    Paper-default coefficients (common in the literature):
      lambda (inv) ~ 25, mu (var) ~ 25, nu (cov) ~ 1

    We keep them configurable in cfg.
    """
    def __init__(self, cfg):
        super().__init__()

        proj_cfg = ProjectorCfg(
            in_dim=2048,
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["model"]["proj_hidden"]),
            layers=int(cfg["model"]["proj_layers"]),
        )
        self.projector = MLPProjector(proj_cfg)

        # Loss weights (defaults match common VICReg settings)
        self.w_inv = float(cfg["vicreg"].get("w_inv", 25.0))
        self.w_var = float(cfg["vicreg"].get("w_var", 25.0))
        self.w_cov = float(cfg["vicreg"].get("w_cov", 1.0))

        # Variance target and numerical epsilon
        self.var_target = float(cfg["vicreg"].get("var_target", 1.0))
        self.eps = float(cfg["vicreg"].get("eps", 1e-4))

        # Whether to compute variance/covariance over the global (all-GPU) batch
        self.gather = bool(cfg["vicreg"].get("gather", True))

    def forward(self, encoder: nn.Module, vs: torch.Tensor):
        """
        vs: (bs, V, C, H, W). VICReg expects exactly two views (V=2).
        """
        bs, V, C, H, W = vs.shape
        if V != 2:
            raise ValueError(f"VICReg requires V=2 views, got V={V}")

        x1 = vs[:, 0]
        x2 = vs[:, 1]
        x  = torch.cat([x1, x2], dim=0)
        f = _get_feat_out(encoder(x))
        h = gap_pool(f)
        z = self.projector(h)
        z1, z2 = z[:bs], z[bs:]

        if self.gather:
            z1g = _gather_cat_autograd(z1)
            z2g = _gather_cat_autograd(z2)
        else:
            z1g, z2g = z1, z2

        # Invariance term (page 2 bardes iclr 2022)
        inv = F.mse_loss(z1, z2)

        
        # Variance term (page 4-5 bardes iclr 2022)
        def _var_loss(z: torch.Tensor) -> torch.Tensor:
            # std over batch for each dim
            std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)
            return torch.mean(F.relu(self.var_target - std))

        var1 = _var_loss(z1g)
        var2 = _var_loss(z2g)
        var = 0.5 * (var1 + var2)

        # 3) Covariance term (page 5 bardes iclr 2022)
        # Choice: penalize off-diagonal covariance of centered embeddings
        # Defense: discourages redundancy/correlation across dimensions (decorrelation),
        # improving conditioning / isotropy-like behavior.
        def _cov_loss(z: torch.Tensor) -> torch.Tensor:
            n, d = z.shape
            zc = z - z.mean(dim=0, keepdim=True)
            cov = (zc.T @ zc) / (n - 1 if n > 1 else 1)  # (D, D)
            return (_off_diagonal(cov).pow(2).sum()) / d

        cov1 = _cov_loss(z1g)
        cov2 = _cov_loss(z2g)
        cov = 0.5 * (cov1 + cov2)

        # Total
        loss = self.w_inv * inv + self.w_var * var + self.w_cov * cov

        logs = {
            "loss": loss,
            "inv": inv,
            "var": var,
            "cov": cov,
            "V": V,
        }
        return loss, logs
