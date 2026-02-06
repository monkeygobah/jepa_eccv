# src/objectives/simclr.py

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from src.projectors import ProjectorCfg, MLPProjector, gap_pool  # rename accordingly
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather  

def _get_feat_out(y):
    return y["out"] if isinstance(y, dict) else y


def ddp_gather_cat_autograd(x: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return x

    try:
        xs = all_gather(x)
        return torch.cat(xs, dim=0)
    except Exception:
        return _GatherLayer.apply(x)

class _GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        world = dist.get_world_size()
        xs = [torch.zeros_like(x) for _ in range(world)]
        dist.all_gather(xs, x.contiguous())
        return torch.cat(xs, dim=0)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        world = dist.get_world_size()
        rank = dist.get_rank()
        return grad_out.chunk(world, dim=0)[rank].contiguous()


class SimCLRObjective(nn.Module):
    """
    SimCLR (InfoNCE) with two views:
      - project -> L2 normalize
      - cosine similarities / temperature
      - cross-entropy where positive is the paired augmented view
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

        self.tau = float(cfg["simclr"].get("tau", 0.2))      
        self.gather = bool(cfg["simclr"].get("gather", True)) 
        self.eps = float(cfg["simclr"].get("eps", 1e-8))     

    def forward(self, encoder: nn.Module, vs: torch.Tensor):
        bs, V, C, H, W = vs.shape
        if V != 2:
            raise ValueError(f"SimCLR requires V=2 views, got V={V}")

        # (bs, C, H, W)
        x1 = vs[:, 0]
        x2 = vs[:, 1]

        # Encode -> pool -> project
        f1 = _get_feat_out(encoder(x1))
        f2 = _get_feat_out(encoder(x2))

        h1 = gap_pool(f1)  
        h2 = gap_pool(f2)

        z1 = self.projector(h1) 
        z2 = self.projector(h2)

        # L2 normalize (cosine similarity)
        z1 = F.normalize(z1, dim=1, eps=self.eps)
        z2 = F.normalize(z2, dim=1, eps=self.eps)

        # Optionally gather for more negatives
        if self.gather:
            z1g = ddp_gather_cat_autograd(z1)
            z2g = ddp_gather_cat_autograd(z2)
            if dist.is_initialized():
                world = dist.get_world_size()
                assert z1g.shape[0] == world * bs, "Need equal per-rank bs (drop_last=True) for correct targets"
                offset = dist.get_rank() * bs
            else:
                offset = 0
        else:
            z1g, z2g = z1, z2
            offset = 0

        # N is global batch for contrastive set
        N = z1g.shape[0]


        logits_12 = (z1 @ z2g.T) / self.tau  # (bs, N)
        logits_21 = (z2 @ z1g.T) / self.tau  # (bs, N)

        targets = torch.arange(bs, device=vs.device) + offset 

        # InfoNCE 
        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        loss = 0.5 * (loss_12 + loss_21)

        logs = {
            "loss": loss,
            "nce_12": loss_12,
            "nce_21": loss_21,
            "tau": self.tau,
            "V": V,
            "N_global": int(N),
            "gather": int(self.gather),
        }
        return loss, logs
