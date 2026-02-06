# src/objectives/cross_view_infonce.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.projectors import ProjectorCfg, MLPProjector, gap_pool


def _get_feat_out(y):
    return y["out"] if isinstance(y, dict) else y


def _ddp_is_init() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_gather_cat_autograd(x: torch.Tensor) -> torch.Tensor:
    """
    Autograd-friendly all_gather + cat(dim=0).
    Assumes every rank provides the same shape.
    """
    if not _ddp_is_init():
        return x

    # Prefer torch.distributed.nn.functional.all_gather when available
    try:
        from torch.distributed.nn.functional import all_gather  # type: ignore
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


class CrossViewInfoNCEObjective(nn.Module):
    """
    Cross-view InfoNCE (SimCLR-like variant) with two views.

    Given two augmented views per sample:
      - Encode both views in a single forward (2*bs) to avoid BN/DDP quirks
      - Pool -> project -> L2 normalize
      - For anchors in view-1, classify the matching view-2 embedding among view-2 negatives
      - Symmetric term for view-2 anchors against view-1 negatives

    Note: This is NOT canonical SimCLR NT-Xent (2N x 2N); negatives come only from the opposite view.
    """

    def __init__(self, cfg):
        super().__init__()

        proj_cfg = ProjectorCfg(
            in_dim=int(cfg["model"].get("feat_dim", 2048)),
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["model"]["proj_hidden"]),
            layers=int(cfg["model"]["proj_layers"]),
        )
        self.projector = MLPProjector(proj_cfg)

        # Put these under cfg["infonce"] (or similar) to avoid “simclr” confusion
        loss_cfg = cfg.get("infonce", cfg.get("simclr", {}))
        self.tau = float(loss_cfg.get("tau", 0.2))
        self.gather = bool(loss_cfg.get("gather", True))
        self.eps = float(loss_cfg.get("eps", 1e-8))

    @staticmethod
    def _assert_equal_local_bs(bs: int, device: torch.device) -> None:
        """
        Ensures all ranks have the same local batch size when using offset indexing.
        If you use drop_last=True this will always pass.
        """
        if not _ddp_is_init():
            return
        t = torch.tensor([bs], device=device, dtype=torch.int64)
        ts = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        dist.all_gather(ts, t)
        bss = [int(x.item()) for x in ts]
        if any(b != bss[0] for b in bss):
            raise RuntimeError(
                f"CrossViewInfoNCE requires equal per-rank batch sizes when gather=True. "
                f"Got per-rank bs={bss}. Use drop_last=True (recommended) or implement variable-size offsets."
            )

    def forward(self, encoder: nn.Module, vs: torch.Tensor):
        bs, V, C, H, W = vs.shape
        if V != 2:
            raise ValueError(f"CrossViewInfoNCE requires V=2 views, got V={V}")

        # Single forward for both views (2bs)
        x1 = vs[:, 0]                      # (bs, C, H, W)
        x2 = vs[:, 1]                      # (bs, C, H, W)
        x  = torch.cat([x1, x2], dim=0)    # (2bs, C, H, W)

        f = _get_feat_out(encoder(x))
        h = gap_pool(f)
        z = F.normalize(self.projector(h), dim=1, eps=self.eps)

        z1, z2 = z[:bs], z[bs:]            # correct pairing


        if self.gather and _ddp_is_init():
            self._assert_equal_local_bs(bs, vs.device)

            z1g = ddp_gather_cat_autograd(z1)      # (B, D)
            z2g = ddp_gather_cat_autograd(z2)      # (B, D)

            B = z1g.shape[0]
            offset = dist.get_rank() * bs
        else:
            z1g, z2g = z1, z2
            B = bs
            offset = 0

        # logits: (bs, B)
        logits_12 = (z1 @ z2g.T) / self.tau
        logits_21 = (z2 @ z1g.T) / self.tau

        # Optional numeric stabilization (doesn't change softmax)
        logits_12 = logits_12 - logits_12.max(dim=1, keepdim=True).values
        logits_21 = logits_21 - logits_21.max(dim=1, keepdim=True).values

        targets = torch.arange(bs, device=vs.device, dtype=torch.long) + offset

        loss_12 = F.cross_entropy(logits_12, targets)
        loss_21 = F.cross_entropy(logits_21, targets)
        loss = 0.5 * (loss_12 + loss_21)

        logs = {
            "loss": loss.detach(),
            "nce_12": loss_12.detach(),
            "nce_21": loss_21.detach(),
            "tau": float(self.tau),
            "gather": int(self.gather),
            "bs_local": int(bs),
            "bs_global_view": int(B),
        }
        return loss, logs












    # def forward(self, encoder: nn.Module, vs: torch.Tensor):
    #     bs, V, C, H, W = vs.shape
    #     if V != 2:
    #         raise ValueError(f"SimCLR requires V=2 views, got V={V}")

    #     # (bs, C, H, W)
    #     x1 = vs[:, 0]
    #     x2 = vs[:, 1]

    #     # Encode -> pool -> project
    #     f1 = _get_feat_out(encoder(x1))
    #     f2 = _get_feat_out(encoder(x2))

    #     h1 = gap_pool(f1)  
    #     h2 = gap_pool(f2)

    #     z1 = self.projector(h1) 
    #     z2 = self.projector(h2)

    #     # L2 normalize (cosine similarity)
    #     z1 = F.normalize(z1, dim=1, eps=self.eps)
    #     z2 = F.normalize(z2, dim=1, eps=self.eps)

    #     # Optionally gather for more negatives
    #     if self.gather:
    #         z1g = ddp_gather_cat_autograd(z1)
    #         z2g = ddp_gather_cat_autograd(z2)
    #         if dist.is_initialized():
    #             world = dist.get_world_size()
    #             assert z1g.shape[0] == world * bs, "Need equal per-rank bs (drop_last=True) for correct targets"
    #             offset = dist.get_rank() * bs
    #         else:
    #             offset = 0
    #     else:
    #         z1g, z2g = z1, z2
    #         offset = 0

    #     # N is global batch for contrastive set
    #     N = z1g.shape[0]


    #     logits_12 = (z1 @ z2g.T) / self.tau  # (bs, N)
    #     logits_21 = (z2 @ z1g.T) / self.tau  # (bs, N)

    #     targets = torch.arange(bs, device=vs.device) + offset 

    #     # InfoNCE 
    #     loss_12 = F.cross_entropy(logits_12, targets)
    #     loss_21 = F.cross_entropy(logits_21, targets)
    #     loss = 0.5 * (loss_12 + loss_21)

    #     logs = {
    #         "loss": loss,
    #         "nce_12": loss_12,
    #         "nce_21": loss_21,
    #         "tau": self.tau,
    #         "V": V,
    #         "N_global": int(N),
    #         "gather": int(self.gather),
    #     }
    #     return loss, logs
