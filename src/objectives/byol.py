# src/objectives/byol.py

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import math

from src.projectors import ProjectorCfg, MLPProjector, gap_pool  # adjust import


def _get_feat_out(y):
    return y["out"] if isinstance(y, dict) else y


@torch.no_grad()
def _ema_update_(target: nn.Module, online: nn.Module, m: float) -> None:
    # target = m*target + (1-m)*online
    for p_t, p_o in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(m).add_(p_o.data, alpha=(1.0 - m))


def _neg_cosine(p: torch.Tensor, z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = F.normalize(p, dim=1, eps=eps)
    z = F.normalize(z, dim=1, eps=eps)
    return 2.0 - 2.0 * (p * z).sum(dim=1).mean()  # = 2 - 2 cos


class BYOLObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Online projector
        proj_cfg = ProjectorCfg(
            in_dim=2048,
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["model"]["proj_hidden"]),
            layers=int(cfg["model"]["proj_layers"]),
        )
        self.projector = MLPProjector(proj_cfg)

        # Online predictor (BYOL uses an extra MLP)
        pred_cfg = ProjectorCfg(
            in_dim=int(cfg["model"]["proj_dim"]),
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["byol"].get("pred_hidden", int(cfg["model"]["proj_hidden"]))),
            layers=int(cfg["byol"].get("pred_layers", 2)),
        )
        self.predictor = MLPProjector(pred_cfg)

        # Target networks (EMA copies of online)
        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

        # EMA schedule
        self.m_base = float(cfg["byol"].get("m", 0.996))
        self.m_final = float(cfg["byol"].get("m_final", 1.0))
        self.eps = float(cfg["byol"].get("eps", 1e-8))

        # Optional: symmetric loss weight (always 0.5/0.5 here)
        self._step = 0

        # This will be set by the trainer once so we can deepcopy encoder safely
        self._target_encoder = None

    @torch.no_grad()
    def init_target_encoder(self, online_encoder: nn.Module) -> None:
        # Call once after DDP wrap is created; pass encoder.module if DDP
        self._target_encoder = copy.deepcopy(online_encoder)
        for p in self._target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, online_encoder: nn.Module, step: int, total_steps: int) -> None:


        # Cosine schedule for momentum is common; defaults m->1
        if self._target_encoder is None:
            raise RuntimeError("Call init_target_encoder() before training.")

        if total_steps <= 1:
            m = self.m_base
        else:
            t = float(step) / float(total_steps)
            # cosine from m_base to m_final
            # m = self.m_final - (self.m_final - self.m_base) * (0.5 * (1.0 + torch.cos(torch.tensor(t * 3.1415926535))).item())
            m = self.m_final - (self.m_final - self.m_base) * (0.5 * (1.0 + math.cos(math.pi * t)))

        _ema_update_(self._target_encoder, online_encoder, m)
        _ema_update_(self.target_projector, self.projector, m)

    def forward(self, encoder: nn.Module, vs: torch.Tensor):
        bs, V, C, H, W = vs.shape
        if V != 2:
            raise ValueError(f"BYOL requires V=2 views, got V={V}")
        if self._target_encoder is None:
            raise RuntimeError("Target encoder not initialized. Call init_target_encoder().")

        x1 = vs[:, 0]
        x2 = vs[:, 1]

        # Online branch
        f1 = _get_feat_out(encoder(x1))
        f2 = _get_feat_out(encoder(x2))
        h1 = gap_pool(f1)
        h2 = gap_pool(f2)

        z1 = self.projector(h1)
        z2 = self.projector(h2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        # Target branch (stop-grad by no_grad + requires_grad False)
        with torch.no_grad():
            self._target_encoder.eval()
            self.target_projector.eval()
            tf1 = _get_feat_out(self._target_encoder(x1))
            tf2 = _get_feat_out(self._target_encoder(x2))
            th1 = gap_pool(tf1)
            th2 = gap_pool(tf2)

            tz1 = self.target_projector(th1)
            tz2 = self.target_projector(th2)

        # Symmetric BYOL loss
        loss_12 = _neg_cosine(p1, tz2, eps=self.eps)
        loss_21 = _neg_cosine(p2, tz1, eps=self.eps)
        loss = 0.5 * (loss_12 + loss_21)

        logs = {
            "loss": loss,
            "byol_12": loss_12,
            "byol_21": loss_21,
            "V": V,
        }
        return loss, logs
