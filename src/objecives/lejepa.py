

import torch
import torch.nn as nn
from src.projectors import MLPProjector, ProjectorCfg, gap_pool
from .sigreg import SIGReg
from collections.abc import Mapping


def lejepa_sim_loss(proj_bvk):
    center = proj_bvk.mean(dim=1, keepdim=True)
    return (center - proj_bvk).square().mean()

def get_feat_out(y):
    return y["out"] if isinstance(y, Mapping) else y


class LeJEPAObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        proj_cfg = ProjectorCfg(
            in_dim=2048,
            proj_dim=int(cfg["model"]["proj_dim"]),
            hidden_dim=int(cfg["model"]["proj_hidden"]),
            layers=int(cfg["model"]["proj_layers"]),
        )


        self.projector = MLPProjector(proj_cfg)

        self.sigreg = SIGReg(
            knots=int(cfg["loss"]["sigreg_knots"]),
            num_slices=int(cfg["loss"]["sigreg_num_slices"]),
        )

        self.lamb = float(cfg["loss"]["lamb"])

    def forward(self, encoder, vs):
        # vs: (bs, V, C, H, W)
        bs, V, C, H, W = vs.shape
        x = vs.view(bs * V, C, H, W)

        feat = get_feat_out(encoder(x))
        emb = gap_pool(feat)
        proj = self.projector(emb)
        K = proj.shape[1]

        proj_bvk = proj.view(bs, V, K)

        sim = lejepa_sim_loss(proj_bvk)
        sr = self.sigreg(proj_bvk)
        loss = (1.0 - self.lamb) * sim + self.lamb * sr

        logs = {
            "loss": loss,
            "sim": sim,
            "sigreg": sr,
            "V": V,
        }

        return loss, logs
