

import torch
import torch.nn as nn
from src.projectors import MLPProjector, ProjectorCfg, gap_pool
from .sigreg import SIGReg
from collections.abc import Mapping

from typing import List, Union, Mapping, Any, Tuple, Dict
import torch
import torch.nn as nn



def lejepa_sim_loss(proj_bvk):
    center = proj_bvk.mean(dim=1, keepdim=True)
    return (center - proj_bvk).square().mean()

def get_feat_out(y):
    return y["out"] if isinstance(y, Mapping) else y

import os

class LeJEPAObjective(nn.Module):
    def __init__(self, cfg):
        super().__init__()


        # print(int(cfg["ssl"]["view_chunk"]))
        # rank = int(os.environ.get("RANK", "0"))
        # if rank == 0:
        #     print(f"[LeJEPAObjective.__init__] view_chunk={cfg['ssl']['view_chunk']}", flush=True)


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
        if isinstance(vs, (list, tuple)):
            return self._forward_multicrop(encoder, vs)
        else:
            return self._forward_local(encoder, vs)

    def _forward_local(self, encoder, vs):
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

        return loss, {"loss": loss, "sim": sim, "sigreg": sr, "V": V}

    def _forward_multicrop(self, encoder, vs):
        # vs: List[Tensor(bs, C, H, W)], length Vg + Vl (mixed resolutions)
        V = len(vs)
        bs = vs[0].shape[0]

        # Group views by spatial resolution and batch them together
        # to avoid redundant forward passes at the same resolution
        groups: dict = {}
        for j, v in enumerate(vs):
            key = tuple(v.shape[-2:])
            groups.setdefault(key, []).append(j)

        emb_per_view: list = [None] * V

        for (H, W), idxs in groups.items():
            # Stack all views of this resolution into one batch
            x = torch.cat([vs[j] for j in idxs], dim=0)   # (bs * len(idxs), C, H, W)
            feat = get_feat_out(encoder(x))
            emb = gap_pool(feat)                            # (bs * len(idxs), 2048)
            # Split back out per-view
            chunks = emb.chunk(len(idxs), dim=0)           # len(idxs) x (bs, 2048)
            for t, j in enumerate(idxs):
                emb_per_view[j] = chunks[t]

        emb_bvk = torch.stack(emb_per_view, dim=1)         # (bs, V, 2048)
        proj_bvk = self.projector(
            emb_bvk.view(bs * V, -1)
        ).view(bs, V, -1)                                   # (bs, V, K)

        sim = lejepa_sim_loss(proj_bvk)
        sr = self.sigreg(proj_bvk)
        loss = (1.0 - self.lamb) * sim + self.lamb * sr

        return loss, {"loss": loss, "sim": sim, "sigreg": sr, "V": V}


    # def forward(self, encoder, vs):
    #     # vs: (bs, V, C, H, W)
    #     bs, V, C, H, W = vs.shape
    #     x = vs.view(bs * V, C, H, W)

    #     feat = get_feat_out(encoder(x))
    #     emb = gap_pool(feat)
    #     proj = self.projector(emb)
    #     K = proj.shape[1]

    #     proj_bvk = proj.view(bs, V, K)

    #     sim = lejepa_sim_loss(proj_bvk)
    #     sr = self.sigreg(proj_bvk)
    #     loss = (1.0 - self.lamb) * sim + self.lamb * sr

    #     logs = {
    #         "loss": loss,
    #         "sim": sim,
    #         "sigreg": sr,
    #         "V": V,
    #     }

    #     return loss, logs



    # def forward_from_emb(self, emb_bvk: torch.Tensor):
    #     # emb_bvk: (bs, V, D=2048)
    #     bs, V, D = emb_bvk.shape
    #     proj = self.projector(emb_bvk.reshape(bs * V, D))
    #     K = proj.shape[1]
    #     proj_bvk = proj.reshape(bs, V, K)

    #     sim = lejepa_sim_loss(proj_bvk)
    #     sr  = self.sigreg(proj_bvk)
    #     loss = (1.0 - self.lamb) * sim + self.lamb * sr

    #     logs = {"loss": loss, "sim": sim, "sigreg": sr, "V": V}
    #     return loss, logs