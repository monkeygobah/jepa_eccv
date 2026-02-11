

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

        ## added to allow more more views and avoid OOM issues
        self.view_chunk = int(cfg["ssl"]["view_chunk"])  


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


    # def forward(self, encoder, vs):
    #         # vs: (bs, V, C, H, W)
    #         bs, V, C, H, W = vs.shape
    #         embs = []
    #         for v0 in range(0, V, self.view_chunk):
    #             v1 = min(V, v0 + self.view_chunk)
    #             x = vs[:, v0:v1].reshape(bs * (v1 - v0), C, H, W)

    #             feat = get_feat_out(encoder(x))
    #             emb = gap_pool(feat)                       # (bs*(v1-v0), 2048)
    #             emb = emb.view(bs, (v1 - v0), -1)          # (bs, chunk, 2048)
    #             embs.append(emb)

    #         emb_bvk = torch.cat(embs, dim=1)               # (bs, V, 2048)

    #         proj = self.projector(emb_bvk.reshape(bs * V, -1))
    #         proj_bvk = proj.view(bs, V, -1)

    #         sim = lejepa_sim_loss(proj_bvk)
    #         sr = self.sigreg(proj_bvk)
    #         loss = (1.0 - self.lamb) * sim + self.lamb * sr

    #         logs = {"loss": loss, "sim": sim, "sigreg": sr, "V": V}
    #         return loss, logs



     
