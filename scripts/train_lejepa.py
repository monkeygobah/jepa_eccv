from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.run_utils import save_checkpoint
from src.config_utils import init_run
from src.dataset_utils import CFCSplitDataset
from src.load_backbones import load_encoder_backbone

from src.transforms import LocalViewsCfg, build_local_views_transform,collate_views_with_meta
from src.projectors import ProjectorCfg, MLPProjector, gap_pool
from src.objectives.sigreg import SIGReg
from collections.abc import Mapping
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


import os
import torch
import torch.distributed as dist


def ddp_init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)
    return device, rank, world_size, local_rank, is_main



def get_feat_out(y):
    return y["out"] if isinstance(y, Mapping) else y


def lejepa_sim_loss(proj_bvk):
    center = proj_bvk.mean(dim=1, keepdim=True)     
    return (center - proj_bvk).square().mean()




def main(args):
    device, rank, world_size, local_rank, is_main = ddp_init()
    cfg, paths, rp = init_run(args,is_main)

    torch.manual_seed(int(cfg["run"]["seed"]))


    tcfg = LocalViewsCfg(
        V=int(cfg["ssl"]["V"]),
        crop_size=int(cfg["ssl"]["crop_size"]),
        scale_min=float(cfg["ssl"]["crop_scale_min"]),
        scale_max=float(cfg["ssl"]["crop_scale_max"]),
        normalize_imagenet=bool(cfg["ssl"]["normalize_imagenet"]),
    )
    transform = build_local_views_transform(tcfg)

    ds = CFCSplitDataset(root=cfg["data"]["root"], transform=transform)


    sampler = DistributedSampler(ds, shuffle=True)

    dl = DataLoader(
        ds,
        batch_size=int(cfg["dataloader"]["batch_size"]),
        sampler=sampler,                
        num_workers=int(cfg["dataloader"]["num_workers"]),
        pin_memory=bool(cfg["dataloader"]["pin_memory"]),
        drop_last=bool(cfg["dataloader"]["drop_last"]),
        collate_fn=collate_views_with_meta,
    )



    init = cfg["model"]["init"]
    encoder = load_encoder_backbone(init=init,seg_ckpt=cfg["model"].get("seg_ckpt", None)).to(device)
    encoder.train()

    proj_cfg = ProjectorCfg(
        in_dim=2048,
        proj_dim=int(cfg["model"]["proj_dim"]),
        hidden_dim=int(cfg["model"]["proj_hidden"]),
        layers=int(cfg["model"]["proj_layers"]),
    )
    projector = MLPProjector(proj_cfg).to(device)

    projector.train()

    encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)
    projector = DDP(projector, device_ids=[local_rank], output_device=local_rank)

    sigreg = SIGReg(knots=int(cfg["loss"]["sigreg_knots"]),num_slices=int(cfg["loss"]["sigreg_num_slices"])).to(device)


    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )

    # warmup_steps = len(dl)
    # total_steps = len(dl) * (cfg["run"]["epochs"])
    
    
    total_steps = int(cfg["run"]["total_steps"])
    warmup_steps = int(cfg["run"]["warmup_steps"])


    s1 = LinearLR(opt, start_factor=float(cfg["optim"]["warmup_factor"]), total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt,T_max=max(1, total_steps - warmup_steps),eta_min=float(cfg["sched"]["final_lr"]))
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    # AMP
    amp_enabled = bool(cfg["amp"]["enabled"])
    amp_dtype = cfg["amp"]["dtype"].lower()
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    scaler = GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)

    lamb = float(cfg["loss"]["lamb"])
    log_every = int(cfg["run"]["log_every"])
    ckpt_every = int(cfg["run"]["ckpt_every"])

    metrics_path = rp.run_dir / "train_metrics.jsonl"

    step = 0
    epoch = 0
    # for epoch in range(cfg["run"]["epochs"]):
    #     encoder.train()
    #     projector.train()
    #     sampler.set_epoch(epoch)

    #     it = dl

    from torchvision.utils import make_grid
    from torchvision.transforms.functional import to_pil_image

    while step < total_steps:
        encoder.train(); projector.train()
        sampler.set_epoch(epoch)

        it = dl
        if is_main:
            it = tqdm(dl, total=len(dl), desc=f"LeJEPA epoch {epoch}")

        for vs, _ in it:

            if step >= total_steps:
                break

            vs = vs.to(device, non_blocking=True)


            bs, V, C, H, W = vs.shape
            x = vs.reshape(bs * V, C, H, W)

            opt.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp_enabled):
                feat = get_feat_out(encoder(x))
                emb = gap_pool(feat)         
                
                proj = projector(emb)                   
                K = proj.shape[1]


                proj_bvk = proj.reshape(bs, V, K)
                sim = lejepa_sim_loss(proj_bvk)

                sr = sigreg(proj_bvk)
                loss = (1.0 - lamb) * sim + lamb * sr

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            scheduler.step()

            if is_main and step % log_every == 0:
                rec = {
                    "step": step,
                    "loss": float(loss.item()),
                    "sim": float(sim.item()),
                    "sigreg": float(sr.item()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "bs": int(bs),
                    "V": int(V),
                }
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")

            if is_main and ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
                save_checkpoint(
                    ckpt_dir=rp.ckpt_dir,
                    step=step,
                    encoder=encoder,
                    projector=projector,
                    opt=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                )

            step += 1

        epoch+=1

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--paths", default='configs/paths.yaml')
    ap.add_argument("--gpu", default=0, type=int)
    args = ap.parse_args()
    main(args)
