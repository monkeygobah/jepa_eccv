from __future__ import annotations

import argparse
import json

import torch
from torch.amp import GradScaler, autocast
from torch.cuda import is_available
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.run_utils import save_checkpoint
from src.config_utils import init_run
from src.dataset_utils import CFCSplitDataset
from src.load_backbones import load_encoder_backbone

from src.transforms import LocalViewsCfg, build_local_views_transform,collate_views_with_meta
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


import os
import torch
import torch.distributed as dist

from src.objectives.simclr import CrossViewInfoNCEObjective
from src.objectives.vicreg import VICRegObjective
from src.objectives.byol import BYOLObjective
from src.objectives.lejepa import LeJEPAObjective

'''
GENERIC SSL LOADER
'''

def ddp_init():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)
    return device, rank, world_size, local_rank, is_main


def load_checkpoint(path, encoder, objective, opt, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    (encoder.module if hasattr(encoder,"module") else encoder).load_state_dict(ckpt["encoder"])
    (objective.module if hasattr(objective,"module") else objective).load_state_dict(ckpt["objective"])
    opt.load_state_dict(ckpt["opt"])
    if scheduler and ckpt.get("scheduler"): scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"): scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt["step"])


def main(args):
    # torch.backends.cudnn.enabled = False

    device, rank, world_size, local_rank, is_main = ddp_init()
    try:
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


        # objective (instantiate ONCE)
        if cfg["ssl"]["method"] == "lejepa":
            objective = LeJEPAObjective(cfg).to(device)
        elif cfg["ssl"]["method"] == "vicreg":
            objective = VICRegObjective(cfg).to(device)
        elif cfg["ssl"]["method"] == "simclr":
            objective = CrossViewInfoNCEObjective(cfg).to(device)
        elif cfg["ssl"]["method"] == "byol":
            objective = BYOLObjective(cfg).to(device)
        else:
            raise ValueError(f"Unknown ssl.method: {cfg['ssl']['method']}")

        # wrap objective ONCE
        objective = DDP(objective, device_ids=[local_rank], output_device=local_rank)

        # if is_main:
        #     print(objective.module.projector)  # or objective.projector if not wrapped yet


        # encoder
        encoder = load_encoder_backbone(init=init, seg_ckpt=cfg["model"].get("seg_ckpt")).to(device)
        if cfg["ssl"]["method"] in ("byol", "simclr", "vicreg"):
            encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(encoder)




        encoder = DDP(encoder, device_ids=[local_rank], output_device=local_rank)

        # BYOL: init target encoder ONCE (after DDP wrap)
        if cfg["ssl"]["method"] == "byol":
            obj = objective.module
            enc = encoder.module
            obj.init_target_encoder(enc)

        if cfg["ssl"]["method"] == "byol":
            opt = torch.optim.AdamW(
                list(encoder.parameters()) + list(objective.module.projector.parameters()) + list(objective.module.predictor.parameters()),
                lr=float(cfg["optim"]["lr"]),
                weight_decay=float(cfg["optim"]["weight_decay"]),)
        else:
            opt = torch.optim.AdamW(
                list(encoder.parameters()) + list(objective.parameters()),
                lr=float(cfg["optim"]["lr"]),
                weight_decay=float(cfg["optim"]["weight_decay"]),
            )
            
        total_steps = int(cfg["run"]["total_steps"])
        warmup_steps = int(cfg["run"]["warmup_steps"])


        s1 = LinearLR(opt, start_factor=float(cfg["optim"]["warmup_factor"]), total_iters=warmup_steps)
        s2 = CosineAnnealingLR(opt,T_max=max(1, total_steps - warmup_steps),eta_min=float(cfg["sched"]["final_lr"]))
        scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

        amp_enabled = bool(cfg["amp"]["enabled"])
        amp_dtype = cfg["amp"]["dtype"].lower()
        autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        scaler = GradScaler(enabled=amp_enabled and autocast_dtype == torch.float16)

        log_every = int(cfg["run"]["log_every"])
        ckpt_every = int(cfg["run"]["ckpt_every"])

        metrics_path = rp.run_dir / "train_metrics.jsonl"

        step = 0
        epoch = 0

        while step < total_steps:
            encoder.train(); objective.train()
            sampler.set_epoch(epoch)

            it = dl
            if is_main:
                it = tqdm(dl, total=len(dl), desc=f"epoch {epoch}")

            for vs, _ in it:
                torch.autograd.set_detect_anomaly(True)

                if step >= total_steps:
                    break

                vs = vs.to(device, non_blocking=True)


                opt.zero_grad(set_to_none=True)

                with autocast(device_type='cuda', dtype=autocast_dtype, enabled=amp_enabled):
                    loss, logs = objective(encoder, vs)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                scheduler.step()

                if cfg["ssl"]["method"] == "byol":
                    obj = objective.module
                    enc = encoder.module
                    obj.update_target(enc, step=step+1, total_steps=total_steps)

                if is_main and step % log_every == 0:
                    rec = {
                        "step": step,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "bs": int(vs.shape[0]),
                    }

                    for k, v in logs.items():
                        if torch.is_tensor(v):
                            rec[k] = float(v.detach().item())
                        else:
                            rec[k] = v

                    with open(metrics_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")


                if is_main and ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
                    save_checkpoint(
                        ckpt_dir=rp.ckpt_dir,
                        step=step,
                        encoder=encoder,
                        objective=objective,
                        opt=opt,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

                step += 1

            epoch+=1
        
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--paths", default='configs/paths.yaml')
    ap.add_argument("--gpu", default=0, type=int)    
    ap.add_argument("--resume", type='store_true')

    args = ap.parse_args()
    main(args)
