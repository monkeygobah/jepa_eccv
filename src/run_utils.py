from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import yaml
import torch
from pathlib import Path
import re


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    ckpt_dir: Path
    log_dir: Path
    fig_dir: Path

    def mkdir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)


def make_run_dir(out_root, run_name, run_id=None, mkdir=True):

    out_root = Path(out_root)

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = out_root / f"{run_id}__{run_name}"

    rp = RunPaths(
        run_dir=run_dir,
        ckpt_dir=run_dir / "checkpoints",
        log_dir=run_dir / "logs",
        fig_dir=run_dir / "figures",
    )

    if mkdir:
        rp.mkdir()

    return rp


def save_config(cfg, run_dir):
    """
    Save a single YAML config snapshot for the run.
    Intended to be called by rank 0 only.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def save_checkpoint(ckpt_dir, step, encoder, objective, opt, epoch, scheduler=None, scaler=None):

    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    enc = encoder.module if hasattr(encoder, "module") else encoder
    obj = objective.module if hasattr(objective, "module") else objective   

    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "encoder": enc.state_dict(),
        "objective": obj.state_dict(),
        "opt": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }

    out = ckpt_dir / f"ckpt_step_{step:07d}.pth"
    torch.save(ckpt, out)

    
def load_checkpoint(path, encoder, objective, opt, scheduler=None, scaler=None, device=None):
    ckpt = torch.load(path, map_location="cpu")

    enc = encoder.module if hasattr(encoder, "module") else encoder
    obj = objective.module if hasattr(objective, "module") else objective

    enc.load_state_dict(ckpt["encoder"], strict=True)
    obj.load_state_dict(ckpt["objective"], strict=True)

    opt.load_state_dict(ckpt["opt"])
    if device is not None:
        optimizer_to_device(opt, device)

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    step0 = int(ckpt.get("step", 0))
    epoch0 = int(ckpt.get("epoch", 0))
    return step0, epoch0



def optimizer_to_device(opt, device):
    for state in opt.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device, non_blocking=True)
