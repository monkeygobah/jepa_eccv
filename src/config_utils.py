from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml
from src.run_utils import RunPaths, make_run_dir, save_config
from src.seed import seed_everything
import torch
import torch.distributed as dist
from datetime import datetime

def load_yaml(path):
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def init_run(args, is_main):
    cfg = load_yaml(args.cfg)
    paths = load_yaml(args.paths)["paths"]

    seed_everything(cfg["run"]["seed"])

    runs_root = Path(paths["runs_root"])
    run_name = cfg["run"]["name"]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") if is_main else None
    if dist.is_initialized():
        obj = [run_id]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]

    rp = make_run_dir(runs_root, run_name, run_id=run_id, mkdir=is_main)

    if is_main:
        save_config({"paths": paths, "cfg": cfg}, rp.run_dir)

    if dist.is_initialized():
        dist.barrier()

    return cfg, paths, rp
