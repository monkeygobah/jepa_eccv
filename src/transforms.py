from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Any, Tuple

import torch
from torchvision.transforms import v2


@dataclass(frozen=True)
class LocalViewsCfg:
    V: int
    crop_size: int
    scale_min: float
    scale_max: float
    normalize_imagenet: bool


@dataclass(frozen=True)
class ViewAugCfg:
    V: int
    crop_size: int
    scale_min: float
    scale_max: float
    normalize_imagenet: bool = True


@dataclass(frozen=True)
class MultiCropCfg:
    global_: ViewAugCfg
    local: ViewAugCfg


def build_local_views_transform(cfg: LocalViewsCfg):
    """
    Returns a callable: PIL -> Tensor[V, C, H, W]
    Local-only multicrop (single resolution), repeated V times.
    Matches LeJEPA README local-view augmentations.
    """
    aug = [
        v2.RandomResizedCrop(cfg.crop_size, scale=(cfg.scale_min, cfg.scale_max)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if cfg.normalize_imagenet:
        aug.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    transform = v2.Compose(aug)

    def _apply(img) -> torch.Tensor:
        views: List[torch.Tensor] = [transform(img) for _ in range(cfg.V)]
        return torch.stack(views, dim=0)  # (V,C,H,W)

    return _apply



def _single_view_tfm(cfg: ViewAugCfg) -> v2.Compose:
    aug: List[Any] = [
        v2.RandomResizedCrop(cfg.crop_size, scale=(cfg.scale_min, cfg.scale_max)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
        v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if cfg.normalize_imagenet:
        aug.append(v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return v2.Compose(aug)


def build_multicrop_transform(cfg: MultiCropCfg):
    """
    Returns: PIL -> List[Tensor], length Vg + Vl
    First Vg are global (224), remaining Vl are local (96).
    """
    t_g = _single_view_tfm(cfg.global_)
    t_l = _single_view_tfm(cfg.local)

    def _apply(img) -> List[torch.Tensor]:
        views: List[torch.Tensor] = []
        for _ in range(cfg.global_.V):
            views.append(t_g(img))
        for _ in range(cfg.local.V):
            views.append(t_l(img))
        return views

    return _apply





def collate_multicrop_with_meta(batch: List[Tuple[List[torch.Tensor], Any]]):
    """
    Each item: (views: List[Tensor(C,H,W)] length V, meta)

    Returns:
      views_batched: List[Tensor(bs,C,H,W)] length V
      metas: List[meta] length bs
    """
    views_list, metas = zip(*batch)  # tuple length bs
    V = len(views_list[0])

    # Sanity: all items must have same number of views
    for v in views_list:
        if len(v) != V:
            raise ValueError("Inconsistent number of views in batch")

    views_batched: List[torch.Tensor] = []
    for j in range(V):
        # stack j-th view across batch; requires same H,W for that view index
        views_batched.append(torch.stack([views_list[i][j] for i in range(len(views_list))], dim=0))

    return views_batched, list(metas)




def collate_views_with_meta(batch: List[Tuple[torch.Tensor, Any]]):
    """
    Each item: (views: Tensor[V,C,H,W], meta)
    Returns:
      vs: Tensor[bs, V, C, H, W]
      metas: list(meta) length bs
    """
    vs, metas = zip(*batch)
    vs_t = torch.stack(vs, dim=0)
    return vs_t, list(metas)
