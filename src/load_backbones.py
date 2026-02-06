from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
from torchvision.models import resnet101


def load_segmentation_encoder(ckpt_path = "hp_tune.pth"):
    model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, 6, kernel_size=1)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    encoder = model.backbone
    return encoder

def load_resnet101_encoder(pretrained: bool = False):
    model = resnet101(weights=None if not pretrained else "IMAGENET1K_V1")
    encoder = nn.Sequential(*list(model.children())[:-2])  # conv features only
    return encoder


def load_encoder_backbone(init, seg_ckpt = None,):
    if init == "seg_init":
        return load_segmentation_encoder(seg_ckpt)
    if init == "imagenet":
        return load_resnet101_encoder(pretrained=True)
    if init == "random":
        return load_resnet101_encoder(pretrained=False)
    raise ValueError(f"Unknown encoder init mode: {init}")
