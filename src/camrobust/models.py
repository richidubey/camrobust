from __future__ import annotations

import torch
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

from .data import NUM_CLASSES


def build_model(architecture: str = "deeplab") -> torch.nn.Module:
    kwargs = {
        "weights": None,
        "weights_backbone": MobileNet_V3_Large_Weights.DEFAULT,
        "num_classes": NUM_CLASSES,
    }
    if architecture == "deeplab":
        return deeplabv3_mobilenet_v3_large(**kwargs)
    if architecture == "lraspp":
        return lraspp_mobilenet_v3_large(**kwargs)
    raise KeyError(f"unknown architecture: {architecture}")


def logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x)
    return output["out"] if isinstance(output, dict) else output

