from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset

CLASS_NAMES = (
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
)
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 11
IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor
    mask: torch.Tensor
    index: int
    name: str


class CamVid(Dataset):
    """Standard SegNet CamVid split, returning unnormalized RGB in [0, 1]."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        size: tuple[int, int] = (288, 384),
        indices: Sequence[int] | None = None,
        cache: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.size = size
        image_dir = self.root / split
        mask_dir = self.root / f"{split}annot"
        self.images = sorted(image_dir.glob("*.png"))
        self.masks = [mask_dir / p.name for p in self.images]
        if indices is not None:
            chosen = list(indices)
            self.images = [self.images[i] for i in chosen]
            self.masks = [self.masks[i] for i in chosen]
            self.original_indices = chosen
        else:
            self.original_indices = list(range(len(self.images)))
        if not self.images or not all(p.exists() for p in self.masks):
            raise FileNotFoundError(f"CamVid split not found under {self.root}: {split}")
        self._cache = [self._load_item(i) for i in range(len(self.images))] if cache else None

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        if self._cache is not None:
            return self._cache[index]
        return self._load_item(index)

    def _load_item(self, index: int) -> dict[str, torch.Tensor | str | int]:
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")
        x = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        x = F.interpolate(x[None], size=self.size, mode="bilinear", align_corners=False)[0]
        y = F.interpolate(y[None, None].float(), size=self.size, mode="nearest")[0, 0].long()
        return {
            "image": x,
            "mask": y,
            "index": self.original_indices[index],
            "name": self.images[index].name,
        }


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
    return (x - mean) / std


def nested_random_subset(total: int, budget: int, seed: int) -> list[int]:
    if not 0 < budget <= total:
        raise ValueError(f"budget must be in [1, {total}], got {budget}")
    rng = np.random.default_rng(seed)
    return rng.permutation(total)[:budget].tolist()


def class_weights(dataset: Dataset, max_weight: float = 3.0) -> torch.Tensor:
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    if isinstance(dataset, CamVid):
        for path in dataset.masks:
            y = np.asarray(Image.open(path), dtype=np.uint8)
            counts += torch.from_numpy(np.bincount(y[y < NUM_CLASSES], minlength=NUM_CLASSES)).double()
    else:
        for sample in dataset:
            y = sample["mask"]
            valid = (y >= 0) & (y < NUM_CLASSES)
            counts += torch.bincount(y[valid].flatten(), minlength=NUM_CLASSES).double()
    freq = counts / counts.sum().clamp_min(1)
    weights = 1.0 / torch.sqrt(freq.clamp_min(1e-8))
    weights /= weights.mean()
    return weights.clamp(0.35, max_weight).float()


def class_presence(dataset: Dataset) -> torch.Tensor:
    presence = torch.zeros((len(dataset), NUM_CLASSES), dtype=torch.bool)
    iterator = dataset.masks if isinstance(dataset, CamVid) else range(len(dataset))
    for i, item in enumerate(iterator):
        if isinstance(dataset, CamVid):
            labels = torch.from_numpy(np.unique(np.asarray(Image.open(item), dtype=np.uint8))).long()
        else:
            labels = torch.unique(dataset[item]["mask"])
        labels = labels[(labels >= 0) & (labels < NUM_CLASSES)]
        presence[i, labels] = True
    return presence
