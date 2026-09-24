from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

from .corruptions import apply_corruption
from .data import CLASS_NAMES, CamVid, normalize
from .models import build_model, logits

PALETTE = np.array(
    [
        [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
        [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
        [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0],
    ], dtype=float
) / 255.0


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_model(config["architecture"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return model, config


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate anonymous qualitative comparison figure")
    parser.add_argument("--erm", type=Path, required=True)
    parser.add_argument("--bacc", type=Path, required=True)
    parser.add_argument("--data-root", default="data/SegNet-Tutorial/CamVid")
    parser.add_argument("--output", type=Path, default=Path("results/aggregate/qualitative.pdf"))
    parser.add_argument("--indices", type=int, nargs="+", default=[8, 91, 177])
    parser.add_argument("--corruption", default="motion_blur")
    parser.add_argument("--severity", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    erm, config = load_checkpoint(args.erm, device)
    bacc, _ = load_checkpoint(args.bacc, device)
    dataset = CamVid(args.data_root, "test", (config["height"], config["width"]))
    cmap = ListedColormap(PALETTE[: len(CLASS_NAMES)])
    fig, axes = plt.subplots(len(args.indices), 5, figsize=(11, 2.1 * len(args.indices)), constrained_layout=True)
    if len(args.indices) == 1:
        axes = axes[None]
    for row, index in enumerate(args.indices):
        sample = dataset[index]
        x = sample["image"][None].to(device)
        y = sample["mask"].numpy()
        corrupted = apply_corruption(x, args.corruption, args.severity)
        pred_erm = logits(erm, normalize(corrupted)).argmax(1)[0].cpu().numpy()
        pred_bacc = logits(bacc, normalize(corrupted)).argmax(1)[0].cpu().numpy()
        panels = [
            x[0].permute(1, 2, 0).cpu().numpy(),
            corrupted[0].permute(1, 2, 0).cpu().numpy(),
            y,
            pred_erm,
            pred_bacc,
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel if col < 2 else panel, cmap=None if col < 2 else cmap, vmin=0 if col >= 2 else None, vmax=10 if col >= 2 else None)
            axes[row, col].axis("off")
    for axis, title in zip(axes[0], ["Clean image", "Corrupted image", "Ground truth", "ERM", "BaCC"]):
        axis.set_title(title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
