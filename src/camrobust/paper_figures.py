from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch.utils.data import DataLoader

from .corruptions import TEST_CORRUPTIONS, apply_corruption
from .data import CLASS_NAMES, CamVid, IGNORE_INDEX, normalize
from .losses import semantic_boundary
from .models import build_model, logits


PALETTE = np.array(
    [
        [128, 128, 128], [128, 0, 0], [192, 192, 128], [128, 64, 128],
        [0, 0, 192], [128, 128, 0], [192, 128, 128], [64, 64, 128],
        [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 0],
    ],
    dtype=float,
) / 255.0
CMAP = ListedColormap(PALETTE[: len(CLASS_NAMES)])

BLUE = "#3B6FB6"
GREEN = "#2A9D6F"
RED = "#D55E5E"
GOLD = "#E9A23B"
PURPLE = "#7656A3"
INK = "#28323C"
LIGHT = "#F3F5F7"


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.3,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8.6, color=INK)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = INK) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.2, color=color))


def _mask_rgb(mask: np.ndarray) -> np.ndarray:
    safe = mask.copy()
    safe[(safe < 0) | (safe >= len(PALETTE))] = len(PALETTE) - 1
    return PALETTE[safe]


def make_overview(data_root: Path, output: Path, index: int = 91) -> None:
    dataset = CamVid(data_root, "test", (288, 384))
    sample = dataset[index]
    x = sample["image"][None]
    y = sample["mask"][None]
    corrupt = apply_corruption(x, "shot_noise", 3, generator=torch.Generator().manual_seed(17))
    boundary = semantic_boundary(y, radius=2)[0].numpy()

    fig = plt.figure(figsize=(9.4, 2.72), facecolor="white")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.10, 1.35, 1.05, 1.45], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    _clean_axis(ax)
    ax.set_title("1  Few labels", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=7)
    ax.imshow(x[0].permute(1, 2, 0).numpy(), extent=(0.02, 0.98, 0.48, 0.97))
    mask_image = _mask_rgb(y[0].numpy())
    mask_image[boundary] = [1.0, 0.78, 0.05]
    ax.imshow(mask_image, extent=(0.02, 0.98, 0.00, 0.45))
    ax.text(0.04, 0.92, "RGB", fontsize=9, color="white", weight="bold", bbox=dict(facecolor=INK, alpha=0.75, pad=2, edgecolor="none"))
    ax.text(0.04, 0.39, "mask + boundary band", fontsize=8.6, color=INK, weight="bold", bbox=dict(facecolor="white", alpha=0.85, pad=2, edgecolor="none"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 1])
    _clean_axis(ax)
    ax.set_title("2  Aligned views", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=7)
    ax.imshow(x[0].permute(1, 2, 0).numpy(), extent=(0.00, 0.43, 0.63, 0.96))
    ax.imshow(corrupt[0].permute(1, 2, 0).numpy(), extent=(0.00, 0.43, 0.14, 0.47))
    ax.text(0.015, 0.89, "clean", color="white", fontsize=8.5, weight="bold")
    ax.text(0.015, 0.40, "corrupted", color="white", fontsize=8.5, weight="bold")
    _box(ax, (0.59, 0.41), 0.35, 0.27, "shared\nsegmenter", BLUE)
    # Box spans y in [0.41, 0.68]; keep both feeds and both outputs symmetric
    # about its centre (0.545) and its horizontal centre (0.765).
    _arrow(ax, (0.43, 0.79), (0.585, 0.58), BLUE)
    _arrow(ax, (0.43, 0.31), (0.585, 0.51), BLUE)
    _arrow(ax, (0.765, 0.69), (0.765, 0.745), BLUE)
    _arrow(ax, (0.765, 0.40), (0.765, 0.345), BLUE)
    ax.text(0.765, 0.765, r"$p$", fontsize=11, color=BLUE, ha="center", va="bottom")
    ax.text(0.765, 0.325, r"$p_t$", fontsize=11, color=BLUE, ha="center", va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 2])
    _clean_axis(ax)
    ax.set_title("3  Four losses", loc="left", fontsize=12.0, fontweight="bold", color=INK, pad=7)
    steps = [
        ("CameraAug", r"$\mathrm{CE}(p_t,y)$", GOLD),
        ("Paired", r"$\frac{1}{2}[\mathrm{CE}(p,y)+\mathrm{CE}(p_t,y)]$", GREEN),
        ("Uniform", r"$+\lambda\,\mathrm{JS}(p,p_t)$", PURPLE),
        ("BaCC", r"$+\lambda\,w_{\partial y}\,\mathrm{JS}(p,p_t)$", RED),
    ]
    for i, (name, formula, color) in enumerate(steps):
        ypos = 0.80 - i * 0.225
        ax.add_patch(FancyBboxPatch((0.02, ypos - 0.09), 0.96, 0.16, boxstyle="round,pad=0.015", facecolor=LIGHT, edgecolor=color, linewidth=1.2))
        ax.text(0.06, ypos + 0.045, name, fontsize=8.8, fontweight="bold", color=color, va="center")
        ax.text(0.50, ypos - 0.035, formula, fontsize=7.5, color=INK, va="center", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 3])
    _clean_axis(ax)
    ax.set_title("4  Test suite", loc="left", fontsize=12.0, fontweight="bold", color=INK, pad=7)
    names = ["brightness", "darkness", "white_balance", "gaussian_noise", "shot_noise", "defocus_blur", "motion_blur", "jpeg"]
    short = ["bright", "dark", "WB", "Gauss.", "shot*", "defocus", "motion*", "JPEG*"]
    for i, (name, label) in enumerate(zip(names, short)):
        row, col = divmod(i, 4)
        left = 0.01 + col * 0.247
        bottom = 0.61 - row * 0.29
        image = apply_corruption(x, name, 3, generator=torch.Generator().manual_seed(100 + i))[0]
        ax.imshow(image.permute(1, 2, 0).numpy(), extent=(left, left + 0.225, bottom, bottom + 0.20))
        ax.text(left + 0.112, bottom - 0.025, label, ha="center", va="top", fontsize=7.2, color=RED if "*" in label else INK, weight="bold" if "*" in label else "normal")
    ax.text(0.50, 0.205, "32 labels", fontsize=7.8, color=INK, weight="bold", ha="center")
    ax.text(0.50, 0.125, "Paired−ERM   +1.44", fontsize=7.8, color=GREEN, weight="bold", ha="center")
    ax.text(0.50, 0.045, "BaCC−uniform   +0.03", fontsize=7.8, color=PURPLE, weight="bold", ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def make_diagnostics(aggregate_csv: Path, output: Path) -> None:
    df = pd.read_csv(aggregate_csv)
    main = df[(df.architecture == "deeplab") & (df.boundary_alpha == 3.0)]
    summary = main.groupby(["budget", "method"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.2), constrained_layout=True)

    ax = axes[0, 0]
    for method, color, marker in [
        ("erm", INK, "o"),
        ("pair", GREEN, "s"),
        ("consistency", PURPLE, "^"),
        ("bacc", RED, "D"),
    ]:
        group = summary[summary.method == method]
        label = {"erm": "ERM", "pair": "Paired", "consistency": "Uniform", "bacc": "BaCC"}[method]
        ax.plot(group.budget, 100 * group.miou_c, color=color, marker=marker, linewidth=2, label=label)
    ax.set(xlabel="Labeled training frames", ylabel="mIoU-C (%)", xticks=[16, 32, 64])
    ax.set_title("(a) More labels dominate method choice", loc="left", fontsize=10.2, fontweight="bold")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 1]
    budget = 32
    order = ["erm", "cameraaug", "pair", "consistency", "bacc"]
    values = [
        100 * summary[(summary.budget == budget) & (summary.method == method)].iloc[0].miou_c
        for method in order
    ]
    labels = ["ERM", "CameraAug", "Paired", "Uniform", "BaCC"]
    colors = [INK, GOLD, GREEN, PURPLE, RED]
    ax.plot(range(len(values)), values, color="#B8BEC5", linewidth=1.5, zorder=1)
    ax.scatter(range(len(values)), values, c=colors, s=55, zorder=2)
    for i, value in enumerate(values):
        ax.text(i, value + 0.20, f"{value:.1f}", ha="center", fontsize=8, color=colors[i], fontweight="bold")
    ax.set_xticks(range(len(values)), labels, rotation=20)
    ax.set_ylim(min(values) - 0.45, max(values) + 0.55)
    ax.set_ylabel("mIoU-C (%)")
    ax.set_title("(b) Paired views give the useful step", loc="left", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y", alpha=0.22)

    ax = axes[1, 0]
    row = summary[(summary.budget == 32)].set_index("method")
    corr_cols = [column for column in row.columns if column.startswith("corr_")]
    delta = 100 * (row.loc["pair", corr_cols] - row.loc["erm", corr_cols])
    short_labels = {
        "brightness": "Bright",
        "darkness": "Dark",
        "defocus_blur": "Defoc.",
        "gaussian_noise": "Gauss.",
        "jpeg": "JPEG*",
        "motion_blur": "Motion*",
        "shot_noise": "Shot*",
        "white_balance": "WB",
    }
    labels = [short_labels[column.removeprefix("corr_")] for column in corr_cols]
    bar_colors = [GREEN if value >= 0 else RED for value in delta]
    ax.bar(range(len(delta)), delta, color=bar_colors)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(len(delta)), labels, fontsize=7.2)
    ax.set_ylabel(r"$\Delta$ mIoU vs. ERM")
    ax.set_title("(c) Paired gains depend on corruption", loc="left", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    for i, value in enumerate(delta):
        ax.text(i, value + (0.16 if value >= 0 else -0.22), f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7.2, color=bar_colors[i], fontweight="bold")

    ax = axes[1, 1]
    pivot = main.pivot_table(index="budget", columns="method", values=["miou_c", "mean_boundary_f1_c"], aggfunc="mean")
    budgets = np.array([16, 32, 64])
    robust_delta = 100 * (pivot["miou_c"]["bacc"] - pivot["miou_c"]["consistency"]).loc[budgets].to_numpy()
    boundary_delta = 100 * (pivot["mean_boundary_f1_c"]["bacc"] - pivot["mean_boundary_f1_c"]["consistency"]).loc[budgets].to_numpy()
    xpos = np.arange(len(budgets))
    width = 0.34
    ax.bar(xpos - width / 2, robust_delta, width, label="mIoU-C", color=PURPLE)
    ax.bar(xpos + width / 2, boundary_delta, width, label="Boundary F1", color=RED)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(xpos, budgets)
    ax.set_xlabel("Labeled training frames")
    ax.set_ylabel(r"BaCC $-$ uniform (points)")
    ax.set_title("(d) BaCC moves boundary F1, mIoU-C stays flat", loc="left", fontsize=10.2, fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    ax.legend(frameon=False, fontsize=8)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _load_checkpoint(path: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload["config"]["architecture"])
    model.load_state_dict(payload["model"])
    return model.to(device).eval()


def _image_miou(pred: np.ndarray, target: np.ndarray) -> float:
    valid = target != IGNORE_INDEX
    classes = np.union1d(np.unique(pred[valid]), np.unique(target[valid]))
    values = []
    for cls in classes:
        if cls >= len(CLASS_NAMES):
            continue
        intersection = np.logical_and(pred == cls, target == cls) & valid
        union = (np.logical_or(pred == cls, target == cls)) & valid
        if union.any():
            values.append(intersection.sum() / union.sum())
    return float(np.mean(values)) if values else 0.0


@torch.inference_mode()
def make_qualitative(
    data_root: Path,
    erm_path: Path,
    pair_path: Path,
    bacc_path: Path,
    output: Path,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "ERM": _load_checkpoint(erm_path, device),
        "Paired": _load_checkpoint(pair_path, device),
        "BaCC": _load_checkpoint(bacc_path, device),
    }
    dataset = CamVid(data_root, "test", (288, 384), cache=True)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    conditions = [("shot_noise", 3, "median positive"), ("motion_blur", 3, "median positive"), ("darkness", 3, "median negative")]
    chosen = []
    for condition_index, (name, severity, selection) in enumerate(conditions):
        records = []
        offset = 0
        generator = torch.Generator(device=device).manual_seed(9000 + condition_index)
        for batch in loader:
            clean = batch["image"].to(device)
            target = batch["mask"].numpy()
            corrupt = apply_corruption(clean, name, severity, generator=generator)
            predictions = {
                label: logits(model, normalize(corrupt)).argmax(1).cpu().numpy()
                for label, model in models.items()
            }
            for local in range(len(target)):
                scores = {label: _image_miou(pred[local], target[local]) for label, pred in predictions.items()}
                records.append(
                    {
                        "index": offset + local,
                        "name": batch["name"][local],
                        "delta": scores["Paired"] - scores["ERM"],
                        "corrupt": corrupt[local].cpu().permute(1, 2, 0).numpy(),
                        "target": target[local],
                        "predictions": {label: pred[local] for label, pred in predictions.items()},
                    }
                )
            offset += len(target)
        candidates = [record for record in records if record["delta"] > 0] if "positive" in selection else [record for record in records if record["delta"] < 0]
        candidates = candidates or records
        candidates.sort(key=lambda record: record["delta"])
        record = candidates[len(candidates) // 2]
        record.update({"condition": name, "severity": severity})
        chosen.append(record)

    fig, axes = plt.subplots(3, 5, figsize=(8.6, 4.9), constrained_layout=True)
    titles = ["Corrupted input", "Ground truth", "ERM", "Paired", "BaCC"]
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12.0, fontweight="bold", color=INK)
    for row, record in enumerate(chosen):
        panels = [
            record["corrupt"],
            record["target"],
            record["predictions"]["ERM"],
            record["predictions"]["Paired"],
            record["predictions"]["BaCC"],
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel, cmap=None if col == 0 else CMAP, vmin=None if col == 0 else 0, vmax=None if col == 0 else len(CLASS_NAMES) - 1)
            axes[row, col].axis("off")
        name = record["condition"].replace("_", " ")
        axes[row, 0].text(
            -0.03,
            0.5,
            f"{name}, s={record['severity']}\nPaired−ERM {100 * record['delta']:+.1f}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=10.0,
            color=GREEN if record["delta"] > 0 else RED,
            fontweight="bold",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    pd.DataFrame(
        [
            {
                "condition": record["condition"],
                "severity": record["severity"],
                "test_index": record["index"],
                "image_name": record["name"],
                "paired_minus_erm_per_image_miou": record["delta"],
            }
            for record in chosen
        ]
    ).to_csv(output.with_name(output.stem + "_selection.csv"), index=False)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-native figures from experiment records")
    parser.add_argument("--data-root", type=Path, default=Path("data/SegNet-Tutorial/CamVid"))
    parser.add_argument("--aggregate", type=Path, default=Path("results/aggregate/all_runs.csv"))
    parser.add_argument("--erm", type=Path, default=Path("artifacts/checkpoints/deeplab_erm_b32_s0/checkpoint.pt"))
    parser.add_argument("--pair", type=Path, default=Path("artifacts/checkpoints/deeplab_pair_b32_s0/checkpoint.pt"))
    parser.add_argument("--bacc", type=Path, default=Path("artifacts/checkpoints/deeplab_bacc_b32_s0_a3/checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/aggregate"))
    args = parser.parse_args()
    make_overview(args.data_root, args.output / "overview.pdf")
    make_diagnostics(args.aggregate, args.output / "diagnostics.pdf")
    make_qualitative(args.data_root, args.erm, args.pair, args.bacc, args.output / "qualitative_main.pdf")


if __name__ == "__main__":
    main()
