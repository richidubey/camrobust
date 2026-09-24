from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .corruptions import TEST_CORRUPTIONS, UNSEEN_CORRUPTIONS, apply_corruption
from .data import CLASS_NAMES, normalize
from .metrics import SegmentationMetrics
from .models import logits


def _condition_seed(name: str, severity: int, seed: int) -> int:
    digest = hashlib.sha256(f"{name}:{severity}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    corruption: str = "clean",
    severity: int = 0,
    seed: int = 0,
    progress: bool = False,
) -> dict[str, object]:
    model.eval()
    metrics = SegmentationMetrics()
    subgroup_metrics: dict[str, SegmentationMetrics] = {}
    generator = torch.Generator(device=device)
    generator.manual_seed(_condition_seed(corruption, severity, seed))
    iterator: Iterable = tqdm(loader, leave=False, desc=f"{corruption}-{severity}") if progress else loader
    for batch in iterator:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        if corruption != "clean":
            x = apply_corruption(x, corruption, severity, generator=generator)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = logits(model, normalize(x))
        metrics.update(output, y)
        if corruption == "clean":
            names = list(batch["name"])
            for sequence in sorted({name.split("_")[0] for name in names}):
                selected = torch.tensor(
                    [name.startswith(f"{sequence}_") for name in names], device=device, dtype=torch.bool
                )
                subgroup_metrics.setdefault(sequence, SegmentationMetrics()).update(output[selected], y[selected])
    result = metrics.compute()
    if subgroup_metrics:
        result["sequence_metrics"] = {name: value.compute() for name, value in subgroup_metrics.items()}
    return result


def evaluate_suite(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    severities: int = 3,
    seed: int = 0,
    progress: bool = False,
) -> dict[str, object]:
    conditions: dict[str, dict[str, object]] = {
        "clean": evaluate_condition(model, loader, device, seed=seed, progress=progress)
    }
    for corruption in TEST_CORRUPTIONS:
        for severity in range(1, severities + 1):
            key = f"{corruption}_s{severity}"
            conditions[key] = evaluate_condition(
                model, loader, device, corruption, severity, seed=seed, progress=progress
            )

    corrupted = [value for key, value in conditions.items() if key != "clean"]
    unseen = [
        value
        for key, value in conditions.items()
        if any(key.startswith(f"{name}_") for name in UNSEEN_CORRUPTIONS)
    ]
    clean_miou = float(conditions["clean"]["miou"])
    miou_c = sum(float(value["miou"]) for value in corrupted) / len(corrupted)
    miou_unseen = sum(float(value["miou"]) for value in unseen) / len(unseen)
    per_corruption = {
        name: sum(float(conditions[f"{name}_s{s}"]["miou"]) for s in range(1, severities + 1))
        / severities
        for name in TEST_CORRUPTIONS
    }
    aggregate = {
        "clean_miou": clean_miou,
        "miou_c": miou_c,
        "unseen_miou_c": miou_unseen,
        "worst_corruption_miou": min(per_corruption.values()),
        "relative_degradation": (clean_miou - miou_c) / max(clean_miou, 1e-8),
        "mean_boundary_f1_c": sum(float(value["boundary_f1"]) for value in corrupted) / len(corrupted),
        "mean_ece_c": sum(float(value["ece"]) for value in corrupted) / len(corrupted),
        "per_corruption_miou": per_corruption,
        "class_names": list(CLASS_NAMES),
    }
    return {"aggregate": aggregate, "conditions": conditions}
