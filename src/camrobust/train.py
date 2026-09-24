from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from .corruptions import random_camera_corruption, random_color_jitter
from .data import CamVid, class_presence, class_weights, nested_random_subset, normalize
from .evaluate import evaluate_condition, evaluate_suite
from .losses import consistency_loss, supervised_loss
from .models import build_model, logits

METHODS = ("erm", "colorjitter", "cameraaug", "pair", "consistency", "bacc")
_EVAL_DATASET_CACHE: dict[tuple[str, str, tuple[int, int]], CamVid] = {}


@dataclass
class TrainConfig:
    data_root: str = "data/SegNet-Tutorial/CamVid"
    output_root: str = "results/runs"
    method: str = "bacc"
    architecture: str = "deeplab"
    budget: int = 32
    seed: int = 0
    steps: int = 400
    batch_size: int = 8
    eval_batch_size: int = 32
    height: int = 288
    width: int = 384
    lr: float = 3e-4
    weight_decay: float = 1e-4
    consistency_weight: float = 1.0
    boundary_alpha: float = 3.0
    boundary_radius: int = 2
    eval_every: int = 100
    eval_severities: int = 3
    workers: int = 4
    eval_workers: int = 0
    save_checkpoint: bool = False
    progress: bool = False

    @property
    def run_name(self) -> str:
        alpha = f"_a{self.boundary_alpha:g}" if self.method == "bacc" else ""
        return f"{self.architecture}_{self.method}_b{self.budget}_s{self.seed}{alpha}"


def seed_everything(seed: int) -> None:
    if not getattr(seed_everything, "threads_configured", False):
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)
        seed_everything.threads_configured = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temp, path)


def _augment_geometry(x: torch.Tensor, y: torch.Tensor, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    for i in range(len(x)):
        if rng.random() < 0.5:
            x[i] = x[i].flip(-1)
            y[i] = y[i].flip(-1)
    return x, y


def _cached_eval_dataset(root: str, split: str, size: tuple[int, int]) -> CamVid:
    key = (str(Path(root).resolve()), split, size)
    if key not in _EVAL_DATASET_CACHE:
        _EVAL_DATASET_CACHE[key] = CamVid(root, split, size, cache=True)
    return _EVAL_DATASET_CACHE[key]


def run(config: TrainConfig) -> dict[str, object]:
    if config.method not in METHODS:
        raise KeyError(config.method)
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (config.height, config.width)
    full_train = CamVid(config.data_root, "train", size)
    indices = nested_random_subset(len(full_train), config.budget, config.seed)
    train_set = CamVid(config.data_root, "train", size, indices)
    val_set = _cached_eval_dataset(config.data_root, "val", size)
    test_set = _cached_eval_dataset(config.data_root, "test", size)
    sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=config.steps * config.batch_size,
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=config.workers > 0,
    )
    eval_kwargs = dict(
        batch_size=config.eval_batch_size,
        num_workers=config.eval_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.eval_workers > 0,
    )
    val_loader = DataLoader(val_set, **eval_kwargs)
    test_loader = DataLoader(test_set, **eval_kwargs)
    weights = class_weights(train_set).to(device)
    presence = class_presence(train_set).any(0).tolist()

    model = build_model(config.architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: max(0.0, 1.0 - step / max(config.steps, 1)) ** 0.9
    )
    rng = random.Random(config.seed + 17)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val = -1.0
    start = time.time()
    iterator = tqdm(train_loader, total=config.steps, disable=not config.progress, desc=config.run_name)
    for step, batch in enumerate(iterator, start=1):
        if step > config.steps:
            break
        model.train()
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        x, y = _augment_geometry(x, y, rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            if config.method == "erm":
                clean_logits = logits(model, normalize(x))
                loss_sup = supervised_loss(clean_logits, y, weights)
                loss_cons = x.new_zeros(())
                loss = loss_sup
            elif config.method == "colorjitter":
                corrupt = random_color_jitter(x, rng)
                corrupt_logits = logits(model, normalize(corrupt))
                loss_sup = supervised_loss(corrupt_logits, y, weights)
                loss_cons = x.new_zeros(())
                loss = loss_sup
            elif config.method == "cameraaug":
                corrupt, _, _ = random_camera_corruption(x, rng)
                corrupt_logits = logits(model, normalize(corrupt))
                loss_sup = supervised_loss(corrupt_logits, y, weights)
                loss_cons = x.new_zeros(())
                loss = loss_sup
            else:
                corrupt, _, _ = random_camera_corruption(x, rng)
                clean_logits = logits(model, normalize(x))
                corrupt_logits = logits(model, normalize(corrupt))
                loss_sup = 0.5 * (
                    supervised_loss(clean_logits, y, weights) + supervised_loss(corrupt_logits, y, weights)
                )
                alpha = config.boundary_alpha if config.method == "bacc" else 0.0
                loss_cons = consistency_loss(
                    clean_logits, corrupt_logits, y, alpha, config.boundary_radius
                )
                loss = loss_sup
                if config.method in ("consistency", "bacc"):
                    loss = loss + config.consistency_weight * loss_cons
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 25 == 0:
            history.append(
                {
                    "step": float(step),
                    "loss": float(loss.detach()),
                    "supervised": float(loss_sup.detach()),
                    "consistency": float(loss_cons.detach()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
        if step % config.eval_every == 0 or step == config.steps:
            val_metrics = evaluate_condition(model, val_loader, device, seed=config.seed)
            val_miou = float(val_metrics["miou"])
            history[-1]["val_miou"] = val_miou
            if val_miou > best_val:
                best_val = val_miou
                best_state = copy.deepcopy(model.state_dict())
        if config.progress:
            iterator.set_postfix(loss=f"{float(loss.detach()):.3f}", best=f"{best_val:.3f}")

    model.load_state_dict(best_state)
    suite = evaluate_suite(
        model,
        test_loader,
        device,
        severities=config.eval_severities,
        seed=config.seed,
        progress=config.progress,
    )
    elapsed = time.time() - start
    result: dict[str, object] = {
        "config": asdict(config),
        "run_name": config.run_name,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "selected_indices": indices,
        "selected_class_presence": presence,
        "class_weights": weights.cpu().tolist(),
        "best_val_miou": best_val,
        "elapsed_seconds": elapsed,
        "history": history,
        "test": suite,
    }
    run_dir = Path(config.output_root) / config.run_name
    _atomic_json(run_dir / "metrics.json", result)
    if config.save_checkpoint:
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": best_state, "config": asdict(config), "selected_indices": indices},
            run_dir / "checkpoint.pt",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/SegNet-Tutorial/CamVid")
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--method", choices=METHODS, default="bacc")
    parser.add_argument("--architecture", choices=("deeplab", "lraspp"), default="deeplab")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--boundary-alpha", type=float, default=3.0)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-severities", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=0)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(TrainConfig(**vars(args)))
    aggregate = result["test"]["aggregate"]
    print(json.dumps({"run": result["run_name"], **aggregate}, indent=2))


if __name__ == "__main__":
    main()
