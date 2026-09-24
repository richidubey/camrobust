from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .train import TrainConfig, run


def _completed(config: TrainConfig) -> bool:
    path = Path(config.output_root) / config.run_name / "metrics.json"
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        return payload.get("config") == config.__dict__
    except (json.JSONDecodeError, OSError):
        return False


def build_suite(name: str, base: TrainConfig) -> list[TrainConfig]:
    configs: list[TrainConfig] = []
    if name in ("main", "all"):
        for budget in (16, 32, 64):
            for seed in (0, 1, 2):
                for method in ("erm", "colorjitter", "cameraaug", "pair", "consistency", "bacc"):
                    configs.append(replace(base, budget=budget, seed=seed, method=method))
    if name in ("ablation", "all"):
        for seed in (0, 1, 2):
            for alpha in (1.0, 6.0):
                configs.append(replace(base, budget=32, seed=seed, method="bacc", boundary_alpha=alpha))
    if name in ("backbone", "all"):
        for seed in (0, 1, 2):
            for method in ("erm", "bacc"):
                configs.append(replace(base, architecture="lraspp", budget=32, seed=seed, method=method))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic CamRobust experiment suites")
    parser.add_argument("--suite", choices=("main", "ablation", "backbone", "all"), default="all")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--data-root", default="data/SegNet-Tutorial/CamVid")
    parser.add_argument("--output-root", default="results/runs")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    base = TrainConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        steps=args.steps,
        progress=args.progress,
    )
    configs = build_suite(args.suite, base)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require shard-count >= 1 and 0 <= shard-index < shard-count")
    indexed_configs = [(i, config) for i, config in enumerate(configs, start=1) if (i - 1) % args.shard_count == args.shard_index]
    for i, config in indexed_configs:
        if _completed(config) and not args.force:
            print(f"[{i}/{len(configs)}] skip {config.run_name}")
            continue
        print(f"[{i}/{len(configs)}] run  {config.run_name}", flush=True)
        result = run(config)
        agg = result["test"]["aggregate"]
        print(
            f"    clean={agg['clean_miou']:.4f} mIoU-C={agg['miou_c']:.4f} "
            f"unseen={agg['unseen_miou_c']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
