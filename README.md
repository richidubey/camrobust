# CamRobust

Code, fixed splits, and run records for **Supervising Both Views: Camera
Robustness under Label Scarcity** (SmallData-CV workshop, ACCV 2026).

With only 16, 32, or 64 labeled CamVid frames, which cheap training choices help
segmentation survive camera corruption? We compare six recipes, each changing
one thing relative to the previous: plain supervised training (ERM), generic
color jitter, camera-realistic augmentation, supervising each frame together
with a corrupted copy of itself, adding a uniform consistency loss between the
two predictions, and concentrating that consistency near class boundaries
(BaCC). Evaluation covers eight corruption families at three severities, three
of which never appear during training.

The short version of the result: supervising both views provides most of the
available robustness, worth 1.1 to 1.4 mIoU-C over ERM at every budget. The
consistency losses added on top, including our own, move mIoU-C by fractions of
a point.

## Reproduce

```bash
cd camrobust
UV_CACHE_DIR=/tmp/camrobust-uv-cache uv sync
git clone --depth 1 https://github.com/alexgkendall/SegNet-Tutorial.git data/SegNet-Tutorial
.venv/bin/pip install -e .

# quick check that everything is wired up
.venv/bin/pytest
camrobust-train --method bacc --budget 16 --seed 0 --steps 20 --eval-severities 1

# the main multi-seed grid
camrobust-grid --suite main
camrobust-analyze

# figures, once representative checkpoints are saved
python -m camrobust.paper_figures
```

`--suite all` adds the boundary-strength sweep and the LR-ASPP runs, which is
what produced the 66 records in `results/runs/`.

CamVid is not redistributed here; the clone step above fetches it. Rebuilding
the paper through `scripts/verify.sh` also needs `tectonic` on your PATH.

## Layout

- `src/camrobust/`: corruptions, losses, training loop, evaluation, analysis
- `results/runs/`: one JSON per run, with config, selected frame indices, class
  weights, training history, and every test condition
- `results/aggregate/`: CSV summaries, the tables and figures used in the
  paper, and the LaTeX macros they feed
- `paper/`: LaTeX sources, plus the hypotheses and claim ledger fixed before
  the grid was run (`RESEARCH_NOTES.md`)

The dataset and model checkpoints stay out of version control.

## The loss

For a labeled image and its pixel-aligned camera-corrupted view:

`L = 0.5 CE(p,y) + 0.5 CE(p_c,y) + lambda * mean(w_boundary * JS(p,p_c))`

`w_boundary` upweights a narrow band around semantic transitions and is
normalized to mean one, so changing its strength redistributes weight rather
than rescaling the loss. Setting the boundary strength to zero recovers uniform
consistency. Only photometric and sensor corruptions are paired, so geometry is
shared and the label stays valid for both views.

## Scope

This is a controlled benchmark on one small dataset and makes no
state-of-the-art claim. Three subset seeds bound how far any of it generalizes, and we treat the
paired tests as exploratory throughout. BaCC is reported here because it did not
work: it changes corruption mIoU by at most 0.24 points against the uniform
consistency baseline it was meant to beat.
