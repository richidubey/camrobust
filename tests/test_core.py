import random

import torch

from camrobust.corruptions import TEST_CORRUPTIONS, apply_corruption, random_camera_corruption
from camrobust.data import IGNORE_INDEX, nested_random_subset
from camrobust.losses import consistency_loss, semantic_boundary
from camrobust.metrics import SegmentationMetrics


def test_nested_subset_is_deterministic_and_nested():
    small = nested_random_subset(100, 16, 7)
    large = nested_random_subset(100, 32, 7)
    assert small == large[:16]
    assert len(set(large)) == 32


def test_semantic_boundary_and_ignore():
    y = torch.zeros(1, 7, 7, dtype=torch.long)
    y[:, :, 4:] = 1
    y[:, 0, :] = IGNORE_INDEX
    boundary = semantic_boundary(y, radius=0)
    assert boundary[:, :, 3:5].any()
    assert not boundary[:, 0, :].any()


def test_corruptions_preserve_shape_range():
    x = torch.rand(2, 3, 32, 48)
    generator = torch.Generator().manual_seed(3)
    for name in TEST_CORRUPTIONS:
        out = apply_corruption(x, name, 2, generator=generator)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        assert float(out.min()) >= 0 and float(out.max()) <= 1
    out, name, severity = random_camera_corruption(x, random.Random(0))
    assert out.shape == x.shape and name and severity > 0


def test_boundary_weighted_consistency_has_gradients():
    a = torch.randn(2, 3, 8, 8, requires_grad=True)
    b = torch.randn(2, 3, 8, 8, requires_grad=True)
    y = torch.zeros(2, 8, 8, dtype=torch.long)
    y[:, :, 4:] = 1
    loss = consistency_loss(a, b, y, boundary_alpha=3, boundary_radius=1)
    loss.backward()
    assert loss > 0 and a.grad is not None and b.grad is not None


def test_perfect_metrics():
    y = torch.tensor([[[0, 0], [1, 1]]])
    logits = torch.full((1, 2, 2, 2), -8.0)
    logits.scatter_(1, y[:, None], 8.0)
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(logits, y)
    result = metrics.compute()
    assert result["miou"] == 1.0
    assert result["pixel_accuracy"] == 1.0

