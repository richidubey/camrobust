from __future__ import annotations

import torch
from torch.nn import functional as F

from .data import IGNORE_INDEX, NUM_CLASSES


def semantic_boundary(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Return a dilated semantic transition band for BHW integer masks."""
    valid = mask != IGNORE_INDEX
    boundary = torch.zeros_like(mask, dtype=torch.bool)
    horizontal = (mask[:, :, 1:] != mask[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    vertical = (mask[:, 1:, :] != mask[:, :-1, :]) & valid[:, 1:, :] & valid[:, :-1, :]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    if radius > 0:
        kernel = 2 * radius + 1
        boundary = F.max_pool2d(boundary[:, None].float(), kernel, stride=1, padding=radius)[:, 0] > 0
    return boundary & valid


def jensen_shannon(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a.float(), dim=1)
    log_b = F.log_softmax(logits_b.float(), dim=1)
    a, b = log_a.exp(), log_b.exp()
    mean = 0.5 * (a + b)
    log_mean = mean.clamp_min(1e-8).log()
    return 0.5 * ((a * (log_a - log_mean)).sum(1) + (b * (log_b - log_mean)).sum(1))


def supervised_loss(logits_: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits_.float(), target, weight=weights.float(), ignore_index=IGNORE_INDEX)


def consistency_loss(
    clean_logits: torch.Tensor,
    corrupt_logits: torch.Tensor,
    target: torch.Tensor,
    boundary_alpha: float,
    boundary_radius: int,
) -> torch.Tensor:
    js = jensen_shannon(clean_logits, corrupt_logits)
    valid = (target >= 0) & (target < NUM_CLASSES)
    weights = torch.ones_like(js)
    if boundary_alpha > 0:
        weights = weights + boundary_alpha * semantic_boundary(target, boundary_radius).float()
    weights = weights * valid
    weights = weights / weights[valid].mean().clamp_min(1e-6)
    return (js * weights).sum() / valid.sum().clamp_min(1)

