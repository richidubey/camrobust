from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .data import IGNORE_INDEX, NUM_CLASSES
from .losses import semantic_boundary


@dataclass
class SegmentationMetrics:
    num_classes: int = NUM_CLASSES
    ece_bins: int = 15
    boundary_tolerance: int = 2
    confusion: torch.Tensor = field(init=False)
    conf_count: torch.Tensor = field(init=False)
    conf_sum: torch.Tensor = field(init=False)
    acc_sum: torch.Tensor = field(init=False)
    boundary_pred: float = 0.0
    boundary_true: float = 0.0
    boundary_pred_match: float = 0.0
    boundary_true_match: float = 0.0

    def __post_init__(self) -> None:
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        self.conf_count = torch.zeros(self.ece_bins, dtype=torch.float64)
        self.conf_sum = torch.zeros(self.ece_bins, dtype=torch.float64)
        self.acc_sum = torch.zeros(self.ece_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probabilities = logits.float().softmax(1)
        confidence, prediction = probabilities.max(1)
        valid = target != IGNORE_INDEX
        encoded = target[valid] * self.num_classes + prediction[valid]
        self.confusion += torch.bincount(encoded.cpu(), minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )
        conf = confidence[valid].cpu()
        correct = (prediction[valid] == target[valid]).double().cpu()
        bins = torch.clamp((conf * self.ece_bins).long(), max=self.ece_bins - 1)
        self.conf_count += torch.bincount(bins, minlength=self.ece_bins)
        self.conf_sum.scatter_add_(0, bins, conf.double())
        self.acc_sum.scatter_add_(0, bins, correct)

        pred_boundary = semantic_boundary(prediction, radius=0)[:, None].float()
        true_boundary = semantic_boundary(target, radius=0)[:, None].float()
        # Predictions inside target void regions are outside the evaluation
        # domain and must not count as boundary false positives.
        pred_boundary = pred_boundary * valid[:, None]
        k = 2 * self.boundary_tolerance + 1
        pred_neighborhood = F.max_pool2d(pred_boundary, k, stride=1, padding=self.boundary_tolerance)
        true_neighborhood = F.max_pool2d(true_boundary, k, stride=1, padding=self.boundary_tolerance)
        self.boundary_pred += float(pred_boundary.sum())
        self.boundary_true += float(true_boundary.sum())
        self.boundary_pred_match += float((pred_boundary * true_neighborhood).sum())
        self.boundary_true_match += float((true_boundary * pred_neighborhood).sum())

    def compute(self) -> dict[str, object]:
        cm = self.confusion.double()
        intersection = cm.diag()
        union = cm.sum(0) + cm.sum(1) - intersection
        iou = intersection / union.clamp_min(1)
        pixel_accuracy = intersection.sum() / cm.sum().clamp_min(1)
        nonempty = union > 0
        precision_b = self.boundary_pred_match / max(self.boundary_pred, 1.0)
        recall_b = self.boundary_true_match / max(self.boundary_true, 1.0)
        bf1 = 2 * precision_b * recall_b / max(precision_b + recall_b, 1e-12)
        populated = self.conf_count > 0
        avg_conf = self.conf_sum[populated] / self.conf_count[populated]
        avg_acc = self.acc_sum[populated] / self.conf_count[populated]
        ece = ((avg_conf - avg_acc).abs() * self.conf_count[populated]).sum() / self.conf_count.sum().clamp_min(1)
        return {
            "miou": float(iou[nonempty].mean()),
            "pixel_accuracy": float(pixel_accuracy),
            "boundary_f1": float(bf1),
            "ece": float(ece),
            "class_iou": [float(value) for value in iou],
            "confusion": self.confusion.tolist(),
        }
