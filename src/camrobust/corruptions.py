from __future__ import annotations

import io
import math
import random

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

TRAIN_CORRUPTIONS = ("exposure", "white_balance", "gaussian_noise", "defocus_blur")
TEST_CORRUPTIONS = (
    "brightness",
    "darkness",
    "white_balance",
    "gaussian_noise",
    "shot_noise",
    "defocus_blur",
    "motion_blur",
    "jpeg",
)
UNSEEN_CORRUPTIONS = ("shot_noise", "motion_blur", "jpeg")


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def _gaussian_kernel(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - kernel_size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel2d = torch.outer(kernel, kernel).to(dtype=dtype)
    return kernel2d.expand(3, 1, kernel_size, kernel_size)


def _blur(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    k = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    return F.conv2d(F.pad(x, (kernel_size // 2,) * 4, mode="reflect"), k, groups=3)


def _motion_blur(x: torch.Tensor, kernel_size: int, vertical: bool = False) -> torch.Tensor:
    kernel = torch.zeros((kernel_size, kernel_size), device=x.device, dtype=x.dtype)
    if vertical:
        kernel[:, kernel_size // 2] = 1 / kernel_size
    else:
        kernel[kernel_size // 2, :] = 1 / kernel_size
    kernel = kernel.expand(3, 1, kernel_size, kernel_size)
    return F.conv2d(F.pad(x, (kernel_size // 2,) * 4, mode="reflect"), kernel, groups=3)


def _jpeg_one_cpu(x: torch.Tensor, quality: int) -> torch.Tensor:
    array = (x.clamp(0, 1) * 255).byte()
    image = Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return pil_to_tensor(decoded).float() / 255.0


def apply_corruption(
    x: torch.Tensor,
    name: str,
    severity: int | float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a label-preserving camera corruption to BCHW RGB in [0, 1]."""
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("expected BCHW RGB tensor")
    s = float(severity)
    if name == "brightness":
        out = x * (1.0 + 0.20 * s)
    elif name == "darkness":
        out = (x ** (1.0 + 0.35 * s)) * (1.0 - 0.12 * s)
    elif name == "exposure":
        gamma = max(0.45, 1.0 + 0.35 * (s - 1.5))
        gain = 1.0 + 0.12 * (s - 1.5)
        out = (x.clamp_min(1e-5) ** gamma) * gain
    elif name == "white_balance":
        warm = 0.08 * s
        gains = x.new_tensor((1.0 + warm, 1.0, max(0.55, 1.0 - warm))).view(1, 3, 1, 1)
        out = x * gains
    elif name == "gaussian_noise":
        sigma = 0.025 * s
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        out = x + sigma * noise
    elif name == "shot_noise":
        scale = max(8.0, 90.0 / (s**1.35))
        # torch.poisson currently has no generator argument on all supported builds.
        out = torch.poisson((x * scale).float(), generator=generator).to(x.dtype) / scale
    elif name == "defocus_blur":
        kernel_size = _odd(1 + 2 * int(math.ceil(s)))
        out = _blur(x, kernel_size, 0.55 + 0.55 * s)
    elif name == "motion_blur":
        kernel_size = _odd(3 + 2 * int(s))
        out = _motion_blur(x, kernel_size, vertical=int(s) % 2 == 0)
    elif name == "jpeg":
        quality = int(max(8, 88 - 22 * s))
        device, dtype = x.device, x.dtype
        # Transfer once per batch. Per-image GPU-to-CPU transfers serialize the
        # CUDA stream and make evaluation roughly an order of magnitude slower.
        cpu = x.detach().float().cpu()
        out = torch.stack([_jpeg_one_cpu(image, quality) for image in cpu]).to(device=device, dtype=dtype)
    elif name == "color_jitter":
        contrast = 1.0 + 0.18 * (s - 1.5)
        saturation = 1.0 - 0.12 * (s - 1.5)
        gray = x.mean(dim=1, keepdim=True)
        out = (x - 0.5) * contrast + 0.5
        out = gray + saturation * (out - gray)
    elif name == "none":
        out = x
    else:
        raise KeyError(f"unknown corruption: {name}")
    return out.clamp(0, 1)


def random_camera_corruption(x: torch.Tensor, rng: random.Random) -> tuple[torch.Tensor, str, float]:
    name = rng.choice(TRAIN_CORRUPTIONS)
    severity = rng.uniform(0.75, 2.75)
    if name == "exposure" and rng.random() < 0.5:
        severity = 3.0 - severity
    return apply_corruption(x, name, severity), name, severity


def random_color_jitter(x: torch.Tensor, rng: random.Random) -> torch.Tensor:
    severity = rng.uniform(0.5, 2.5)
    if rng.random() < 0.5:
        severity = 3.0 - severity
    return apply_corruption(x, "color_jitter", severity)
