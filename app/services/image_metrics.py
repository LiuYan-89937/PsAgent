"""Small deterministic image metrics shared by reviews and guards."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def compute_image_metrics(image_path: str, mask_path: str | None = None) -> dict[str, float]:
    """Compute lightweight luminance and saturation metrics for an image."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])
    rgb01 = image_np / 255.0
    rgb_max = rgb01.max(axis=2)
    rgb_min = rgb01.min(axis=2)
    chroma = rgb_max - rgb_min
    saturation = np.divide(chroma, np.maximum(rgb_max, 1e-6), out=np.zeros_like(chroma), where=rgb_max > 1e-6)

    if mask_path:
        mask = Image.open(mask_path).convert("L").resize(image.size, Image.Resampling.BILINEAR)
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        active = mask_np > 0.05
        if bool(active.any()):
            gray_values = gray[active]
            sat_values = saturation[active]
        else:
            gray_values = gray.reshape(-1)
            sat_values = saturation.reshape(-1)
    else:
        gray_values = gray.reshape(-1)
        sat_values = saturation.reshape(-1)

    return {
        "brightness_mean": float(gray_values.mean()),
        "brightness_std": float(gray_values.std()),
        "shadow_ratio": float((gray_values < 28).mean()),
        "highlight_ratio": float((gray_values > 235).mean()),
        "saturation_mean": float(sat_values.mean()),
        "saturation_std": float(sat_values.std()),
    }


def summarize_execution_counts(trace: list[dict[str, Any]]) -> dict[str, int]:
    """Return standard execution counters from JSON trace payloads."""

    return {
        "num_operations": len(trace),
        "success_count": sum(1 for item in trace if item.get("ok")),
        "failure_count": sum(1 for item in trace if item.get("ok") is False),
        "fallback_count": sum(1 for item in trace if item.get("fallback_used")),
    }
