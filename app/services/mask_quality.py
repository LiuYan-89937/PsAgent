"""Deterministic quality scoring for generated segmentation masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.graph.state import MaskQuality

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is available in normal runtime.
    cv2 = None


def _safe_bbox(mask: np.ndarray) -> dict[str, int]:
    """Return a foreground bounding box for a boolean mask."""

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return {"x": 0, "y": 0, "width": 0, "height": 0}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return {"x": x0, "y": y0, "width": x1 - x0 + 1, "height": y1 - y0 + 1}


def _connected_components(mask_uint8: np.ndarray) -> int:
    """Count connected foreground components using OpenCV when available."""

    if cv2 is None:
        return 1 if bool(mask_uint8.any()) else 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if count <= 1:
        return 0
    image_area = mask_uint8.shape[0] * mask_uint8.shape[1]
    min_area = max(int(image_area * 0.0008), 12)
    return int(sum(1 for index in range(1, count) if int(stats[index, cv2.CC_STAT_AREA]) >= min_area))


def evaluate_mask_quality(mask_path: str) -> MaskQuality:
    """Score whether a mask is usable enough for local tool execution."""

    path = Path(mask_path)
    if not path.exists():
        return MaskQuality(
            score=0.0,
            flags=["mask_file_missing"],
            rejected=True,
        )

    mask_image = Image.open(path).convert("L")
    mask = np.asarray(mask_image, dtype=np.uint8) > 127
    height, width = mask.shape
    image_area = max(width * height, 1)
    foreground = int(mask.sum())
    area_ratio = float(foreground / image_area)
    bbox = _safe_bbox(mask)
    bbox_area_ratio = float((bbox["width"] * bbox["height"]) / image_area) if foreground else 0.0
    mask_uint8 = mask.astype(np.uint8)
    connected_components = _connected_components(mask_uint8)

    if cv2 is not None and foreground:
        eroded = cv2.erode(mask_uint8, np.ones((3, 3), dtype=np.uint8), iterations=1)
        boundary = mask_uint8 - eroded
        edge_density = float(boundary.sum() / max(foreground, 1))
    else:
        edge_density = 0.0

    flags: list[str] = []
    if area_ratio <= 0.0015:
        flags.append("empty_or_too_small")
    if area_ratio >= 0.94:
        flags.append("nearly_full_image")
    if bbox_area_ratio >= 0.96 and area_ratio >= 0.72:
        flags.append("bbox_too_large")
    if connected_components >= 8 and area_ratio < 0.25:
        flags.append("fragmented")
    if edge_density > 0.34 and area_ratio < 0.18:
        flags.append("noisy_edges")

    score = 1.0
    if "empty_or_too_small" in flags:
        score -= 0.75
    if "nearly_full_image" in flags or "bbox_too_large" in flags:
        score -= 0.55
    if "fragmented" in flags:
        score -= 0.22
    if "noisy_edges" in flags:
        score -= 0.15
    if area_ratio < 0.01:
        score -= 0.15
    if area_ratio > 0.86:
        score -= 0.12
    if connected_components <= 2 and 0.01 <= area_ratio <= 0.72:
        score += 0.08

    rejected = any(flag in flags for flag in ("empty_or_too_small", "nearly_full_image", "bbox_too_large"))
    if score < 0.35:
        rejected = True

    return MaskQuality(
        score=max(0.0, min(1.0, round(score, 4))),
        area_ratio=round(area_ratio, 6),
        bbox=bbox,
        connected_components=connected_components,
        edge_density=round(edge_density, 6),
        flags=flags,
        rejected=rejected,
    )


__all__ = ["evaluate_mask_quality"]
