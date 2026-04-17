"""Shared deterministic image-op helpers used across tool families."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def prepare_blend_mask(
    mask_path: str | None,
    image_size: tuple[int, int],
    *,
    feather_radius: float = 0.0,
) -> Image.Image | None:
    """Load and optionally feather a grayscale mask for local adjustments."""

    if not mask_path:
        return None

    mask = Image.open(mask_path).convert("L")
    if mask.size != image_size:
        mask = mask.resize(image_size)

    feather_radius = max(0.0, float(feather_radius))
    if feather_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return mask


def save_result_image(result: Image.Image, output_path: str) -> str:
    """Persist an edited PIL image and return the saved file path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return str(output)


def save_result_array(result: np.ndarray, output_path: str) -> str:
    """Persist an RGB uint8 array and return the saved file path."""

    image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), mode="RGB")
    return save_result_image(image, output_path)


def prepare_blend_mask_np(
    mask_path: str | None,
    image_size: tuple[int, int],
    *,
    feather_radius: float = 0.0,
) -> np.ndarray | None:
    """Return a float32 blend mask in [0, 1] or None."""

    mask = prepare_blend_mask(
        mask_path,
        image_size,
        feather_radius=feather_radius,
    )
    if mask is None:
        return None
    return np.asarray(mask, dtype=np.float32) / 255.0


def blend_rgb_result(
    adjusted_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask_np: np.ndarray | None,
) -> np.ndarray:
    """Blend an adjusted RGB image back onto the original with an optional mask."""

    if mask_np is None:
        return adjusted_rgb
    if adjusted_rgb.ndim == 2:
        adjusted_rgb = np.repeat(adjusted_rgb[:, :, None], 3, axis=2)
    if original_rgb.ndim == 2:
        original_rgb = np.repeat(original_rgb[:, :, None], 3, axis=2)
    return adjusted_rgb * mask_np[:, :, None] + original_rgb * (1.0 - mask_np[:, :, None])


def rgb_to_lab_float(image_float: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0, 1] into float32 Lab."""

    return cv2.cvtColor(image_float.astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float32)


def lab_float_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert float32 Lab back to RGB [0, 1]."""

    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)


def highlights_mask(luminance: np.ndarray, rolloff: float) -> np.ndarray:
    """Build a soft highlight mask from normalized luminance."""

    rolloff = float(np.clip(rolloff, 0.08, 0.95))
    threshold = 1.0 - rolloff
    return np.power(
        np.clip((luminance - threshold) / max(rolloff, 1e-6), 0.0, 1.0),
        1.45,
    ).astype(np.float32)


def shadows_mask(luminance: np.ndarray, rolloff: float) -> np.ndarray:
    """Build a soft shadow mask from normalized luminance."""

    rolloff = float(np.clip(rolloff, 0.08, 0.95))
    return np.power(
        np.clip((rolloff - luminance) / max(rolloff, 1e-6), 0.0, 1.0),
        1.45,
    ).astype(np.float32)


def midtones_mask(luminance: np.ndarray, width: float = 0.7) -> np.ndarray:
    """Build a midtone-favoring mask."""

    width = float(np.clip(width, 0.2, 1.0))
    distance = np.abs(luminance - 0.5) / max(width / 2.0, 1e-6)
    return np.power(np.clip(1.0 - distance, 0.0, 1.0), 1.2).astype(np.float32)


def color_range_mask(
    hue: np.ndarray,
    saturation: np.ndarray,
    value: np.ndarray,
    *,
    center: float,
    half_width: float,
    saturation_floor: float = 0.08,
    value_floor: float = 0.06,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Build a soft hue-range mask in HSV space."""

    half_width = float(np.clip(half_width, 8.0, 80.0))
    distance = np.abs(hue - center)
    distance = np.minimum(distance, 360.0 - distance)
    hue_gate = np.power(np.clip(1.0 - distance / half_width, 0.0, 1.0), 1.4)

    sat_gate = np.power(
        np.clip((saturation - saturation_floor) / max(1.0 - saturation_floor, 1e-6), 0.0, 1.0),
        1.1,
    )
    value_gate = np.power(
        np.clip((value - value_floor) / max(1.0 - value_floor, 1e-6), 0.0, 1.0),
        1.05,
    )

    mask = (hue_gate * sat_gate * value_gate).astype(np.float32)
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(
            mask,
            ksize=(0, 0),
            sigmaX=blur_sigma,
            sigmaY=blur_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def soft_binary_mask(binary_mask: np.ndarray, *, blur_sigma: float = 3.0) -> np.ndarray:
    """Convert a hard binary mask into a soft blend mask."""

    mask = binary_mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    if blur_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def auto_subject_mask_np(image_rgb: np.ndarray) -> np.ndarray:
    """Build a soft foreground mask with GrabCut and a center-weighted fallback."""

    height, width = image_rgb.shape[:2]
    if min(height, width) < 24:
        mask = np.zeros((height, width), dtype=np.float32)
        y0 = int(height * 0.15)
        y1 = int(height * 0.85)
        x0 = int(width * 0.15)
        x1 = int(width * 0.85)
        mask[y0:y1, x0:x1] = 1.0
        return soft_binary_mask(mask, blur_sigma=max(min(height, width) / 20.0, 1.0))

    bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    rect = (
        int(width * 0.08),
        int(height * 0.08),
        max(int(width * 0.84), 1),
        max(int(height * 0.84), 1),
    )
    grabcut_mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(bgr, grabcut_mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
        foreground = np.where(
            (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
            1.0,
            0.0,
        ).astype(np.float32)
        if foreground.mean() < 0.03:
            raise ValueError("Foreground mask coverage too small.")
        return soft_binary_mask(foreground, blur_sigma=max(min(height, width) / 90.0, 1.5))
    except Exception:
        yy, xx = np.mgrid[0:height, 0:width]
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        rx = max(width * 0.36, 1.0)
        ry = max(height * 0.42, 1.0)
        ellipse = 1.0 - ((xx - cx) ** 2 / (rx * rx) + (yy - cy) ** 2 / (ry * ry))
        fallback = np.clip(ellipse, 0.0, 1.0).astype(np.float32)
        return soft_binary_mask(fallback, blur_sigma=max(min(height, width) / 70.0, 1.2))
