"""Extended deterministic image operations for portrait, repair, and optics tools."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image

from app.tools.image_ops_support import (
    auto_subject_mask_np,
    blend_rgb_result,
    color_range_mask,
    highlights_mask,
    lab_float_to_rgb,
    midtones_mask,
    prepare_blend_mask_np,
    rgb_to_lab_float,
    save_result_array,
    shadows_mask,
)


def _load_rgb_float(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    return image_np, image_np.astype(np.float32) / 255.0


def _soft_mask_or_full(
    image_size: tuple[int, int],
    *,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> np.ndarray:
    mask_np = prepare_blend_mask_np(mask_path, image_size, feather_radius=feather_radius)
    if mask_np is None:
        return np.ones((image_size[1], image_size[0]), dtype=np.float32)
    return np.clip(mask_np, 0.0, 1.0).astype(np.float32)


def _blend_and_save(
    original_rgb: np.ndarray,
    adjusted_rgb: np.ndarray,
    output_path: str,
    *,
    mask_np: np.ndarray | None = None,
) -> str:
    blended = blend_rgb_result(adjusted_rgb, original_rgb, mask_np)
    return save_result_array(np.clip(blended * 255.0, 0, 255), output_path)


def _auto_defect_mask(
    image_rgb: np.ndarray,
    *,
    sensitivity: float,
    small_spot_bias: float,
) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur_small = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    blur_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.5, sigmaY=5.5)
    residual = np.abs(blur_small - blur_large)
    threshold = 0.035 + (1.0 - float(np.clip(sensitivity, 0.0, 1.0))) * 0.06
    defect_mask = (residual > threshold).astype(np.uint8) * 255
    kernel_size = 3 if small_spot_bias >= 0.45 else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    defect_mask = cv2.dilate(defect_mask, kernel, iterations=1)
    return defect_mask


def apply_remove_heal(
    image_path: str,
    output_path: str,
    *,
    strength: float,
    radius_px: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    detail_protection: float = 0.35,
    method: str = "telea",
    auto_detect: bool = True,
    small_spot_bias: float = 0.55,
) -> str:
    """Heal or remove small distracting areas with inpainting."""

    image_rgb, image_float = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    soft_mask = prepare_blend_mask_np(mask_path, (width, height), feather_radius=feather_radius)
    binary_mask = None
    if soft_mask is not None:
        binary_mask = (soft_mask > 0.18).astype(np.uint8) * 255
    elif auto_detect:
        binary_mask = _auto_defect_mask(
            image_rgb,
            sensitivity=abs(strength),
            small_spot_bias=small_spot_bias,
        )

    if binary_mask is None or not np.any(binary_mask):
        return save_result_array(image_rgb, output_path)

    inpaint_flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA
    inpaint_radius = float(np.clip(radius_px + abs(strength) * 4.0, 1.0, 18.0))
    repaired_bgr = cv2.inpaint(
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        binary_mask,
        inpaint_radius,
        inpaint_flag,
    )
    repaired_rgb = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    if detail_protection > 0:
        detail_base = cv2.GaussianBlur(image_float, (0, 0), sigmaX=1.0, sigmaY=1.0)
        detail = image_float - detail_base
        repaired_rgb = np.clip(repaired_rgb + detail * detail_protection * 0.16, 0.0, 1.0)

    return _blend_and_save(image_float, repaired_rgb, output_path, mask_np=soft_mask)


def _edge_preserving_smooth(
    image_rgb: np.ndarray,
    *,
    smooth_strength: float,
    detail_protection: float,
) -> np.ndarray:
    filtered = cv2.bilateralFilter(
        image_rgb.astype(np.uint8),
        d=0,
        sigmaColor=12 + smooth_strength * 36,
        sigmaSpace=10 + smooth_strength * 24,
    ).astype(np.float32) / 255.0
    original = image_rgb.astype(np.float32) / 255.0
    detail = original - cv2.GaussianBlur(original, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return np.clip(filtered + detail * detail_protection * 0.45, 0.0, 1.0)


def apply_skin_smooth(
    image_path: str,
    output_path: str,
    *,
    strength: float,
    smooth_strength: float,
    detail_protection: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    saturation_protection: float = 0.2,
) -> str:
    """Apply restrained skin smoothing with detail protection."""

    image_rgb, image_float = _load_rgb_float(image_path)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    smoothed = _edge_preserving_smooth(
        image_rgb,
        smooth_strength=smooth_strength,
        detail_protection=detail_protection,
    )

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    sat_gate = np.clip(1.0 - saturation * saturation_protection, 0.72, 1.0)
    mixed = np.clip(image_float * (1.0 - smooth_strength * 0.58) + smoothed * (smooth_strength * 0.58), 0.0, 1.0)
    adjusted = np.clip(mixed * sat_gate[:, :, None] + image_float * (1.0 - sat_gate[:, :, None]), 0.0, 1.0)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=mask_np)


_POINT_COLOR_PRESETS: dict[str, tuple[float, float, float, float]] = {
    "red": (0.0, 26.0, 0.08, 0.08),
    "orange": (28.0, 28.0, 0.08, 0.08),
    "yellow": (55.0, 26.0, 0.08, 0.08),
    "green": (125.0, 34.0, 0.1, 0.08),
    "aqua": (180.0, 26.0, 0.1, 0.08),
    "blue": (220.0, 30.0, 0.1, 0.08),
    "purple": (275.0, 28.0, 0.1, 0.08),
    "magenta": (320.0, 28.0, 0.1, 0.08),
    "skin": (26.0, 24.0, 0.06, 0.12),
}


def apply_point_color_adjustment(
    image_path: str,
    output_path: str,
    *,
    target_color: str,
    target_hue: float | None,
    range_width: float,
    hue_shift: float,
    saturation_shift: float,
    luminance_shift: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    preserve_neutrals: float = 0.2,
) -> str:
    """Apply a narrower point-color style adjustment than HSL mixer."""

    image_rgb, image_float = _load_rgb_float(image_path)
    hsv = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    color_key = str(target_color or "").strip().lower()
    if color_key == "white":
        target_mask = np.clip((0.22 - saturation) / 0.22, 0.0, 1.0) * np.clip((value - 0.58) / 0.42, 0.0, 1.0)
    else:
        preset = _POINT_COLOR_PRESETS.get(color_key, _POINT_COLOR_PRESETS["orange"])
        center = float(target_hue if target_hue is not None else preset[0])
        half_width = float(range_width if range_width > 0 else preset[1])
        sat_floor = max(preset[2] - preserve_neutrals * 0.05, 0.02)
        value_floor = preset[3]
        target_mask = color_range_mask(
            hue,
            saturation,
            value,
            center=center,
            half_width=half_width,
            saturation_floor=sat_floor,
            value_floor=value_floor,
            blur_sigma=1.4,
        )

    local_mask = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    if local_mask is not None:
        target_mask = np.clip(target_mask * local_mask, 0.0, 1.0)

    adjusted_hsv = hsv.copy()
    adjusted_hsv[:, :, 0] = (adjusted_hsv[:, :, 0] + (hue_shift / 2.0) * target_mask) % 180.0
    adjusted_hsv[:, :, 1] = np.clip(adjusted_hsv[:, :, 1] * (1.0 + saturation_shift * target_mask), 0.0, 255.0)
    adjusted_hsv[:, :, 2] = np.clip(adjusted_hsv[:, :, 2] * (1.0 + luminance_shift * target_mask), 0.0, 255.0)
    adjusted_rgb = cv2.cvtColor(adjusted_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return _blend_and_save(image_float, adjusted_rgb, output_path, mask_np=None)


def apply_regional_enhancement(
    image_path: str,
    output_path: str,
    *,
    exposure_boost: float = 0.0,
    saturation_boost: float = 0.0,
    warmth_shift: float = 0.0,
    clarity_boost: float = 0.0,
    smooth_amount: float = 0.0,
    sharpen_amount: float = 0.0,
    highlight_protection: float = 0.2,
    shadow_lift: float = 0.0,
    yellow_suppression: float = 0.0,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a mix of portrait-region enhancement adjustments."""

    image_rgb, image_float = _load_rgb_float(image_path)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)

    hsv = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation_boost), 0.0, 255.0)

    if yellow_suppression > 0:
        hue = hsv[:, :, 0] * 2.0
        saturation = hsv[:, :, 1] / 255.0
        value = hsv[:, :, 2] / 255.0
        yellow_mask = color_range_mask(
            hue,
            saturation,
            value,
            center=48.0,
            half_width=28.0,
            saturation_floor=0.06,
            value_floor=0.1,
            blur_sigma=1.4,
        )
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - yellow_mask * yellow_suppression), 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 + yellow_mask * yellow_suppression * 0.22), 0.0, 255.0)

    adjusted_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    if exposure_boost != 0.0:
        adjusted_rgb = np.clip(adjusted_rgb * (2 ** exposure_boost), 0.0, 1.0)

    if warmth_shift != 0.0 or shadow_lift != 0.0:
        lab = rgb_to_lab_float(adjusted_rgb)
        luminance = lab[:, :, 0] / 100.0
        if warmth_shift != 0.0:
            lab[:, :, 2] = np.clip(lab[:, :, 2] + warmth_shift * 8.0, -127.0, 127.0)
        if shadow_lift != 0.0:
            shadow_gate = shadows_mask(luminance, 0.48)
            lab[:, :, 0] = np.clip(lab[:, :, 0] + shadow_gate * shadow_lift * 10.0, 0.0, 100.0)
        adjusted_rgb = lab_float_to_rgb(lab)

    if smooth_amount > 0.0:
        smoothed = _edge_preserving_smooth(
            (adjusted_rgb * 255.0).astype(np.uint8),
            smooth_strength=smooth_amount,
            detail_protection=0.76,
        )
        adjusted_rgb = np.clip(adjusted_rgb * (1.0 - smooth_amount * 0.52) + smoothed * (smooth_amount * 0.52), 0.0, 1.0)

    if clarity_boost > 0.0 or sharpen_amount > 0.0:
        luminance = cv2.cvtColor((adjusted_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] / 255.0
        local = cv2.GaussianBlur(luminance, (0, 0), sigmaX=1.0 + clarity_boost * 2.6, sigmaY=1.0 + clarity_boost * 2.6)
        detail = luminance - local
        highlight_gate = 1.0 - highlights_mask(luminance, 0.34 + highlight_protection * 0.2)
        amount = clarity_boost * 0.22 + sharpen_amount * 0.34
        new_l = np.clip(luminance + detail * amount * highlight_gate, 0.0, 1.0)
        lab = cv2.cvtColor((adjusted_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[:, :, 0] = new_l * 255.0
        adjusted_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    return _blend_and_save(image_float, adjusted_rgb, output_path, mask_np=mask_np)


def _radial_distortion_map(
    width: int,
    height: int,
    *,
    k1: float,
    k2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    r2 = nx * nx + ny * ny
    factor = 1.0 + k1 * r2 + k2 * r2 * r2
    map_x = nx * factor * max(cx, 1.0) + cx
    map_y = ny * factor * max(cy, 1.0) + cy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def apply_lens_correction(
    image_path: str,
    output_path: str,
    *,
    distortion_amount: float,
    edge_scale: float,
) -> str:
    """Apply a restrained barrel/pincushion correction."""

    image_rgb, _ = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    map_x, map_y = _radial_distortion_map(
        width,
        height,
        k1=-distortion_amount * 0.22,
        k2=distortion_amount * 0.04,
    )
    corrected = cv2.remap(
        image_rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    ).astype(np.float32) / 255.0
    if edge_scale > 1.0:
        corrected = cv2.resize(corrected, None, fx=edge_scale, fy=edge_scale, interpolation=cv2.INTER_LINEAR)
        corrected = cv2.resize(corrected, (width, height), interpolation=cv2.INTER_LINEAR)
    return save_result_array(corrected * 255.0, output_path)


def apply_remove_chromatic_aberration(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    radial_bias: float,
) -> str:
    """Reduce simple radial chromatic aberration by re-aligning red and blue channels."""

    image_rgb, image_float = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    nx = (xx - cx) / max(cx, 1.0)
    ny = (yy - cy) / max(cy, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    shift = radius * amount * (1.6 + radial_bias * 1.2)
    red_map_x = xx - nx * shift
    red_map_y = yy - ny * shift
    blue_map_x = xx + nx * shift
    blue_map_y = yy + ny * shift

    red = cv2.remap(image_float[:, :, 0], red_map_x, red_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    green = image_float[:, :, 1]
    blue = cv2.remap(image_float[:, :, 2], blue_map_x, blue_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    corrected = np.stack([red, green, blue], axis=2)
    return save_result_array(corrected * 255.0, output_path)


def apply_defringe(
    image_path: str,
    output_path: str,
    *,
    purple_amount: float,
    green_amount: float,
    edge_threshold: float,
) -> str:
    """Suppress purple and green fringing near strong edges."""

    image_rgb, image_float = _load_rgb_float(image_path)
    hsv = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2.0
    saturation = hsv[:, :, 1] / 255.0
    luminance = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] / 255.0
    edge = cv2.Laplacian(luminance, cv2.CV_32F, ksize=3)
    edge_gate = np.clip((np.abs(edge) - edge_threshold) / max(1.0 - edge_threshold, 1e-6), 0.0, 1.0)
    purple_gate = color_range_mask(hue, saturation, luminance, center=290.0, half_width=34.0, blur_sigma=0.8)
    green_gate = color_range_mask(hue, saturation, luminance, center=128.0, half_width=30.0, blur_sigma=0.8)
    suppress = np.clip(purple_gate * purple_amount + green_gate * green_amount, 0.0, 1.0) * edge_gate
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 - suppress * 0.85), 0.0, 255.0)
    corrected = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    return save_result_array(corrected * 255.0, output_path)


def apply_perspective_correction(
    image_path: str,
    output_path: str,
    *,
    vertical_amount: float,
    horizontal_amount: float,
) -> str:
    """Apply a simple keystone correction."""

    image_rgb, _ = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    top_shift = vertical_amount * width * 0.08
    side_shift = horizontal_amount * height * 0.08
    src = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    dst = np.float32(
        [
            [0 + top_shift, 0 + side_shift],
            [width - 1 - top_shift, 0 - side_shift],
            [width - 1 + top_shift, height - 1 + side_shift],
            [0 - top_shift, height - 1 - side_shift],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    corrected = cv2.warpPerspective(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return save_result_array(corrected, output_path)


def apply_auto_upright(
    image_path: str,
    output_path: str,
    *,
    strength: float,
    max_angle: float,
) -> str:
    """Estimate a dominant line angle and rotate the image back toward upright."""

    image_rgb, _ = _load_rgb_float(image_path)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=50, minLineLength=max(min(gray.shape) // 6, 20), maxLineGap=12)

    angles: list[float] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = line.tolist()
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if -45.0 <= angle <= 45.0:
                angles.append(angle)
            elif angle < -45.0:
                angles.append(angle + 90.0)
            else:
                angles.append(angle - 90.0)

    dominant_angle = float(np.median(angles)) if angles else 0.0
    correction_angle = float(np.clip(-dominant_angle * strength, -max_angle, max_angle))
    center = ((image_rgb.shape[1] - 1) / 2.0, (image_rgb.shape[0] - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)
    rotated = cv2.warpAffine(
        image_rgb,
        matrix,
        (image_rgb.shape[1], image_rgb.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    return save_result_array(rotated, output_path)


def apply_vignette(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    midpoint: float,
    roundness: float,
    feather: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a Lightroom-style vignette."""

    image_rgb, image_float = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    rx = max(width * (0.5 + roundness * 0.18), 1.0)
    ry = max(height * (0.5 - roundness * 0.12), 1.0)
    radius = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    midpoint = float(np.clip(midpoint, 0.15, 0.95))
    feather = float(np.clip(feather, 0.05, 1.0))
    gate = np.clip((radius - midpoint) / max(1.0 - midpoint, 1e-6), 0.0, 1.0)
    gate = np.power(gate, 1.0 / feather)
    factor = 1.0 - gate * amount * 0.6
    adjusted = np.clip(image_float * factor[:, :, None], 0.0, 1.0)
    mask_np = prepare_blend_mask_np(mask_path, (width, height), feather_radius=feather_radius)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=mask_np)


def apply_grain(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    size: float,
    roughness: float,
    color_amount: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Add restrained film-like grain."""

    image_rgb, image_float = _load_rgb_float(image_path)
    rng = np.random.default_rng(42)
    base_noise = rng.normal(0.0, 1.0, size=image_float.shape[:2]).astype(np.float32)
    sigma = max(0.3, 1.8 - size)
    luma_noise = cv2.GaussianBlur(base_noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    luma_noise = luma_noise[:, :, None]
    color_noise = rng.normal(0.0, 1.0, size=image_float.shape).astype(np.float32)
    color_noise = cv2.GaussianBlur(color_noise, (0, 0), sigmaX=sigma * 0.8, sigmaY=sigma * 0.8)
    adjusted = np.clip(
        image_float
        + luma_noise * amount * (0.035 + roughness * 0.03)
        + color_noise * amount * color_amount * 0.015,
        0.0,
        1.0,
    )
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=mask_np)


def apply_moire_reduction(
    image_path: str,
    output_path: str,
    *,
    amount: float,
) -> str:
    """Reduce visible moire by attenuating chroma and very high frequencies."""

    image_rgb, image_float = _load_rgb_float(image_path)
    lab = rgb_to_lab_float(image_float)
    chroma_a = cv2.GaussianBlur(lab[:, :, 1], (0, 0), sigmaX=amount * 1.8, sigmaY=amount * 1.8)
    chroma_b = cv2.GaussianBlur(lab[:, :, 2], (0, 0), sigmaX=amount * 1.8, sigmaY=amount * 1.8)
    luminance = lab[:, :, 0]
    local = cv2.GaussianBlur(luminance, (0, 0), sigmaX=1.0 + amount * 0.8, sigmaY=1.0 + amount * 0.8)
    detail = luminance - local
    lab[:, :, 0] = np.clip(luminance - detail * amount * 0.22, 0.0, 100.0)
    lab[:, :, 1] = chroma_a
    lab[:, :, 2] = chroma_b
    adjusted = lab_float_to_rgb(lab)
    return save_result_array(adjusted * 255.0, output_path)


def _hue_to_rgb(hue_deg: float, saturation: float) -> np.ndarray:
    hsv = np.array([[[hue_deg / 2.0, np.clip(saturation, 0.0, 1.0) * 255.0, 255.0]]], dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)[0, 0] / 255.0


def apply_color_grading(
    image_path: str,
    output_path: str,
    *,
    shadow_hue: float,
    shadow_saturation: float,
    midtone_hue: float,
    midtone_saturation: float,
    highlight_hue: float,
    highlight_saturation: float,
    balance: float,
    blending: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply split-toning / color-grading style tinting."""

    image_rgb, image_float = _load_rgb_float(image_path)
    lab = rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0 + balance * 0.08, 0.0, 1.0)
    shadow_gate = shadows_mask(luminance, 0.45)
    highlight_gate = highlights_mask(luminance, 0.42)
    midtone_gate = midtones_mask(luminance, 0.68)

    shadow_color = _hue_to_rgb(shadow_hue, shadow_saturation)
    midtone_color = _hue_to_rgb(midtone_hue, midtone_saturation)
    highlight_color = _hue_to_rgb(highlight_hue, highlight_saturation)

    tint = (
        shadow_color[None, None, :] * shadow_gate[:, :, None] * shadow_saturation
        + midtone_color[None, None, :] * midtone_gate[:, :, None] * midtone_saturation
        + highlight_color[None, None, :] * highlight_gate[:, :, None] * highlight_saturation
    )
    adjusted = np.clip(image_float * (1.0 - blending * 0.18) + tint * blending * 0.18 + image_float, 0.0, 1.0)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=mask_np)


def apply_lut_preset(
    image_path: str,
    output_path: str,
    *,
    preset: str,
    strength: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a lightweight preset-like look."""

    image_rgb, image_float = _load_rgb_float(image_path)
    preset_name = str(preset or "clean_portrait").strip().lower()
    adjusted = image_float.copy()
    if preset_name == "warm_film":
        adjusted = np.clip(adjusted * np.array([1.04, 1.0, 0.95], dtype=np.float32), 0.0, 1.0)
        adjusted = np.clip(adjusted + 0.02, 0.0, 1.0)
    elif preset_name == "cool_fade":
        adjusted = np.clip(adjusted * np.array([0.98, 1.0, 1.05], dtype=np.float32), 0.0, 1.0)
        adjusted = np.clip(0.06 + adjusted * 0.94, 0.0, 1.0)
    else:
        adjusted = np.clip(adjusted * np.array([1.01, 1.0, 0.99], dtype=np.float32), 0.0, 1.0)
        adjusted = np.clip(adjusted + 0.015, 0.0, 1.0)

    mixed = np.clip(image_float * (1.0 - strength) + adjusted * strength, 0.0, 1.0)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    return _blend_and_save(image_float, mixed, output_path, mask_np=mask_np)


def apply_convert_black_white(
    image_path: str,
    output_path: str,
    *,
    contrast: float,
    filter_color: str,
    tone_amount: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Convert to black and white with a restrained color filter and split tone."""

    image_rgb, image_float = _load_rgb_float(image_path)
    filter_key = str(filter_color or "neutral").strip().lower()
    weights = {
        "red": np.array([0.45, 0.35, 0.2], dtype=np.float32),
        "yellow": np.array([0.4, 0.42, 0.18], dtype=np.float32),
        "green": np.array([0.3, 0.5, 0.2], dtype=np.float32),
        "blue": np.array([0.22, 0.38, 0.4], dtype=np.float32),
        "neutral": np.array([0.299, 0.587, 0.114], dtype=np.float32),
    }.get(filter_key, np.array([0.299, 0.587, 0.114], dtype=np.float32))
    luminance = np.tensordot(image_float, weights, axes=([2], [0]))
    luminance = np.clip((luminance - 0.5) * (1.0 + contrast * 0.45) + 0.5, 0.0, 1.0)
    bw = np.repeat(luminance[:, :, None], 3, axis=2)
    if tone_amount > 0:
        shadow_gate = shadows_mask(luminance, 0.5)
        highlight_gate = highlights_mask(luminance, 0.38)
        bw[:, :, 0] = np.clip(bw[:, :, 0] + shadow_gate * tone_amount * 0.03, 0.0, 1.0)
        bw[:, :, 2] = np.clip(bw[:, :, 2] + highlight_gate * tone_amount * 0.05, 0.0, 1.0)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    return _blend_and_save(image_float, bw, output_path, mask_np=mask_np)


def apply_camera_calibration(
    image_path: str,
    output_path: str,
    *,
    red_bias: float,
    green_bias: float,
    blue_bias: float,
    saturation_bias: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply restrained per-primary calibration shifts."""

    image_rgb, image_float = _load_rgb_float(image_path)
    adjusted = image_float.copy()
    adjusted[:, :, 0] = np.clip(adjusted[:, :, 0] * (1.0 + red_bias * 0.12), 0.0, 1.0)
    adjusted[:, :, 1] = np.clip(adjusted[:, :, 1] * (1.0 + green_bias * 0.12), 0.0, 1.0)
    adjusted[:, :, 2] = np.clip(adjusted[:, :, 2] * (1.0 + blue_bias * 0.12), 0.0, 1.0)
    hsv = cv2.cvtColor((adjusted * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation_bias * 0.28), 0.0, 255.0)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=mask_np)


def apply_background_blur(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    highlight_boost: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Blur the background automatically, or blur the provided masked region."""

    image_rgb, image_float = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    local_mask = prepare_blend_mask_np(mask_path, (width, height), feather_radius=feather_radius)
    if local_mask is None:
        subject_mask = auto_subject_mask_np(image_rgb)
        blur_mask = 1.0 - subject_mask
    else:
        blur_mask = np.clip(local_mask, 0.0, 1.0)

    blur_sigma = 1.5 + amount * 7.5
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    if highlight_boost > 0:
        luminance = cv2.cvtColor((blurred * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] / 255.0
        highlight_gate = highlights_mask(luminance, 0.28)
        blurred = np.clip(blurred + highlight_gate[:, :, None] * highlight_boost * 0.05, 0.0, 1.0)
    return _blend_and_save(image_float, blurred, output_path, mask_np=blur_mask)


def apply_lens_blur(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    highlight_bloom: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Simulate lens blur with stronger blur falloff and bloom."""

    image_rgb, image_float = _load_rgb_float(image_path)
    height, width = image_rgb.shape[:2]
    local_mask = prepare_blend_mask_np(mask_path, (width, height), feather_radius=feather_radius)
    if local_mask is None:
        subject_mask = auto_subject_mask_np(image_rgb)
        blur_mask = 1.0 - subject_mask
    else:
        blur_mask = np.clip(local_mask, 0.0, 1.0)

    base_sigma = 2.4 + amount * 10.0
    heavily_blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=base_sigma, sigmaY=base_sigma)
    medium_blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=max(base_sigma * 0.45, 1.0), sigmaY=max(base_sigma * 0.45, 1.0))
    mixed = np.clip(heavily_blurred * 0.65 + medium_blurred * 0.35, 0.0, 1.0)
    if highlight_bloom > 0:
        luminance = cv2.cvtColor((mixed * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] / 255.0
        mixed = np.clip(mixed + highlights_mask(luminance, 0.24)[:, :, None] * highlight_bloom * 0.07, 0.0, 1.0)
    return _blend_and_save(image_float, mixed, output_path, mask_np=blur_mask)


def apply_glow_highlight(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    threshold: float,
    warmth: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply bloom/glow to highlight regions."""

    image_rgb, image_float = _load_rgb_float(image_path)
    luminance = cv2.cvtColor((image_float * 255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] / 255.0
    bright_mask = np.clip((luminance - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    glow_source = image_float * bright_mask[:, :, None]
    glow = cv2.GaussianBlur(glow_source, (0, 0), sigmaX=4.0 + amount * 12.0, sigmaY=4.0 + amount * 12.0)
    glow[:, :, 0] = np.clip(glow[:, :, 0] * (1.0 + warmth * 0.12), 0.0, 1.0)
    glow[:, :, 2] = np.clip(glow[:, :, 2] * (1.0 - warmth * 0.08), 0.0, 1.0)
    adjusted = np.clip(1.0 - (1.0 - image_float) * (1.0 - glow * amount * 0.72), 0.0, 1.0)
    mask_np = prepare_blend_mask_np(mask_path, (image_rgb.shape[1], image_rgb.shape[0]), feather_radius=feather_radius)
    if mask_np is not None:
        bright_mask = np.clip(bright_mask * mask_np, 0.0, 1.0)
    return _blend_and_save(image_float, adjusted, output_path, mask_np=bright_mask)
