"""Deterministic image operations used by tool packages."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter


def _prepare_blend_mask(
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
        # CV / 图像处理知识：
        # 羽化本质上是在 mask 上做低通平滑，这里用高斯模糊近似。
        # 模糊后的灰度 mask 会让边缘从“硬切换”变成“渐进过渡”，
        # 使局部曝光更接近 PS 里 adjustment layer + feather 的感觉。
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return mask


def _save_result_image(result: Image.Image, output_path: str) -> str:
    """Persist an edited PIL image and return the saved file path."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.save(output)
    return str(output)


def _prepare_blend_mask_np(
    mask_path: str | None,
    image_size: tuple[int, int],
    *,
    feather_radius: float = 0.0,
) -> np.ndarray | None:
    """Return a float32 blend mask in [0, 1] or None."""

    mask = _prepare_blend_mask(
        mask_path,
        image_size,
        feather_radius=feather_radius,
    )
    if mask is None:
        return None
    return np.asarray(mask, dtype=np.float32) / 255.0


def _blend_rgb_result(
    adjusted_rgb: np.ndarray,
    original_rgb: np.ndarray,
    mask_np: np.ndarray | None,
) -> np.ndarray:
    """Blend an adjusted RGB image back onto the original with an optional mask."""

    if mask_np is None:
        return adjusted_rgb
    return adjusted_rgb * mask_np[:, :, None] + original_rgb * (1.0 - mask_np[:, :, None])


def _rgb_to_lab_float(image_float: np.ndarray) -> np.ndarray:
    """Convert float32 RGB [0, 1] into float32 Lab."""

    return cv2.cvtColor(image_float.astype(np.float32), cv2.COLOR_RGB2LAB).astype(np.float32)


def _lab_float_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert float32 Lab back to RGB [0, 1]."""

    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)


def _highlights_mask(luminance: np.ndarray, rolloff: float) -> np.ndarray:
    """Build a soft highlight mask from normalized luminance."""

    rolloff = float(np.clip(rolloff, 0.08, 0.95))
    threshold = 1.0 - rolloff
    return np.power(
        np.clip((luminance - threshold) / max(rolloff, 1e-6), 0.0, 1.0),
        1.45,
    ).astype(np.float32)


def _shadows_mask(luminance: np.ndarray, rolloff: float) -> np.ndarray:
    """Build a soft shadow mask from normalized luminance."""

    rolloff = float(np.clip(rolloff, 0.08, 0.95))
    return np.power(
        np.clip((rolloff - luminance) / max(rolloff, 1e-6), 0.0, 1.0),
        1.45,
    ).astype(np.float32)


def _midtones_mask(luminance: np.ndarray, width: float = 0.7) -> np.ndarray:
    """Build a midtone-favoring mask."""

    width = float(np.clip(width, 0.2, 1.0))
    distance = np.abs(luminance - 0.5) / max(width / 2.0, 1e-6)
    return np.power(np.clip(1.0 - distance, 0.0, 1.0), 1.2).astype(np.float32)


def _color_range_mask(
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
    """Build a soft hue-range mask in HSV space.

    这是后续 Color Mixer / HSL 的基础算法遮罩：
    1. 用色相角决定主范围；
    2. 用饱和度和值抑制接近灰色和过暗区域；
    3. 再对 mask 本身做轻微平滑，避免色块边界生硬。
    """

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


def apply_exposure_adjustment(
    image_path: str,
    output_path: str,
    *,
    multiplier: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a simple exposure adjustment and save the result.

    The implementation uses a multiplicative exposure model:
    values above 1.0 brighten the image, values below 1.0 darken it.
    When a mask is provided, only the masked area is blended with the
    adjusted result. The mask can be feathered to create a softer local
    transition, which is closer to how Photoshop adjustment masks feel.
    """

    image = Image.open(image_path).convert("RGB")
    # CV / 图像处理知识：
    # 这里对每个像素通道做 point-wise transform（逐点变换）。
    # 数学上可以写成：
    #   I' = clip(I * multiplier, 0, 255)
    # 其中 I 是输入像素值，I' 是输出像素值。
    #
    # 这类乘法模型更接近“曝光”或“增益”调整；
    # 如果用 I' = I + beta，则更像简单亮度平移。
    #
    # clip 的作用是做饱和裁剪，避免像素越界。
    adjusted = image.point(lambda value: max(0, min(255, int(round(value * multiplier)))))

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        # CV / 图像处理知识：
        # 这里是最基础的区域混合（masked compositing）。
        # mask 白色区域取 adjusted，黑色区域保留原图。
        # 等价于：
        #   result = mask * adjusted + (1 - mask) * image
        # 只是 Pillow 在内部帮我们完成了归一化和混合。
        result = Image.composite(adjusted, image, mask)
    else:
        result = adjusted

    return _save_result_image(result, output_path)


def apply_highlights_shadows_adjustment(
    image_path: str,
    output_path: str,
    *,
    shadow_amount: float,
    highlight_amount: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    midtone_contrast: float = 0.12,
    local_radius: float = 36.0,
    shadow_tonal_width: float = 0.45,
    highlight_tonal_width: float = 0.45,
    detail_amount: float = 0.3,
) -> str:
    """Apply a more professional highlight/shadow adjustment and save the result.

    Implementation notes:
    - Compute local illumination from the luminance channel.
    - Build separate shadow/highlight masks from local brightness.
    - Adjust only luminance, then recompose with original chroma.
    - Add restrained detail recovery to avoid the typical gray-fog look.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    shadow_amount = float(np.clip(shadow_amount, -1.0, 1.0))
    highlight_amount = float(np.clip(highlight_amount, -1.0, 1.0))
    midtone_contrast = float(np.clip(midtone_contrast, 0.0, 0.5))
    local_radius = float(np.clip(local_radius, 4.0, 160.0))
    shadow_tonal_width = float(np.clip(shadow_tonal_width, 0.1, 0.8))
    highlight_tonal_width = float(np.clip(highlight_tonal_width, 0.1, 0.8))
    detail_amount = float(np.clip(detail_amount, 0.0, 1.0))

    height, width = image_np.shape[:2]
    # 小图如果还用大半径，会把整张图都模糊成一块平均亮度，
    # 导致高光/阴影的局部判断失真。这里按图像尺寸收一层上限。
    effective_local_radius = min(local_radius, max(1.2, min(height, width) / 6.0))

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0] / 255.0

    # CV / 图像处理知识：
    # 这里先对亮度通道做大尺度高斯模糊，近似“局部照明”。
    # 与直接看单个像素亮度不同，局部照明能判断某块区域整体偏暗还是偏亮，
    # 更接近 PS/ACR 的高光阴影思路，而不是简单整体压亮度。
    local_luminance = cv2.GaussianBlur(
        luminance,
        ksize=(0, 0),
        sigmaX=effective_local_radius,
        sigmaY=effective_local_radius,
        borderType=cv2.BORDER_REPLICATE,
    )

    shadow_map = np.clip((shadow_tonal_width - local_luminance) / shadow_tonal_width, 0.0, 1.0)
    highlight_threshold = 1.0 - highlight_tonal_width
    highlight_map = np.clip(
        (local_luminance - highlight_threshold) / max(highlight_tonal_width, 1e-6),
        0.0,
        1.0,
    )

    # 暗部和亮部都再做一次幂次收紧，让调整更多集中在目标区域，
    # 避免整张图一层薄灰那种“全局都被动到了”的感觉。
    shadow_map = np.power(shadow_map, 1.5)
    highlight_map = np.power(highlight_map, 1.7)

    adjusted_luminance = luminance.copy()
    if shadow_amount >= 0:
        adjusted_luminance += (1.0 - adjusted_luminance) * shadow_amount * shadow_map
    else:
        adjusted_luminance -= adjusted_luminance * abs(shadow_amount) * shadow_map

    if highlight_amount >= 0:
        adjusted_luminance -= adjusted_luminance * highlight_amount * highlight_map
    else:
        adjusted_luminance += (1.0 - adjusted_luminance) * abs(highlight_amount) * highlight_map

    # 细节回灌：
    # 先从原始亮度里提一个小半径 detail layer，再按调整强度轻微加回去。
    # 这样可以减轻“阴影拉起来之后整片发灰、失去质感”的问题。
    detail_base = cv2.GaussianBlur(
        luminance,
        ksize=(0, 0),
        sigmaX=max(effective_local_radius * 0.12, 1.2),
        sigmaY=max(effective_local_radius * 0.12, 1.2),
        borderType=cv2.BORDER_REPLICATE,
    )
    detail_layer = luminance - detail_base
    affected_map = np.maximum(shadow_map, highlight_map)
    adjusted_luminance += detail_layer * detail_amount * (0.25 + 0.75 * affected_map)

    if midtone_contrast > 0:
        center_weight = 1.0 - np.clip(np.abs(adjusted_luminance - 0.5) / 0.5, 0.0, 1.0)
        adjusted_luminance += (adjusted_luminance - 0.5) * midtone_contrast * center_weight

    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    # 只替换 LAB 的 L 通道，保留原始 a/b 色度信息，比直接改 RGB 更不容易发灰、偏色。
    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 255.0
    adjusted_rgb = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = adjusted_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = adjusted_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_contrast_adjustment(
    image_path: str,
    output_path: str,
    *,
    contrast_amount: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    pivot: float = 0.5,
    protect_highlights: float = 0.2,
    protect_shadows: float = 0.2,
) -> str:
    """Apply a more professional contrast adjustment around the luminance pivot.

    The implementation works on the LAB luminance channel and preserves chroma.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    contrast_amount = float(np.clip(contrast_amount, -1.0, 1.0))
    pivot = float(np.clip(pivot, 0.25, 0.75))
    protect_highlights = float(np.clip(protect_highlights, 0.0, 0.8))
    protect_shadows = float(np.clip(protect_shadows, 0.0, 0.8))

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0] / 255.0

    # CV / 图像处理知识：
    # 对比度本质上是在中灰点附近拉开或压缩亮度分布。
    # 这里以 pivot 作为“中灰基准”，对亮度做围绕 pivot 的伸缩：
    #   y = pivot + (x - pivot) * slope
    #
    # 为了避免黑位死黑、白位炸掉，再对极暗/极亮区域加保护权重，
    # 让对比度主要作用在中间调，这会更接近真实修图软件的手感。
    slope = 1.0 + contrast_amount * 0.85
    protected = pivot + (luminance - pivot) * slope

    shadow_protection_map = np.power(np.clip((protect_shadows - luminance) / max(protect_shadows, 1e-6), 0.0, 1.0), 1.5)
    highlight_threshold = 1.0 - protect_highlights
    highlight_protection_map = np.power(
        np.clip(
            (luminance - highlight_threshold) / max(protect_highlights, 1e-6),
            0.0,
            1.0,
        ),
        1.5,
    )
    protection = np.maximum(shadow_protection_map, highlight_protection_map)
    adjusted_luminance = protected * (1.0 - protection) + luminance * protection
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 255.0
    adjusted_rgb = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = adjusted_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = adjusted_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_vibrance_saturation_adjustment(
    image_path: str,
    output_path: str,
    *,
    vibrance_amount: float,
    saturation_amount: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    protect_highlights: float = 0.22,
    protect_skin: float = 0.28,
    protect_shadows: float = 0.24,
    chroma_denoise: float = 0.32,
    max_chroma: float = 92.0,
    neutral_floor: float = 6.0,
    neutral_softness: float = 14.0,
) -> str:
    """Apply a restrained vibrance/saturation adjustment and save the result.

    Implementation notes:
    - Work in float32 Lab space so the adjustment mainly touches chroma.
    - Let vibrance prefer low-chroma colors, which is usually safer than
      uniformly boosting all colors.
    - Add highlight protection, approximate skin-tone protection, chroma
      smoothing and soft clipping to reduce chroma noise and block artifacts.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    vibrance_amount = float(np.clip(vibrance_amount, -1.0, 1.0))
    saturation_amount = float(np.clip(saturation_amount, -1.0, 1.0))
    protect_highlights = float(np.clip(protect_highlights, 0.0, 0.8))
    protect_skin = float(np.clip(protect_skin, 0.0, 0.8))
    protect_shadows = float(np.clip(protect_shadows, 0.0, 0.8))
    chroma_denoise = float(np.clip(chroma_denoise, 0.0, 1.0))
    max_chroma = float(np.clip(max_chroma, 48.0, 128.0))
    neutral_floor = float(np.clip(neutral_floor, 0.0, 24.0))
    neutral_softness = float(np.clip(neutral_softness, 2.0, 32.0))

    # CV / 图像处理知识：
    # OpenCV 在 float32 RGB -> Lab 转换时，会尽量保留连续颜色信息，
    # 比在 8-bit HSV 上来回转换更不容易出现量化小色块。
    lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # 关键改动：
    # 不再直接平滑 a/b 色度通道本身。对 JPEG 图来说，这样很容易把压缩块状结构
    # 变成可见的小色块。更稳的做法是：
    # 1. 用原始 a/b 计算色度与方向；
    # 2. 只对“色度增益图”做轻度平滑；
    # 3. 再用这个平滑后的 gain 去缩放原始 a/b。
    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    chroma_safe = np.maximum(chroma, 1e-6)
    unit_a = a_channel / chroma_safe
    unit_b = b_channel / chroma_safe

    # 统一在 Lab 颜色空间里建立“色相角”与保护图。
    hue_angle = (np.degrees(np.arctan2(b_channel, a_channel)) + 360.0) % 360.0
    luminance = np.clip(l_channel / 100.0, 0.0, 1.0)

    chroma_norm = np.clip(chroma / max_chroma, 0.0, 1.0)
    low_chroma_weight = np.power(1.0 - chroma_norm, 1.7)

    # 中性色与暗部保护：
    # 很多“莫名其妙的红点/彩噪”其实来自近灰色像素或暗部 JPEG 压缩噪声。
    # 这两类区域不应该像正常低饱和颜色那样被 vibrance 优先放大。
    neutral_gate = np.power(
        np.clip((chroma - neutral_floor) / max(neutral_softness, 1e-6), 0.0, 1.0),
        1.3,
    )
    shadow_gate = np.power(
        np.clip((luminance - protect_shadows) / max(0.42 - protect_shadows, 1e-6), 0.0, 1.0),
        1.15,
    )
    chroma_presence_gate = np.clip((chroma - 2.5) / 6.0, 0.0, 1.0)
    vibrance_gate = neutral_gate * shadow_gate * chroma_presence_gate

    target_chroma = chroma.copy()
    if vibrance_amount >= 0:
        target_chroma += (max_chroma - target_chroma) * vibrance_amount * low_chroma_weight * vibrance_gate
    else:
        negative_weight = 0.35 + 0.65 * np.maximum(chroma_norm, 1.0 - neutral_gate * 0.6)
        target_chroma *= 1.0 + vibrance_amount * negative_weight

    target_chroma *= 1.0 + saturation_amount

    highlight_threshold = 1.0 - protect_highlights
    highlight_map = np.power(
        np.clip(
            (luminance - highlight_threshold) / max(protect_highlights, 1e-6),
            0.0,
            1.0,
        ),
        1.8,
    )

    # 粗略肤色保护改到 Lab 色彩空间：
    # 暖色肤色通常在 a/b 平面落在正 a、正 b 的一段夹角里。
    # 这里只做保守抑制，避免人脸和皮肤一下子被冲得太橙太红。
    skin_center = 50.0
    skin_width = 28.0
    skin_angle_distance = np.abs(hue_angle - skin_center)
    skin_angle_distance = np.minimum(skin_angle_distance, 360.0 - skin_angle_distance)
    skin_hue_weight = np.clip(1.0 - skin_angle_distance / skin_width, 0.0, 1.0)
    skin_chroma_gate = np.clip((chroma - 8.0) / 18.0, 0.0, 1.0) * np.clip(
        (72.0 - chroma) / 24.0,
        0.0,
        1.0,
    )
    skin_luma_gate = np.clip((luminance - 0.18) / 0.18, 0.0, 1.0) * np.clip(
        (0.92 - luminance) / 0.18,
        0.0,
        1.0,
    )
    skin_map = skin_hue_weight * skin_chroma_gate * skin_luma_gate

    protection = np.clip(
        highlight_map * protect_highlights
        + skin_map * protect_skin
        + (1.0 - shadow_gate) * protect_shadows * 0.9
        + (1.0 - neutral_gate) * 0.42,
        0.0,
        0.95,
    )
    protected_chroma = target_chroma * (1.0 - protection) + chroma * protection

    # Soft clipping：
    # 接近上限时不再线性增长，而是柔和压缩，避免颜色冲出可用范围后
    # 出现看起来像 JPEG 色块被放大的感觉。
    soft_knee = max_chroma * 0.78
    excess = np.clip(protected_chroma - soft_knee, 0.0, None)
    compressed_excess = excess / (1.0 + excess / max(max_chroma - soft_knee, 1e-6))
    final_chroma = np.where(
        protected_chroma > soft_knee,
        soft_knee + compressed_excess,
        protected_chroma,
    )
    final_chroma = np.clip(final_chroma, 0.0, max_chroma)

    gain = final_chroma / chroma_safe
    if chroma_denoise > 0:
        gain_sigma = 0.6 + chroma_denoise * 2.2
        gain = cv2.GaussianBlur(
            gain.astype(np.float32),
            ksize=(0, 0),
            sigmaX=gain_sigma,
            sigmaY=gain_sigma,
            borderType=cv2.BORDER_REPLICATE,
        )
    gain = np.clip(gain, 0.0, 2.5)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 1] = a_channel * gain
    adjusted_lab[:, :, 2] = b_channel * gain
    adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    adjusted_rgb = np.clip(adjusted_rgb, 0.0, 1.0)

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = adjusted_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = adjusted_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_whites_blacks_adjustment(
    image_path: str,
    output_path: str,
    *,
    whites_amount: float,
    blacks_amount: float,
    highlight_rolloff: float = 0.32,
    shadow_rolloff: float = 0.34,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Adjust whites and blacks on the luminance channel."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    lab = _rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)

    white_mask = _highlights_mask(luminance, highlight_rolloff)
    black_mask = _shadows_mask(luminance, shadow_rolloff)
    midtone_guard = 1.0 - 0.45 * _midtones_mask(luminance, 0.8)

    adjusted_luminance = luminance.copy()
    if whites_amount >= 0:
        adjusted_luminance += (1.0 - adjusted_luminance) * whites_amount * white_mask * midtone_guard
    else:
        adjusted_luminance -= adjusted_luminance * abs(whites_amount) * white_mask * midtone_guard

    if blacks_amount >= 0:
        adjusted_luminance -= adjusted_luminance * blacks_amount * black_mask * midtone_guard
    else:
        adjusted_luminance += (1.0 - adjusted_luminance) * abs(blacks_amount) * black_mask * midtone_guard

    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)
    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 100.0
    adjusted_rgb = _lab_float_to_rgb(adjusted_lab)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_curves_adjustment(
    image_path: str,
    output_path: str,
    *,
    shadow_lift: float,
    midtone_gamma: float,
    highlight_compress: float,
    contrast_bias: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a restrained curves-style luminance shaping."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    lab = _rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)

    shadow_mask = _shadows_mask(luminance, 0.48)
    highlight_mask = _highlights_mask(luminance, 0.42)
    midtones = _midtones_mask(luminance, 0.76)

    adjusted_luminance = luminance.copy()
    adjusted_luminance += shadow_lift * shadow_mask * (1.0 - adjusted_luminance) * 0.72
    adjusted_luminance -= highlight_compress * highlight_mask * adjusted_luminance * 0.62
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    if midtone_gamma != 1.0:
        adjusted_luminance = np.power(np.clip(adjusted_luminance, 1e-6, 1.0), 1.0 / midtone_gamma)

    adjusted_luminance += (adjusted_luminance - 0.5) * contrast_bias * 0.42 * midtones
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 100.0
    adjusted_rgb = _lab_float_to_rgb(adjusted_lab)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_clarity_adjustment(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    radius_scale: float = 1.0,
    highlight_protection: float = 0.22,
    shadow_protection: float = 0.22,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply midtone local-contrast enhancement."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    lab = _rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)

    sigma = 2.2 + radius_scale * 5.2
    base = cv2.GaussianBlur(luminance, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    detail = luminance - base

    midtones = _midtones_mask(luminance, 0.82)
    highlights = _highlights_mask(luminance, max(0.14, highlight_protection + 0.18))
    shadows = _shadows_mask(luminance, max(0.14, shadow_protection + 0.18))
    protection = np.clip(highlights * highlight_protection + shadows * shadow_protection, 0.0, 0.92)

    adjusted_luminance = luminance + detail * amount * 1.35 * midtones * (1.0 - protection)
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 100.0
    adjusted_rgb = _lab_float_to_rgb(adjusted_lab)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_texture_adjustment(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    detail_scale: float = 1.0,
    noise_protection: float = 0.4,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply medium-scale texture enhancement or softening."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    lab = _rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)

    sigma_small = 0.9 + detail_scale * 1.6
    sigma_large = sigma_small * 2.2
    small_base = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=sigma_small,
        sigmaY=sigma_small,
        borderType=cv2.BORDER_REPLICATE,
    )
    large_base = cv2.GaussianBlur(
        luminance,
        (0, 0),
        sigmaX=sigma_large,
        sigmaY=sigma_large,
        borderType=cv2.BORDER_REPLICATE,
    )
    detail = small_base - large_base
    detail_magnitude = np.abs(detail)
    noise_floor = 0.004 + noise_protection * 0.03
    detail_gate = np.clip((detail_magnitude - noise_floor) / max(0.08 - noise_floor, 1e-6), 0.0, 1.0)
    midtones = _midtones_mask(luminance, 0.9)

    adjusted_luminance = luminance + detail * amount * 1.55 * detail_gate * midtones
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 100.0
    adjusted_rgb = _lab_float_to_rgb(adjusted_lab)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_dehaze_adjustment(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    luminance_protection: float = 0.26,
    color_protection: float = 0.3,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a conservative, perceptual dehaze adjustment."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    lab = _rgb_to_lab_float(image_float)
    luminance = np.clip(lab[:, :, 0] / 100.0, 0.0, 1.0)
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    veil = cv2.GaussianBlur(luminance, (0, 0), sigmaX=11.0, sigmaY=11.0, borderType=cv2.BORDER_REPLICATE)
    local_contrast = luminance - veil
    midtones = _midtones_mask(luminance, 0.92)
    extreme_protection = np.clip(
        _highlights_mask(luminance, 0.28) * luminance_protection
        + _shadows_mask(luminance, 0.28) * luminance_protection,
        0.0,
        0.92,
    )

    adjusted_luminance = luminance + amount * local_contrast * 2.1 * midtones * (1.0 - extreme_protection)
    adjusted_luminance = np.clip(adjusted_luminance, 0.0, 1.0)

    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    chroma_safe = np.maximum(chroma, 1e-6)
    chroma_gain = 1.0 + amount * 0.11 * (1.0 - np.clip(chroma / 96.0, 0.0, 1.0) * color_protection)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = adjusted_luminance * 100.0
    adjusted_lab[:, :, 1] = a_channel * chroma_gain
    adjusted_lab[:, :, 2] = b_channel * chroma_gain
    adjusted_rgb = _lab_float_to_rgb(adjusted_lab)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_color_mixer_adjustment(
    image_path: str,
    output_path: str,
    *,
    channel_settings: dict[str, dict[str, float]],
    saturation_protection: float = 0.3,
    luminance_protection: float = 0.22,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
) -> str:
    """Apply a professional Color Mixer / HSL adjustment."""

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(image_float, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    color_bands = {
        "red": (0.0, 24.0),
        "orange": (30.0, 20.0),
        "yellow": (58.0, 18.0),
        "green": (120.0, 28.0),
        "aqua": (180.0, 22.0),
        "blue": (230.0, 26.0),
        "purple": (275.0, 22.0),
        "magenta": (320.0, 24.0),
    }

    for color_name, (center, half_width) in color_bands.items():
        settings = channel_settings.get(color_name) or {}
        hue_shift_deg = float(settings.get("hue_shift_deg", 0.0))
        saturation_shift = float(settings.get("saturation_shift", 0.0))
        luminance_shift = float(settings.get("luminance_shift", 0.0))

        if abs(hue_shift_deg) < 1e-6 and abs(saturation_shift) < 1e-6 and abs(luminance_shift) < 1e-6:
            continue

        band_mask = _color_range_mask(
            hue,
            saturation,
            value,
            center=center,
            half_width=half_width,
            saturation_floor=0.06,
            value_floor=0.06,
            blur_sigma=1.2,
        )
        sat_guard = 1.0 - np.clip(saturation * saturation_protection, 0.0, 0.88)
        lum_guard = 1.0 - np.clip(np.abs(value - 0.5) * 2.0 * luminance_protection, 0.0, 0.85)
        effective_mask = band_mask * sat_guard * lum_guard

        hue = (hue + hue_shift_deg * effective_mask) % 360.0
        saturation = np.clip(saturation * (1.0 + saturation_shift * effective_mask), 0.0, 1.0)

        if luminance_shift >= 0:
            value = np.clip(value + (1.0 - value) * luminance_shift * effective_mask, 0.0, 1.0)
        else:
            value = np.clip(value - value * abs(luminance_shift) * effective_mask, 0.0, 1.0)

    adjusted_hsv = np.stack([hue, saturation, value], axis=-1).astype(np.float32)
    adjusted_rgb = np.clip(cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)

    mask_np = _prepare_blend_mask_np(mask_path, image.size, feather_radius=feather_radius)
    result_np = _blend_rgb_result(adjusted_rgb, image_float, mask_np)
    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_white_balance_adjustment(
    image_path: str,
    output_path: str,
    *,
    temperature_shift: float,
    tint_shift: float = 0.0,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    protect_saturated: float = 0.3,
) -> str:
    """Apply a restrained white-balance adjustment and save the result.

    Implementation notes:
    - Work in float32 Lab space so we can primarily adjust chroma axes.
    - Temperature mainly moves along the blue-yellow axis (b channel).
    - Tint mainly moves along the green-magenta axis (a channel).
    - Saturated colors receive less correction than near-neutral colors to
      keep the result closer to a conservative photo workflow.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    temperature_shift = float(np.clip(temperature_shift, -24.0, 24.0))
    tint_shift = float(np.clip(tint_shift, -18.0, 18.0))
    protect_saturated = float(np.clip(protect_saturated, 0.0, 0.85))

    # CV / 图像处理知识：
    # 在 Lab 颜色空间里，L 是亮度，a 近似绿-洋红轴，b 近似蓝-黄轴。
    # 白平衡本质上主要是校正色偏，所以这里优先动 a/b，不直接改 L。
    lab = cv2.cvtColor(image_float, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    chroma = np.sqrt(a_channel * a_channel + b_channel * b_channel)
    chroma_norm = np.clip(chroma / 80.0, 0.0, 1.0)

    # 越接近中性色，越应该接受白平衡修正；已经很饱和的颜色则保守一些，
    # 避免天空、霓虹灯或高饱和物体被整体带偏。
    cast_weight = 1.0 - chroma_norm * protect_saturated
    cast_weight = np.clip(cast_weight, 0.15, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = l_channel
    adjusted_lab[:, :, 1] = np.clip(a_channel + tint_shift * cast_weight, -128.0, 127.0)
    adjusted_lab[:, :, 2] = np.clip(b_channel + temperature_shift * cast_weight, -128.0, 127.0)

    adjusted_rgb = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    adjusted_rgb = np.clip(adjusted_rgb, 0.0, 1.0)

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = adjusted_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = adjusted_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_crop_and_straighten(
    image_path: str,
    output_path: str,
    *,
    crop_ratio: float,
    straighten_angle: float = 0.0,
    crop_guard: float = 0.04,
    min_scale: float = 0.72,
) -> str:
    """Apply a restrained crop/straighten adjustment and save the result.

    Implementation notes:
    - Crop is currently center-based and conservative.
    - Straighten rotates with bicubic resampling and uses a rotated content
      mask to shrink the final center crop until black corners are avoided.
    - This is a stable MVP implementation, not a full smart-composition crop.
    """

    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    aspect_ratio = original_width / max(original_height, 1)

    crop_ratio = float(np.clip(crop_ratio, 0.0, 0.35))
    straighten_angle = float(np.clip(straighten_angle, -15.0, 15.0))
    crop_guard = float(np.clip(crop_guard, 0.0, 0.12))
    min_scale = float(np.clip(min_scale, 0.45, 1.0))

    if abs(straighten_angle) > 1e-3:
        rotated = image.rotate(
            -straighten_angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(0, 0, 0),
        )
        content_mask = Image.new("L", image.size, 255).rotate(
            -straighten_angle,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
    else:
        rotated = image.copy()
        content_mask = Image.new("L", image.size, 255)

    # CV / 图像处理知识：
    # 拉直后的黑角问题，本质上来自“旋转后的有效内容区域”变成了倾斜多边形。
    # 这里不去求解析几何里的最大内接矩形，而是：
    # 1. 先按原图比例给一个目标裁剪尺寸；
    # 2. 再用旋转后的 content mask 检查这块区域是否全部有效；
    # 3. 如果边缘还碰到空区，就继续轻微缩小。
    base_scale = max(1.0 - crop_ratio - crop_guard, min_scale)
    if abs(straighten_angle) > 1e-3:
        base_scale -= min(abs(straighten_angle) * 0.006, 0.12)
        base_scale = max(base_scale, min_scale)

    target_width = max(32, int(round(original_width * base_scale)))
    target_height = max(32, int(round(original_height * base_scale)))

    if target_width / target_height > aspect_ratio:
        target_width = max(32, int(round(target_height * aspect_ratio)))
    else:
        target_height = max(32, int(round(target_width / aspect_ratio)))

    rotated_width, rotated_height = rotated.size
    target_width = min(target_width, rotated_width)
    target_height = min(target_height, rotated_height)

    mask_np = np.asarray(content_mask, dtype=np.uint8)
    center_x = rotated_width // 2
    center_y = rotated_height // 2

    for _ in range(24):
        left = max(0, center_x - target_width // 2)
        top = max(0, center_y - target_height // 2)
        right = min(rotated_width, left + target_width)
        bottom = min(rotated_height, top + target_height)

        if right - left != target_width:
            left = max(0, right - target_width)
        if bottom - top != target_height:
            top = max(0, bottom - target_height)

        crop_mask = mask_np[top:bottom, left:right]
        if crop_mask.size == 0:
            raise ValueError("Invalid crop region computed during straighten.")

        if crop_mask.min() > 0:
            result = rotated.crop((left, top, right, bottom))
            return _save_result_image(result, output_path)

        target_width = max(32, int(round(target_width * 0.97)))
        target_height = max(32, int(round(target_height * 0.97)))
        if target_width / target_height > aspect_ratio:
            target_width = max(32, int(round(target_height * aspect_ratio)))
        else:
            target_height = max(32, int(round(target_width / aspect_ratio)))

    # 最后兜底：
    # 如果还没找到全有效区域，就直接中心裁剪当前尺寸，至少保证流程稳定。
    left = max(0, center_x - target_width // 2)
    top = max(0, center_y - target_height // 2)
    right = min(rotated_width, left + target_width)
    bottom = min(rotated_height, top + target_height)
    result = rotated.crop((left, top, right, bottom))
    return _save_result_image(result, output_path)


def apply_denoise_adjustment(
    image_path: str,
    output_path: str,
    *,
    luma_strength: float,
    chroma_strength: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    detail_protection: float = 0.22,
    template_window_size: int = 7,
    search_window_size: int = 21,
) -> str:
    """Apply a restrained photo denoise adjustment and save the result.

    Implementation notes:
    - Use OpenCV's non-local means color denoising as the main engine.
    - Blend a little of the original image back to reduce waxy texture loss.
    - When a mask is provided, denoise first and then feather-blend only the
      requested region so local denoise stays controlled.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    luma_strength = float(np.clip(luma_strength, 0.0, 24.0))
    chroma_strength = float(np.clip(chroma_strength, 0.0, 24.0))
    detail_protection = float(np.clip(detail_protection, 0.0, 0.75))

    # OpenCV 要求这些窗口参数为奇数；这里做一次兜底归一化。
    template_window_size = int(np.clip(template_window_size, 3, 15))
    search_window_size = int(np.clip(search_window_size, 7, 31))
    if template_window_size % 2 == 0:
        template_window_size += 1
    if search_window_size % 2 == 0:
        search_window_size += 1

    # CV / 图像处理知识：
    # fastNlMeansDenoisingColored 是经典的“非局部均值”去噪。
    # 和直接高斯模糊不同，它会在更大范围内寻找相似纹理块再做平均，
    # 对照片噪声通常更自然，也更适合作为第一版的照片去噪工具。
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        image_bgr,
        None,
        h=luma_strength,
        hColor=chroma_strength,
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 细节保护：
    # 去噪过强时容易发“糊”和“蜡”，所以这里把原图轻微混回去，
    # 让第一版更接近保守修图而不是极限降噪。
    denoised_rgb = denoised_rgb * (1.0 - detail_protection) + image_float * detail_protection
    denoised_rgb = np.clip(denoised_rgb, 0.0, 1.0)

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = denoised_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = denoised_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


def apply_sharpen_adjustment(
    image_path: str,
    output_path: str,
    *,
    amount: float,
    radius: float,
    threshold: float,
    mask_path: str | None = None,
    feather_radius: float = 0.0,
    highlight_protection: float = 0.24,
) -> str:
    """Apply a restrained sharpen adjustment and save the result.

    Implementation notes:
    - Use unsharp masking on the luminance channel only.
    - Apply a threshold gate so tiny noise does not get sharpened.
    - Protect bright highlights to reduce brittle digital halos.
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)
    image_float = image_np.astype(np.float32) / 255.0

    amount = float(np.clip(amount, 0.0, 2.4))
    radius = float(np.clip(radius, 0.4, 6.0))
    threshold = float(np.clip(threshold, 0.0, 0.2))
    highlight_protection = float(np.clip(highlight_protection, 0.0, 0.85))

    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    luminance = lab[:, :, 0] / 255.0

    # CV / 图像处理知识：
    # unsharp mask 的核心是：
    # 1. 先得到一个模糊版本
    # 2. 原图 - 模糊图 = 高频细节
    # 3. 再把这层高频细节按 amount 加回去
    blurred = cv2.GaussianBlur(
        luminance,
        ksize=(0, 0),
        sigmaX=radius,
        sigmaY=radius,
        borderType=cv2.BORDER_REPLICATE,
    )
    detail = luminance - blurred

    # 阈值门控：
    # 很小的亮度扰动多数是噪声，不应该被锐化放大。
    threshold_gate = np.clip((np.abs(detail) - threshold) / max(0.08, threshold * 4.0 + 1e-6), 0.0, 1.0)

    # 高光保护：
    # 接近纯亮区如果继续锐化，很容易出现数字感很强的脆边和 halo。
    highlight_threshold = 1.0 - highlight_protection
    highlight_gate = 1.0 - np.power(
        np.clip(
            (luminance - highlight_threshold) / max(highlight_protection, 1e-6),
            0.0,
            1.0,
        ),
        1.5,
    )

    sharpened_luminance = luminance + detail * amount * threshold_gate * highlight_gate
    sharpened_luminance = np.clip(sharpened_luminance, 0.0, 1.0)

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0] = sharpened_luminance * 255.0
    adjusted_rgb = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0

    mask = _prepare_blend_mask(mask_path, image.size, feather_radius=feather_radius)
    if mask:
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        mask_np = mask_np[:, :, None]
        result_np = adjusted_rgb * mask_np + image_float * (1.0 - mask_np)
    else:
        result_np = adjusted_rgb

    result = Image.fromarray(np.clip(result_np * 255.0, 0, 255).astype(np.uint8), mode="RGB")
    return _save_result_image(result, output_path)


from app.tools.image_ops_extended import (  # noqa: E402
    apply_auto_upright,
    apply_background_blur,
    apply_camera_calibration,
    apply_color_grading,
    apply_convert_black_white,
    apply_defringe,
    apply_glow_highlight,
    apply_grain,
    apply_lens_blur,
    apply_lens_correction,
    apply_lut_preset,
    apply_moire_reduction,
    apply_perspective_correction,
    apply_point_color_adjustment,
    apply_regional_enhancement,
    apply_remove_chromatic_aberration,
    apply_remove_heal,
    apply_skin_smooth,
    apply_vignette,
)
