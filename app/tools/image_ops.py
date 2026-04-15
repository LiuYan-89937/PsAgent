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
