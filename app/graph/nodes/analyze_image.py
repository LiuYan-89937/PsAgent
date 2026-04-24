"""Analyze input image content and issues."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import AnalyzeImageResult, EditState, ImageQualityMetrics
from app.services.analyze_image_model import (
    analyze_image_model_available,
    generate_image_analysis_with_qwen,
)


def _compute_basic_image_analysis(image_path: str) -> dict[str, Any]:
    """Compute a small set of deterministic image facts.

    当前版本先不用多模态模型，先给 planner 提供稳定的数值型图像事实：
    1. 基本尺寸和方向；
    2. 亮度、对比度、暗部/高光占比；
    3. 一个保守的默认分割提示。
    """

    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    rgb01 = image_np / 255.0
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])

    height, width = gray.shape
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    shadow_ratio = float((gray < 28).mean())
    highlight_ratio = float((gray > 235).mean())
    midtone_ratio = float(((gray >= 56) & (gray <= 205)).mean())
    dynamic_range = float(np.percentile(gray, 95) - np.percentile(gray, 5))

    # 亮度直方图只保留归一化比例，避免把大数组塞进 state。
    histogram, _ = np.histogram(gray, bins=12, range=(0, 255))
    histogram_total = max(int(histogram.sum()), 1)
    exposure_histogram = [round(float(value) / histogram_total, 4) for value in histogram]

    # 轻量 HSV 饱和度估计，不依赖 OpenCV，保证分析节点稳定可跑。
    rgb_max = rgb01.max(axis=2)
    rgb_min = rgb01.min(axis=2)
    chroma = rgb_max - rgb_min
    saturation = np.divide(chroma, np.maximum(rgb_max, 1e-6), out=np.zeros_like(chroma), where=rgb_max > 1e-6)
    saturation_mean = float(saturation.mean())
    saturation_std = float(saturation.std())

    # 局部对比用相邻像素梯度近似，足够支撑“平/糊/低反差”的基础判断。
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    local_contrast_mean = float((grad_x.mean() + grad_y.mean()) / 2.0)

    channel_means = image_np.mean(axis=(0, 1))
    neutral_mean = float(channel_means.mean())
    color_cast_rgb = {
        "red": round(float(channel_means[0] - neutral_mean), 4),
        "green": round(float(channel_means[1] - neutral_mean), 4),
        "blue": round(float(channel_means[2] - neutral_mean), 4),
    }

    # 没有分割时，用中心区域/边缘区域给主体和背景亮度一个保守代理值。
    center_y0, center_y1 = int(height * 0.25), int(height * 0.75)
    center_x0, center_x1 = int(width * 0.25), int(width * 0.75)
    center_gray = gray[center_y0:center_y1, center_x0:center_x1]
    border_mask = np.ones_like(gray, dtype=bool)
    border_margin_y = max(int(height * 0.12), 1)
    border_margin_x = max(int(width * 0.12), 1)
    border_mask[border_margin_y : height - border_margin_y, border_margin_x : width - border_margin_x] = False
    background_gray = gray[border_mask]

    # 简单肤色代理只作为分析提示，不作为最终人像判断依据。
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    skin_mask = (r > 0.32) & (g > 0.22) & (b > 0.16) & (r > b * 1.05) & (r < 0.98) & (saturation > 0.08)
    skin_luminance_mean = float(gray[skin_mask].mean()) if bool(skin_mask.any()) else None

    issues: list[str] = []
    if brightness_mean < 95:
        issues.append("underexposed")
    elif brightness_mean > 180:
        issues.append("overexposed")

    if brightness_std < 42:
        issues.append("flat_contrast")
    if shadow_ratio > 0.18:
        issues.append("crushed_shadows")
    if highlight_ratio > 0.08:
        issues.append("clipped_highlights")
    if saturation_mean < 0.08:
        issues.append("low_saturation")
    if local_contrast_mean < 5.5:
        issues.append("low_local_contrast")
    if dynamic_range < 95:
        issues.append("compressed_tonal_range")

    summary = "画面整体正常。"
    if issues:
        summary = f"检测到的基础问题：{', '.join(issues)}。"

    return {
        "source_image": image_path,
        "filename": Path(image_path).name,
        "width": width,
        "height": height,
        "orientation": "portrait" if height > width else "landscape",
        "domain": "general",
        "scene_tags": [],
        "issues": issues,
        "subjects": ["primary visible subject"],
        "segmentation_hints": ["primary visible subject"],
        "main_issues": list(issues),
        "primary_subject": "primary visible subject",
        "has_portrait": None,
        "needs_local_editing": bool(issues),
        "has_background_distraction": False,
        "summary": summary,
        "metrics": {
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
            "midtone_ratio": midtone_ratio,
            "saturation_mean": saturation_mean,
            "saturation_std": saturation_std,
            "local_contrast_mean": local_contrast_mean,
            "dynamic_range": dynamic_range,
            "color_cast_rgb": color_cast_rgb,
            "exposure_histogram": exposure_histogram,
            "subject_luminance_mean": float(center_gray.mean()) if center_gray.size else None,
            "background_luminance_mean": float(background_gray.mean()) if background_gray.size else None,
            "skin_luminance_mean": skin_luminance_mean,
        },
    }


def analyze_image(state: EditState) -> dict:
    """Analyze image domain, tags, and quality issues.

    这一步先做最小稳定版：
    1. 如果 state 里已经有 image_analysis，就直接透传；
    2. 否则读取第一张输入图，生成基础分析结果；
    3. 不在这里做任何模型级决策，只产出“图像事实”。
    """

    existing = state.get("image_analysis")
    if existing:
        validated = AnalyzeImageResult.model_validate(existing)
        return {"image_analysis": validated.model_dump(mode="json")}

    input_images = state.get("input_images") or []
    if not input_images:
        return {
            "image_analysis": AnalyzeImageResult(
                domain="general",
                scene_tags=[],
                issues=[],
                subjects=[],
                segmentation_hints=[],
                summary="当前没有输入图片。",
                metrics=ImageQualityMetrics(
                    brightness_mean=0.0,
                    brightness_std=0.0,
                    shadow_ratio=0.0,
                    highlight_ratio=0.0,
                ),
            ).model_dump(mode="json"),
        }

    basic_analysis = _compute_basic_image_analysis(input_images[0])
    if analyze_image_model_available():
        try:
            model_analysis = generate_image_analysis_with_qwen(
                image_path=input_images[0],
                request_text=str(state.get("request_text") or ""),
                basic_metrics=basic_analysis["metrics"],
            )
            merged_analysis = dict(basic_analysis)
            for key in ("domain", "scene_tags", "issues", "subjects", "segmentation_hints", "summary"):
                if key in model_analysis:
                    merged_analysis[key] = model_analysis[key]
            merged_analysis["main_issues"] = list(model_analysis.get("main_issues") or merged_analysis.get("issues") or [])
            merged_analysis["primary_subject"] = model_analysis.get("primary_subject") or (
                (merged_analysis.get("subjects") or [None])[0]
            )
            merged_analysis["has_portrait"] = model_analysis.get("has_portrait")
            merged_analysis["needs_local_editing"] = model_analysis.get("needs_local_editing")
            merged_analysis["has_background_distraction"] = model_analysis.get("has_background_distraction")
            merged_analysis["model_analysis"] = model_analysis
            validated = AnalyzeImageResult.model_validate(merged_analysis)
            return {"image_analysis": validated.model_dump(mode="json")}
        except RuntimeError as error:
            fallback_trace = append_fallback_trace(
                state.get("fallback_trace"),
                round_id=None,
                focus=None,
                source="analyze_image_model",
                location="image_analysis",
                strategy="basic_image_analysis",
                message="图像分析模型不可用，改用基础图像分析。",
                error=str(error),
            )
            validated = AnalyzeImageResult.model_validate(basic_analysis)
            return {
                "image_analysis": validated.model_dump(mode="json"),
                "fallback_trace": fallback_trace,
            }
    else:
        fallback_trace = append_fallback_trace(
            state.get("fallback_trace"),
            round_id=None,
            focus=None,
            source="analyze_image_model",
            location="image_analysis",
            strategy="basic_image_analysis",
            message="图像分析模型不可用，改用基础图像分析。",
            error=None,
        )
        validated = AnalyzeImageResult.model_validate(basic_analysis)
        return {
            "image_analysis": validated.model_dump(mode="json"),
            "fallback_trace": fallback_trace,
        }

    validated = AnalyzeImageResult.model_validate(basic_analysis)
    return {
        "image_analysis": validated.model_dump(mode="json"),
    }
