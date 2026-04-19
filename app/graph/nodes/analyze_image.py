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
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])

    height, width = gray.shape
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    shadow_ratio = float((gray < 28).mean())
    highlight_ratio = float((gray > 235).mean())

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
        "summary": summary,
        "metrics": {
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std,
            "shadow_ratio": shadow_ratio,
            "highlight_ratio": highlight_ratio,
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
            merged_analysis["model_analysis"] = model_analysis
            validated = AnalyzeImageResult.model_validate(merged_analysis)
            return {"image_analysis": validated.model_dump(mode="json")}
        except RuntimeError as error:
            fallback_trace = append_fallback_trace(
                state.get("fallback_trace"),
                stage="analyze_image",
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
            stage="analyze_image",
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
