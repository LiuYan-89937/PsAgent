"""Common helpers shared by stage pipeline nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
from langgraph.config import get_stream_writer
from PIL import Image

from app.graph.state import EditState, PhaseArtifact, StageKey
from app.services.stage_policy import STAGE_LABELS


def safe_stream_writer():
    """Return a stream writer when inside LangGraph runtime, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        # 非 LangGraph 运行时（例如单测）下，返回一个空 writer，
        # 这样调用方就不需要到处判空。
        return lambda *_args, **_kwargs: None


def current_image_path(state: EditState) -> str:
    """Return the current working image path."""

    # 当前链路始终遵循“上一步输出图覆盖下一步输入图”的规则，
    # 所以优先取 selected_output；如果还没有执行过任何步骤，就退回原始输入图。
    input_images = state.get("input_images") or []
    if not input_images and not state.get("selected_output"):
        raise ValueError("No input image available.")
    return str(state.get("selected_output") or input_images[0])


def compute_image_metrics(image_path: str, *, mask_path: str | None = None) -> dict[str, float]:
    """Compute lightweight brightness and clipping statistics for stage guards."""

    # 这里的指标不是为了做精确美学评价，而是给阶段 guard 提供
    # 一个快速、稳定、可解释的硬阈值判断依据。
    image = Image.open(image_path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114])

    if mask_path:
        # 局部阶段 guard 只看 mask 覆盖到的区域，这样不会被整图统计稀释掉。
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        selector = mask > 0
        if selector.any():
            gray = gray[selector]
            rgb = image_np[selector]
        else:
            rgb = image_np.reshape(-1, 3)
    else:
        rgb = image_np.reshape(-1, 3)

    if gray.size == 0:
        # 极端情况下如果 mask 为空，返回一组保守的 0 值，
        # 由上层逻辑决定是否继续或跳过。
        return {
            "brightness_mean": 0.0,
            "shadow_ratio": 0.0,
            "highlight_ratio": 0.0,
            "saturation_mean": 0.0,
        }

    max_rgb = rgb.max(axis=1)
    min_rgb = rgb.min(axis=1)
    saturation = np.where(max_rgb == 0, 0.0, (max_rgb - min_rgb) / np.maximum(max_rgb, 1.0))
    return {
        "brightness_mean": float(gray.mean()),
        "shadow_ratio": float((gray < 28).mean()),
        "highlight_ratio": float((gray > 235).mean()),
        "saturation_mean": float(saturation.mean()),
    }


def base_phase_artifact(stage_key: StageKey) -> PhaseArtifact:
    """Return a default phase artifact for a stage."""

    # 每个阶段第一次被访问时，都从一个统一的空壳开始，
    # 后续 planner / execution / summary 都是在这个壳上逐步填充。
    return PhaseArtifact(key=stage_key, label=STAGE_LABELS[stage_key])
