"""Native adjust_contrast tool."""

from __future__ import annotations

import tempfile
from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common.contracts import ToolExecutionResult, ToolSpec, build_planner_schema
from app.tools.common.mask_contracts import MASK_PARAMS_SCHEMA
from app.tools.image_ops import apply_contrast_adjustment


def _temp_output_path(prefix: str) -> str:
    """Build a temporary PNG output path for deterministic edits."""

    return tempfile.mktemp(prefix=prefix, suffix=".png")


@tool
def adjust_contrast(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.28,
            ge=0.0,
            le=1.0,
            description="Primary contrast push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.28,
    contrast_scale: Annotated[
        float,
        Field(
            default=0.8,
            ge=0.25,
            le=1.5,
            description="Scaling factor that maps the primary strength into LAB luminance contrast.",
        ),
    ] = 0.8,
    pivot: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.25,
            le=0.75,
            description="Midtone pivot used by the contrast curve.",
        ),
    ] = 0.5,
    feather_radius: Annotated[
        float,
        Field(
            default=18.0,
            ge=0.0,
            le=64.0,
            description="Mask feather radius in pixels when running on a local region.",
        ),
    ] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image looks flat, hazy, low-contrast, or needs stronger separation between light and dark areas. It changes luminance contrast around a pivot, so it is better for overall punch and structure than for narrow color or brightness-only corrections."""

    # 对比度工具沿用统一主强度语义，
    # 再通过 contrast_scale 控制映射到真实 luminance 伸缩幅度。
    contrast_amount = float(strength) * float(contrast_scale)
    output_path = _temp_output_path("psagent_contrast_")
    saved_path = apply_contrast_adjustment(
        image_path,
        output_path,
        contrast_amount=contrast_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
        pivot=pivot,
        protect_highlights=0.32,
        protect_shadows=0.32,
    )
    return ToolExecutionResult(
        ok=True,
        tool="adjust_contrast",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "contrast_scale": contrast_scale,
            "contrast_amount": contrast_amount,
            "pivot": pivot,
            "feather_radius": feather_radius,
        },
        artifacts={
            "input_image": image_path,
            "mask_path": mask_path,
        },
    ).model_dump(mode="json")


ADJUST_CONTRAST_SPEC = ToolSpec(
    # spec 只描述能力和默认值，不参与图像处理。
    name="adjust_contrast",
    label="对比度",
    description="Adjust whole-image or masked contrast around a controlled luminance pivot.",
    family="tone",
    focus_affinity=[
        "global_tone",
        "global_tone",
        "subject_separation",
        "subject_cleanup",
        "finish",
    ],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "strength": 0.28,
        "contrast_scale": 0.8,
        "pivot": 0.5,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        # runtime-only 参数不暴露给 planner，mask 相关字段通过公共 schema 合并进来。
        adjust_contrast,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="strength",
    risk_level="low",
    status_label="正在调整对比度",
    keywords=("对比度", "层次", "反差"),
)


__all__ = [
    "ADJUST_CONTRAST_SPEC",
    "adjust_contrast",
]
