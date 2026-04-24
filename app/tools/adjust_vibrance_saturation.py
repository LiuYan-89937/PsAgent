"""Native adjust_vibrance_saturation tool."""

from __future__ import annotations

import tempfile
from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common.contracts import ToolExecutionResult, ToolSpec, build_planner_schema
from app.tools.common.mask_contracts import MASK_PARAMS_SCHEMA
from app.tools.image_ops import apply_vibrance_saturation_adjustment


def _temp_output_path(prefix: str) -> str:
    """Build a temporary PNG output path for deterministic edits."""

    return tempfile.mktemp(prefix=prefix, suffix=".png")


@tool
def adjust_vibrance_saturation(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Primary color push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.5,
    vibrance_scale: Annotated[
        float,
        Field(
            default=0.8,
            ge=0.2,
            le=1.2,
            description="Vibrance emphasis applied to lower-chroma colors first.",
        ),
    ] = 0.8,
    saturation_scale: Annotated[
        float,
        Field(
            default=0.28,
            ge=0.0,
            le=0.6,
            description="Additional global saturation lift layered on top of vibrance.",
        ),
    ] = 0.28,
    protect_highlights: Annotated[
        float,
        Field(
            default=0.26,
            ge=0.0,
            le=0.8,
            description="Highlight protection that restrains color boosts in bright areas.",
        ),
    ] = 0.26,
    protect_skin: Annotated[
        float,
        Field(
            default=0.34,
            ge=0.0,
            le=0.8,
            description="Skin-tone protection weight to avoid plastic or oversaturated skin.",
        ),
    ] = 0.34,
    protect_shadows: Annotated[
        float,
        Field(
            default=0.24,
            ge=0.0,
            le=0.8,
            description="Shadow protection weight to keep dark chroma noise under control.",
        ),
    ] = 0.24,
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
    """Use this tool when colors need to feel richer, cleaner, or more vivid without fully restyling the image. It boosts lower-chroma colors first and can also increase saturation, so it is useful for general color lift but not for precise single-color repair."""

    # 颜色推进拆成 vibrance 和 saturation 两层：
    # vibrance 更偏向低饱和区域，saturation 做更均匀的补充。
    vibrance_amount = float(strength) * float(vibrance_scale)
    saturation_amount = float(strength) * float(saturation_scale)
    output_path = _temp_output_path("psagent_vibrance_saturation_")
    saved_path = apply_vibrance_saturation_adjustment(
        image_path,
        output_path,
        vibrance_amount=vibrance_amount,
        saturation_amount=saturation_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
        protect_highlights=protect_highlights,
        protect_skin=protect_skin,
        protect_shadows=protect_shadows,
    )
    return ToolExecutionResult(
        ok=True,
        tool="adjust_vibrance_saturation",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "vibrance_scale": vibrance_scale,
            "saturation_scale": saturation_scale,
            "vibrance_amount": vibrance_amount,
            "saturation_amount": saturation_amount,
            "protect_highlights": protect_highlights,
            "protect_skin": protect_skin,
            "protect_shadows": protect_shadows,
            "feather_radius": feather_radius,
        },
        artifacts={
            "input_image": image_path,
            "mask_path": mask_path,
        },
    ).model_dump(mode="json")


ADJUST_VIBRANCE_SATURATION_SPEC = ToolSpec(
    # 这份 spec 决定 planner 如何认识这个工具，
    # 例如默认值、家族归类、mask 能力和主参数。
    name="adjust_vibrance_saturation",
    label="自然饱和度",
    description="Adjust whole-image or masked vibrance and saturation in color space while protecting highlights, skin, and shadows.",
    family="color",
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
        "strength": 0.5,
        "vibrance_scale": 0.8,
        "saturation_scale": 0.28,
        "protect_highlights": 0.26,
        "protect_skin": 0.34,
        "protect_shadows": 0.24,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        # 对 planner 来说，这里看到的是“可规划参数”，
        # 不是最终运行时完整签名。
        adjust_vibrance_saturation,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="strength",
    risk_level="low",
    status_label="正在调整色彩",
    keywords=("饱和度", "色彩", "鲜艳", "自然饱和度", "vibrance"),
)


__all__ = [
    "ADJUST_VIBRANCE_SATURATION_SPEC",
    "adjust_vibrance_saturation",
]
