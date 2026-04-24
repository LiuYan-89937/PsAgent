"""Native adjust_curves tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_curves_adjustment


@tool
def adjust_curves(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    shadow_lift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Shadow lift amount.")] = 0.0,
    midtone_gamma: Annotated[float, Field(default=1.0, ge=0.35, le=2.8, description="Midtone gamma response.")] = 1.0,
    highlight_compress: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Highlight compression amount.")] = 0.0,
    contrast_bias: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Overall contrast bias.")] = 0.0,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want curve-like control over shadows, midtones, highlights, and overall contrast bias. It is more expressive than levels for shaping tone, so use it for stylized or carefully tuned luminance moves rather than quick automatic cleanup."""

    output_path = temp_output_path("psagent_curves_")
    saved_path = apply_curves_adjustment(
        image_path,
        output_path,
        shadow_lift=shadow_lift,
        midtone_gamma=midtone_gamma,
        highlight_compress=highlight_compress,
        contrast_bias=contrast_bias,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_curves",
        output_image=saved_path,
        applied_params={
            "shadow_lift": shadow_lift,
            "midtone_gamma": midtone_gamma,
            "highlight_compress": highlight_compress,
            "contrast_bias": contrast_bias,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_CURVES_SPEC = ToolSpec(
    name="adjust_curves",
    label="曲线",
    description="Apply restrained parametric curve shaping on luminance.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "shadow_lift": 0.0,
        "midtone_gamma": 1.0,
        "highlight_compress": 0.0,
        "contrast_bias": 0.0,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_curves,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="contrast_bias",
    risk_level="low",
    status_label="正在调整曲线",
    keywords=("曲线", "tone curve"),
)


__all__ = ["ADJUST_CURVES_SPEC", "adjust_curves"]
