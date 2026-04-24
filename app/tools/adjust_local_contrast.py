"""Native adjust_local_contrast tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_clarity_adjustment


@tool
def adjust_local_contrast(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=-1.0, le=1.0, description="Local contrast amount.")] = 0.5,
    radius: Annotated[float, Field(default=16.0, ge=1.0, le=48.0, description="Local contrast radius.")] = 16.0,
    edge_protection: Annotated[float, Field(default=0.2, ge=0.0, le=0.85, description="Protection on extremes and edges.")] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image needs more local punch, separation, and three-dimensional structure without changing global exposure too much. It enhances mid-scale contrast, so it is useful for texture and depth but not for fixing overall brightness or color cast problems."""

    output_path = temp_output_path("psagent_local_contrast_")
    radius_scale = max(radius / 12.0, 0.25)
    saved_path = apply_clarity_adjustment(
        image_path,
        output_path,
        amount=amount,
        radius_scale=radius_scale,
        highlight_protection=edge_protection,
        shadow_protection=edge_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_local_contrast",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "radius": radius,
            "edge_protection": edge_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_LOCAL_CONTRAST_SPEC = ToolSpec(
    name="adjust_local_contrast",
    label="局部对比",
    description="Enhance local contrast on mid-scale detail structures.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "amount": 0.5,
        "radius": 16.0,
        "edge_protection": 0.2,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_local_contrast,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="amount",
    risk_level="low",
    status_label="正在增强局部对比",
    keywords=("局部对比", "通透", "层次", "清透"),
)


__all__ = ["ADJUST_LOCAL_CONTRAST_SPEC", "adjust_local_contrast"]
