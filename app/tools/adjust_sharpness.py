"""Native adjust_sharpness tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_sharpen_adjustment


@tool
def adjust_sharpness(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.38, ge=0.0, le=2.4)] = 0.38,
    radius: Annotated[float, Field(default=1.0, ge=0.4, le=6.0)] = 1.0,
    threshold: Annotated[float, Field(default=0.02, ge=0.0, le=0.2)] = 0.02,
    highlight_protection: Annotated[float, Field(default=0.42, ge=0.0, le=0.85)] = 0.42,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when edge definition needs to look crisper and more resolved after other tonal or color adjustments. It is for sharpening visible structure, not for making low-contrast images brighter or more vivid."""

    output_path = temp_output_path("psagent_sharpness_")
    saved_path = apply_sharpen_adjustment(
        image_path,
        output_path,
        amount=amount,
        radius=radius,
        threshold=threshold,
        highlight_protection=highlight_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_sharpness",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "radius": radius,
            "threshold": threshold,
            "highlight_protection": highlight_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SHARPNESS_SPEC = ToolSpec(
    name="adjust_sharpness",
    label="锐化",
    description="Apply restrained luminance sharpen with threshold gating.",
    family="detail",
    focus_affinity=["subject_cleanup", "finish"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.38, "radius": 1.0, "threshold": 0.02, "highlight_protection": 0.42, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_sharpness, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在锐化",
    keywords=("锐化", "sharpness", "更锐"),
)


__all__ = ["ADJUST_SHARPNESS_SPEC", "adjust_sharpness"]
