"""Native adjust_dehaze tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_dehaze_adjustment


@tool
def adjust_dehaze(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.24, ge=-1.0, le=1.0)] = 0.24,
    luminance_protection: Annotated[float, Field(default=0.36, ge=0.0, le=1.0)] = 0.36,
    color_protection: Annotated[float, Field(default=0.42, ge=0.0, le=1.0)] = 0.42,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image looks veiled, low-contrast, atmospheric, or washed by haze and needs cleaner separation. It increases clarity and depth in a haze-oriented way, so it is stronger and more scene-structural than a simple contrast move."""

    output_path = temp_output_path("psagent_dehaze_")
    saved_path = apply_dehaze_adjustment(
        image_path,
        output_path,
        amount=amount,
        luminance_protection=luminance_protection,
        color_protection=color_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_dehaze",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "luminance_protection": luminance_protection,
            "color_protection": color_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_DEHAZE_SPEC = ToolSpec(
    name="adjust_dehaze",
    label="去灰雾",
    description="Apply perceptual dehaze with luminance and color protection.",
    family="detail",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.24, "luminance_protection": 0.36, "color_protection": 0.42, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_dehaze, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在去灰雾",
    keywords=("去灰雾", "dehaze", "空气感"),
)


__all__ = ["ADJUST_DEHAZE_SPEC", "adjust_dehaze"]
