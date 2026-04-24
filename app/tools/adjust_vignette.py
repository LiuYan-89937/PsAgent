"""Native adjust_vignette tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_vignette


@tool
def adjust_vignette(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=-1.0, le=1.0)] = 0.5,
    midpoint: Annotated[float, Field(default=0.5, ge=0.15, le=0.95)] = 0.5,
    roundness: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    feather: Annotated[float, Field(default=0.5, ge=0.05, le=1.0)] = 0.5,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want the frame edges darker or lighter to guide attention toward the center or subject. It is a framing and finishing effect, not a true exposure correction for the whole image."""

    output_path = temp_output_path("psagent_vignette_")
    saved_path = apply_vignette(
        image_path,
        output_path,
        amount=amount,
        midpoint=midpoint,
        roundness=roundness,
        feather=feather,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_vignette",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "midpoint": midpoint,
            "roundness": roundness,
            "feather": feather,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_VIGNETTE_SPEC = ToolSpec(
    name="adjust_vignette",
    label="暗角",
    description="Apply a restrained post-crop vignette.",
    family="effects",
    focus_affinity=["finish"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.5, "midpoint": 0.5, "roundness": 0.5, "feather": 0.5, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_vignette, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在调整暗角",
    keywords=("暗角", "vignette"),
)


__all__ = ["ADJUST_VIGNETTE_SPEC", "adjust_vignette"]
