"""Native adjust_glow_highlights tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_glow_highlight


@tool
def adjust_glow_highlights(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    threshold: Annotated[float, Field(default=0.75, ge=0.0, le=1.0)] = 0.75,
    warmth: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when bright areas should bloom, glow, or feel more luminous and dreamy. It affects highlight regions specifically, so it is useful for finishing atmosphere and not for technical brightness or contrast correction."""

    output_path = temp_output_path("psagent_glow_highlights_")
    saved_path = apply_glow_highlight(
        image_path,
        output_path,
        amount=amount,
        threshold=threshold,
        warmth=warmth,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_glow_highlights",
        output_image=saved_path,
        applied_params={"amount": amount, "threshold": threshold, "warmth": warmth, "feather_radius": feather_radius},
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_GLOW_HIGHLIGHTS_SPEC = ToolSpec(
    name="adjust_glow_highlights",
    label="高光发光",
    description="Apply bloom/glow to highlight regions.",
    family="effects",
    focus_affinity=["finish"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.4, "threshold": 0.75, "warmth": 0.0, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_glow_highlights, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在增强高光发光",
    keywords=("发光", "glow", "bloom", "氛围"),
)


__all__ = ["ADJUST_GLOW_HIGHLIGHTS_SPEC", "adjust_glow_highlights"]
