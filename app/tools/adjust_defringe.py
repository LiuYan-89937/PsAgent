"""Native adjust_defringe tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_defringe


@tool
def adjust_defringe(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    purple_amount: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    green_amount: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    edge_threshold: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1,
) -> dict:
    """Use this tool when high-contrast edges show purple or green fringing from optics or heavy contrast. It suppresses edge color fringing, so it is more appropriate for lens artifacts than for normal color cleanup."""

    output_path = temp_output_path("psagent_defringe_")
    saved_path = apply_defringe(
        image_path,
        output_path,
        purple_amount=purple_amount,
        green_amount=green_amount,
        edge_threshold=edge_threshold,
    )
    return build_result(
        tool_name="adjust_defringe",
        output_image=saved_path,
        applied_params={
            "purple_amount": purple_amount,
            "green_amount": green_amount,
            "edge_threshold": edge_threshold,
        },
        image_path=image_path,
        mask_path=None,
    )


ADJUST_DEFRINGE_SPEC = ToolSpec(
    name="adjust_defringe",
    label="边色修正",
    description="Suppress purple and green fringing near strong edges.",
    family="detail",
    focus_affinity=["global_tone", "subject_cleanup"],
    supports_mask=False,
    supports_whole_image=True,
    default_params={"purple_amount": 0.5, "green_amount": 0.5, "edge_threshold": 0.1},
    planner_schema=build_planner_schema(adjust_defringe, supports_mask=False, excluded_fields={"image_path", "mask_path"}),
    primary_param="purple_amount",
    risk_level="low",
    status_label="正在修正边色",
    keywords=("紫边", "绿边", "defringe"),
)


__all__ = ["ADJUST_DEFRINGE_SPEC", "adjust_defringe"]
