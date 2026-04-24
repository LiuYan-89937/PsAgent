"""Native adjust_moire_reduction tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_moire_reduction


@tool
def adjust_moire_reduction(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
) -> dict:
    """Use this tool when repeating fabric, screen, or fine pattern interference creates visible moire artifacts. It is specifically for suppressing moire and should not be used as a general blur or denoise substitute."""

    output_path = temp_output_path("psagent_moire_reduction_")
    saved_path = apply_moire_reduction(image_path, output_path, amount=amount)
    return build_result(
        tool_name="adjust_moire_reduction",
        output_image=saved_path,
        applied_params={"amount": amount},
        image_path=image_path,
        mask_path=None,
    )


ADJUST_MOIRE_REDUCTION_SPEC = ToolSpec(
    name="adjust_moire_reduction",
    label="摩尔纹抑制",
    description="Reduce visible moire by attenuating chroma and high frequencies.",
    family="detail",
    focus_affinity=["global_tone", "subject_cleanup"],
    supports_mask=False,
    supports_whole_image=True,
    default_params={"amount": 0.5},
    planner_schema=build_planner_schema(adjust_moire_reduction, supports_mask=False, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在抑制摩尔纹",
    keywords=("摩尔纹", "moire"),
)


__all__ = ["ADJUST_MOIRE_REDUCTION_SPEC", "adjust_moire_reduction"]
