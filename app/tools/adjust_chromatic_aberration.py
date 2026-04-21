"""Native adjust_chromatic_aberration tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_remove_chromatic_aberration


@tool
def adjust_chromatic_aberration(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    radial_bias: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
) -> dict:
    """Use this tool when red-cyan or blue-yellow edge separation from lens aberration is visible near image edges. It is an optics repair tool, not a creative color tool, and should be used for lens-style channel misalignment rather than cast correction."""

    output_path = temp_output_path("psagent_chromatic_aberration_")
    saved_path = apply_remove_chromatic_aberration(
        image_path,
        output_path,
        amount=amount,
        radial_bias=radial_bias,
    )
    return build_result(
        tool_name="adjust_chromatic_aberration",
        output_image=saved_path,
        applied_params={"amount": amount, "radial_bias": radial_bias},
        image_path=image_path,
        mask_path=None,
    )


ADJUST_CHROMATIC_ABERRATION_SPEC = ToolSpec(
    name="adjust_chromatic_aberration",
    label="色差校正",
    description="Reduce simple radial chromatic aberration by re-aligning RGB channels.",
    family="effects",
    stage_affinity=["technical_prep"],
    supports_mask=False,
    supports_whole_image=True,
    default_params={"amount": 0.5, "radial_bias": 0.5},
    planner_schema=build_planner_schema(adjust_chromatic_aberration, supports_mask=False, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在校正色差",
    keywords=("色差", "chromatic aberration"),
)


__all__ = ["ADJUST_CHROMATIC_ABERRATION_SPEC", "adjust_chromatic_aberration"]
