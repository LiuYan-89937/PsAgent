"""Native apply_color_lookup tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_lut_preset


@tool
def apply_color_lookup(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    preset: Annotated[str, Field(default="clean_portrait", description="Preset / LUT name.")] = "clean_portrait",
    strength: Annotated[float, Field(default=0.5, ge=0.0, le=1.0, description="Preset blend strength.")] = 0.5,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want to apply a preset-like LUT or lookup-based look quickly. It is best for final color style direction and should not be the first choice for local repair, precise hue targeting, or technical cleanup."""

    output_path = temp_output_path("psagent_color_lookup_")
    saved_path = apply_lut_preset(
        image_path,
        output_path,
        preset=preset,
        strength=strength,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="apply_color_lookup",
        output_image=saved_path,
        applied_params={
            "preset": preset,
            "strength": strength,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


APPLY_COLOR_LOOKUP_SPEC = ToolSpec(
    name="apply_color_lookup",
    label="色彩查找",
    description="Apply a lightweight LUT / color-lookup preset.",
    family="color",
    stage_affinity=["finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "preset": "clean_portrait",
        "strength": 0.5,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        apply_color_lookup,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="strength",
    risk_level="medium",
    status_label="正在应用色彩查找",
    keywords=("LUT", "色彩查找", "预设", "风格预设"),
)


__all__ = ["APPLY_COLOR_LOOKUP_SPEC", "apply_color_lookup"]
