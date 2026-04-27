"""Native adjust_midtones tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_midtones_adjustment


@tool
def adjust_midtones(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    midtone_shift: Annotated[float, Field(default=0.0, ge=-0.4, le=0.4, description="Midtone lift/compress amount.")] = 0.0,
    midtone_width: Annotated[float, Field(default=0.4, ge=0.2, le=1.0, description="Midtone band width.")] = 0.4,
    preserve_shadows: Annotated[float, Field(default=0.34, ge=0.0, le=0.85, description="Shadow protection.")] = 0.34,
    preserve_highlights: Annotated[float, Field(default=0.34, ge=0.0, le=0.85, description="Highlight protection.")] = 0.34,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the middle brightness range needs to be lifted or lowered without strongly disturbing deep shadows and highlights. It is best for perceived midtone presence and facial or subject visibility, not for changing overall dynamic range endpoints."""

    output_path = temp_output_path("psagent_midtones_")
    saved_path = apply_midtones_adjustment(
        image_path,
        output_path,
        midtone_shift=midtone_shift,
        midtone_width=midtone_width,
        preserve_shadows=preserve_shadows,
        preserve_highlights=preserve_highlights,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_midtones",
        output_image=saved_path,
        applied_params={
            "midtone_shift": midtone_shift,
            "midtone_width": midtone_width,
            "preserve_shadows": preserve_shadows,
            "preserve_highlights": preserve_highlights,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_MIDTONES_SPEC = ToolSpec(
    name="adjust_midtones",
    label="中间调",
    description="Adjust midtone luminance while protecting shadows and highlights.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "midtone_shift": 0.0,
        "midtone_width": 0.4,
        "preserve_shadows": 0.34,
        "preserve_highlights": 0.34,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_midtones,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="midtone_shift",
    risk_level="low",
    status_label="正在调整中间调",
    keywords=("中间调", "midtone"),
)


__all__ = ["ADJUST_MIDTONES_SPEC", "adjust_midtones"]
