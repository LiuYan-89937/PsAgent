"""Native adjust_highlights_shadows tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_highlights_shadows_adjustment


@tool
def adjust_highlights_shadows(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    shadow_amount: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Shadow lift/compress amount.")] = 0.0,
    highlight_amount: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Highlight recovery/boost amount.")] = 0.0,
    midtone_contrast: Annotated[float, Field(default=0.12, ge=0.0, le=0.5, description="Midtone contrast recovery.")] = 0.12,
    local_radius: Annotated[float, Field(default=36.0, ge=4.0, le=160.0, description="Local illumination estimation radius.")] = 36.0,
    shadow_tonal_width: Annotated[float, Field(default=0.45, ge=0.1, le=0.8, description="Shadow tonal width.")] = 0.45,
    highlight_tonal_width: Annotated[float, Field(default=0.45, ge=0.1, le=0.8, description="Highlight tonal width.")] = 0.45,
    detail_amount: Annotated[float, Field(default=0.3, ge=0.0, le=1.0, description="Detail recovery amount.")] = 0.3,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when highlights are too harsh, shadows are too blocked, or you need to recover dynamic range in a more photographic way. It separately reshapes bright and dark regions, so it is better for tonal recovery than for simple brightness or contrast pushes."""

    output_path = temp_output_path("psagent_highlights_shadows_")
    saved_path = apply_highlights_shadows_adjustment(
        image_path,
        output_path,
        shadow_amount=shadow_amount,
        highlight_amount=highlight_amount,
        midtone_contrast=midtone_contrast,
        local_radius=local_radius,
        shadow_tonal_width=shadow_tonal_width,
        highlight_tonal_width=highlight_tonal_width,
        detail_amount=detail_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_highlights_shadows",
        output_image=saved_path,
        applied_params={
            "shadow_amount": shadow_amount,
            "highlight_amount": highlight_amount,
            "midtone_contrast": midtone_contrast,
            "local_radius": local_radius,
            "shadow_tonal_width": shadow_tonal_width,
            "highlight_tonal_width": highlight_tonal_width,
            "detail_amount": detail_amount,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_HIGHLIGHTS_SHADOWS_SPEC = ToolSpec(
    name="adjust_highlights_shadows",
    label="高光阴影",
    description="Recover highlights and lift shadows with local illumination awareness.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "shadow_amount": 0.0,
        "highlight_amount": 0.0,
        "midtone_contrast": 0.12,
        "local_radius": 36.0,
        "shadow_tonal_width": 0.45,
        "highlight_tonal_width": 0.45,
        "detail_amount": 0.3,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_highlights_shadows,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="shadow_amount",
    risk_level="low",
    status_label="正在调整高光和阴影",
    keywords=("高光", "阴影", "压高光", "提暗部"),
)


__all__ = ["ADJUST_HIGHLIGHTS_SHADOWS_SPEC", "adjust_highlights_shadows"]
