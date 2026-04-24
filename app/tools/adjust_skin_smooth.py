"""Native adjust_skin_smooth tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, require_mask_path, temp_output_path
from app.tools.image_ops import apply_skin_smooth


@tool
def adjust_skin_smooth(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    smooth_strength: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    detail_protection: Annotated[float, Field(default=0.6, ge=0.0, le=1.0)] = 0.6,
    saturation_protection: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when skin should look smoother and more polished while preserving edges and major facial structure. It is a skin-only local retouch tool, so it should be paired with a skin mask and not applied to the full image."""

    require_mask_path("adjust_skin_smooth", mask_path, recommended_prompt="skin")
    output_path = temp_output_path("psagent_skin_smooth_")
    saved_path = apply_skin_smooth(
        image_path,
        output_path,
        strength=strength,
        smooth_strength=smooth_strength,
        detail_protection=detail_protection,
        saturation_protection=saturation_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_skin_smooth",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "smooth_strength": smooth_strength,
            "detail_protection": detail_protection,
            "saturation_protection": saturation_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SKIN_SMOOTH_SPEC = ToolSpec(
    name="adjust_skin_smooth",
    label="皮肤柔化",
    description="Apply restrained skin smoothing with detail protection.",
    family="portrait",
    focus_affinity=["subject_cleanup"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="skin",
    default_params={"strength": 0.4, "smooth_strength": 0.4, "detail_protection": 0.6, "saturation_protection": 0.2, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_skin_smooth, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="smooth_strength",
    risk_level="medium",
    status_label="正在柔化皮肤",
    keywords=("磨皮", "柔肤", "皮肤柔化"),
)


__all__ = ["ADJUST_SKIN_SMOOTH_SPEC", "adjust_skin_smooth"]
