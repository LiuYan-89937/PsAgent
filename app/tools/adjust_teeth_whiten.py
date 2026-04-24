"""Native adjust_teeth_whiten tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, require_mask_path, temp_output_path
from app.tools.image_ops import apply_teeth_whiten_adjustment


@tool
def adjust_teeth_whiten(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    yellow_reduce: Annotated[float, Field(default=0.25, ge=0.0, le=1.0)] = 0.25,
    brightness_increase: Annotated[float, Field(default=0.15, ge=0.0, le=1.0)] = 0.15,
    neutralize_gray: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1,
    preserve_contrast: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    feather_radius: Annotated[float, Field(default=10.0, ge=0.0, le=64.0)] = 10.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when teeth need a restrained whitening effect by reducing yellow cast and slightly lifting brightness. It is a teeth-only local portrait tool and should never be treated as a general global brightening or color-cleanup tool."""

    require_mask_path("adjust_teeth_whiten", mask_path, recommended_prompt="teeth")
    output_path = temp_output_path("psagent_teeth_whiten_")
    saved_path = apply_teeth_whiten_adjustment(
        image_path,
        output_path,
        yellow_reduce=yellow_reduce,
        brightness_increase=brightness_increase,
        neutralize_gray=neutralize_gray,
        preserve_contrast=preserve_contrast,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_teeth_whiten",
        output_image=saved_path,
        applied_params={
            "yellow_reduce": yellow_reduce,
            "brightness_increase": brightness_increase,
            "neutralize_gray": neutralize_gray,
            "preserve_contrast": preserve_contrast,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_TEETH_WHITEN_SPEC = ToolSpec(
    name="adjust_teeth_whiten",
    label="牙齿美白",
    description="Whiten teeth by reducing yellow cast and lifting restrained luminance.",
    family="portrait",
    focus_affinity=["subject_cleanup"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="teeth",
    default_params={
        "yellow_reduce": 0.25,
        "brightness_increase": 0.15,
        "neutralize_gray": 0.1,
        "preserve_contrast": 0.5,
        "feather_radius": 10.0,
    },
    planner_schema=build_planner_schema(
        adjust_teeth_whiten,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="yellow_reduce",
    risk_level="medium",
    status_label="正在美白牙齿",
    keywords=("牙齿美白", "牙齿", "美白"),
)


__all__ = ["ADJUST_TEETH_WHITEN_SPEC", "adjust_teeth_whiten"]
