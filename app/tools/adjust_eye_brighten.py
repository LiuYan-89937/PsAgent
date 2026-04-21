"""Native adjust_eye_brighten tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_eye_brighten_adjustment


@tool
def adjust_eye_brighten(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    brightness_increase: Annotated[float, Field(default=0.15, ge=0.0, le=1.0)] = 0.15,
    clarity_boost: Annotated[float, Field(default=0.15, ge=0.0, le=1.0)] = 0.15,
    saturation_boost: Annotated[float, Field(default=0.05, ge=-1.0, le=1.0)] = 0.05,
    highlight_boost: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1,
    feather_radius: Annotated[float, Field(default=10.0, ge=0.0, le=64.0)] = 10.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when eyes should look brighter, clearer, and more alive through subtle lift, clarity, saturation, or catchlight emphasis. It is an eye-region portrait tool and should be paired with an eye mask instead of being applied across the whole image."""

    output_path = temp_output_path("psagent_eye_brighten_")
    saved_path = apply_eye_brighten_adjustment(
        image_path,
        output_path,
        brightness_increase=brightness_increase,
        clarity_boost=clarity_boost,
        saturation_boost=saturation_boost,
        highlight_boost=highlight_boost,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_eye_brighten",
        output_image=saved_path,
        applied_params={
            "brightness_increase": brightness_increase,
            "clarity_boost": clarity_boost,
            "saturation_boost": saturation_boost,
            "highlight_boost": highlight_boost,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_EYE_BRIGHTEN_SPEC = ToolSpec(
    name="adjust_eye_brighten",
    label="眼睛提亮",
    description="Brighten eyes and lightly boost clarity, saturation, and catchlights.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="eye",
    default_params={
        "brightness_increase": 0.15,
        "clarity_boost": 0.15,
        "saturation_boost": 0.05,
        "highlight_boost": 0.1,
        "feather_radius": 10.0,
    },
    planner_schema=build_planner_schema(
        adjust_eye_brighten,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="brightness_increase",
    risk_level="medium",
    status_label="正在提亮眼睛",
    keywords=("眼睛提亮", "眼白", "眼神光"),
)


__all__ = ["ADJUST_EYE_BRIGHTEN_SPEC", "adjust_eye_brighten"]
