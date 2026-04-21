"""Native adjust_temperature_tint tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_white_balance_adjustment


@tool
def adjust_temperature_tint(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    temperature_shift: Annotated[float, Field(default=0.0, ge=-24.0, le=24.0, description="Blue-yellow axis shift.")] = 0.0,
    tint_shift: Annotated[float, Field(default=0.0, ge=-18.0, le=18.0, description="Green-magenta axis shift.")] = 0.0,
    protect_saturated: Annotated[float, Field(default=0.3, ge=0.0, le=0.85, description="Saturated-color protection.")] = 0.3,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image feels too warm, too cool, too green, or too magenta and needs white-balance style correction. It changes blue-yellow and green-magenta balance, so it is the right choice for color cast repair rather than saturation or style grading."""

    output_path = temp_output_path("psagent_temperature_tint_")
    saved_path = apply_white_balance_adjustment(
        image_path,
        output_path,
        temperature_shift=temperature_shift,
        tint_shift=tint_shift,
        protect_saturated=protect_saturated,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_temperature_tint",
        output_image=saved_path,
        applied_params={
            "temperature_shift": temperature_shift,
            "tint_shift": tint_shift,
            "protect_saturated": protect_saturated,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_TEMPERATURE_TINT_SPEC = ToolSpec(
    name="adjust_temperature_tint",
    label="色温色调",
    description="Adjust blue-yellow and green-magenta balance with saturation protection.",
    family="tone",
    stage_affinity=["global_base", "local_balance"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "temperature_shift": 0.0,
        "tint_shift": 0.0,
        "protect_saturated": 0.3,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_temperature_tint,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="temperature_shift",
    risk_level="low",
    status_label="正在调整色温和色调",
    keywords=("色温", "色调", "偏黄", "偏蓝", "偏绿", "偏紫"),
)


__all__ = ["ADJUST_TEMPERATURE_TINT_SPEC", "adjust_temperature_tint"]
