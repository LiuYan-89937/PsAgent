"""Native adjust_grain tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_grain


@tool
def adjust_grain(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    size: Annotated[float, Field(default=1.0, ge=0.2, le=2.5)] = 1.0,
    roughness: Annotated[float, Field(default=0.5, ge=0.0, le=1.0)] = 0.5,
    color_amount: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image needs controlled film-like grain for texture, atmosphere, or a less digital finish. It adds stylized grain and should be treated as a look-building effect rather than a repair step."""

    output_path = temp_output_path("psagent_grain_")
    saved_path = apply_grain(
        image_path,
        output_path,
        amount=amount,
        size=size,
        roughness=roughness,
        color_amount=color_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_grain",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "size": size,
            "roughness": roughness,
            "color_amount": color_amount,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_GRAIN_SPEC = ToolSpec(
    name="adjust_grain",
    label="颗粒",
    description="Add restrained film-like grain.",
    family="effects",
    stage_affinity=["finish_output"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.5, "size": 1.0, "roughness": 0.5, "color_amount": 0.2, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_grain, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在添加颗粒",
    keywords=("颗粒", "grain", "胶片颗粒"),
)


__all__ = ["ADJUST_GRAIN_SPEC", "adjust_grain"]
