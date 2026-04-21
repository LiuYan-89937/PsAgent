"""Native adjust_skin_texture_reduce tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_skin_smooth


@tool
def adjust_skin_texture_reduce(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    radius: Annotated[float, Field(default=4.0, ge=0.5, le=12.0)] = 4.0,
    detail_preserve: Annotated[float, Field(default=0.6, ge=0.0, le=1.0)] = 0.6,
    tone_preserve: Annotated[float, Field(default=0.8, ge=0.0, le=1.0)] = 0.8,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when fine skin texture should be softened while preserving larger edges and overall tone structure. It is a local skin-detail softening tool and should be used with a skin mask instead of the whole frame."""

    output_path = temp_output_path("psagent_skin_texture_reduce_")
    smooth_strength = min(max(radius / 8.0, 0.1), 1.0) * amount
    saved_path = apply_skin_smooth(
        image_path,
        output_path,
        strength=amount,
        smooth_strength=smooth_strength,
        detail_protection=detail_preserve * tone_preserve,
        saturation_protection=0.2,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_skin_texture_reduce",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "radius": radius,
            "detail_preserve": detail_preserve,
            "tone_preserve": tone_preserve,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SKIN_TEXTURE_REDUCE_SPEC = ToolSpec(
    name="adjust_skin_texture_reduce",
    label="皮肤纹理减弱",
    description="Reduce fine skin texture while preserving tone and edges.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="skin",
    default_params={"amount": 0.4, "radius": 4.0, "detail_preserve": 0.6, "tone_preserve": 0.8, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_skin_texture_reduce, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="medium",
    status_label="正在减弱皮肤纹理",
    keywords=("皮肤纹理", "细纹减弱", "皮肤细腻"),
)


__all__ = ["ADJUST_SKIN_TEXTURE_REDUCE_SPEC", "adjust_skin_texture_reduce"]
