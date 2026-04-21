"""Native adjust_soften_local_contrast tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_clarity_adjustment


@tool
def adjust_soften_local_contrast(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    radius: Annotated[float, Field(default=8.0, ge=0.6, le=20.0)] = 8.0,
    highlight_preserve: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when a local area feels too crunchy, too textured, or too harsh and needs softer mid-scale contrast. It is the inverse of clarity-style structure enhancement, so it is useful for calming skin or harsh detail without fully blurring the image."""

    output_path = temp_output_path("psagent_soften_local_contrast_")
    radius_scale = max(0.2, radius / 8.0)
    saved_path = apply_clarity_adjustment(
        image_path,
        output_path,
        amount=-amount,
        radius_scale=radius_scale,
        highlight_protection=highlight_preserve,
        shadow_protection=0.18,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_soften_local_contrast",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "radius": radius,
            "highlight_preserve": highlight_preserve,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SOFTEN_LOCAL_CONTRAST_SPEC = ToolSpec(
    name="adjust_soften_local_contrast",
    label="局部降清晰度",
    description="Soften local contrast by applying restrained negative clarity.",
    family="detail",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "amount": 0.3,
        "radius": 8.0,
        "highlight_preserve": 0.3,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_soften_local_contrast,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="amount",
    risk_level="low",
    status_label="正在降低局部清晰度",
    keywords=("局部降清晰度", "柔一点", "软化局部反差"),
)


__all__ = ["ADJUST_SOFTEN_LOCAL_CONTRAST_SPEC", "adjust_soften_local_contrast"]
