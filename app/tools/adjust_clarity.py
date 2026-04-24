"""Native adjust_clarity tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_clarity_adjustment


@tool
def adjust_clarity(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.5, ge=-1.0, le=1.0)] = 0.5,
    radius_scale: Annotated[float, Field(default=1.0, ge=0.2, le=3.0)] = 1.0,
    highlight_protection: Annotated[float, Field(default=0.22, ge=0.0, le=0.85)] = 0.22,
    shadow_protection: Annotated[float, Field(default=0.22, ge=0.0, le=0.85)] = 0.22,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image needs more mid-scale punch, local separation, and perceived crispness. It acts on larger structure than texture, so it is useful for shape definition and depth but can become harsh if used where softness is desired."""

    output_path = temp_output_path("psagent_clarity_")
    saved_path = apply_clarity_adjustment(
        image_path,
        output_path,
        amount=amount,
        radius_scale=radius_scale,
        highlight_protection=highlight_protection,
        shadow_protection=shadow_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_clarity",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "radius_scale": radius_scale,
            "highlight_protection": highlight_protection,
            "shadow_protection": shadow_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_CLARITY_SPEC = ToolSpec(
    name="adjust_clarity",
    label="清晰度",
    description="Adjust midtone local contrast and clarity.",
    family="detail",
    focus_affinity=["subject_cleanup"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.5, "radius_scale": 1.0, "highlight_protection": 0.22, "shadow_protection": 0.22, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_clarity, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在增强清晰度",
    keywords=("清晰度", "clarity", "通透"),
)


__all__ = ["ADJUST_CLARITY_SPEC", "adjust_clarity"]
