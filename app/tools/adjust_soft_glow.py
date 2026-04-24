"""Native adjust_soft_glow tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_soft_glow_adjustment


@tool
def adjust_soft_glow(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.3, ge=0.0, le=1.0)] = 0.3,
    blur_radius: Annotated[float, Field(default=6.0, ge=0.6, le=20.0)] = 6.0,
    contrast_restore: Annotated[float, Field(default=0.2, ge=0.0, le=1.0)] = 0.2,
    highlight_bias: Annotated[float, Field(default=0.4, ge=0.0, le=1.0)] = 0.4,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image should feel softer, dreamier, or more diffused overall or in a masked area. It creates a gentle soft-focus glow, so it is a finishing effect rather than a sharpness, clarity, or repair tool."""

    output_path = temp_output_path("psagent_soft_glow_")
    saved_path = apply_soft_glow_adjustment(
        image_path,
        output_path,
        amount=amount,
        blur_radius=blur_radius,
        contrast_restore=contrast_restore,
        highlight_bias=highlight_bias,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_soft_glow",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "blur_radius": blur_radius,
            "contrast_restore": contrast_restore,
            "highlight_bias": highlight_bias,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SOFT_GLOW_SPEC = ToolSpec(
    name="adjust_soft_glow",
    label="柔焦氛围",
    description="Apply restrained soft glow / focus-softening.",
    family="effects",
    focus_affinity=["finish"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.3, "blur_radius": 6.0, "contrast_restore": 0.2, "highlight_bias": 0.4, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_soft_glow, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在增强柔焦氛围",
    keywords=("柔焦", "soft glow", "梦幻"),
)


__all__ = ["ADJUST_SOFT_GLOW_SPEC", "adjust_soft_glow"]
