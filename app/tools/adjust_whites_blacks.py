"""Native adjust_whites_blacks tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_whites_blacks_adjustment


@tool
def adjust_whites_blacks(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    whites_amount: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="White point shift amount.")] = 0.0,
    blacks_amount: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0, description="Black point shift amount.")] = 0.0,
    highlight_rolloff: Annotated[float, Field(default=0.32, ge=0.08, le=0.95, description="Highlight rolloff.")] = 0.32,
    shadow_rolloff: Annotated[float, Field(default=0.34, ge=0.08, le=0.95, description="Shadow rolloff.")] = 0.34,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0, description="Mask feather radius.")] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want to anchor the endpoints of the tonal range by setting brighter whites or deeper blacks. It mainly affects the tonal extremes, so use it for cleaner black/white points and not for midtone brightness or broad color work."""

    output_path = temp_output_path("psagent_whites_blacks_")
    saved_path = apply_whites_blacks_adjustment(
        image_path,
        output_path,
        whites_amount=whites_amount,
        blacks_amount=blacks_amount,
        highlight_rolloff=highlight_rolloff,
        shadow_rolloff=shadow_rolloff,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_whites_blacks",
        output_image=saved_path,
        applied_params={
            "whites_amount": whites_amount,
            "blacks_amount": blacks_amount,
            "highlight_rolloff": highlight_rolloff,
            "shadow_rolloff": shadow_rolloff,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_WHITES_BLACKS_SPEC = ToolSpec(
    name="adjust_whites_blacks",
    label="白场黑场",
    description="Adjust white and black points on the luminance channel.",
    family="tone",
    focus_affinity=["global_tone", "subject_separation"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "whites_amount": 0.0,
        "blacks_amount": 0.0,
        "highlight_rolloff": 0.32,
        "shadow_rolloff": 0.34,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_whites_blacks,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="whites_amount",
    risk_level="low",
    status_label="正在调整白场和黑场",
    keywords=("白场", "黑场", "白位", "黑位"),
)


__all__ = ["ADJUST_WHITES_BLACKS_SPEC", "adjust_whites_blacks"]
