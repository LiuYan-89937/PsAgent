"""Native adjust_color_cleanup tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_cleanup_adjustment


@tool
def adjust_color_cleanup(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    yellow_reduce: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    green_reduce: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    magenta_balance: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    shadow_desaturate: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    highlight_neutralize: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when a local area has dirty yellow, green, magenta, or muddy color contamination and needs to look cleaner rather than more stylized. It is for local color cleanup and is not the right choice for broad creative grading across the full frame."""

    output_path = temp_output_path("psagent_color_cleanup_")
    saved_path = apply_color_cleanup_adjustment(
        image_path,
        output_path,
        yellow_reduce=yellow_reduce,
        green_reduce=green_reduce,
        magenta_balance=magenta_balance,
        shadow_desaturate=shadow_desaturate,
        highlight_neutralize=highlight_neutralize,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_cleanup",
        output_image=saved_path,
        applied_params={
            "yellow_reduce": yellow_reduce,
            "green_reduce": green_reduce,
            "magenta_balance": magenta_balance,
            "shadow_desaturate": shadow_desaturate,
            "highlight_neutralize": highlight_neutralize,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_CLEANUP_SPEC = ToolSpec(
    name="adjust_color_cleanup",
    label="局部色彩净化",
    description="Clean dirty yellow, green, and magenta casts with restrained chroma cleanup.",
    family="color",
    stage_affinity=["local_balance", "subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="face",
    default_params={
        "yellow_reduce": 0.0,
        "green_reduce": 0.0,
        "magenta_balance": 0.0,
        "shadow_desaturate": 0.0,
        "highlight_neutralize": 0.0,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_color_cleanup,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="yellow_reduce",
    risk_level="medium",
    status_label="正在净化局部色彩",
    keywords=("色彩净化", "去脏黄", "去脏绿", "去脏色"),
)


__all__ = ["ADJUST_COLOR_CLEANUP_SPEC", "adjust_color_cleanup"]
