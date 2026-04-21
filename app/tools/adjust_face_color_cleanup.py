"""Native adjust_face_color_cleanup tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_color_cleanup_adjustment


@tool
def adjust_face_color_cleanup(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    yellow_reduce: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1,
    magenta_balance: Annotated[float, Field(default=0.05, ge=0.0, le=1.0)] = 0.05,
    green_reduce: Annotated[float, Field(default=0.0, ge=0.0, le=1.0)] = 0.0,
    shadow_desaturate: Annotated[float, Field(default=0.08, ge=0.0, le=1.0)] = 0.08,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the face specifically has dirty yellow, green, magenta, or muddy color contamination and needs cleaner skin tone rendering. It is a face-region cleanup tool and should be paired with a face mask, not used globally."""

    output_path = temp_output_path("psagent_face_color_cleanup_")
    saved_path = apply_color_cleanup_adjustment(
        image_path,
        output_path,
        yellow_reduce=yellow_reduce,
        green_reduce=green_reduce,
        magenta_balance=magenta_balance,
        shadow_desaturate=shadow_desaturate,
        highlight_neutralize=0.1,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_face_color_cleanup",
        output_image=saved_path,
        applied_params={
            "yellow_reduce": yellow_reduce,
            "magenta_balance": magenta_balance,
            "green_reduce": green_reduce,
            "shadow_desaturate": shadow_desaturate,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_FACE_COLOR_CLEANUP_SPEC = ToolSpec(
    name="adjust_face_color_cleanup",
    label="面部脏色清理",
    description="Clean dirty yellow, green, and magenta casts in the face region.",
    family="portrait",
    stage_affinity=["subject_refine"],
    supports_mask=True,
    requires_mask=True,
    supports_whole_image=False,
    recommended_mask_prompt="face",
    default_params={
        "yellow_reduce": 0.1,
        "magenta_balance": 0.05,
        "green_reduce": 0.0,
        "shadow_desaturate": 0.08,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_face_color_cleanup,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="yellow_reduce",
    risk_level="medium",
    status_label="正在清理面部脏色",
    keywords=("面部脏色", "脸部净化", "去黄去脏"),
)


__all__ = ["ADJUST_FACE_COLOR_CLEANUP_SPEC", "adjust_face_color_cleanup"]
