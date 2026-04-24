"""Native adjust_selective_color tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_selective_color_adjustment


@tool
def adjust_selective_color(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    target_band: Annotated[str, Field(default="neutrals", description="Target hue or neutral band.")] = "neutrals",
    cyan_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    magenta_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    yellow_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    black_shift: Annotated[float, Field(default=0.0, ge=-1.0, le=1.0)] = 0.0,
    relative_mode: Annotated[bool, Field(default=True)] = True,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when you want Photoshop-style selective color behavior, especially for removing yellow, green, magenta, or dirty color contamination inside a chosen color class. It is better for component-style color cleanup than for bold creative hue shifts."""

    output_path = temp_output_path("psagent_selective_color_")
    saved_path = apply_selective_color_adjustment(
        image_path,
        output_path,
        target_band=target_band,
        cyan_shift=cyan_shift,
        magenta_shift=magenta_shift,
        yellow_shift=yellow_shift,
        black_shift=black_shift,
        relative_mode=relative_mode,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_selective_color",
        output_image=saved_path,
        applied_params={
            "target_band": target_band,
            "cyan_shift": cyan_shift,
            "magenta_shift": magenta_shift,
            "yellow_shift": yellow_shift,
            "black_shift": black_shift,
            "relative_mode": relative_mode,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_SELECTIVE_COLOR_SPEC = ToolSpec(
    name="adjust_selective_color",
    label="选择性颜色",
    description="Adjust CMYK-like shifts on a selected color or neutral band.",
    family="color",
    focus_affinity=["subject_separation", "subject_cleanup", "finish"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "target_band": "neutrals",
        "cyan_shift": 0.0,
        "magenta_shift": 0.0,
        "yellow_shift": 0.0,
        "black_shift": 0.0,
        "relative_mode": True,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        adjust_selective_color,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="yellow_shift",
    risk_level="medium",
    status_label="正在调整选择性颜色",
    keywords=("选择性颜色", "白裙去黄", "中性色净化"),
)


__all__ = ["ADJUST_SELECTIVE_COLOR_SPEC", "adjust_selective_color"]
