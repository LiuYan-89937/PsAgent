"""Native adjust_color_noise_reduction tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_denoise_adjustment


@tool
def adjust_color_noise_reduction(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    chroma_strength: Annotated[float, Field(default=10.0, ge=0.0, le=24.0)] = 10.0,
    detail_protection: Annotated[float, Field(default=0.35, ge=0.0, le=0.75)] = 0.35,
    template_window_size: Annotated[int, Field(default=7, ge=3, le=15)] = 7,
    search_window_size: Annotated[int, Field(default=21, ge=7, le=31)] = 21,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the image has colored speckles or chroma blotches, especially in darker areas, but edge detail should stay intact. It focuses more on color noise than luminance noise, so use it when the image looks dirty or blotchy rather than grainy in brightness only."""

    output_path = temp_output_path("psagent_color_noise_reduction_")
    saved_path = apply_denoise_adjustment(
        image_path,
        output_path,
        luma_strength=2.0,
        chroma_strength=chroma_strength,
        detail_protection=detail_protection,
        template_window_size=template_window_size,
        search_window_size=search_window_size,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_color_noise_reduction",
        output_image=saved_path,
        applied_params={
            "chroma_strength": chroma_strength,
            "detail_protection": detail_protection,
            "template_window_size": template_window_size,
            "search_window_size": search_window_size,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_COLOR_NOISE_REDUCTION_SPEC = ToolSpec(
    name="adjust_color_noise_reduction",
    label="色彩噪声抑制",
    description="Reduce chroma noise while keeping luminance detail stable.",
    family="detail",
    stage_affinity=["technical_prep", "subject_refine"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"chroma_strength": 10.0, "detail_protection": 0.35, "template_window_size": 7, "search_window_size": 21, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_color_noise_reduction, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="chroma_strength",
    risk_level="low",
    status_label="正在抑制色彩噪声",
    keywords=("色彩噪声", "彩噪", "color noise"),
)


__all__ = ["ADJUST_COLOR_NOISE_REDUCTION_SPEC", "adjust_color_noise_reduction"]
