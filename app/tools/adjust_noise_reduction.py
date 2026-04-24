"""Native adjust_noise_reduction tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_denoise_adjustment


@tool
def adjust_noise_reduction(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    luma_strength: Annotated[float, Field(default=6.0, ge=0.0, le=24.0)] = 6.0,
    chroma_strength: Annotated[float, Field(default=6.0, ge=0.0, le=24.0)] = 6.0,
    detail_protection: Annotated[float, Field(default=0.22, ge=0.0, le=0.75)] = 0.22,
    template_window_size: Annotated[int, Field(default=7, ge=3, le=15)] = 7,
    search_window_size: Annotated[int, Field(default=21, ge=7, le=31)] = 21,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when luminance or color noise is distracting and the image needs to look cleaner before or after tonal work. It reduces noise while trying to preserve detail, so it is better for cleanup than for softening skin or lowering local contrast on purpose."""

    output_path = temp_output_path("psagent_noise_reduction_")
    saved_path = apply_denoise_adjustment(
        image_path,
        output_path,
        luma_strength=luma_strength,
        chroma_strength=chroma_strength,
        detail_protection=detail_protection,
        template_window_size=template_window_size,
        search_window_size=search_window_size,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_noise_reduction",
        output_image=saved_path,
        applied_params={
            "luma_strength": luma_strength,
            "chroma_strength": chroma_strength,
            "detail_protection": detail_protection,
            "template_window_size": template_window_size,
            "search_window_size": search_window_size,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_NOISE_REDUCTION_SPEC = ToolSpec(
    name="adjust_noise_reduction",
    label="降噪",
    description="Apply non-local means noise reduction with detail protection.",
    family="detail",
    focus_affinity=["global_tone", "subject_cleanup"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"luma_strength": 6.0, "chroma_strength": 6.0, "detail_protection": 0.22, "template_window_size": 7, "search_window_size": 21, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_noise_reduction, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="luma_strength",
    risk_level="low",
    status_label="正在降噪",
    keywords=("降噪", "去噪", "noise reduction"),
)


__all__ = ["ADJUST_NOISE_REDUCTION_SPEC", "adjust_noise_reduction"]
