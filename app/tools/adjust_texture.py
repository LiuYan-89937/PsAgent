"""Native adjust_texture tool."""

from __future__ import annotations

from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common import MASK_PARAMS_SCHEMA, ToolSpec, build_planner_schema, build_result, temp_output_path
from app.tools.image_ops import apply_texture_adjustment


@tool
def adjust_texture(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    amount: Annotated[float, Field(default=0.22, ge=-1.0, le=1.0)] = 0.22,
    detail_scale: Annotated[float, Field(default=1.0, ge=0.2, le=3.0)] = 1.0,
    noise_protection: Annotated[float, Field(default=0.52, ge=0.0, le=1.0)] = 0.52,
    feather_radius: Annotated[float, Field(default=18.0, ge=0.0, le=64.0)] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when fine surface detail needs to feel stronger or softer without changing overall exposure too much. It targets smaller texture-scale detail than clarity, so it is useful for skin texture, fabric, hair detail, and small surface structure."""

    output_path = temp_output_path("psagent_texture_")
    saved_path = apply_texture_adjustment(
        image_path,
        output_path,
        amount=amount,
        detail_scale=detail_scale,
        noise_protection=noise_protection,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return build_result(
        tool_name="adjust_texture",
        output_image=saved_path,
        applied_params={
            "amount": amount,
            "detail_scale": detail_scale,
            "noise_protection": noise_protection,
            "feather_radius": feather_radius,
        },
        image_path=image_path,
        mask_path=mask_path,
    )


ADJUST_TEXTURE_SPEC = ToolSpec(
    name="adjust_texture",
    label="纹理",
    description="Adjust high-frequency texture detail.",
    family="detail",
    focus_affinity=["subject_cleanup"],
    supports_mask=True,
    supports_whole_image=True,
    default_params={"amount": 0.22, "detail_scale": 1.0, "noise_protection": 0.52, "feather_radius": 18.0},
    planner_schema=build_planner_schema(adjust_texture, supports_mask=True, mask_schema=MASK_PARAMS_SCHEMA, excluded_fields={"image_path", "mask_path"}),
    primary_param="amount",
    risk_level="low",
    status_label="正在调整纹理",
    keywords=("纹理", "皮肤纹理", "材质"),
)


__all__ = ["ADJUST_TEXTURE_SPEC", "adjust_texture"]
