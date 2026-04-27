"""Native adjust_exposure tool."""

from __future__ import annotations

import tempfile
from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.common.contracts import ToolExecutionResult, ToolSpec, build_planner_schema
from app.tools.common.mask_contracts import MASK_PARAMS_SCHEMA
from app.tools.image_ops import apply_exposure_adjustment


def _temp_output_path(prefix: str) -> str:
    """Build a temporary PNG output path for deterministic edits."""

    return tempfile.mktemp(prefix=prefix, suffix=".png")


@tool
def adjust_exposure(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.28,
            ge=0.0,
            le=1.0,
            description="Primary exposure push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.28,
    max_stops: Annotated[
        float,
        Field(
            default=1.4,
            ge=0.5,
            le=3.0,
            description="Maximum stop range used to map the primary strength into exposure gain.",
        ),
    ] = 1.4,
    feather_radius: Annotated[
        float,
        Field(
            default=18.0,
            ge=0.0,
            le=64.0,
            description="Mask feather radius in pixels when running on a local region.",
        ),
    ] = 18.0,
    mask_path: Annotated[str | None, Field(description="Optional runtime mask path.")] = None,
) -> dict:
    """Use this tool when the whole image or a masked region is clearly underexposed or overexposed and needs an exposure-style push. It changes overall light intensity with a stop-like feel, so it is better for true brightness/exposure correction than for delicate midtone shaping."""

    # 把抽象强度映射成曝光档位，再转成乘法增益。
    # 这样“0.2/0.5/0.8”的手感能跨工具保持一致。
    exposure_stops = float(strength) * float(max_stops)
    exposure_multiplier = 2 ** exposure_stops
    output_path = _temp_output_path("psagent_exposure_")
    saved_path = apply_exposure_adjustment(
        image_path,
        output_path,
        multiplier=exposure_multiplier,
        mask_path=mask_path,
        feather_radius=feather_radius,
    )
    return ToolExecutionResult(
        ok=True,
        tool="adjust_exposure",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "max_stops": max_stops,
            "feather_radius": feather_radius,
            "exposure_stops": exposure_stops,
            "exposure_multiplier": exposure_multiplier,
        },
        artifacts={
            "input_image": image_path,
            "mask_path": mask_path,
        },
    ).model_dump(mode="json")


ADJUST_EXPOSURE_SPEC = ToolSpec(
    # spec 是 planner/API/前端看到的元数据，不是执行逻辑。
    name="adjust_exposure",
    label="曝光",
    description="Adjust whole-image or masked exposure with stable strength semantics.",
    family="tone",
    focus_affinity=[
        "global_tone",
        "global_tone",
        "subject_separation",
        "subject_cleanup",
        "finish",
    ],
    supports_mask=True,
    supports_whole_image=True,
    default_params={
        "strength": 0.28,
        "max_stops": 1.4,
        "feather_radius": 18.0,
    },
    planner_schema=build_planner_schema(
        # planner schema 会显式剔除 runtime-only 的 image_path/mask_path，
        # 但会合并共享 mask schema，方便 planner 正常填 mask_*。
        adjust_exposure,
        supports_mask=True,
        mask_schema=MASK_PARAMS_SCHEMA,
        excluded_fields={"image_path", "mask_path"},
    ),
    primary_param="strength",
    risk_level="low",
    status_label="正在调整曝光",
    keywords=("曝光", "提亮", "压暗", "变亮", "变暗"),
)


__all__ = [
    "ADJUST_EXPOSURE_SPEC",
    "adjust_exposure",
]
