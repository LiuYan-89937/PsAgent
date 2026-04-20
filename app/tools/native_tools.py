"""Native LangChain tools used by the stage pipeline."""

from __future__ import annotations

import tempfile
from typing import Annotated

from langchain.tools import tool
from pydantic import Field

from app.tools.image_ops import (
    apply_contrast_adjustment,
    apply_exposure_adjustment,
    apply_vibrance_saturation_adjustment,
)
from app.tools.tool_specs import ToolExecutionResult


def _temp_output_path(prefix: str) -> str:
    """Build a temporary PNG output path for deterministic edits."""

    return tempfile.mktemp(prefix=prefix, suffix=".png")


@tool
def adjust_exposure(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Primary exposure push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.5,
    max_stops: Annotated[
        float,
        Field(
            default=2.0,
            ge=0.5,
            le=3.0,
            description="Maximum stop range used to map the primary strength into exposure gain.",
        ),
    ] = 2.0,
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
    """Adjust exposure on the provided image."""

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


@tool
def adjust_contrast(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Primary contrast push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.5,
    contrast_scale: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.25,
            le=1.5,
            description="Scaling factor that maps the primary strength into LAB luminance contrast.",
        ),
    ] = 1.0,
    pivot: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.25,
            le=0.75,
            description="Midtone pivot used by the contrast curve.",
        ),
    ] = 0.5,
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
    """Adjust contrast on the provided image."""

    contrast_amount = float(strength) * float(contrast_scale)
    output_path = _temp_output_path("psagent_contrast_")
    saved_path = apply_contrast_adjustment(
        image_path,
        output_path,
        contrast_amount=contrast_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
        pivot=pivot,
        protect_highlights=0.22,
        protect_shadows=0.22,
    )
    return ToolExecutionResult(
        ok=True,
        tool="adjust_contrast",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "contrast_scale": contrast_scale,
            "contrast_amount": contrast_amount,
            "pivot": pivot,
            "feather_radius": feather_radius,
        },
        artifacts={
            "input_image": image_path,
            "mask_path": mask_path,
        },
    ).model_dump(mode="json")


@tool
def adjust_vibrance_saturation(
    image_path: Annotated[str, Field(description="Runtime image path.")],
    strength: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Primary color push strength. 0.2=light visible, 0.5=obvious, 0.8=strong.",
        ),
    ] = 0.5,
    vibrance_scale: Annotated[
        float,
        Field(
            default=0.8,
            ge=0.2,
            le=1.2,
            description="Vibrance emphasis applied to lower-chroma colors first.",
        ),
    ] = 0.8,
    saturation_scale: Annotated[
        float,
        Field(
            default=0.28,
            ge=0.0,
            le=0.6,
            description="Additional global saturation lift layered on top of vibrance.",
        ),
    ] = 0.28,
    protect_highlights: Annotated[
        float,
        Field(
            default=0.26,
            ge=0.0,
            le=0.8,
            description="Highlight protection that restrains color boosts in bright areas.",
        ),
    ] = 0.26,
    protect_skin: Annotated[
        float,
        Field(
            default=0.34,
            ge=0.0,
            le=0.8,
            description="Skin-tone protection weight to avoid plastic or oversaturated skin.",
        ),
    ] = 0.34,
    protect_shadows: Annotated[
        float,
        Field(
            default=0.24,
            ge=0.0,
            le=0.8,
            description="Shadow protection weight to keep dark chroma noise under control.",
        ),
    ] = 0.24,
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
    """Adjust vibrance and saturation in color space on the provided image."""

    vibrance_amount = float(strength) * float(vibrance_scale)
    saturation_amount = float(strength) * float(saturation_scale)
    output_path = _temp_output_path("psagent_vibrance_saturation_")
    saved_path = apply_vibrance_saturation_adjustment(
        image_path,
        output_path,
        vibrance_amount=vibrance_amount,
        saturation_amount=saturation_amount,
        mask_path=mask_path,
        feather_radius=feather_radius,
        protect_highlights=protect_highlights,
        protect_skin=protect_skin,
        protect_shadows=protect_shadows,
    )
    return ToolExecutionResult(
        ok=True,
        tool="adjust_vibrance_saturation",
        output_image=saved_path,
        applied_params={
            "strength": strength,
            "vibrance_scale": vibrance_scale,
            "saturation_scale": saturation_scale,
            "vibrance_amount": vibrance_amount,
            "saturation_amount": saturation_amount,
            "protect_highlights": protect_highlights,
            "protect_skin": protect_skin,
            "protect_shadows": protect_shadows,
            "feather_radius": feather_radius,
        },
        artifacts={
            "input_image": image_path,
            "mask_path": mask_path,
        },
    ).model_dump(mode="json")


__all__ = [
    "adjust_contrast",
    "adjust_exposure",
    "adjust_vibrance_saturation",
]
