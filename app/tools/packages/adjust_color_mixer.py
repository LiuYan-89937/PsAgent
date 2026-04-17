"""Color mixer (HSL-like) adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_color_mixer_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustColorMixerParams(PackageParamsModel):
    """Planner-fillable params for Color Mixer / HSL style adjustment."""

    red_hue: float = Field(0.0, ge=-1.0, le=1.0)
    red_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    red_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    orange_hue: float = Field(0.0, ge=-1.0, le=1.0)
    orange_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    orange_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    yellow_hue: float = Field(0.0, ge=-1.0, le=1.0)
    yellow_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    yellow_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    green_hue: float = Field(0.0, ge=-1.0, le=1.0)
    green_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    green_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    aqua_hue: float = Field(0.0, ge=-1.0, le=1.0)
    aqua_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    aqua_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    blue_hue: float = Field(0.0, ge=-1.0, le=1.0)
    blue_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    blue_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    purple_hue: float = Field(0.0, ge=-1.0, le=1.0)
    purple_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    purple_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    magenta_hue: float = Field(0.0, ge=-1.0, le=1.0)
    magenta_saturation: float = Field(0.0, ge=-1.0, le=1.0)
    magenta_luminance: float = Field(0.0, ge=-1.0, le=1.0)
    saturation_protection: float = Field(0.3, ge=0.0, le=0.85)
    luminance_protection: float = Field(0.22, ge=0.0, le=0.85)
    feather_radius: float = Field(18.0, ge=0.0, le=64.0)


class AdjustColorMixerPackage(ToolPackage):
    """Professional Color Mixer / HSL package."""

    spec = PackageSpec(
        name="adjust_color_mixer",
        description="Adjust hue, saturation and luminance by color group.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={
            "saturation_protection": 0.3,
            "luminance_protection": 0.22,
            "feather_radius": 18.0,
        },
    )
    params_model = AdjustColorMixerParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for color mixer adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustColorMixerParams):
            raise ValueError("Color mixer params model is not configured.")

        raw_params = parsed.model_dump()
        channel_settings: dict[str, dict[str, float]] = {}
        for name in ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"):
            channel_settings[name] = {
                "hue_shift_deg": raw_params[f"{name}_hue"] * 12.0,
                "saturation_shift": raw_params[f"{name}_saturation"] * 0.28,
                "luminance_shift": raw_params[f"{name}_luminance"] * 0.16,
            }

        return {
            "region": operation.get("region") or "whole_image",
            "params": raw_params,
            "channel_settings": channel_settings,
            "saturation_protection": parsed.saturation_protection,
            "luminance_protection": parsed.luminance_protection,
            "feather_radius": parsed.feather_radius,
        }

    def execute(self, operation: dict[str, Any], context: OperationContext) -> PackageResult:
        try:
            self.validate(operation, context)
            normalized = self.normalize(operation, context)
            requirements = self.resolve_requirements(operation, context)

            mask_path: str | None = None
            if requirements["requires_mask"]:
                required_region = requirements["required_region"]
                mask_path = context.masks.get(required_region) if required_region else None
                if not mask_path:
                    raise ValueError(f"Mask is required for region: {required_region}")

            output_path = tempfile.mktemp(prefix="psagent_color_mixer_", suffix=".png")
            saved_path = apply_color_mixer_adjustment(
                context.image_path or "",
                output_path,
                channel_settings=normalized["channel_settings"],
                saturation_protection=normalized["saturation_protection"],
                luminance_protection=normalized["luminance_protection"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
            )

            return PackageResult(
                ok=True,
                package=self.name,
                output_image=saved_path,
                applied_params=normalized,
                artifacts={
                    "input_image": context.image_path,
                    "mask_path": mask_path,
                    "requirements": requirements,
                },
            )
        except Exception as error:  # pragma: no cover
            return self.fallback(error, operation, context)
