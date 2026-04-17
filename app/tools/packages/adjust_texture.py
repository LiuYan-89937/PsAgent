"""Texture adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_texture_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustTextureParams(PackageParamsModel):
    """Planner-fillable params for texture adjustment."""

    amount: float = Field(..., ge=-1.0, le=1.0, description="纹理增强强度")
    detail_scale: float = Field(1.0, ge=0.6, le=2.8, description="纹理尺度倍率")
    noise_protection: float = Field(0.4, ge=0.0, le=0.9, description="噪点保护")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustTexturePackage(ToolPackage):
    """Professional texture package."""

    spec = PackageSpec(
        name="adjust_texture",
        description="Enhance or soften medium-scale texture globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={
            "detail_scale": 1.0,
            "noise_protection": 0.4,
            "feather_radius": 18.0,
        },
    )
    params_model = AdjustTextureParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for texture adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustTextureParams):
            raise ValueError("Texture params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "amount": parsed.amount,
            "detail_scale": parsed.detail_scale,
            "noise_protection": parsed.noise_protection,
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

            output_path = tempfile.mktemp(prefix="psagent_texture_", suffix=".png")
            saved_path = apply_texture_adjustment(
                context.image_path or "",
                output_path,
                amount=normalized["amount"],
                detail_scale=normalized["detail_scale"],
                noise_protection=normalized["noise_protection"],
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
