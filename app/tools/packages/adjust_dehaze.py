"""Dehaze adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_dehaze_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustDehazeParams(PackageParamsModel):
    """Planner-fillable params for dehaze adjustment."""

    amount: float = Field(..., ge=-1.0, le=1.0, description="去灰雾强度")
    luminance_protection: float = Field(0.26, ge=0.0, le=0.85, description="亮度保护")
    color_protection: float = Field(0.3, ge=0.0, le=0.9, description="颜色保护")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustDehazePackage(ToolPackage):
    """Professional dehaze package."""

    spec = PackageSpec(
        name="adjust_dehaze",
        description="Reduce haze and recover clarity globally or in the background.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["landscape", "general", "portrait"],
        risk_level="medium",
        default_params={
            "luminance_protection": 0.26,
            "color_protection": 0.3,
            "feather_radius": 18.0,
        },
    )
    params_model = AdjustDehazeParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for dehaze adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustDehazeParams):
            raise ValueError("Dehaze params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "amount": parsed.amount,
            "luminance_protection": parsed.luminance_protection,
            "color_protection": parsed.color_protection,
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

            output_path = tempfile.mktemp(prefix="psagent_dehaze_", suffix=".png")
            saved_path = apply_dehaze_adjustment(
                context.image_path or "",
                output_path,
                amount=normalized["amount"],
                luminance_protection=normalized["luminance_protection"],
                color_protection=normalized["color_protection"],
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
