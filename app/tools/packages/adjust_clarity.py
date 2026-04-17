"""Clarity adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_clarity_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustClarityParams(PackageParamsModel):
    """Planner-fillable params for clarity adjustment."""

    amount: float = Field(..., ge=-1.0, le=1.0, description="清晰度强度")
    radius_scale: float = Field(1.0, ge=0.5, le=3.0, description="局部对比半径倍率")
    highlight_protection: float = Field(0.22, ge=0.0, le=0.8, description="高光保护")
    shadow_protection: float = Field(0.22, ge=0.0, le=0.8, description="阴影保护")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustClarityPackage(ToolPackage):
    """Professional clarity package."""

    spec = PackageSpec(
        name="adjust_clarity",
        description="Enhance midtone local contrast globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={
            "radius_scale": 1.0,
            "highlight_protection": 0.22,
            "shadow_protection": 0.22,
            "feather_radius": 18.0,
        },
    )
    params_model = AdjustClarityParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for clarity adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustClarityParams):
            raise ValueError("Clarity params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "amount": parsed.amount,
            "radius_scale": parsed.radius_scale,
            "highlight_protection": parsed.highlight_protection,
            "shadow_protection": parsed.shadow_protection,
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

            output_path = tempfile.mktemp(prefix="psagent_clarity_", suffix=".png")
            saved_path = apply_clarity_adjustment(
                context.image_path or "",
                output_path,
                amount=normalized["amount"],
                radius_scale=normalized["radius_scale"],
                highlight_protection=normalized["highlight_protection"],
                shadow_protection=normalized["shadow_protection"],
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
