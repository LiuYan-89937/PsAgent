"""Curves adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_curves_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustCurvesParams(PackageParamsModel):
    """Planner-fillable params for a restrained curves adjustment."""

    shadow_lift: float = Field(0.0, ge=-1.0, le=1.0, description="阴影提拉或压暗强度")
    midtone_gamma: float = Field(1.0, ge=0.75, le=1.35, description="中间调 gamma")
    highlight_compress: float = Field(0.0, ge=-1.0, le=1.0, description="高光压缩或展开强度")
    contrast_bias: float = Field(0.0, ge=-1.0, le=1.0, description="整体 S 曲线偏置")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustCurvesPackage(ToolPackage):
    """Professional first-pass curves adjustment."""

    spec = PackageSpec(
        name="adjust_curves",
        description="Shape tone with a restrained curves model globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"feather_radius": 18.0},
    )
    params_model = AdjustCurvesParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for curves adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustCurvesParams):
            raise ValueError("Curves params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "shadow_lift": parsed.shadow_lift,
            "midtone_gamma": parsed.midtone_gamma,
            "highlight_compress": parsed.highlight_compress,
            "contrast_bias": parsed.contrast_bias,
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

            output_path = tempfile.mktemp(prefix="psagent_curves_", suffix=".png")
            saved_path = apply_curves_adjustment(
                context.image_path or "",
                output_path,
                shadow_lift=normalized["shadow_lift"],
                midtone_gamma=normalized["midtone_gamma"],
                highlight_compress=normalized["highlight_compress"],
                contrast_bias=normalized["contrast_bias"],
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
