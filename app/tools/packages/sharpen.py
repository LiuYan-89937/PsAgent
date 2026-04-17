"""Sharpen package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_sharpen_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class SharpenParams(PackageParamsModel):
    """Planner-fillable params for sharpen adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主锐化强度，建议使用非负值")
    amount_scale: float = Field(1.1, ge=0.2, le=2.4, description="锐化量映射倍率")
    radius_scale: float = Field(1.4, ge=0.4, le=4.0, description="锐化半径映射倍率")
    threshold_scale: float = Field(0.018, ge=0.0, le=0.08, description="锐化阈值映射倍率")
    feather_radius: float = Field(12.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    highlight_protection: float = Field(0.24, ge=0.0, le=0.85, description="高光保护强度")


class SharpenPackage(ToolPackage):
    """Minimal package for sharpening operations."""

    # 锐化和去噪类似，局部模式后续会比全局模式更保守。
    spec = PackageSpec(
        name="sharpen",
        description="Increase detail or sharpness globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "amount_scale": 1.1,
            "radius_scale": 1.4,
            "threshold_scale": 0.018,
            "feather_radius": 12.0,
            "highlight_protection": 0.24,
        },
    )
    params_model = SharpenParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 锐化先做统一边界校验；局部风险在归一化阶段收紧。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for sharpen adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 非 whole_image 时后续统一走 dispatcher 补 mask。
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 锐化只接受正向值；局部模式自动更保守，降低噪声被一起拉起来的风险。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, SharpenParams):
            raise ValueError("Sharpen params model is not configured.")
        clipped_strength = parsed.strength
        params = parsed.model_dump()

        region = operation.get("region") or "whole_image"
        local_mode = self.operation_requires_mask(operation, context)
        region_scale = 0.72 if local_mode else 1.0

        amount_scale = parsed.amount_scale
        radius_scale = parsed.radius_scale
        threshold_scale = parsed.threshold_scale
        feather_radius = parsed.feather_radius
        highlight_protection = parsed.highlight_protection

        sharpen_strength = max(0.0, clipped_strength)
        amount = sharpen_strength * amount_scale * region_scale
        radius = 0.6 + sharpen_strength * radius_scale * 0.9
        threshold = sharpen_strength * threshold_scale + (0.006 if local_mode else 0.0)

        return {
            "region": region,
            "strength": clipped_strength,
            "params": params,
            "sharpen_strength": sharpen_strength,
            "amount_scale": amount_scale,
            "radius_scale": radius_scale,
            "threshold_scale": threshold_scale,
            "amount": amount,
            "radius": radius,
            "threshold": threshold,
            "feather_radius": feather_radius,
            "highlight_protection": highlight_protection,
        }

    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
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

            output_path = tempfile.mktemp(prefix="psagent_sharpen_", suffix=".png")
            saved_path = apply_sharpen_adjustment(
                context.image_path or "",
                output_path,
                amount=normalized["amount"],
                radius=normalized["radius"],
                threshold=normalized["threshold"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                highlight_protection=normalized["highlight_protection"],
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
        except Exception as error:  # pragma: no cover - fallback path is verified via result state
            return self.fallback(error, operation, context)
