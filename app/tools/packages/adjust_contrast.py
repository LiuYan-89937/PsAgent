"""Contrast adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_contrast_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustContrastParams(PackageParamsModel):
    """Planner-fillable params for contrast adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主对比度强度，正值增强，负值减弱")
    contrast_scale: float = Field(0.7, ge=0.1, le=1.5, description="主对比度映射倍率")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    pivot: float = Field(0.5, ge=0.25, le=0.75, description="中灰锚点")
    protect_highlights: float = Field(0.22, ge=0.0, le=0.8, description="高光保护强度")
    protect_shadows: float = Field(0.22, ge=0.0, le=0.8, description="阴影保护强度")


class AdjustContrastPackage(ToolPackage):
    """Minimal professional contrast package."""

    # 对比度后续也可以扩展到主体/背景等局部区域。
    spec = PackageSpec(
        name="adjust_contrast",
        description="Adjust contrast globally or for selected regions.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "contrast_scale": 0.7,
            "feather_radius": 18.0,
            "pivot": 0.5,
            "protect_highlights": 0.22,
            "protect_shadows": 0.22,
        },
    )
    params_model = AdjustContrastParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 对比度和曝光一样，需要先做最基础的边界限制。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for contrast adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # dispatcher 会根据这里的声明去准备前置依赖。
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
        # 最小实现里把一个抽象 strength 映射成围绕中灰点的对比度伸缩量。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustContrastParams):
            raise ValueError("Contrast params model is not configured.")

        clipped_strength = parsed.strength
        params = parsed.model_dump()
        contrast_scale = parsed.contrast_scale
        feather_radius = parsed.feather_radius
        pivot = parsed.pivot
        protect_highlights = parsed.protect_highlights
        protect_shadows = parsed.protect_shadows
        contrast_amount = clipped_strength * contrast_scale
        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "contrast_scale": contrast_scale,
            "contrast_amount": contrast_amount,
            "feather_radius": feather_radius,
            "pivot": pivot,
            "protect_highlights": protect_highlights,
            "protect_shadows": protect_shadows,
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

            output_path = tempfile.mktemp(prefix="psagent_contrast_", suffix=".png")
            saved_path = apply_contrast_adjustment(
                context.image_path or "",
                output_path,
                contrast_amount=normalized["contrast_amount"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                pivot=normalized["pivot"],
                protect_highlights=normalized["protect_highlights"],
                protect_shadows=normalized["protect_shadows"],
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
