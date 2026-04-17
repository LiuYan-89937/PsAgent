"""Vibrance and saturation adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_vibrance_saturation_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustVibranceSaturationParams(PackageParamsModel):
    """Planner-fillable params for vibrance/saturation adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主色彩增强强度")
    vibrance_scale: float = Field(0.52, ge=0.1, le=1.5, description="vibrance 映射倍率")
    saturation_scale: float = Field(0.14, ge=0.0, le=1.0, description="saturation 映射倍率")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    protect_highlights: float = Field(0.26, ge=0.0, le=0.8, description="高光保护")
    protect_skin: float = Field(0.34, ge=0.0, le=0.8, description="肤色保护")
    protect_shadows: float = Field(0.24, ge=0.0, le=0.8, description="暗部保护")
    chroma_denoise: float = Field(0.34, ge=0.0, le=1.0, description="色度平滑强度")
    max_chroma: float = Field(92.0, ge=48.0, le=128.0, description="色度软上限")
    neutral_floor: float = Field(6.0, ge=0.0, le=24.0, description="中性色保护起点")
    neutral_softness: float = Field(14.0, ge=2.0, le=32.0, description="中性色保护过渡宽度")


class AdjustVibranceSaturationPackage(ToolPackage):
    """Minimal package for vibrance and saturation adjustments."""

    # 饱和度类操作后续适合支持主体/背景等局部区域，但第一阶段先跑全局。
    spec = PackageSpec(
        name="adjust_vibrance_saturation",
        description="Adjust vibrance or saturation globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "vibrance_scale": 0.52,
            "saturation_scale": 0.14,
            "feather_radius": 18.0,
            "protect_highlights": 0.26,
            "protect_skin": 0.34,
            "protect_shadows": 0.24,
            "chroma_denoise": 0.34,
            "max_chroma": 92.0,
            "neutral_floor": 6.0,
            "neutral_softness": 14.0,
        },
    )
    params_model = AdjustVibranceSaturationParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 色彩类操作也先做最基础的边界校验。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for vibrance/saturation adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # region 不是 whole_image 时，后续由 dispatcher 准备 mask。
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
        # 默认优先走 vibrance，让低饱和区域先起来；
        # saturation 只做保守补充，避免整体颜色冲得太猛。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustVibranceSaturationParams):
            raise ValueError("Vibrance/saturation params model is not configured.")
        clipped_strength = parsed.strength
        params = parsed.model_dump()
        vibrance_scale = parsed.vibrance_scale
        saturation_scale = parsed.saturation_scale
        feather_radius = parsed.feather_radius
        protect_highlights = parsed.protect_highlights
        protect_skin = parsed.protect_skin
        protect_shadows = parsed.protect_shadows
        chroma_denoise = parsed.chroma_denoise
        max_chroma = parsed.max_chroma
        neutral_floor = parsed.neutral_floor
        neutral_softness = parsed.neutral_softness

        vibrance_amount = clipped_strength * vibrance_scale
        saturation_amount = clipped_strength * saturation_scale

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "vibrance_scale": vibrance_scale,
            "saturation_scale": saturation_scale,
            "vibrance_amount": vibrance_amount,
            "saturation_amount": saturation_amount,
            "feather_radius": feather_radius,
            "protect_highlights": protect_highlights,
            "protect_skin": protect_skin,
            "protect_shadows": protect_shadows,
            "chroma_denoise": chroma_denoise,
            "max_chroma": max_chroma,
            "neutral_floor": neutral_floor,
            "neutral_softness": neutral_softness,
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

            output_path = tempfile.mktemp(prefix="psagent_vibrance_saturation_", suffix=".png")
            saved_path = apply_vibrance_saturation_adjustment(
                context.image_path or "",
                output_path,
                vibrance_amount=normalized["vibrance_amount"],
                saturation_amount=normalized["saturation_amount"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                protect_highlights=normalized["protect_highlights"],
                protect_skin=normalized["protect_skin"],
                protect_shadows=normalized["protect_shadows"],
                chroma_denoise=normalized["chroma_denoise"],
                max_chroma=normalized["max_chroma"],
                neutral_floor=normalized["neutral_floor"],
                neutral_softness=normalized["neutral_softness"],
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
