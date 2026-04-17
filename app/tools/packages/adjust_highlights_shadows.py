"""Highlights/shadows adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_highlights_shadows_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustHighlightsShadowsParams(PackageParamsModel):
    """Planner-fillable params for highlights/shadows adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主层次平衡强度")
    tone_amount: float = Field(0.26, ge=0.05, le=0.9, description="整体层次调整量")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    midtone_contrast: float = Field(0.12, ge=0.0, le=0.5, description="中间调对比补偿")
    local_radius: float = Field(36.0, ge=4.0, le=160.0, description="局部亮度估计半径")
    shadow_tonal_width: float = Field(0.42, ge=0.1, le=0.8, description="阴影影响范围")
    highlight_tonal_width: float = Field(0.38, ge=0.1, le=0.8, description="高光影响范围")
    detail_amount: float = Field(0.32, ge=0.0, le=1.0, description="细节回灌强度")
    highlight_balance: float = Field(0.6, ge=0.2, le=1.2, description="高光侧权重")


class AdjustHighlightsShadowsPackage(ToolPackage):
    """Minimal package for highlight and shadow balancing."""

    # 高光/阴影和曝光类似，天然支持全局与局部两种模式。
    spec = PackageSpec(
        name="adjust_highlights_shadows",
        description="Balance highlights and shadows globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "tone_amount": 0.26,
            "feather_radius": 18.0,
            "midtone_contrast": 0.12,
            "local_radius": 36.0,
            "shadow_tonal_width": 0.42,
            "highlight_tonal_width": 0.38,
            "detail_amount": 0.32,
            "highlight_balance": 0.6,
        },
    )
    params_model = AdjustHighlightsShadowsParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 先做最基础的边界校验，保证最小实现可控。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for highlights/shadows adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前先按 region 是否为 whole_image 来判断要不要 mask。
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
        # 专业版里除了基本强度，还要把局部估计半径、作用 tonal width、
        # 细节回灌量等一起纳入内部参数，才能更接近真实修图软件的手感。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustHighlightsShadowsParams):
            raise ValueError("Highlights/shadows params model is not configured.")
        clipped_strength = parsed.strength
        params = parsed.model_dump()
        tone_amount = parsed.tone_amount
        feather_radius = parsed.feather_radius
        midtone_contrast = parsed.midtone_contrast
        local_radius = parsed.local_radius
        shadow_tonal_width = parsed.shadow_tonal_width
        highlight_tonal_width = parsed.highlight_tonal_width
        detail_amount = parsed.detail_amount
        highlight_balance = parsed.highlight_balance

        shadow_amount = clipped_strength * tone_amount
        highlight_amount = clipped_strength * tone_amount * highlight_balance

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "tone_amount": tone_amount,
            "shadow_amount": shadow_amount,
            "highlight_amount": highlight_amount,
            "feather_radius": feather_radius,
            "midtone_contrast": midtone_contrast,
            "local_radius": local_radius,
            "shadow_tonal_width": shadow_tonal_width,
            "highlight_tonal_width": highlight_tonal_width,
            "detail_amount": detail_amount,
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

            output_path = tempfile.mktemp(prefix="psagent_highlights_shadows_", suffix=".png")
            saved_path = apply_highlights_shadows_adjustment(
                context.image_path or "",
                output_path,
                shadow_amount=normalized["shadow_amount"],
                highlight_amount=normalized["highlight_amount"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                midtone_contrast=normalized["midtone_contrast"],
                local_radius=normalized["local_radius"],
                shadow_tonal_width=normalized["shadow_tonal_width"],
                highlight_tonal_width=normalized["highlight_tonal_width"],
                detail_amount=normalized["detail_amount"],
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
