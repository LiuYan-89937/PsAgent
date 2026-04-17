"""White balance adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_white_balance_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustWhiteBalanceParams(PackageParamsModel):
    """Planner-fillable params for white-balance adjustment."""

    strength: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="主冷暖方向，正值更暖，负值更冷",
    )
    temperature_scale: float = Field(12.0, ge=2.0, le=24.0, description="温度映射尺度")
    tint: float = Field(0.0, ge=-1.0, le=1.0, description="绿-洋红偏移")
    tint_scale: float = Field(8.0, ge=0.0, le=18.0, description="色调映射尺度")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    protect_saturated: float = Field(0.3, ge=0.0, le=0.85, description="高饱和颜色保护")


class AdjustWhiteBalancePackage(ToolPackage):
    """Minimal package for white-balance adjustments."""

    # 白平衡第一阶段先做 whole_image，后续再评估局部模式的稳定性。
    spec = PackageSpec(
        name="adjust_white_balance",
        description="Adjust color temperature and tint globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "temperature_scale": 12.0,
            "tint_scale": 8.0,
            "tint": 0.0,
            "feather_radius": 18.0,
            "protect_saturated": 0.3,
        },
    )
    params_model = AdjustWhiteBalanceParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 白平衡也先做最基础的边界校验。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for white-balance adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前统一按 region 是否为 whole_image 判断局部依赖。
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
        # 默认 strength 只负责冷暖方向：
        # 正值更暖，负值更冷。tint 先作为可选附加参数暴露给后续 planner。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustWhiteBalanceParams):
            raise ValueError("White-balance params model is not configured.")

        clipped_strength = parsed.strength
        params = parsed.model_dump()
        temperature_scale = parsed.temperature_scale
        tint_scale = parsed.tint_scale
        feather_radius = parsed.feather_radius
        protect_saturated = parsed.protect_saturated
        temperature_shift = clipped_strength * temperature_scale
        tint_shift = parsed.tint * tint_scale

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "temperature_scale": temperature_scale,
            "tint_scale": tint_scale,
            "tint": parsed.tint,
            "temperature_shift": temperature_shift,
            "tint_shift": tint_shift,
            "feather_radius": feather_radius,
            "protect_saturated": protect_saturated,
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

            output_path = tempfile.mktemp(prefix="psagent_white_balance_", suffix=".png")
            saved_path = apply_white_balance_adjustment(
                context.image_path or "",
                output_path,
                temperature_shift=normalized["temperature_shift"],
                tint_shift=normalized["tint_shift"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                protect_saturated=normalized["protect_saturated"],
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
