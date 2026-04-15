"""White balance adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_white_balance_adjustment
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class AdjustWhiteBalancePackage(ToolPackage):
    """Minimal package for white-balance adjustments."""

    # 白平衡第一阶段先做 whole_image，后续再评估局部模式的稳定性。
    spec = PackageSpec(
        name="adjust_white_balance",
        description="Adjust color temperature and tint globally or by region.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "temperature_scale": 12.0,
            "tint_scale": 8.0,
            "tint_bias": 0.0,
            "feather_radius": 18.0,
            "protect_saturated": 0.3,
        },
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # 给 planner 的导出始终保持简洁。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 白平衡也先做最基础的边界校验。
        region = operation.get("region") or "whole_image"
        strength = operation.get("strength", 0.0)

        if region not in self.spec.supported_regions:
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        if not -1.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between -1.0 and 1.0")
        if not context.image_path:
            raise ValueError("image_path is required for white-balance adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前统一按 region 是否为 whole_image 判断局部依赖。
        region = operation.get("region") or "whole_image"
        return {
            "requires_mask": region != "whole_image",
            "required_region": None if region == "whole_image" else region,
        }

    def normalize(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 默认 strength 只负责冷暖方向：
        # 正值更暖，负值更冷。tint 先作为可选附加参数暴露给后续 planner。
        raw_strength = float(operation.get("strength", 0.0))
        clipped_strength = max(-1.0, min(1.0, raw_strength))
        params = dict(self.spec.default_params)
        params.update(operation.get("params", {}))

        temperature_scale = float(
            params.get("temperature_scale", self.spec.default_params["temperature_scale"])
        )
        temperature_scale = max(2.0, min(24.0, temperature_scale))
        tint_scale = float(params.get("tint_scale", self.spec.default_params["tint_scale"]))
        tint_scale = max(0.0, min(18.0, tint_scale))
        tint_bias = float(params.get("tint_bias", self.spec.default_params["tint_bias"]))
        tint_bias = max(-1.0, min(1.0, tint_bias))
        feather_radius = float(params.get("feather_radius", self.spec.default_params["feather_radius"]))
        feather_radius = max(0.0, min(64.0, feather_radius))
        protect_saturated = float(
            params.get("protect_saturated", self.spec.default_params["protect_saturated"])
        )
        protect_saturated = max(0.0, min(0.85, protect_saturated))

        temperature_shift = clipped_strength * temperature_scale
        tint_shift = tint_bias * tint_scale

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "temperature_scale": temperature_scale,
            "tint_scale": tint_scale,
            "tint_bias": tint_bias,
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
