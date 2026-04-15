"""Exposure adjustment package skeleton."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.tools.image_ops import apply_exposure_adjustment
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class AdjustExposurePackage(ToolPackage):
    """Skeleton package for exposure adjustments."""

    # 曝光既能全局调，也能在后续阶段支持局部区域。
    spec = PackageSpec(
        name="adjust_exposure",
        description="Adjust global or regional exposure in a controlled way.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"max_stops": 1.5, "feather_radius": 18.0},
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # 给 planner 暴露最小能力集合，避免把底层实现细节泄漏给模型。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 先做最基础的边界校验，保证第一个包具备最小可用性。
        region = operation.get("region") or "whole_image"
        strength = operation.get("strength", 0.0)

        if region not in self.spec.supported_regions:
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        if not -1.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between -1.0 and 1.0")
        if not context.image_path:
            raise ValueError("image_path is required for exposure adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # whole_image 直接执行；局部 region 后续由 dispatcher 统一补 mask。
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
        # 后续这里负责把抽象 strength 映射成真实内部参数。
        raw_strength = float(operation.get("strength", 0.0))
        clipped_strength = max(-1.0, min(1.0, raw_strength))
        params = dict(self.spec.default_params)
        params.update(operation.get("params", {}))

        # CV / 图像处理知识：
        # 这里把抽象强度映射成“曝光档位(stops)”。
        # 1 stop 约等于亮度乘以 2，-1 stop 约等于亮度乘以 0.5。
        # 因此这里用 2 ** stops 把抽象强度转成像素乘法系数。
        # 这是一种很适合 MVP 的曝光近似模型，直观、稳定、可测试。
        #
        # max_stops 控制最大曝光变化档位，保持在保守范围内。
        max_stops = float(params.get("max_stops", self.spec.default_params["max_stops"]))
        max_stops = max(0.25, min(3.0, max_stops))
        exposure_multiplier = 2 ** (clipped_strength * max_stops)
        feather_radius = float(params.get("feather_radius", self.spec.default_params["feather_radius"]))
        feather_radius = max(0.0, min(64.0, feather_radius))

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "max_stops": max_stops,
            "exposure_multiplier": exposure_multiplier,
            "feather_radius": feather_radius,
        }

    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        # 第一个包直接做最小真实落地：支持 whole_image 与 mask 局部模式。
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

            output_path = tempfile.mktemp(prefix="psagent_exposure_", suffix=".png")
            # 这里真正进入底层图像处理：
            # 传入原图、归一化后的曝光乘法系数，以及可选 mask。
            saved_path = apply_exposure_adjustment(
                context.image_path or "",
                output_path,
                multiplier=normalized["exposure_multiplier"],
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
        except Exception as error:  # pragma: no cover - fallback path is tested via result state
            return self.fallback(error, operation, context)
