"""Exposure adjustment package skeleton."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.tools.image_ops import apply_exposure_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustExposureParams(PackageParamsModel):
    """Planner-fillable params for exposure adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主曝光强度，正值提亮，负值压暗")
    max_stops: float = Field(1.5, ge=0.25, le=3.0, description="最大曝光档位")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustExposurePackage(ToolPackage):
    """Skeleton package for exposure adjustments."""

    # 曝光既能全局调，也能在后续阶段支持局部区域。
    spec = PackageSpec(
        name="adjust_exposure",
        description="Adjust global or regional exposure in a controlled way.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"max_stops": 1.5, "feather_radius": 18.0},
    )
    params_model = AdjustExposureParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 先做最基础的边界校验，保证第一个包具备最小可用性。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for exposure adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # whole_image 直接执行；局部 region 后续由 dispatcher 统一补 mask。
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
        # 后续这里负责把抽象 strength 映射成真实内部参数。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustExposureParams):
            raise ValueError("Exposure params model is not configured.")

        clipped_strength = parsed.strength
        params = parsed.model_dump()

        # CV / 图像处理知识：
        # 这里把抽象强度映射成“曝光档位(stops)”。
        # 1 stop 约等于亮度乘以 2，-1 stop 约等于亮度乘以 0.5。
        # 因此这里用 2 ** stops 把抽象强度转成像素乘法系数。
        # 这是一种很适合 MVP 的曝光近似模型，直观、稳定、可测试。
        #
        # max_stops 控制最大曝光变化档位，保持在保守范围内。
        max_stops = parsed.max_stops
        exposure_multiplier = 2 ** (clipped_strength * max_stops)
        feather_radius = parsed.feather_radius

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
