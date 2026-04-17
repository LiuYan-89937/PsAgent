"""Crop and straighten package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_crop_and_straighten
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class CropAndStraightenParams(PackageParamsModel):
    """Planner-fillable params for crop/straighten adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主裁剪强度，负值将被裁到 0")
    max_crop_ratio: float = Field(0.16, ge=0.02, le=0.35, description="最大裁剪比例")
    max_straighten_angle: float = Field(8.0, ge=1.0, le=15.0, description="最大拉直角度")
    straighten_bias: float = Field(0.0, ge=-1.0, le=1.0, description="拉直方向偏置")
    crop_guard: float = Field(0.04, ge=0.0, le=0.12, description="黑边安全边距")
    min_scale: float = Field(0.72, ge=0.45, le=1.0, description="最小保留比例")


class CropAndStraightenPackage(ToolPackage):
    """Minimal package for crop and straighten operations."""

    # 裁剪/拉直天然只作用于整张图，因此不支持局部 region。
    spec = PackageSpec(
        name="crop_and_straighten",
        description="Adjust framing, crop, and horizon alignment.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "max_crop_ratio": 0.16,
            "max_straighten_angle": 8.0,
            "straighten_bias": 0.0,
            "crop_guard": 0.04,
            "min_scale": 0.72,
        },
    )
    params_model = CropAndStraightenParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 裁剪/拉直只支持 whole_image，同时校验抽象强度范围。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for crop/straighten adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 这个包永远不需要 mask。
        return {
            "requires_mask": False,
            "required_region": None,
        }

    def normalize(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 这里采用保守映射：
        # 1. strength 只控制裁剪强度，不允许负裁剪；
        # 2. straighten_angle 单独通过 params 传入，便于后续 planner 更细化控制。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, CropAndStraightenParams):
            raise ValueError("Crop params model is not configured.")
        clipped_strength = parsed.strength
        params = parsed.model_dump()
        max_crop_ratio = parsed.max_crop_ratio
        max_straighten_angle = parsed.max_straighten_angle
        straighten_bias = parsed.straighten_bias
        crop_guard = parsed.crop_guard
        min_scale = parsed.min_scale

        crop_strength = max(0.0, clipped_strength)
        crop_ratio = crop_strength * max_crop_ratio
        straighten_angle = straighten_bias * max_straighten_angle

        return {
            "region": "whole_image",
            "strength": clipped_strength,
            "params": params,
            "crop_strength": crop_strength,
            "crop_ratio": crop_ratio,
            "max_crop_ratio": max_crop_ratio,
            "max_straighten_angle": max_straighten_angle,
            "straighten_bias": straighten_bias,
            "straighten_angle": straighten_angle,
            "crop_guard": crop_guard,
            "min_scale": min_scale,
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

            output_path = tempfile.mktemp(prefix="psagent_crop_straighten_", suffix=".png")
            saved_path = apply_crop_and_straighten(
                context.image_path or "",
                output_path,
                crop_ratio=normalized["crop_ratio"],
                straighten_angle=normalized["straighten_angle"],
                crop_guard=normalized["crop_guard"],
                min_scale=normalized["min_scale"],
            )

            return PackageResult(
                ok=True,
                package=self.name,
                output_image=saved_path,
                applied_params=normalized,
                artifacts={
                    "input_image": context.image_path,
                    "requirements": requirements,
                },
            )
        except Exception as error:  # pragma: no cover - fallback path is verified via result state
            return self.fallback(error, operation, context)
