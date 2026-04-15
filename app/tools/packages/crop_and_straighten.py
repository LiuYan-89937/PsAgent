"""Crop and straighten package skeleton."""

from __future__ import annotations

from typing import Any

from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class CropAndStraightenPackage(ToolPackage):
    """Skeleton package for crop and straighten operations."""

    # 裁剪/拉直天然只作用于整张图，因此不支持局部 region。
    spec = PackageSpec(
        name="crop_and_straighten",
        description="Adjust framing, crop, and horizon alignment.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={},
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # 这是最典型的 whole_image-only 包。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 后续可在这里限制 crop ratio、straighten angle 等参数范围。
        return None

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
        # 后续这里负责把抽象强度映射成裁剪策略或拉直角度。
        return {
            "strength": operation.get("strength", 0.0),
            "params": operation.get("params", {}),
        }

    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        # 先占位，避免误用成真实裁剪实现。
        return PackageResult(
            ok=False,
            package=self.name,
            applied_params=self.normalize(operation, context),
            warnings=["Package skeleton not implemented yet."],
            error="CropAndStraightenPackage is a skeleton.",
        )
