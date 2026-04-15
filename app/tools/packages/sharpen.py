"""Sharpen package skeleton."""

from __future__ import annotations

from typing import Any

from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class SharpenPackage(ToolPackage):
    """Skeleton package for sharpening operations."""

    # 锐化和去噪类似，局部模式后续会比全局模式更保守。
    spec = PackageSpec(
        name="sharpen",
        description="Increase detail or sharpness globally or by region.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={},
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # 让 planner 明确知道这个包支持哪些 region。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 后续在这里控制局部锐化的风险边界。
        return None

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 非 whole_image 时后续统一走 dispatcher 补 mask。
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
        # 这里后续负责锐化强度到内部参数的转换。
        return {
            "strength": operation.get("strength", 0.0),
            "params": operation.get("params", {}),
        }

    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        # 当前只保留骨架，等待后续接真实锐化实现。
        return PackageResult(
            ok=False,
            package=self.name,
            applied_params=self.normalize(operation, context),
            warnings=["Package skeleton not implemented yet."],
            error="SharpenPackage is a skeleton.",
        )
