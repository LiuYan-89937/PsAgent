"""Denoise package skeleton."""

from __future__ import annotations

from typing import Any

from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class DenoisePackage(ToolPackage):
    """Skeleton package for denoising operations."""

    # 去噪后续可以支持局部，但局部模式通常要比全局更保守。
    spec = PackageSpec(
        name="denoise",
        description="Reduce image noise globally or by region.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={},
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # 只导出 planner 选包需要的关键信息。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 后续在这里控制不同区域下的最大强度。
        return None

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前先统一用 whole_image / 非 whole_image 两档依赖判断。
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
        # 这里后续接去噪强度到内部参数的映射。
        return {
            "strength": operation.get("strength", 0.0),
            "params": operation.get("params", {}),
        }

    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        # 未实现阶段统一返回未执行结果。
        return PackageResult(
            ok=False,
            package=self.name,
            applied_params=self.normalize(operation, context),
            warnings=["Package skeleton not implemented yet."],
            error="DenoisePackage is a skeleton.",
        )
