"""Contrast adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_contrast_adjustment
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class AdjustContrastPackage(ToolPackage):
    """Minimal professional contrast package."""

    # 对比度后续也可以扩展到主体/背景等局部区域。
    spec = PackageSpec(
        name="adjust_contrast",
        description="Adjust contrast globally or for selected regions.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "contrast_scale": 0.7,
            "feather_radius": 18.0,
            "pivot": 0.5,
            "protect_highlights": 0.22,
            "protect_shadows": 0.22,
        },
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # planner 只看能力声明，不看执行细节。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 对比度和曝光一样，需要先做最基础的边界限制。
        region = operation.get("region") or "whole_image"
        strength = operation.get("strength", 0.0)

        if region not in self.spec.supported_regions:
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        if not -1.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between -1.0 and 1.0")
        if not context.image_path:
            raise ValueError("image_path is required for contrast adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # dispatcher 会根据这里的声明去准备前置依赖。
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
        # 最小实现里把一个抽象 strength 映射成围绕中灰点的对比度伸缩量。
        raw_strength = float(operation.get("strength", 0.0))
        clipped_strength = max(-1.0, min(1.0, raw_strength))
        params = dict(self.spec.default_params)
        params.update(operation.get("params", {}))

        contrast_scale = float(params.get("contrast_scale", self.spec.default_params["contrast_scale"]))
        contrast_scale = max(0.1, min(1.5, contrast_scale))
        feather_radius = float(params.get("feather_radius", self.spec.default_params["feather_radius"]))
        feather_radius = max(0.0, min(64.0, feather_radius))
        pivot = float(params.get("pivot", self.spec.default_params["pivot"]))
        pivot = max(0.25, min(0.75, pivot))
        protect_highlights = float(
            params.get("protect_highlights", self.spec.default_params["protect_highlights"])
        )
        protect_highlights = max(0.0, min(0.8, protect_highlights))
        protect_shadows = float(
            params.get("protect_shadows", self.spec.default_params["protect_shadows"])
        )
        protect_shadows = max(0.0, min(0.8, protect_shadows))

        contrast_amount = clipped_strength * contrast_scale
        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "contrast_scale": contrast_scale,
            "contrast_amount": contrast_amount,
            "feather_radius": feather_radius,
            "pivot": pivot,
            "protect_highlights": protect_highlights,
            "protect_shadows": protect_shadows,
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

            output_path = tempfile.mktemp(prefix="psagent_contrast_", suffix=".png")
            saved_path = apply_contrast_adjustment(
                context.image_path or "",
                output_path,
                contrast_amount=normalized["contrast_amount"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                pivot=normalized["pivot"],
                protect_highlights=normalized["protect_highlights"],
                protect_shadows=normalized["protect_shadows"],
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
