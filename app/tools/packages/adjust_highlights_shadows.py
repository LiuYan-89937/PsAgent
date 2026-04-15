"""Highlights/shadows adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_highlights_shadows_adjustment
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class AdjustHighlightsShadowsPackage(ToolPackage):
    """Minimal package for highlight and shadow balancing."""

    # 高光/阴影和曝光类似，天然支持全局与局部两种模式。
    spec = PackageSpec(
        name="adjust_highlights_shadows",
        description="Balance highlights and shadows globally or by region.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
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

    def get_llm_schema(self) -> dict[str, Any]:
        # 只导出选择包所需的信息。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 先做最基础的边界校验，保证最小实现可控。
        region = operation.get("region") or "whole_image"
        strength = operation.get("strength", 0.0)

        if region not in self.spec.supported_regions:
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        if not -1.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between -1.0 and 1.0")
        if not context.image_path:
            raise ValueError("image_path is required for highlights/shadows adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前先按 region 是否为 whole_image 来判断要不要 mask。
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
        # 专业版里除了基本强度，还要把局部估计半径、作用 tonal width、
        # 细节回灌量等一起纳入内部参数，才能更接近真实修图软件的手感。
        raw_strength = float(operation.get("strength", 0.0))
        clipped_strength = max(-1.0, min(1.0, raw_strength))
        params = dict(self.spec.default_params)
        params.update(operation.get("params", {}))

        tone_amount = float(params.get("tone_amount", self.spec.default_params["tone_amount"]))
        tone_amount = max(0.05, min(0.9, tone_amount))
        feather_radius = float(params.get("feather_radius", self.spec.default_params["feather_radius"]))
        feather_radius = max(0.0, min(64.0, feather_radius))
        midtone_contrast = float(
            params.get("midtone_contrast", self.spec.default_params["midtone_contrast"])
        )
        midtone_contrast = max(0.0, min(0.5, midtone_contrast))
        local_radius = float(params.get("local_radius", self.spec.default_params["local_radius"]))
        local_radius = max(4.0, min(160.0, local_radius))
        shadow_tonal_width = float(
            params.get("shadow_tonal_width", self.spec.default_params["shadow_tonal_width"])
        )
        shadow_tonal_width = max(0.1, min(0.8, shadow_tonal_width))
        highlight_tonal_width = float(
            params.get("highlight_tonal_width", self.spec.default_params["highlight_tonal_width"])
        )
        highlight_tonal_width = max(0.1, min(0.8, highlight_tonal_width))
        detail_amount = float(params.get("detail_amount", self.spec.default_params["detail_amount"]))
        detail_amount = max(0.0, min(1.0, detail_amount))
        highlight_balance = float(
            params.get("highlight_balance", self.spec.default_params["highlight_balance"])
        )
        highlight_balance = max(0.2, min(1.2, highlight_balance))

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
