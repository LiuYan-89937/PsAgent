"""Vibrance and saturation adjustment package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_vibrance_saturation_adjustment
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage


class AdjustVibranceSaturationPackage(ToolPackage):
    """Minimal package for vibrance and saturation adjustments."""

    # 饱和度类操作后续适合支持主体/背景等局部区域，但第一阶段先跑全局。
    spec = PackageSpec(
        name="adjust_vibrance_saturation",
        description="Adjust vibrance or saturation globally or by region.",
        supported_regions=["whole_image", "person", "main_subject", "background"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "vibrance_scale": 0.52,
            "saturation_scale": 0.14,
            "feather_radius": 18.0,
            "protect_highlights": 0.26,
            "protect_skin": 0.34,
            "protect_shadows": 0.24,
            "chroma_denoise": 0.34,
            "max_chroma": 92.0,
            "neutral_floor": 6.0,
            "neutral_softness": 14.0,
        },
    )

    def get_llm_schema(self) -> dict[str, Any]:
        # planner 只需要知道这个包的选择边界。
        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 色彩类操作也先做最基础的边界校验。
        region = operation.get("region") or "whole_image"
        strength = operation.get("strength", 0.0)

        if region not in self.spec.supported_regions:
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        if not -1.0 <= float(strength) <= 1.0:
            raise ValueError("strength must be between -1.0 and 1.0")
        if not context.image_path:
            raise ValueError("image_path is required for vibrance/saturation adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # region 不是 whole_image 时，后续由 dispatcher 准备 mask。
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
        # 默认优先走 vibrance，让低饱和区域先起来；
        # saturation 只做保守补充，避免整体颜色冲得太猛。
        raw_strength = float(operation.get("strength", 0.0))
        clipped_strength = max(-1.0, min(1.0, raw_strength))
        params = dict(self.spec.default_params)
        params.update(operation.get("params", {}))

        vibrance_scale = float(params.get("vibrance_scale", self.spec.default_params["vibrance_scale"]))
        vibrance_scale = max(0.1, min(1.5, vibrance_scale))
        saturation_scale = float(
            params.get("saturation_scale", self.spec.default_params["saturation_scale"])
        )
        saturation_scale = max(0.0, min(1.0, saturation_scale))
        feather_radius = float(params.get("feather_radius", self.spec.default_params["feather_radius"]))
        feather_radius = max(0.0, min(64.0, feather_radius))
        protect_highlights = float(
            params.get("protect_highlights", self.spec.default_params["protect_highlights"])
        )
        protect_highlights = max(0.0, min(0.8, protect_highlights))
        protect_skin = float(params.get("protect_skin", self.spec.default_params["protect_skin"]))
        protect_skin = max(0.0, min(0.8, protect_skin))
        protect_shadows = float(
            params.get("protect_shadows", self.spec.default_params["protect_shadows"])
        )
        protect_shadows = max(0.0, min(0.8, protect_shadows))
        chroma_denoise = float(params.get("chroma_denoise", self.spec.default_params["chroma_denoise"]))
        chroma_denoise = max(0.0, min(1.0, chroma_denoise))
        max_chroma = float(params.get("max_chroma", self.spec.default_params["max_chroma"]))
        max_chroma = max(48.0, min(128.0, max_chroma))
        neutral_floor = float(params.get("neutral_floor", self.spec.default_params["neutral_floor"]))
        neutral_floor = max(0.0, min(24.0, neutral_floor))
        neutral_softness = float(
            params.get("neutral_softness", self.spec.default_params["neutral_softness"])
        )
        neutral_softness = max(2.0, min(32.0, neutral_softness))

        vibrance_amount = clipped_strength * vibrance_scale
        saturation_amount = clipped_strength * saturation_scale

        return {
            "region": operation.get("region") or "whole_image",
            "strength": clipped_strength,
            "params": params,
            "vibrance_scale": vibrance_scale,
            "saturation_scale": saturation_scale,
            "vibrance_amount": vibrance_amount,
            "saturation_amount": saturation_amount,
            "feather_radius": feather_radius,
            "protect_highlights": protect_highlights,
            "protect_skin": protect_skin,
            "protect_shadows": protect_shadows,
            "chroma_denoise": chroma_denoise,
            "max_chroma": max_chroma,
            "neutral_floor": neutral_floor,
            "neutral_softness": neutral_softness,
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

            output_path = tempfile.mktemp(prefix="psagent_vibrance_saturation_", suffix=".png")
            saved_path = apply_vibrance_saturation_adjustment(
                context.image_path or "",
                output_path,
                vibrance_amount=normalized["vibrance_amount"],
                saturation_amount=normalized["saturation_amount"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                protect_highlights=normalized["protect_highlights"],
                protect_skin=normalized["protect_skin"],
                protect_shadows=normalized["protect_shadows"],
                chroma_denoise=normalized["chroma_denoise"],
                max_chroma=normalized["max_chroma"],
                neutral_floor=normalized["neutral_floor"],
                neutral_softness=normalized["neutral_softness"],
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
