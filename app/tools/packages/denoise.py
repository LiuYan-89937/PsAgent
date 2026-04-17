"""Denoise package skeleton."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_denoise_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class DenoiseParams(PackageParamsModel):
    """Planner-fillable params for denoise adjustment."""

    strength: float = Field(..., ge=-1.0, le=1.0, description="主去噪强度，建议使用非负值")
    luma_scale: float = Field(12.0, ge=2.0, le=24.0, description="亮度去噪尺度")
    chroma_scale: float = Field(9.0, ge=2.0, le=24.0, description="色度去噪尺度")
    feather_radius: float = Field(14.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")
    detail_protection: float = Field(0.24, ge=0.0, le=0.75, description="细节保护强度")
    template_window_size: int = Field(7, ge=3, le=15, description="去噪模板窗口大小")
    search_window_size: int = Field(21, ge=7, le=31, description="去噪搜索窗口大小")


class DenoisePackage(ToolPackage):
    """Minimal package for denoising operations."""

    # 去噪后续可以支持局部，但局部模式通常要比全局更保守。
    spec = PackageSpec(
        name="denoise",
        description="Reduce image noise globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "luma_scale": 12.0,
            "chroma_scale": 9.0,
            "feather_radius": 14.0,
            "detail_protection": 0.24,
            "template_window_size": 7,
            "search_window_size": 21,
        },
    )
    params_model = DenoiseParams

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        # 去噪先做统一的数值边界校验；局部模式的保守性放到参数映射里处理。
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)

        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for denoise adjustment")

    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        # 当前先统一用 whole_image / 非 whole_image 两档依赖判断。
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
        # 去噪强度只接受正向值；负值在语义上没有意义，统一裁到 0。
        parsed = self.parse_params(operation)
        if not isinstance(parsed, DenoiseParams):
            raise ValueError("Denoise params model is not configured.")
        clipped_strength = parsed.strength
        params = parsed.model_dump()

        region = operation.get("region") or "whole_image"
        local_mode = self.operation_requires_mask(operation, context)
        region_scale = 0.78 if local_mode else 1.0

        luma_scale = parsed.luma_scale
        chroma_scale = parsed.chroma_scale
        feather_radius = parsed.feather_radius
        detail_protection = parsed.detail_protection
        template_window_size = parsed.template_window_size
        search_window_size = parsed.search_window_size
        denoise_strength = max(0.0, clipped_strength)
        luma_strength = denoise_strength * luma_scale * region_scale
        chroma_strength = denoise_strength * chroma_scale * region_scale

        return {
            "region": region,
            "strength": clipped_strength,
            "params": params,
            "denoise_strength": denoise_strength,
            "luma_scale": luma_scale,
            "chroma_scale": chroma_scale,
            "luma_strength": luma_strength,
            "chroma_strength": chroma_strength,
            "feather_radius": feather_radius,
            "detail_protection": detail_protection,
            "template_window_size": template_window_size,
            "search_window_size": search_window_size,
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

            output_path = tempfile.mktemp(prefix="psagent_denoise_", suffix=".png")
            saved_path = apply_denoise_adjustment(
                context.image_path or "",
                output_path,
                luma_strength=normalized["luma_strength"],
                chroma_strength=normalized["chroma_strength"],
                mask_path=mask_path,
                feather_radius=normalized["feather_radius"],
                detail_protection=normalized["detail_protection"],
                template_window_size=normalized["template_window_size"],
                search_window_size=normalized["search_window_size"],
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
