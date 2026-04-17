"""Whites and blacks adjustment package."""

from __future__ import annotations

import tempfile
from typing import Any

from app.tools.image_ops import apply_whites_blacks_adjustment
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage
from pydantic import Field


class AdjustWhitesBlacksParams(PackageParamsModel):
    """Planner-fillable params for whites/blacks adjustment."""

    whites_amount: float = Field(..., ge=-1.0, le=1.0, description="白场调整强度，正值提亮白场，负值压白场")
    blacks_amount: float = Field(..., ge=-1.0, le=1.0, description="黑场调整强度，正值压黑场，负值抬黑场")
    highlight_rolloff: float = Field(0.32, ge=0.1, le=0.85, description="白场作用范围滚降")
    shadow_rolloff: float = Field(0.34, ge=0.1, le=0.85, description="黑场作用范围滚降")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部模式下的羽化半径")


class AdjustWhitesBlacksPackage(ToolPackage):
    """Professional first-pass whites/blacks adjustment."""

    spec = PackageSpec(
        name="adjust_whites_blacks",
        description="Set white point and black point globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={
            "highlight_rolloff": 0.32,
            "shadow_rolloff": 0.34,
            "feather_radius": 18.0,
        },
    )
    params_model = AdjustWhitesBlacksParams

    def coerce_legacy_strength_params(
        self,
        legacy_strength: Any,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map a coarse top-level strength into both white and black endpoint shifts."""

        if "whites_amount" in params or "blacks_amount" in params:
            return {}

        strength = float(legacy_strength)
        return {
            "whites_amount": strength * 0.8,
            "blacks_amount": strength * 0.6,
        }

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError("image_path is required for whites/blacks adjustment")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, AdjustWhitesBlacksParams):
            raise ValueError("Whites/blacks params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "whites_amount": parsed.whites_amount,
            "blacks_amount": parsed.blacks_amount,
            "highlight_rolloff": parsed.highlight_rolloff,
            "shadow_rolloff": parsed.shadow_rolloff,
            "feather_radius": parsed.feather_radius,
        }

    def execute(self, operation: dict[str, Any], context: OperationContext) -> PackageResult:
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

            output_path = tempfile.mktemp(prefix="psagent_whites_blacks_", suffix=".png")
            saved_path = apply_whites_blacks_adjustment(
                context.image_path or "",
                output_path,
                whites_amount=normalized["whites_amount"],
                blacks_amount=normalized["blacks_amount"],
                highlight_rolloff=normalized["highlight_rolloff"],
                shadow_rolloff=normalized["shadow_rolloff"],
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
        except Exception as error:  # pragma: no cover
            return self.fallback(error, operation, context)
