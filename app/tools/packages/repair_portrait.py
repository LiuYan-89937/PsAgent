"""Repair and portrait-oriented deterministic tool packages."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.tools.image_ops_extended import (
    apply_point_color_adjustment,
    apply_regional_enhancement,
    apply_remove_heal,
    apply_skin_smooth,
)
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageSpec
from app.tools.packages.extended_base import DeterministicImageOpPackage


class RepairParams(PackageParamsModel):
    """Shared params for inpaint-based repair tools."""

    strength: float = Field(0.35, ge=-1.0, le=1.0, description="修复主强度。")
    radius_px: float = Field(3.0, ge=1.0, le=24.0, description="修复半径。")
    feather_radius: float = Field(8.0, ge=0.0, le=64.0, description="局部羽化半径。")
    detail_protection: float = Field(0.35, ge=0.0, le=1.0, description="细节保护。")
    auto_detect: bool = Field(True, description="无遮罩时自动找缺陷。")
    small_spot_bias: float = Field(0.55, ge=0.0, le=1.0, description="偏向小点修复。")


class SkinSmoothParams(PackageParamsModel):
    """Shared params for skin smoothing tools."""

    strength: float = Field(0.35, ge=-1.0, le=1.0, description="柔肤主强度。")
    preserve_detail: float = Field(0.78, ge=0.0, le=1.0, description="纹理保留。")
    saturation_protection: float = Field(0.22, ge=0.0, le=1.0, description="饱和度保护。")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部羽化半径。")


class PointColorParams(PackageParamsModel):
    """Params for point-color style targeted adjustments."""

    strength: float = Field(0.25, ge=-1.0, le=1.0, description="精准调色主强度。")
    target_color: str = Field("orange", min_length=2, max_length=32, description="目标颜色名，如 skin、orange、blue、white。")
    target_hue: float | None = Field(default=None, ge=0.0, le=360.0, description="手动色相中心角。")
    range_width: float = Field(28.0, ge=8.0, le=90.0, description="颜色选择宽度。")
    hue_shift: float = Field(0.0, ge=-1.0, le=1.0, description="色相偏移。")
    saturation_shift: float = Field(0.0, ge=-1.0, le=1.0, description="饱和度偏移。")
    luminance_shift: float = Field(0.0, ge=-1.0, le=1.0, description="亮度偏移。")
    preserve_neutrals: float = Field(0.2, ge=0.0, le=1.0, description="中性色保护。")
    feather_radius: float = Field(16.0, ge=0.0, le=64.0, description="局部羽化半径。")


class RegionalEnhanceParams(PackageParamsModel):
    """Shared params for portrait-local enhancement tools."""

    strength: float = Field(0.3, ge=-1.0, le=1.0, description="局部增强主强度。")
    exposure_boost: float = Field(0.0, ge=-1.0, le=1.0, description="曝光增减。")
    saturation_boost: float = Field(0.0, ge=-1.0, le=1.0, description="饱和度增减。")
    warmth_shift: float = Field(0.0, ge=-1.0, le=1.0, description="冷暖偏移。")
    clarity_boost: float = Field(0.0, ge=-1.0, le=1.0, description="清晰度增减。")
    smooth_amount: float = Field(0.0, ge=0.0, le=1.0, description="柔化量。")
    sharpen_amount: float = Field(0.0, ge=0.0, le=1.0, description="锐化量。")
    highlight_protection: float = Field(0.22, ge=0.0, le=1.0, description="高光保护。")
    shadow_lift: float = Field(0.0, ge=0.0, le=1.0, description="暗部提亮。")
    yellow_suppression: float = Field(0.0, ge=0.0, le=1.0, description="压黄量。")
    feather_radius: float = Field(16.0, ge=0.0, le=64.0, description="局部羽化半径。")


class BaseRepairPackage(DeterministicImageOpPackage):
    """Shared normalization for repair primitives."""

    params_model = RepairParams
    image_op = staticmethod(apply_remove_heal)
    repair_method: str = "telea"
    output_prefix = "psagent_repair_"
    param_aliases = {
        "radius_px": ("radius", "heal_radius", "spot_size", "remove_radius"),
        "detail_protection": ("preserve_detail", "structure_protection"),
        "small_spot_bias": ("spot_bias", "blemish_bias"),
    }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, RepairParams):
            raise ValueError("Repair params model is not configured.")
        strength = abs(parsed.strength)
        params = parsed.model_dump()
        return {
            "region": operation.get("region") or "whole_image",
            "params": params,
            "strength": strength,
            "radius_px": parsed.radius_px * (0.9 + strength * 0.8),
            "feather_radius": parsed.feather_radius,
            "detail_protection": parsed.detail_protection,
            "auto_detect": parsed.auto_detect,
            "small_spot_bias": parsed.small_spot_bias,
        }

    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        return {
            "strength": normalized["strength"],
            "radius_px": normalized["radius_px"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
            "detail_protection": normalized["detail_protection"],
            "method": self.repair_method,
            "auto_detect": normalized["auto_detect"],
            "small_spot_bias": normalized["small_spot_bias"],
        }


class BaseSkinSmoothPackage(DeterministicImageOpPackage):
    """Shared normalization for skin smoothing tools."""

    params_model = SkinSmoothParams
    image_op = staticmethod(apply_skin_smooth)
    output_prefix = "psagent_skin_"
    smooth_scale: float = 0.72
    detail_scale: float = 1.0
    param_aliases = {
        "preserve_detail": ("detail_protection", "preserve_texture"),
        "saturation_protection": ("protect_saturation",),
    }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, SkinSmoothParams):
            raise ValueError("Skin smooth params model is not configured.")
        strength = max(0.0, parsed.strength)
        params = parsed.model_dump()
        return {
            "region": operation.get("region") or "whole_image",
            "params": params,
            "strength": strength,
            "smooth_strength": min(1.0, strength * self.smooth_scale + 0.1),
            "detail_protection": min(1.0, parsed.preserve_detail * self.detail_scale),
            "saturation_protection": parsed.saturation_protection,
            "feather_radius": parsed.feather_radius,
        }

    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        return {
            "strength": normalized["strength"],
            "smooth_strength": normalized["smooth_strength"],
            "detail_protection": normalized["detail_protection"],
            "saturation_protection": normalized["saturation_protection"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class BaseRegionalEnhancePackage(DeterministicImageOpPackage):
    """Shared normalization for local portrait enhancement tools."""

    params_model = RegionalEnhanceParams
    image_op = staticmethod(apply_regional_enhancement)
    output_prefix = "psagent_regional_"
    feature_defaults: dict[str, float] = {}
    param_aliases = {
        "exposure_boost": ("brightness_boost", "lighten"),
        "saturation_boost": ("saturation", "color_boost"),
        "warmth_shift": ("warmth", "temperature"),
        "clarity_boost": ("clarity", "detail_boost"),
        "smooth_amount": ("smooth", "soften"),
        "sharpen_amount": ("sharpen", "detail_sharpen"),
        "shadow_lift": ("lift_shadows",),
        "yellow_suppression": ("reduce_yellow", "deyellow"),
    }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, RegionalEnhanceParams):
            raise ValueError("Regional enhancement params model is not configured.")
        strength = parsed.strength
        params = parsed.model_dump()
        normalized = {
            "region": operation.get("region") or "whole_image",
            "params": params,
            "strength": strength,
            "feather_radius": parsed.feather_radius,
            "highlight_protection": parsed.highlight_protection,
        }
        for key in (
            "exposure_boost",
            "saturation_boost",
            "warmth_shift",
            "clarity_boost",
            "smooth_amount",
            "sharpen_amount",
            "shadow_lift",
            "yellow_suppression",
        ):
            base_value = getattr(parsed, key)
            normalized[key] = float(np_clip(base_value + self.feature_defaults.get(key, 0.0) * strength, -1.0, 1.0))
        normalized["smooth_amount"] = float(np_clip(normalized["smooth_amount"], 0.0, 1.0))
        normalized["sharpen_amount"] = float(np_clip(normalized["sharpen_amount"], 0.0, 1.0))
        normalized["shadow_lift"] = float(np_clip(normalized["shadow_lift"], 0.0, 1.0))
        normalized["yellow_suppression"] = float(np_clip(normalized["yellow_suppression"], 0.0, 1.0))
        return normalized

    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        return {
            "exposure_boost": normalized["exposure_boost"],
            "saturation_boost": normalized["saturation_boost"],
            "warmth_shift": normalized["warmth_shift"],
            "clarity_boost": normalized["clarity_boost"],
            "smooth_amount": normalized["smooth_amount"],
            "sharpen_amount": normalized["sharpen_amount"],
            "highlight_protection": normalized["highlight_protection"],
            "shadow_lift": normalized["shadow_lift"],
            "yellow_suppression": normalized["yellow_suppression"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


def np_clip(value: float, low: float, high: float) -> float:
    """Clip a float without importing NumPy into every package body."""

    return min(max(float(value), low), high)


class RemoveHealPackage(BaseRepairPackage):
    spec = PackageSpec(
        name="remove_heal",
        description="Heal or remove small distracting objects globally or by mask.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"radius_px": 3.0, "feather_radius": 8.0, "detail_protection": 0.35},
    )
    output_prefix = "psagent_remove_heal_"


class BlemishRemovePackage(BaseRepairPackage):
    spec = PackageSpec(
        name="blemish_remove",
        description="Remove pimples or small blemishes with restrained healing.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="medium",
        default_params={"radius_px": 2.4, "feather_radius": 6.0, "detail_protection": 0.42, "small_spot_bias": 0.85},
    )
    output_prefix = "psagent_blemish_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        normalized = super().normalize(operation, context)
        normalized["radius_px"] = min(normalized["radius_px"], 5.8)
        normalized["small_spot_bias"] = max(normalized["small_spot_bias"], 0.8)
        return normalized


class SpotHealPackage(BaseRepairPackage):
    spec = PackageSpec(
        name="spot_heal",
        description="Spot-heal tiny dust marks or localized defects.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"radius_px": 2.0, "feather_radius": 4.0, "detail_protection": 0.46, "small_spot_bias": 0.92},
    )
    output_prefix = "psagent_spot_heal_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        normalized = super().normalize(operation, context)
        normalized["radius_px"] = min(normalized["radius_px"], 4.2)
        normalized["small_spot_bias"] = max(normalized["small_spot_bias"], 0.9)
        return normalized


class CloneStampPackage(BaseRepairPackage):
    spec = PackageSpec(
        name="clone_stamp",
        description="Clone-like repair for patterned regions using structure-aware inpainting.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"radius_px": 3.4, "feather_radius": 8.0, "detail_protection": 0.4},
    )
    repair_method = "ns"
    output_prefix = "psagent_clone_stamp_"


class SkinSmoothPackage(BaseSkinSmoothPackage):
    spec = PackageSpec(
        name="skin_smooth",
        description="Smooth skin while retaining natural contours and pores where possible.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="medium",
        default_params={"preserve_detail": 0.78, "saturation_protection": 0.22, "feather_radius": 18.0},
    )
    output_prefix = "psagent_skin_smooth_"
    param_aliases = {
        **BaseSkinSmoothPackage.param_aliases,
        "strength": ("amount", "smooth_strength"),
    }


class SkinTextureReducePackage(BaseSkinSmoothPackage):
    spec = PackageSpec(
        name="skin_texture_reduce",
        description="Reduce micro texture on skin without heavy beauty filtering.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="low",
        default_params={"preserve_detail": 0.86, "saturation_protection": 0.16, "feather_radius": 16.0},
    )
    output_prefix = "psagent_skin_texture_"
    smooth_scale = 0.48
    detail_scale = 1.08


class PointColorPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="point_color",
        description="Apply targeted point-color style adjustments to a narrow color range.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"target_color": "orange", "range_width": 28.0, "preserve_neutrals": 0.2, "feather_radius": 16.0},
    )
    params_model = PointColorParams
    image_op = staticmethod(apply_point_color_adjustment)
    output_prefix = "psagent_point_color_"
    param_aliases = {
        "target_color": ("color", "color_name", "sampled_color"),
        "target_hue": ("hue_center", "center_hue"),
        "range_width": ("range", "hue_width", "selection_width"),
        "saturation_shift": ("saturation", "sat_shift"),
        "luminance_shift": ("luminance", "brightness_shift", "value_shift"),
        "preserve_neutrals": ("neutral_protection",),
    }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, PointColorParams):
            raise ValueError("Point color params model is not configured.")
        params = parsed.model_dump()
        strength = parsed.strength
        hue_shift = parsed.hue_shift * 16.0
        saturation_shift = parsed.saturation_shift if parsed.saturation_shift != 0.0 else strength * 0.42
        luminance_shift = parsed.luminance_shift if parsed.luminance_shift != 0.0 else strength * 0.26
        return {
            "region": operation.get("region") or "whole_image",
            "params": params,
            "strength": strength,
            "target_color": parsed.target_color,
            "target_hue": parsed.target_hue,
            "range_width": parsed.range_width,
            "hue_shift": hue_shift,
            "saturation_shift": saturation_shift,
            "luminance_shift": luminance_shift,
            "preserve_neutrals": parsed.preserve_neutrals,
            "feather_radius": parsed.feather_radius,
        }

    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        return {
            "target_color": normalized["target_color"],
            "target_hue": normalized["target_hue"],
            "range_width": normalized["range_width"],
            "hue_shift": normalized["hue_shift"],
            "saturation_shift": normalized["saturation_shift"],
            "luminance_shift": normalized["luminance_shift"],
            "preserve_neutrals": normalized["preserve_neutrals"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class UnderEyeBrightenPackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="under_eye_brighten",
        description="Brighten under-eye shadows while keeping the result natural.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="low",
        default_params={"highlight_protection": 0.28, "feather_radius": 14.0},
    )
    output_prefix = "psagent_under_eye_"
    feature_defaults = {"exposure_boost": 0.14, "shadow_lift": 0.18, "smooth_amount": 0.03, "saturation_boost": -0.04}


class TeethWhitenPackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="teeth_whiten",
        description="Whiten teeth by reducing yellow saturation and gently lifting brightness.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="low",
        default_params={"highlight_protection": 0.2, "feather_radius": 10.0},
    )
    output_prefix = "psagent_teeth_"
    feature_defaults = {"exposure_boost": 0.08, "saturation_boost": -0.18, "yellow_suppression": 0.55}


class EyeBrightenPackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="eye_brighten",
        description="Brighten and sharpen eyes in a restrained portrait-retouch style.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="medium",
        default_params={"highlight_protection": 0.18, "feather_radius": 12.0},
    )
    output_prefix = "psagent_eye_"
    feature_defaults = {"exposure_boost": 0.1, "clarity_boost": 0.22, "sharpen_amount": 0.22, "saturation_boost": 0.04}


class HairEnhancePackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="hair_enhance",
        description="Add texture and definition to hair with restrained sharpening.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="medium",
        default_params={"highlight_protection": 0.18, "feather_radius": 18.0},
    )
    output_prefix = "psagent_hair_"
    feature_defaults = {"clarity_boost": 0.24, "sharpen_amount": 0.28, "saturation_boost": 0.04}


class LipEnhancePackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="lip_enhance",
        description="Enhance lip color and local clarity without over-saturating skin.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "general"],
        risk_level="low",
        default_params={"highlight_protection": 0.18, "feather_radius": 12.0},
    )
    output_prefix = "psagent_lips_"
    feature_defaults = {"saturation_boost": 0.18, "warmth_shift": 0.12, "clarity_boost": 0.08}


class ReflectionReducePackage(BaseRegionalEnhancePackage):
    spec = PackageSpec(
        name="reflection_reduce",
        description="Reduce harsh reflections and glare while preserving structure.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"highlight_protection": 0.44, "feather_radius": 18.0},
    )
    output_prefix = "psagent_reflection_"
    feature_defaults = {"saturation_boost": -0.12, "clarity_boost": -0.08, "shadow_lift": 0.05}
