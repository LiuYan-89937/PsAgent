"""Optics, color-fx, and blur oriented deterministic tool packages."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.tools.image_ops_extended import (
    apply_auto_upright,
    apply_background_blur,
    apply_camera_calibration,
    apply_color_grading,
    apply_convert_black_white,
    apply_defringe,
    apply_glow_highlight,
    apply_grain,
    apply_lens_blur,
    apply_lens_correction,
    apply_lut_preset,
    apply_moire_reduction,
    apply_perspective_correction,
    apply_remove_chromatic_aberration,
    apply_vignette,
)
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageSpec
from app.tools.packages.extended_base import DeterministicImageOpPackage


class LensCorrectionParams(PackageParamsModel):
    strength: float = Field(0.4, ge=-1.0, le=1.0, description="畸变校正强度。")
    edge_scale: float = Field(1.0, ge=1.0, le=1.2, description="边缘缩放补偿。")


class ChromaticAberrationParams(PackageParamsModel):
    strength: float = Field(0.35, ge=0.0, le=1.0, description="色差校正强度。")
    radial_bias: float = Field(0.35, ge=0.0, le=1.0, description="边缘优先度。")


class DefringeParams(PackageParamsModel):
    strength: float = Field(0.35, ge=0.0, le=1.0, description="去边色主强度。")
    purple_amount: float = Field(0.6, ge=0.0, le=1.0, description="紫边抑制。")
    green_amount: float = Field(0.35, ge=0.0, le=1.0, description="绿边抑制。")
    edge_threshold: float = Field(0.06, ge=0.0, le=0.6, description="边缘阈值。")


class PerspectiveParams(PackageParamsModel):
    vertical_amount: float = Field(0.0, ge=-1.0, le=1.0, description="垂直透视偏移。")
    horizontal_amount: float = Field(0.0, ge=-1.0, le=1.0, description="水平透视偏移。")
    strength: float = Field(0.4, ge=0.0, le=1.0, description="透视校正总强度。")


class AutoUprightParams(PackageParamsModel):
    strength: float = Field(0.7, ge=0.0, le=1.0, description="自动扶正强度。")
    max_angle: float = Field(8.0, ge=0.0, le=20.0, description="最大旋转角度。")


class VignetteParams(PackageParamsModel):
    amount: float = Field(0.25, ge=-1.0, le=1.0, description="暗角强度。")
    midpoint: float = Field(0.62, ge=0.15, le=0.95, description="暗角起点。")
    roundness: float = Field(0.0, ge=-1.0, le=1.0, description="暗角圆润度。")
    feather: float = Field(0.65, ge=0.05, le=1.0, description="暗角过渡柔和度。")
    feather_radius: float = Field(20.0, ge=0.0, le=64.0, description="局部羽化半径。")


class GrainParams(PackageParamsModel):
    amount: float = Field(0.2, ge=0.0, le=1.0, description="颗粒强度。")
    size: float = Field(0.9, ge=0.2, le=2.0, description="颗粒尺寸。")
    roughness: float = Field(0.45, ge=0.0, le=1.0, description="颗粒粗糙度。")
    color_amount: float = Field(0.18, ge=0.0, le=1.0, description="彩色颗粒占比。")
    feather_radius: float = Field(12.0, ge=0.0, le=64.0, description="局部羽化半径。")


class MoireParams(PackageParamsModel):
    amount: float = Field(0.28, ge=0.0, le=1.0, description="摩尔纹抑制强度。")


class ColorGradingParams(PackageParamsModel):
    shadow_hue: float = Field(220.0, ge=0.0, le=360.0, description="阴影色相。")
    shadow_saturation: float = Field(0.12, ge=0.0, le=1.0, description="阴影饱和度。")
    midtone_hue: float = Field(32.0, ge=0.0, le=360.0, description="中间调色相。")
    midtone_saturation: float = Field(0.1, ge=0.0, le=1.0, description="中间调饱和度。")
    highlight_hue: float = Field(48.0, ge=0.0, le=360.0, description="高光色相。")
    highlight_saturation: float = Field(0.12, ge=0.0, le=1.0, description="高光饱和度。")
    balance: float = Field(0.0, ge=-1.0, le=1.0, description="阴影/高光平衡。")
    blending: float = Field(0.55, ge=0.0, le=1.0, description="分级融合强度。")
    feather_radius: float = Field(16.0, ge=0.0, le=64.0, description="局部羽化半径。")


class LutParams(PackageParamsModel):
    preset: str = Field("clean_portrait", min_length=2, max_length=32, description="预设名，如 clean_portrait、warm_film、cool_fade。")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="预设强度。")
    feather_radius: float = Field(16.0, ge=0.0, le=64.0, description="局部羽化半径。")


class BlackWhiteParams(PackageParamsModel):
    contrast: float = Field(0.24, ge=0.0, le=1.0, description="黑白对比强度。")
    filter_color: str = Field("neutral", min_length=2, max_length=16, description="黑白滤镜色，如 neutral、red、yellow、green、blue。")
    tone_amount: float = Field(0.18, ge=0.0, le=1.0, description="黑白分色力度。")
    feather_radius: float = Field(16.0, ge=0.0, le=64.0, description="局部羽化半径。")


class CalibrationParams(PackageParamsModel):
    red_bias: float = Field(0.0, ge=-1.0, le=1.0, description="红原色偏移。")
    green_bias: float = Field(0.0, ge=-1.0, le=1.0, description="绿原色偏移。")
    blue_bias: float = Field(0.0, ge=-1.0, le=1.0, description="蓝原色偏移。")
    saturation_bias: float = Field(0.18, ge=-1.0, le=1.0, description="整体饱和度偏置。")
    feather_radius: float = Field(14.0, ge=0.0, le=64.0, description="局部羽化半径。")


class BlurParams(PackageParamsModel):
    amount: float = Field(0.38, ge=0.0, le=1.0, description="背景虚化强度。")
    highlight_boost: float = Field(0.12, ge=0.0, le=1.0, description="虚化高光增强。")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部羽化半径。")


class LensBlurParams(PackageParamsModel):
    amount: float = Field(0.42, ge=0.0, le=1.0, description="镜头虚化强度。")
    highlight_bloom: float = Field(0.16, ge=0.0, le=1.0, description="散景高光增强。")
    feather_radius: float = Field(20.0, ge=0.0, le=64.0, description="局部羽化半径。")


class GlowParams(PackageParamsModel):
    amount: float = Field(0.32, ge=0.0, le=1.0, description="Glow 强度。")
    threshold: float = Field(0.62, ge=0.0, le=0.95, description="高光阈值。")
    warmth: float = Field(0.18, ge=-1.0, le=1.0, description="Glow 冷暖偏移。")
    feather_radius: float = Field(18.0, ge=0.0, le=64.0, description="局部羽化半径。")


class WholeImageDeterministicPackage(DeterministicImageOpPackage):
    """Helper for whole-image only optics tools."""

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        return {"requires_mask": False, "required_region": None}

    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        return normalized["image_op_kwargs"]


class LensCorrectionPackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="lens_correction",
        description="Correct mild barrel or pincushion distortion globally.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["landscape", "architecture", "general"],
        risk_level="medium",
        default_params={"edge_scale": 1.0},
    )
    params_model = LensCorrectionParams
    image_op = staticmethod(apply_lens_correction)
    output_prefix = "psagent_lens_correction_"
    param_aliases = {"edge_scale": ("scale", "crop_scale")}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, LensCorrectionParams)
        distortion_amount = parsed.strength * 0.9
        return {
            "region": "whole_image",
            "params": parsed.model_dump(),
            "image_op_kwargs": {"distortion_amount": distortion_amount, "edge_scale": parsed.edge_scale},
        }


class RemoveChromaticAberrationPackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="remove_chromatic_aberration",
        description="Reduce simple radial chromatic aberration and color fringing.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["landscape", "architecture", "general"],
        risk_level="medium",
        default_params={"radial_bias": 0.35},
    )
    params_model = ChromaticAberrationParams
    image_op = staticmethod(apply_remove_chromatic_aberration)
    output_prefix = "psagent_ca_"
    param_aliases = {"radial_bias": ("edge_bias",)}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, ChromaticAberrationParams)
        return {
            "region": "whole_image",
            "params": parsed.model_dump(),
            "image_op_kwargs": {"amount": parsed.strength, "radial_bias": parsed.radial_bias},
        }


class DefringePackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="defringe",
        description="Suppress purple and green edge fringing in high-contrast areas.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"purple_amount": 0.6, "green_amount": 0.35, "edge_threshold": 0.06},
    )
    params_model = DefringeParams
    image_op = staticmethod(apply_defringe)
    output_prefix = "psagent_defringe_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, DefringeParams)
        return {
            "region": "whole_image",
            "params": parsed.model_dump(),
            "image_op_kwargs": {
                "purple_amount": parsed.purple_amount * parsed.strength,
                "green_amount": parsed.green_amount * parsed.strength,
                "edge_threshold": parsed.edge_threshold,
            },
        }


class PerspectiveCorrectionPackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="perspective_correction",
        description="Apply simple vertical and horizontal keystone correction.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["architecture", "interior", "general"],
        risk_level="high",
        default_params={},
    )
    params_model = PerspectiveParams
    image_op = staticmethod(apply_perspective_correction)
    output_prefix = "psagent_perspective_"
    param_aliases = {
        "vertical_amount": ("vertical", "keystone_vertical"),
        "horizontal_amount": ("horizontal", "keystone_horizontal"),
    }

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, PerspectiveParams)
        return {
            "region": "whole_image",
            "params": parsed.model_dump(),
            "image_op_kwargs": {
                "vertical_amount": parsed.vertical_amount * parsed.strength,
                "horizontal_amount": parsed.horizontal_amount * parsed.strength,
            },
        }


class AutoUprightPackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="auto_upright",
        description="Automatically rotate toward the dominant upright perspective.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["architecture", "general"],
        risk_level="medium",
        default_params={"max_angle": 8.0},
    )
    params_model = AutoUprightParams
    image_op = staticmethod(apply_auto_upright)
    output_prefix = "psagent_auto_upright_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, AutoUprightParams)
        return {
            "region": "whole_image",
            "params": parsed.model_dump(),
            "image_op_kwargs": {"strength": parsed.strength, "max_angle": parsed.max_angle},
        }


class VignettePackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="vignette",
        description="Apply a Lightroom-style vignette globally or inside a mask.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"midpoint": 0.62, "roundness": 0.0, "feather": 0.65, "feather_radius": 20.0},
    )
    params_model = VignetteParams
    image_op = staticmethod(apply_vignette)
    output_prefix = "psagent_vignette_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, VignetteParams)
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "amount": parsed.amount,
            "midpoint": parsed.midpoint,
            "roundness": parsed.roundness,
            "feather": parsed.feather,
            "feather_radius": parsed.feather_radius,
        }

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "amount": normalized["amount"],
            "midpoint": normalized["midpoint"],
            "roundness": normalized["roundness"],
            "feather": normalized["feather"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class GrainPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="grain",
        description="Add restrained film-style grain globally or by region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"size": 0.9, "roughness": 0.45, "color_amount": 0.18, "feather_radius": 12.0},
    )
    params_model = GrainParams
    image_op = staticmethod(apply_grain)
    output_prefix = "psagent_grain_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, GrainParams)
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "amount": parsed.amount,
            "size": parsed.size,
            "roughness": parsed.roughness,
            "color_amount": parsed.color_amount,
            "feather_radius": parsed.feather_radius,
        }

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "amount": normalized["amount"],
            "size": normalized["size"],
            "roughness": normalized["roughness"],
            "color_amount": normalized["color_amount"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class MoireReducePackage(WholeImageDeterministicPackage):
    spec = PackageSpec(
        name="moire_reduce",
        description="Reduce moire by attenuating chroma and the highest-frequency detail.",
        supported_regions=["whole_image"],
        mask_policy="none",
        supported_domains=["fashion", "architecture", "general"],
        risk_level="medium",
        default_params={"amount": 0.28},
    )
    params_model = MoireParams
    image_op = staticmethod(apply_moire_reduction)
    output_prefix = "psagent_moire_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, MoireParams)
        return {"region": "whole_image", "params": parsed.model_dump(), "image_op_kwargs": {"amount": parsed.amount}}


class ColorGradingPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="color_grading",
        description="Apply split-toning style color grading to shadows, midtones, and highlights.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"blending": 0.55, "balance": 0.0, "feather_radius": 16.0},
    )
    params_model = ColorGradingParams
    image_op = staticmethod(apply_color_grading)
    output_prefix = "psagent_color_grading_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, ColorGradingParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "shadow_hue": normalized["shadow_hue"],
            "shadow_saturation": normalized["shadow_saturation"],
            "midtone_hue": normalized["midtone_hue"],
            "midtone_saturation": normalized["midtone_saturation"],
            "highlight_hue": normalized["highlight_hue"],
            "highlight_saturation": normalized["highlight_saturation"],
            "balance": normalized["balance"],
            "blending": normalized["blending"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class ApplyLutPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="apply_lut",
        description="Apply a lightweight LUT-like preset look globally or by mask.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"preset": "clean_portrait", "strength": 0.5, "feather_radius": 16.0},
    )
    params_model = LutParams
    image_op = staticmethod(apply_lut_preset)
    output_prefix = "psagent_lut_"
    param_aliases = {"preset": ("look", "style_preset")}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, LutParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "preset": normalized["preset"],
            "strength": normalized["strength"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class ConvertBlackWhitePackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="convert_black_white",
        description="Convert the image to black and white with optional toning.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
        default_params={"contrast": 0.24, "filter_color": "neutral", "tone_amount": 0.18, "feather_radius": 16.0},
    )
    params_model = BlackWhiteParams
    image_op = staticmethod(apply_convert_black_white)
    output_prefix = "psagent_bw_"
    param_aliases = {"filter_color": ("color_filter", "filter")}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, BlackWhiteParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "contrast": normalized["contrast"],
            "filter_color": normalized["filter_color"],
            "tone_amount": normalized["tone_amount"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class CameraCalibrationPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="camera_calibration",
        description="Apply restrained primary-color calibration style adjustments.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
        default_params={"saturation_bias": 0.18, "feather_radius": 14.0},
    )
    params_model = CalibrationParams
    image_op = staticmethod(apply_camera_calibration)
    output_prefix = "psagent_camera_calibration_"
    param_aliases = {"saturation_bias": ("saturation", "vibrance")}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, CalibrationParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "red_bias": normalized["red_bias"],
            "green_bias": normalized["green_bias"],
            "blue_bias": normalized["blue_bias"],
            "saturation_bias": normalized["saturation_bias"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class BackgroundBlurPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="background_blur",
        description="Blur background automatically, or blur the supplied masked region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "product", "general"],
        risk_level="medium",
        default_params={"amount": 0.38, "highlight_boost": 0.12, "feather_radius": 18.0},
    )
    params_model = BlurParams
    image_op = staticmethod(apply_background_blur)
    output_prefix = "psagent_background_blur_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, BlurParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "amount": normalized["amount"],
            "highlight_boost": normalized["highlight_boost"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class LensBlurPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="lens_blur",
        description="Simulate stronger lens blur or depth-of-field style blur.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "product", "general"],
        risk_level="medium",
        default_params={"amount": 0.42, "highlight_bloom": 0.16, "feather_radius": 20.0},
    )
    params_model = LensBlurParams
    image_op = staticmethod(apply_lens_blur)
    output_prefix = "psagent_lens_blur_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, LensBlurParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "amount": normalized["amount"],
            "highlight_bloom": normalized["highlight_bloom"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }


class GlowHighlightPackage(DeterministicImageOpPackage):
    spec = PackageSpec(
        name="glow_highlight",
        description="Add bloom-like glow to highlights globally or inside a masked region.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="optional",
        supported_domains=["portrait", "night", "general"],
        risk_level="low",
        default_params={"amount": 0.32, "threshold": 0.62, "warmth": 0.18, "feather_radius": 18.0},
    )
    params_model = GlowParams
    image_op = staticmethod(apply_glow_highlight)
    output_prefix = "psagent_glow_"

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        assert isinstance(parsed, GlowParams)
        payload = parsed.model_dump()
        payload["region"] = operation.get("region") or "whole_image"
        return payload

    def build_image_op_kwargs(self, normalized: dict[str, Any], context: OperationContext, *, mask_path: str | None) -> dict[str, Any]:
        return {
            "amount": normalized["amount"],
            "threshold": normalized["threshold"],
            "warmth": normalized["warmth"],
            "mask_path": mask_path,
            "feather_radius": normalized["feather_radius"],
        }
