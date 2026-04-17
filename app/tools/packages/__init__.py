"""Tool-package exports and default registry builder."""

from __future__ import annotations

from app.tools.packages.adjust_contrast import AdjustContrastPackage
from app.tools.packages.adjust_dehaze import AdjustDehazePackage
from app.tools.packages.adjust_exposure import AdjustExposurePackage
from app.tools.packages.adjust_highlights_shadows import AdjustHighlightsShadowsPackage
from app.tools.packages.adjust_clarity import AdjustClarityPackage
from app.tools.packages.adjust_color_mixer import AdjustColorMixerPackage
from app.tools.packages.adjust_curves import AdjustCurvesPackage
from app.tools.packages.adjust_vibrance_saturation import AdjustVibranceSaturationPackage
from app.tools.packages.adjust_white_balance import AdjustWhiteBalancePackage
from app.tools.packages.adjust_texture import AdjustTexturePackage
from app.tools.packages.adjust_whites_blacks import AdjustWhitesBlacksPackage
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage
from app.tools.packages.creative_optics import (
    ApplyLutPackage,
    AutoUprightPackage,
    BackgroundBlurPackage,
    CameraCalibrationPackage,
    ColorGradingPackage,
    ConvertBlackWhitePackage,
    DefringePackage,
    GlowHighlightPackage,
    GrainPackage,
    LensBlurPackage,
    LensCorrectionPackage,
    MoireReducePackage,
    PerspectiveCorrectionPackage,
    RemoveChromaticAberrationPackage,
    VignettePackage,
)
from app.tools.packages.crop_and_straighten import CropAndStraightenPackage
from app.tools.packages.denoise import DenoisePackage
from app.tools.packages.macros import (
    CleanupDistractingObjectsPackage,
    CleanupSkinBlemishesPackage,
    PortraitBacklightRepairPackage,
    PortraitHairDetailBoostPackage,
    PortraitNaturalWhiteningPackage,
    PortraitRetouchPackage,
    PortraitSkinCleanTonePackage,
    ProductSpecularEnhancePackage,
    RemovePassersbyPackage,
    SummerAiryLookPackage,
    WeddingDressProtectPackage,
)
from app.tools.packages.repair_portrait import (
    BlemishRemovePackage,
    CloneStampPackage,
    EyeBrightenPackage,
    HairEnhancePackage,
    LipEnhancePackage,
    PointColorPackage,
    ReflectionReducePackage,
    RemoveHealPackage,
    SkinSmoothPackage,
    SkinTextureReducePackage,
    SpotHealPackage,
    TeethWhitenPackage,
    UnderEyeBrightenPackage,
)
from app.tools.packages.registry import PackageRegistry
from app.tools.packages.sharpen import SharpenPackage


def build_default_package_registry() -> PackageRegistry:
    """Build a registry containing the first batch of package skeletons."""

    # 第一批默认注册 8 个核心参数工具包。
    registry = PackageRegistry()
    registry.register(AdjustExposurePackage())
    registry.register(AdjustHighlightsShadowsPackage())
    registry.register(AdjustContrastPackage())
    registry.register(AdjustWhitesBlacksPackage())
    registry.register(AdjustCurvesPackage())
    registry.register(AdjustClarityPackage())
    registry.register(AdjustTexturePackage())
    registry.register(AdjustDehazePackage())
    registry.register(AdjustColorMixerPackage())
    registry.register(AdjustWhiteBalancePackage())
    registry.register(AdjustVibranceSaturationPackage())
    registry.register(CropAndStraightenPackage())
    registry.register(DenoisePackage())
    registry.register(SharpenPackage())
    registry.register(RemoveHealPackage())
    registry.register(BlemishRemovePackage())
    registry.register(SkinSmoothPackage())
    registry.register(PointColorPackage())
    registry.register(SpotHealPackage())
    registry.register(CloneStampPackage())
    registry.register(SkinTextureReducePackage())
    registry.register(UnderEyeBrightenPackage())
    registry.register(TeethWhitenPackage())
    registry.register(EyeBrightenPackage())
    registry.register(HairEnhancePackage())
    registry.register(LipEnhancePackage())
    registry.register(ReflectionReducePackage())
    registry.register(LensCorrectionPackage())
    registry.register(RemoveChromaticAberrationPackage())
    registry.register(DefringePackage())
    registry.register(PerspectiveCorrectionPackage())
    registry.register(AutoUprightPackage())
    registry.register(VignettePackage())
    registry.register(GrainPackage())
    registry.register(MoireReducePackage())
    registry.register(ColorGradingPackage())
    registry.register(ApplyLutPackage())
    registry.register(ConvertBlackWhitePackage())
    registry.register(CameraCalibrationPackage())
    registry.register(BackgroundBlurPackage())
    registry.register(LensBlurPackage())
    registry.register(GlowHighlightPackage())
    registry.register(PortraitNaturalWhiteningPackage())
    registry.register(PortraitSkinCleanTonePackage())
    registry.register(PortraitBacklightRepairPackage())
    registry.register(WeddingDressProtectPackage())
    registry.register(SummerAiryLookPackage())
    registry.register(PortraitRetouchPackage())
    registry.register(PortraitHairDetailBoostPackage())
    registry.register(ProductSpecularEnhancePackage())
    registry.register(CleanupSkinBlemishesPackage())
    registry.register(CleanupDistractingObjectsPackage())
    registry.register(RemovePassersbyPackage())
    return registry


__all__ = [
    "AdjustContrastPackage",
    "AdjustDehazePackage",
    "AdjustExposurePackage",
    "AdjustHighlightsShadowsPackage",
    "AdjustClarityPackage",
    "AdjustColorMixerPackage",
    "AdjustCurvesPackage",
    "AdjustVibranceSaturationPackage",
    "AdjustWhiteBalancePackage",
    "AdjustTexturePackage",
    "AdjustWhitesBlacksPackage",
    "ApplyLutPackage",
    "AutoUprightPackage",
    "BackgroundBlurPackage",
    "BlemishRemovePackage",
    "CameraCalibrationPackage",
    "CleanupDistractingObjectsPackage",
    "CleanupSkinBlemishesPackage",
    "CloneStampPackage",
    "ColorGradingPackage",
    "ConvertBlackWhitePackage",
    "CropAndStraightenPackage",
    "DefringePackage",
    "DenoisePackage",
    "EyeBrightenPackage",
    "GlowHighlightPackage",
    "GrainPackage",
    "HairEnhancePackage",
    "LensBlurPackage",
    "LensCorrectionPackage",
    "LipEnhancePackage",
    "MoireReducePackage",
    "OperationContext",
    "PackageRegistry",
    "PackageResult",
    "PackageSpec",
    "PerspectiveCorrectionPackage",
    "PointColorPackage",
    "PortraitBacklightRepairPackage",
    "PortraitHairDetailBoostPackage",
    "PortraitNaturalWhiteningPackage",
    "PortraitRetouchPackage",
    "PortraitSkinCleanTonePackage",
    "ProductSpecularEnhancePackage",
    "ReflectionReducePackage",
    "RemoveChromaticAberrationPackage",
    "RemoveHealPackage",
    "RemovePassersbyPackage",
    "SharpenPackage",
    "SkinSmoothPackage",
    "SkinTextureReducePackage",
    "SpotHealPackage",
    "SummerAiryLookPackage",
    "TeethWhitenPackage",
    "ToolPackage",
    "UnderEyeBrightenPackage",
    "VignettePackage",
    "WeddingDressProtectPackage",
    "build_default_package_registry",
]
