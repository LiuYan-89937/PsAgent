"""Tool-package exports and default registry builder."""

from __future__ import annotations

from app.tools.packages.adjust_contrast import AdjustContrastPackage
from app.tools.packages.adjust_exposure import AdjustExposurePackage
from app.tools.packages.adjust_highlights_shadows import AdjustHighlightsShadowsPackage
from app.tools.packages.adjust_vibrance_saturation import AdjustVibranceSaturationPackage
from app.tools.packages.adjust_white_balance import AdjustWhiteBalancePackage
from app.tools.packages.base import OperationContext, PackageResult, PackageSpec, ToolPackage
from app.tools.packages.crop_and_straighten import CropAndStraightenPackage
from app.tools.packages.denoise import DenoisePackage
from app.tools.packages.registry import PackageRegistry
from app.tools.packages.sharpen import SharpenPackage


def build_default_package_registry() -> PackageRegistry:
    """Build a registry containing the first batch of package skeletons."""

    # 第一批默认注册 8 个核心参数工具包。
    registry = PackageRegistry()
    registry.register(AdjustExposurePackage())
    registry.register(AdjustHighlightsShadowsPackage())
    registry.register(AdjustContrastPackage())
    registry.register(AdjustWhiteBalancePackage())
    registry.register(AdjustVibranceSaturationPackage())
    registry.register(CropAndStraightenPackage())
    registry.register(DenoisePackage())
    registry.register(SharpenPackage())
    return registry


__all__ = [
    "AdjustContrastPackage",
    "AdjustExposurePackage",
    "AdjustHighlightsShadowsPackage",
    "AdjustVibranceSaturationPackage",
    "AdjustWhiteBalancePackage",
    "CropAndStraightenPackage",
    "DenoisePackage",
    "OperationContext",
    "PackageRegistry",
    "PackageResult",
    "PackageSpec",
    "SharpenPackage",
    "ToolPackage",
    "build_default_package_registry",
]
