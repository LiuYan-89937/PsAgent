"""Macro tool packages that expand into primitive deterministic operations."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import Field

from app.tools import MACRO_TOOL_NAMES
from app.tools.packages.base import OperationContext, PackageParamsModel, PackageResult, PackageSpec, ToolPackage


class MacroParams(PackageParamsModel):
    """Shared params for macro-style tools."""

    strength: float = Field(0.32, ge=0.0, le=1.0)
    preserve_natural: bool = True


def _mask_payload(prompt: str, negative_prompt: str | None = None) -> dict[str, Any]:
    payload = {
        "mask_provider": "fal_sam3",
        "mask_prompt": prompt,
        "mask_semantic_type": True,
    }
    if negative_prompt:
        payload["mask_negative_prompt"] = negative_prompt
    return payload


def _macro_strength(operation: dict[str, Any]) -> float:
    raw = (operation.get("params") or {}).get("strength", operation.get("strength", 0.32))
    try:
        return max(0.0, min(float(raw), 1.0))
    except (TypeError, ValueError):
        return 0.32


def is_macro_tool(name: str) -> bool:
    """Return whether a tool name is one of the registered macro tools."""

    return name in MACRO_TOOL_NAMES


class MacroPackage(ToolPackage):
    """Base class for macro packages that expand into primitive operations."""

    params_model = MacroParams
    spec: PackageSpec

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        self.parse_params(operation)
        if not context.image_path:
            raise ValueError(f"image_path is required for macro package {self.name}")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        return {"requires_mask": False, "required_region": None}

    def normalize(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        parsed = self.parse_params(operation)
        if not isinstance(parsed, MacroParams):
            raise ValueError("Macro params model is not configured.")
        return {
            "region": operation.get("region") or "whole_image",
            "params": parsed.model_dump(),
            "strength": parsed.strength,
            "preserve_natural": parsed.preserve_natural,
        }

    @abstractmethod
    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        """Expand a macro operation into primitive operations."""

    def execute(self, operation: dict[str, Any], context: OperationContext) -> PackageResult:
        self.validate(operation, context)
        from app.tools.packages import build_default_package_registry

        registry = build_default_package_registry()
        current_image = context.image_path or ""
        substeps: list[dict[str, Any]] = []
        for sub_operation in self.expand(operation, context):
            sub_context = OperationContext(
                image_path=current_image,
                image_analysis=context.image_analysis,
                retrieved_prefs=context.retrieved_prefs,
                masks=context.masks,
                thread_id=context.thread_id,
                audit=context.audit,
            )
            result = registry.require(sub_operation["op"]).execute(sub_operation, sub_context)
            substeps.append(
                {
                    "op": sub_operation["op"],
                    "region": sub_operation.get("region", "whole_image"),
                    "ok": result.ok,
                    "output_image": result.output_image,
                    "error": result.error,
                }
            )
            if not result.ok:
                return PackageResult(
                    ok=False,
                    package=self.name,
                    output_image=current_image or None,
                    applied_params={"operation": operation},
                    artifacts={"substeps": substeps},
                    error=result.error,
                )
            if result.output_image:
                current_image = result.output_image

        return PackageResult(
            ok=True,
            package=self.name,
            output_image=current_image or None,
            applied_params={"operation": operation},
            artifacts={"substeps": substeps},
        )


class PortraitNaturalWhiteningPackage(MacroPackage):
    spec = PackageSpec(
        name="portrait_natural_whitening",
        description="Natural portrait whitening using skin-safe local tonal and color cleanup.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        mask = _mask_payload(
            "the subject's face, neck skin, and other visible skin",
            "hair, clothing, accessories, background",
        )
        return [
            {"op": "point_color", "region": "skin whitening area", "params": {"strength": strength * 0.5, "target_color": "skin", "luminance_shift": 0.12 + strength * 0.08, "saturation_shift": -0.05 - strength * 0.04, **mask}},
            {"op": "adjust_white_balance", "region": "skin whitening area", "params": {"strength": -0.03 - strength * 0.05, "protect_saturated": 0.42, **mask}},
            {"op": "skin_smooth", "region": "skin whitening area", "params": {"strength": 0.08 + strength * 0.16, "preserve_detail": 0.82, **mask}},
        ]


class PortraitSkinCleanTonePackage(MacroPackage):
    spec = PackageSpec(
        name="portrait_skin_clean_tone",
        description="Clean up skin tone with restrained point color and texture control.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        mask = _mask_payload(
            "the subject's face, neck skin, and other visible skin",
            "hair, clothing, accessories, background",
        )
        return [
            {"op": "point_color", "region": "clean skin tone area", "params": {"strength": strength * 0.42, "target_color": "skin", "saturation_shift": -0.04, "luminance_shift": 0.06 + strength * 0.05, **mask}},
            {"op": "skin_texture_reduce", "region": "clean skin tone area", "params": {"strength": 0.1 + strength * 0.15, **mask}},
            {"op": "adjust_exposure", "region": "clean skin tone area", "params": {"strength": 0.04 + strength * 0.08, "feather_radius": 18.0, **mask}},
        ]


class PortraitBacklightRepairPackage(MacroPackage):
    spec = PackageSpec(
        name="portrait_backlight_repair",
        description="Repair a backlit portrait with local exposure, shadow, and skin balance control.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        subject_mask = _mask_payload(
            "the subject's face, neck skin, and visible upper body affected by backlight",
            "hair, background, unrelated objects",
        )
        return [
            {"op": "adjust_exposure", "region": "backlit portrait subject", "params": {"strength": 0.16 + strength * 0.22, "max_stops": 1.45, **subject_mask}},
            {"op": "adjust_highlights_shadows", "region": "backlit portrait subject", "params": {"strength": 0.14 + strength * 0.16, "tone_amount": 0.3, **subject_mask}},
            {"op": "point_color", "region": "backlit portrait subject", "params": {"strength": strength * 0.3, "target_color": "skin", "luminance_shift": 0.04, "saturation_shift": 0.03, **subject_mask}},
        ]


class WeddingDressProtectPackage(MacroPackage):
    spec = PackageSpec(
        name="wedding_dress_protect",
        description="Protect white dress detail with highlight control and precise point color cleanup.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait", "wedding"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        dress_mask = _mask_payload(
            "the subject's wedding dress or white dress area",
            "face, skin, hair, background",
        )
        return [
            {"op": "adjust_highlights_shadows", "region": "wedding dress area", "params": {"strength": 0.14 + strength * 0.12, "tone_amount": 0.24, **dress_mask}},
            {"op": "point_color", "region": "wedding dress area", "params": {"strength": strength * 0.24, "target_color": "white", "saturation_shift": -0.08, "luminance_shift": 0.1, **dress_mask}},
            {"op": "adjust_texture", "region": "wedding dress area", "params": {"amount": 0.05 + strength * 0.08, "detail_scale": 0.9, **dress_mask}},
        ]


class SummerAiryLookPackage(MacroPackage):
    spec = PackageSpec(
        name="summer_airy_look",
        description="Create a summer airy look with restrained warmth, vibrance, and background cleanup.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="low",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        background_mask = _mask_payload(
            "the background behind the main subject",
            "person, face, skin, foreground objects",
        )
        return [
            {"op": "adjust_white_balance", "region": "whole_image", "params": {"strength": 0.06 + strength * 0.08, "protect_saturated": 0.34}},
            {"op": "adjust_vibrance_saturation", "region": "whole_image", "params": {"strength": 0.08 + strength * 0.14, "protect_skin": 0.42}},
            {"op": "adjust_dehaze", "region": "summer airy background", "params": {"amount": 0.08 + strength * 0.08, "luminance_protection": 0.34, **background_mask}},
            {"op": "vignette", "region": "whole_image", "params": {"amount": -0.08 - strength * 0.06, "midpoint": 0.72}},
        ]


class PortraitRetouchPackage(MacroPackage):
    spec = PackageSpec(
        name="portrait_retouch",
        description="Full portrait retouch macro combining blemish cleanup, skin work, and facial feature finishing.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="high",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        skin_mask = _mask_payload("the subject's face, neck skin, and other visible skin", "hair, clothing, accessories, background")
        under_eye_mask = _mask_payload("the subject's under-eye shadow area", "eyebrows, hair, background")
        eye_mask = _mask_payload("the subject's eyes and irises", "skin, hair, background")
        teeth_mask = _mask_payload("the subject's visible teeth", "lips, skin, background")
        hair_mask = _mask_payload("the subject's hair and loose hair edges", "face, skin, clothing, background")
        return [
            {"op": "blemish_remove", "region": "portrait blemish area", "params": {"strength": 0.12 + strength * 0.22, **skin_mask}},
            {"op": "under_eye_brighten", "region": "portrait under eye area", "params": {"strength": 0.14 + strength * 0.18, **under_eye_mask}},
            {"op": "skin_smooth", "region": "portrait skin area", "params": {"strength": 0.08 + strength * 0.12, "preserve_detail": 0.84, **skin_mask}},
            {"op": "eye_brighten", "region": "portrait eye area", "params": {"strength": 0.1 + strength * 0.16, **eye_mask}},
            {"op": "teeth_whiten", "region": "portrait teeth area", "params": {"strength": 0.12 + strength * 0.14, **teeth_mask}},
            {"op": "hair_enhance", "region": "portrait hair area", "params": {"strength": 0.08 + strength * 0.12, **hair_mask}},
        ]


class PortraitHairDetailBoostPackage(MacroPackage):
    spec = PackageSpec(
        name="portrait_hair_detail_boost",
        description="Enhance hair detail with a hair-focused mask and restrained sharpening.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        hair_mask = _mask_payload("the subject's hair and loose hair edges", "face, skin, background")
        return [
            {"op": "hair_enhance", "region": "hair detail area", "params": {"strength": 0.14 + strength * 0.18, **hair_mask}},
            {"op": "sharpen", "region": "hair detail area", "params": {"strength": 0.08 + strength * 0.1, "highlight_protection": 0.18, **hair_mask}},
        ]


class ProductSpecularEnhancePackage(MacroPackage):
    spec = PackageSpec(
        name="product_specular_enhance",
        description="Enhance glossy product highlights and crisp edges without clipping detail.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["product", "general"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        spec_mask = _mask_payload("product specular highlights, glossy edges, and reflective surfaces", "background, person")
        return [
            {"op": "glow_highlight", "region": "product specular area", "params": {"amount": 0.12 + strength * 0.12, "threshold": 0.68, **spec_mask}},
            {"op": "sharpen", "region": "product specular area", "params": {"strength": 0.1 + strength * 0.12, "highlight_protection": 0.12, **spec_mask}},
            {"op": "adjust_highlights_shadows", "region": "product specular area", "params": {"strength": 0.06 + strength * 0.08, "tone_amount": 0.2, **spec_mask}},
        ]


class CleanupSkinBlemishesPackage(MacroPackage):
    spec = PackageSpec(
        name="cleanup_skin_blemishes",
        description="Clean up skin blemishes and lightly smooth the cleaned area.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        skin_mask = _mask_payload("the subject's visible skin with small blemishes", "hair, clothing, background")
        return [
            {"op": "blemish_remove", "region": "skin blemish area", "params": {"strength": 0.14 + strength * 0.24, **skin_mask}},
            {"op": "skin_smooth", "region": "skin blemish area", "params": {"strength": 0.05 + strength * 0.08, **skin_mask}},
        ]


class CleanupDistractingObjectsPackage(MacroPackage):
    spec = PackageSpec(
        name="cleanup_distracting_objects",
        description="Clean up distracting objects with a single remove-heal pass.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait", "landscape", "general"],
        risk_level="medium",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        object_mask = _mask_payload("the distracting object or clutter that should be removed", "main subject, important foreground")
        return [{"op": "remove_heal", "region": "distracting object area", "params": {"strength": 0.18 + strength * 0.3, **object_mask}}]


class RemovePassersbyPackage(MacroPackage):
    spec = PackageSpec(
        name="remove_passersby",
        description="Remove passersby in the background with a people-removal repair pass.",
        supported_regions=["whole_image", "masked_region"],
        mask_policy="none",
        supported_domains=["portrait", "travel", "general"],
        risk_level="high",
    )

    def expand(self, operation: dict[str, Any], context: OperationContext) -> list[dict[str, Any]]:
        strength = _macro_strength(operation)
        people_mask = _mask_payload("background passersby and distracting people", "main subject, foreground objects")
        return [{"op": "remove_heal", "region": "passersby area", "params": {"strength": 0.22 + strength * 0.32, "radius_px": 4.0, **people_mask}}]


def expand_macro_operation(
    operation: dict[str, Any],
    *,
    context: OperationContext | None = None,
) -> list[dict[str, Any]]:
    """Expand a macro operation into primitive operations, or return it unchanged."""

    op_name = str(operation.get("op") or "")
    if op_name not in MACRO_TOOL_NAMES:
        return [operation]

    local_context = context or OperationContext()
    registry = {
        "portrait_natural_whitening": PortraitNaturalWhiteningPackage(),
        "portrait_skin_clean_tone": PortraitSkinCleanTonePackage(),
        "portrait_backlight_repair": PortraitBacklightRepairPackage(),
        "wedding_dress_protect": WeddingDressProtectPackage(),
        "summer_airy_look": SummerAiryLookPackage(),
        "portrait_retouch": PortraitRetouchPackage(),
        "portrait_hair_detail_boost": PortraitHairDetailBoostPackage(),
        "product_specular_enhance": ProductSpecularEnhancePackage(),
        "cleanup_skin_blemishes": CleanupSkinBlemishesPackage(),
        "cleanup_distracting_objects": CleanupDistractingObjectsPackage(),
        "remove_passersby": RemovePassersbyPackage(),
    }
    return registry[op_name].expand(operation, local_context)


def expand_macro_operations(
    operations: list[dict[str, Any]],
    *,
    context: OperationContext | None = None,
) -> list[dict[str, Any]]:
    """Expand any macro operations inside a list into primitive operations."""

    expanded: list[dict[str, Any]] = []
    for operation in operations:
        sub_operations = expand_macro_operation(operation, context=context)
        for sub_operation in sub_operations:
            if is_macro_tool(str(sub_operation.get("op") or "")):
                expanded.extend(expand_macro_operations([sub_operation], context=context))
            else:
                expanded.append(sub_operation)
    return expanded


def operations_require_hybrid(operations: list[dict[str, Any]], *, context: OperationContext | None = None) -> bool:
    """Return whether a plan expands into any mask-guided local work."""

    for operation in expand_macro_operations(operations, context=context):
        params = operation.get("params") or {}
        if params.get("mask_prompt"):
            return True
    return False
