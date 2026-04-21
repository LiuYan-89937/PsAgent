"""Real-portrait fixture helpers for full native tool output testing."""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.segmentation_tools import resolve_region_mask
from app.tools import (
    COLOR_TOOL_NAMES,
    DETAIL_TOOL_NAMES,
    EFFECT_TOOL_NAMES,
    PORTRAIT_TOOL_NAMES,
    TONE_TOOL_NAMES,
    require_tool,
    require_tool_spec,
)


TESTS_DIR = Path("/Users/liuyan/Desktop/PsAgent/tests")
FIXTURE_DIR = TESTS_DIR / "fixtures" / "real_portrait"
FIXTURE_IMAGE_PATH = FIXTURE_DIR / "source_test.jpg"
FIXTURE_MASK_DIR = FIXTURE_DIR / "masks_sam"
OUTPUT_DIR = TESTS_DIR / "output" / "real_portrait_tools"

MASK_PATHS = {
    "person": FIXTURE_MASK_DIR / "person.png",
    "face": FIXTURE_MASK_DIR / "face.png",
    "hair": FIXTURE_MASK_DIR / "hair.png",
    "eye": FIXTURE_MASK_DIR / "eye.png",
    "under_eye": FIXTURE_MASK_DIR / "under_eye.png",
    "lips": FIXTURE_MASK_DIR / "lips.png",
    "teeth": FIXTURE_MASK_DIR / "teeth.png",
    "skin": FIXTURE_MASK_DIR / "skin.png",
    "background": FIXTURE_MASK_DIR / "background.png",
}

TOOL_MASK_NAME = {
    "adjust_skin_smooth": "skin",
    "adjust_skin_brightness": "skin",
    "adjust_under_eye_brighten": "under_eye",
    "adjust_teeth_whiten": "teeth",
    "adjust_eye_brighten": "eye",
    "adjust_lip_enhance": "lips",
    "adjust_hair_enhance": "hair",
    "adjust_face_color_cleanup": "face",
    "adjust_skin_tone_balance": "skin",
    "adjust_color_cleanup": "face",
}

SAM_PROMPT_BY_MASK_NAME = {
    "person": "person",
    "face": "face",
    "hair": "hair",
    "eye": "eye",
    "under_eye": "under eye",
    "lips": "lips",
    "teeth": "teeth",
    "skin": "skin",
    "background": "background",
}

_REAL_OUTPUT_ROOT_PREPARED = False

BASE_REAL_EFFECT_ARGS_OVERRIDES: dict[str, dict] = {
    "adjust_brightness": {"brightness_offset": 0.12},
    "adjust_highlights_shadows": {"shadow_amount": 0.35, "highlight_amount": 0.22},
    "adjust_whites_blacks": {"white_point_shift": 0.22, "black_point_shift": 0.18},
    "adjust_levels": {"input_black": 0.06, "input_white": 0.94, "gamma": 1.15},
    "adjust_curves": {
        "shadow_lift": 0.18,
        "midtone_gamma": 1.1,
        "highlight_compress": 0.12,
        "contrast_bias": 0.18,
    },
    "adjust_midtones": {"midtone_shift": 0.18},
    "adjust_temperature_tint": {"temperature_shift": 8.0, "tint_shift": 4.0},
    "adjust_hue_saturation": {"hue_shift": 12.0, "saturation_shift": 0.18, "lightness_shift": 0.06},
    "adjust_color_balance": {"midtone_yellow_blue": 0.28, "shadow_cyan_red": -0.12},
    "adjust_color_mixer": {"blue_saturation": 0.35, "orange_luminance": 0.18},
    "adjust_point_color": {"target_color": "skin", "saturation_shift": -0.18, "luminance_shift": 0.16},
    "adjust_selective_color": {"target_band": "neutrals", "yellow_shift": -0.22, "black_shift": 0.12},
    "adjust_single_color_shift": {
        "target_hue": 50.0,
        "hue_shift": 14.0,
        "saturation_shift": -0.2,
        "luminance_shift": 0.12,
    },
    "adjust_neutral_clean_tone": {
        "yellow_blue_shift": -0.18,
        "green_magenta_shift": 0.08,
        "brightness_shift": 0.08,
    },
    "adjust_skin_tone_balance": {
        "skin_hue_shift": 6.0,
        "skin_saturation_shift": -0.12,
        "skin_luminance_shift": 0.12,
    },
    "adjust_color_cleanup": {
        "yellow_reduce": 0.24,
        "green_reduce": 0.18,
        "magenta_balance": 0.12,
        "shadow_desaturate": 0.16,
    },
    "adjust_color_overlay": {
        "overlay_hue": 35.0,
        "overlay_saturation": 0.65,
        "overlay_luminance": 0.65,
        "opacity": 0.32,
    },
    "adjust_channel_mixer": {
        "red_from_red": 1.08,
        "red_from_green": 0.1,
        "green_from_green": 0.92,
        "blue_from_blue": 1.08,
    },
    "adjust_color_grading": {
        "shadow_hue": 220.0,
        "shadow_saturation": 0.12,
        "highlight_hue": 36.0,
        "highlight_saturation": 0.14,
        "blending": 0.55,
    },
    "adjust_chromatic_aberration": {"amount": 0.8, "radial_bias": 0.8},
    "adjust_defringe": {"purple_amount": 0.8, "green_amount": 0.8, "edge_threshold": 0.06},
    "adjust_glow_highlights": {"amount": 0.75, "threshold": 0.55, "warmth": 0.2},
}

REAL_EFFECT_ARGS_OVERRIDES: dict[str, dict] = {
    **BASE_REAL_EFFECT_ARGS_OVERRIDES,
    "adjust_exposure": {"strength": 0.58, "max_stops": 2.2},
    "adjust_contrast": {"strength": 0.5, "contrast_scale": 1.15},
    "adjust_vibrance_saturation": {"strength": 0.45, "vibrance_scale": 1.0, "saturation_scale": 0.6},
    "adjust_local_contrast": {"amount": 0.38, "radius": 18.0, "edge_protection": 0.28},
    "adjust_dehaze": {"amount": 0.42, "highlight_protection": 0.22, "color_protection": 0.2},
    "adjust_noise_reduction": {"luma_strength": 8.0, "chroma_strength": 7.0, "detail_protection": 0.3},
    "adjust_color_noise_reduction": {"chroma_strength": 11.0, "detail_protection": 0.35},
    "adjust_texture": {"amount": 0.42, "detail_scale": 1.2},
    "adjust_clarity": {"amount": 0.35, "radius_scale": 1.1},
    "adjust_sharpness": {"amount": 0.75, "radius": 1.1, "threshold": 0.02},
    "adjust_skin_smooth": {"strength": 0.45, "smooth_strength": 0.42, "detail_protection": 0.72},
    "adjust_skin_brightness": {"brightness_shift": 0.16, "saturation_shift": -0.06},
    "adjust_under_eye_brighten": {"amount": 0.32, "contrast_soften": 0.2, "shadow_lift": 0.28},
    "adjust_teeth_whiten": {"yellow_reduce": 0.32, "brightness_increase": 0.22, "neutralize_gray": 0.14},
    "adjust_eye_brighten": {"brightness_increase": 0.2, "clarity_boost": 0.18, "highlight_boost": 0.12},
    "adjust_lip_enhance": {"saturation_boost": 0.18, "brightness_shift": 0.08, "gloss_boost": 0.08},
    "adjust_hair_enhance": {"texture_boost": 0.36, "clarity_boost": 0.24, "highlight_control": 0.28},
    "adjust_face_color_cleanup": {"yellow_reduce": 0.14, "magenta_balance": 0.08, "shadow_desaturate": 0.1},
    "adjust_soften_local_contrast": {"amount": 0.34, "radius": 10.0, "highlight_preserve": 0.35},
}


def ensure_real_portrait_fixture() -> None:
    """Ensure the copied real-portrait fixture and masks exist."""

    missing = [path for path in [FIXTURE_IMAGE_PATH] if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing real portrait fixture files: {joined}")


def mean_abs_diff(path_a: str | Path, path_b: str | Path) -> float:
    """Return mean absolute RGB difference between two images."""

    image_a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    image_b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    return float(np.abs(image_a - image_b).mean())


def ensure_real_portrait_sam_masks() -> None:
    """Ensure cached real-portrait SAM masks exist, generating them on first run if needed."""

    ensure_real_portrait_fixture()
    generated: list[dict[str, str]] = []
    required = list(MASK_PATHS.values())
    if not all(path.exists() for path in required):
        FIXTURE_MASK_DIR.mkdir(parents=True, exist_ok=True)
        generation_root = FIXTURE_DIR / "sam_generated"
        generation_root.mkdir(parents=True, exist_ok=True)
        for mask_name, prompt in SAM_PROMPT_BY_MASK_NAME.items():
            prompt_dir = generation_root / mask_name
            prompt_dir.mkdir(parents=True, exist_ok=True)
            result = resolve_region_mask(
                str(FIXTURE_IMAGE_PATH),
                prompt,
                provider="fal_sam3",
                prompt=prompt,
                output_dir=str(prompt_dir),
            )
            shutil.copy2(result.binary_mask_path, MASK_PATHS[mask_name])
            generated.append(
                {
                    "mask_name": mask_name,
                    "prompt": prompt,
                    "provider": result.provider,
                    "binary_mask_path": str(MASK_PATHS[mask_name]),
                }
            )
    else:
        for mask_name, prompt in SAM_PROMPT_BY_MASK_NAME.items():
            generated.append(
                {
                    "mask_name": mask_name,
                    "prompt": prompt,
                    "provider": "fal_sam3",
                    "binary_mask_path": str(MASK_PATHS[mask_name]),
                }
            )

    (FIXTURE_MASK_DIR / "meta.json").write_text(
        json.dumps(
            {
                "provider": "fal_sam3",
                "image": str(FIXTURE_IMAGE_PATH),
                "generated": generated,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def real_mask_name_for_tool(tool_name: str) -> str:
    """Return the most suitable real-portrait mask for a tool."""

    return TOOL_MASK_NAME.get(tool_name, "person")


def real_effect_args_for(tool_name: str) -> dict:
    """Return a slightly stronger visible argument set for the real portrait fixture."""

    return dict(REAL_EFFECT_ARGS_OVERRIDES.get(tool_name, {}))


def masked_region_diffs(
    original_path: str | Path,
    adjusted_path: str | Path,
    mask_path: str | Path,
) -> tuple[float, float]:
    """Return mean RGB delta inside and outside a mask."""

    original = np.asarray(Image.open(original_path).convert("RGB"), dtype=np.float32)
    adjusted = np.asarray(Image.open(adjusted_path).convert("RGB"), dtype=np.float32)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    delta = np.abs(adjusted - original).mean(axis=2)
    inside = float(delta[mask >= 0.5].mean()) if np.any(mask >= 0.5) else 0.0
    outside = float(delta[mask < 0.5].mean()) if np.any(mask < 0.5) else 0.0
    return inside, outside


class RealPortraitToolOutputsMixin(unittest.TestCase):
    """Shared real-image integration behavior for native tools."""

    TOOL_NAMES: tuple[str, ...] = ()

    @classmethod
    def setUpClass(cls) -> None:
        global _REAL_OUTPUT_ROOT_PREPARED
        ensure_real_portrait_fixture()
        ensure_real_portrait_sam_masks()
        if not _REAL_OUTPUT_ROOT_PREPARED:
            if OUTPUT_DIR.exists():
                shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            _REAL_OUTPUT_ROOT_PREPARED = True

    def assert_real_portrait_tool_outputs(self, tool_name: str) -> None:
        """Run one tool on the real portrait fixture and persist outputs to tests/output."""

        tool = require_tool(tool_name)
        spec = require_tool_spec(tool_name)
        effect_args = real_effect_args_for(tool_name)
        tool_dir = OUTPUT_DIR / tool_name
        tool_dir.mkdir(parents=True, exist_ok=True)

        whole_result = tool.invoke({"image_path": str(FIXTURE_IMAGE_PATH), **effect_args})
        self.assertTrue(whole_result["ok"], tool_name)
        whole_output_path = tool_dir / "whole.png"
        shutil.copy2(whole_result["output_image"], whole_output_path)
        self.assertTrue(whole_output_path.exists(), tool_name)
        self.assertGreater(mean_abs_diff(str(FIXTURE_IMAGE_PATH), str(whole_output_path)), 0.03, tool_name)

        meta: dict[str, object] = {
            "tool": tool_name,
            "family": spec.family,
            "input_image": str(FIXTURE_IMAGE_PATH),
            "whole_output": str(whole_output_path),
            "whole_params": effect_args,
            "supports_mask": spec.supports_mask,
        }

        if spec.supports_mask:
            mask_name = real_mask_name_for_tool(tool_name)
            mask_path = MASK_PATHS[mask_name]
            schema_properties = tool.get_input_schema().model_json_schema().get("properties", {})
            masked_args = {"image_path": str(FIXTURE_IMAGE_PATH), "mask_path": str(mask_path), **effect_args}
            if "feather_radius" in schema_properties:
                masked_args["feather_radius"] = 0.0

            masked_result = tool.invoke(masked_args)
            self.assertTrue(masked_result["ok"], tool_name)
            masked_output_path = tool_dir / f"masked__{mask_name}.png"
            shutil.copy2(masked_result["output_image"], masked_output_path)
            self.assertTrue(masked_output_path.exists(), tool_name)

            inside_diff, outside_diff = masked_region_diffs(
                FIXTURE_IMAGE_PATH,
                masked_output_path,
                mask_path,
            )
            self.assertGreater(inside_diff, outside_diff + 0.01, tool_name)
            self.assertGreater(inside_diff, 0.01, tool_name)

            meta.update(
                {
                    "mask_name": mask_name,
                    "mask_path": str(mask_path),
                    "masked_output": str(masked_output_path),
                    "masked_inside_diff": inside_diff,
                    "masked_outside_diff": outside_diff,
                }
            )

        (tool_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class RealPortraitToneToolsTest(RealPortraitToolOutputsMixin):
    """Run tone tools on the provided real portrait image and persist outputs."""

    TOOL_NAMES = tuple(TONE_TOOL_NAMES)

    def test_real_portrait_tone_tools(self) -> None:
        for tool_name in self.TOOL_NAMES:
            with self.subTest(tool=tool_name):
                self.assert_real_portrait_tool_outputs(tool_name)


class RealPortraitColorToolsTest(RealPortraitToolOutputsMixin):
    """Run color tools on the provided real portrait image and persist outputs."""

    TOOL_NAMES = tuple(COLOR_TOOL_NAMES)

    def test_real_portrait_color_tools(self) -> None:
        for tool_name in self.TOOL_NAMES:
            with self.subTest(tool=tool_name):
                self.assert_real_portrait_tool_outputs(tool_name)


class RealPortraitDetailToolsTest(RealPortraitToolOutputsMixin):
    """Run detail tools on the provided real portrait image and persist outputs."""

    TOOL_NAMES = tuple(DETAIL_TOOL_NAMES)

    def test_real_portrait_detail_tools(self) -> None:
        for tool_name in self.TOOL_NAMES:
            with self.subTest(tool=tool_name):
                self.assert_real_portrait_tool_outputs(tool_name)


class RealPortraitEffectToolsTest(RealPortraitToolOutputsMixin):
    """Run effect tools on the provided real portrait image and persist outputs."""

    TOOL_NAMES = tuple(EFFECT_TOOL_NAMES)

    def test_real_portrait_effect_tools(self) -> None:
        for tool_name in self.TOOL_NAMES:
            with self.subTest(tool=tool_name):
                self.assert_real_portrait_tool_outputs(tool_name)


class RealPortraitPortraitToolsTest(RealPortraitToolOutputsMixin):
    """Run portrait tools on the provided real portrait image and persist outputs."""

    TOOL_NAMES = tuple(PORTRAIT_TOOL_NAMES)

    def test_real_portrait_portrait_tools(self) -> None:
        for tool_name in self.TOOL_NAMES:
            with self.subTest(tool=tool_name):
                self.assert_real_portrait_tool_outputs(tool_name)
