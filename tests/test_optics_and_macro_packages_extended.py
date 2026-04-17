"""Tests for optics/fx packages and macro expansion helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import (
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
    OperationContext,
)
from app.tools.packages.macros import expand_macro_operation, operations_require_hybrid


class OpticsAndMacroPackagesExtendedTest(unittest.TestCase):
    """Verify optics/fx tools execute and macros expand into primitive plans."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "scene.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        image = Image.new("RGB", (64, 64), (110, 150, 210))
        for x in range(18, 46):
            for y in range(18, 46):
                image.putpixel((x, y), (225, 220, 210))
        image.save(self.image_path)

        mask = Image.new("L", (64, 64), 0)
        for x in range(8, 56):
            for y in range(8, 56):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

        self.context = OperationContext(image_path=self.image_path, masks={"focus": self.mask_path})

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_whole_image_only_optics_tools_execute(self) -> None:
        tools = [
            (LensCorrectionPackage(), {"strength": 0.3}),
            (RemoveChromaticAberrationPackage(), {"strength": 0.2}),
            (DefringePackage(), {"strength": 0.25}),
            (PerspectiveCorrectionPackage(), {"vertical_amount": 0.3, "horizontal_amount": -0.2}),
            (AutoUprightPackage(), {"strength": 0.5}),
            (MoireReducePackage(), {"amount": 0.25}),
        ]

        for package, params in tools:
            with self.subTest(package=package.name):
                result = package.execute({"op": package.name, "region": "whole_image", "params": params}, self.context)
                self.assertTrue(result.ok, msg=f"{package.name} failed with: {result.error}")

    def test_local_capable_fx_tools_execute(self) -> None:
        tools = [
            (VignettePackage(), {"amount": 0.2}),
            (GrainPackage(), {"amount": 0.16}),
            (ColorGradingPackage(), {"shadow_hue": 220.0, "highlight_hue": 36.0}),
            (ApplyLutPackage(), {"preset": "warm_film", "strength": 0.45}),
            (ConvertBlackWhitePackage(), {"contrast": 0.25, "filter_color": "yellow"}),
            (CameraCalibrationPackage(), {"red_bias": 0.15, "blue_bias": -0.08}),
            (BackgroundBlurPackage(), {"amount": 0.3}),
            (LensBlurPackage(), {"amount": 0.32}),
            (GlowHighlightPackage(), {"amount": 0.22}),
        ]

        for package, params in tools:
            with self.subTest(package=package.name):
                result = package.execute({"op": package.name, "region": "focus", "params": params}, self.context)
                self.assertTrue(result.ok, msg=f"{package.name} failed with: {result.error}")

    def test_macro_expansion_produces_expected_substeps_and_hybrid_requirement(self) -> None:
        operation = {"op": "portrait_retouch", "region": "whole_image", "params": {"strength": 0.35}}
        expanded = expand_macro_operation(operation)

        self.assertEqual(
            [item["op"] for item in expanded],
            ["blemish_remove", "under_eye_brighten", "skin_smooth", "eye_brighten", "teeth_whiten", "hair_enhance"],
        )
        self.assertTrue(operations_require_hybrid([operation]))


if __name__ == "__main__":
    unittest.main()
