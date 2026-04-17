"""Unit tests for the curves package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustCurvesPackage, OperationContext


class AdjustCurvesPackageTest(unittest.TestCase):
    """Validate curves adjustment behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustCurvesPackage()

        self.gradient_path = str(Path(self.tmpdir.name) / "gradient.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        gradient = Image.new("RGB", (16, 1))
        for x in range(16):
            value = int(round(55 + (x / 15) * 145))
            gradient.putpixel((x, 0), (value, value, value))
        gradient = gradient.resize((16, 16))
        gradient.save(self.gradient_path)

        masked = Image.new("RGB", (8, 8), (120, 120, 120))
        for x in range(4):
            for y in range(8):
                masked.putpixel((x, y), (150, 150, 150))
        masked.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_returns_curves_payload(self) -> None:
        normalized = self.package.normalize(
            {
                "op": "adjust_curves",
                "params": {
                    "shadow_lift": 0.2,
                    "midtone_gamma": 0.95,
                    "highlight_compress": 0.15,
                    "contrast_bias": 0.3,
                },
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertAlmostEqual(normalized["shadow_lift"], 0.2)
        self.assertAlmostEqual(normalized["contrast_bias"], 0.3)

    def test_execute_shapes_global_curve(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_curves",
                "region": "whole_image",
                "params": {
                    "shadow_lift": 0.2,
                    "midtone_gamma": 0.9,
                    "highlight_compress": 0.18,
                    "contrast_bias": 0.28,
                },
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((8, 8)), Image.open(self.gradient_path).convert("L").getpixel((8, 8)))

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_curves",
                "region": "main_subject",
                "params": {
                    "shadow_lift": 0.15,
                    "midtone_gamma": 0.92,
                    "highlight_compress": 0.12,
                    "contrast_bias": 0.2,
                    "feather_radius": 0,
                },
            },
            OperationContext(image_path=self.masked_image_path, masks={"main_subject": self.mask_path}),
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((1, 4)), 150)
        self.assertEqual(output.getpixel((6, 4)), 120)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_curves",
                "region": "main_subject",
                "params": {
                    "shadow_lift": 0.15,
                    "midtone_gamma": 0.92,
                    "highlight_compress": 0.12,
                    "contrast_bias": 0.2,
                },
            },
            OperationContext(image_path=self.masked_image_path),
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
