"""Unit tests for the contrast package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustContrastPackage, OperationContext


class AdjustContrastPackageTest(unittest.TestCase):
    """Validate the minimal professional contrast implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustContrastPackage()

        self.gradient_path = str(Path(self.tmpdir.name) / "gradient.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked_input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        gradient = Image.new("RGB", (16, 1))
        for x in range(16):
            value = int(round(80 + (x / 15) * 80))
            gradient.putpixel((x, 0), (value, value, value))
        gradient = gradient.resize((16, 16))
        gradient.save(self.gradient_path)

        masked_image = Image.new("RGB", (8, 8), (120, 120, 120))
        for x in range(4):
            for y in range(8):
                masked_image.putpixel((x, y), (150, 150, 150))
        masked_image.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_contrast_amount(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_contrast", "strength": 0.5, "params": {}},
            OperationContext(image_path=self.gradient_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["contrast_amount"], 0.0)
        self.assertAlmostEqual(normalized["contrast_scale"], 0.7)

    def test_execute_increases_gradient_contrast(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_contrast",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        original = Image.open(self.gradient_path).convert("L")
        output = Image.open(result.output_image).convert("L")

        original_span = original.getpixel((15, 8)) - original.getpixel((0, 8))
        output_span = output.getpixel((15, 8)) - output.getpixel((0, 8))
        self.assertGreater(output_span, original_span)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_contrast",
                "region": "main_subject",
                "strength": 0.8,
                "params": {"feather_radius": 0},
            },
            OperationContext(
                image_path=self.masked_image_path,
                masks={"main_subject": self.mask_path},
            ),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        output = Image.open(result.output_image).convert("L")
        # 左侧被提升对比后应比原值更偏离中灰，右侧保持不变。
        self.assertNotEqual(output.getpixel((1, 4)), 150)
        self.assertEqual(output.getpixel((6, 4)), 120)

    def test_execute_returns_fallback_result_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_contrast",
                "region": "main_subject",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.masked_image_path),
        )

        self.assertFalse(result.ok)
        self.assertTrue(result.fallback_used)
        self.assertIsNotNone(result.error)


if __name__ == "__main__":
    unittest.main()
