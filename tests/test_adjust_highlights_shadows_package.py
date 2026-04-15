"""Unit tests for the minimal highlights/shadows package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustHighlightsShadowsPackage, OperationContext


class AdjustHighlightsShadowsPackageTest(unittest.TestCase):
    """Validate the minimal tone-balance implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustHighlightsShadowsPackage()

        self.gradient_path = str(Path(self.tmpdir.name) / "gradient.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked_input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        gradient = Image.new("RGB", (16, 1))
        for x in range(16):
            value = int(round((x / 15) * 255))
            gradient.putpixel((x, 0), (value, value, value))
        gradient = gradient.resize((16, 16))
        gradient.save(self.gradient_path)

        masked_image = Image.new("RGB", (8, 8), (220, 220, 220))
        for x in range(4):
            for y in range(8):
                masked_image.putpixel((x, y), (40, 40, 40))
        masked_image.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_shadow_and_highlight_terms(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_highlights_shadows", "strength": 0.5, "params": {}},
            OperationContext(image_path=self.gradient_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["shadow_amount"], 0.0)
        self.assertGreater(normalized["highlight_amount"], 0.0)
        self.assertAlmostEqual(normalized["tone_amount"], 0.26)
        self.assertAlmostEqual(normalized["local_radius"], 36.0)

    def test_execute_lifts_shadows_and_recovers_highlights(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_highlights_shadows",
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

        # 左侧暗部应该被提亮，右侧亮部应该被压低一些。
        self.assertGreater(output.getpixel((1, 8)), original.getpixel((1, 8)))
        self.assertLess(output.getpixel((14, 8)), original.getpixel((14, 8)))

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_highlights_shadows",
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
        # mask 左半边暗部应该被提亮；右半边不应被改动。
        self.assertGreater(output.getpixel((1, 4)), 40)
        self.assertEqual(output.getpixel((6, 4)), 220)

    def test_execute_returns_fallback_result_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_highlights_shadows",
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
