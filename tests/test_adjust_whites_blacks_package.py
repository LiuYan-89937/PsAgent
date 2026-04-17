"""Unit tests for the whites/blacks package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustWhitesBlacksPackage, OperationContext


class AdjustWhitesBlacksPackageTest(unittest.TestCase):
    """Validate whites/blacks adjustment behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustWhitesBlacksPackage()

        self.gradient_path = str(Path(self.tmpdir.name) / "gradient.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked_input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        gradient = Image.new("RGB", (16, 1))
        for x in range(16):
            value = int(round(45 + (x / 15) * 170))
            gradient.putpixel((x, 0), (value, value, value))
        gradient = gradient.resize((16, 16))
        gradient.save(self.gradient_path)

        masked = Image.new("RGB", (8, 8), (110, 110, 110))
        for x in range(4):
            for y in range(8):
                masked.putpixel((x, y), (190, 190, 190))
        masked.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_exposes_params(self) -> None:
        normalized = self.package.normalize(
            {
                "op": "adjust_whites_blacks",
                "params": {"whites_amount": 0.5, "blacks_amount": 0.3},
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertAlmostEqual(normalized["whites_amount"], 0.5)
        self.assertAlmostEqual(normalized["blacks_amount"], 0.3)
        self.assertAlmostEqual(normalized["highlight_rolloff"], 0.32)

    def test_normalize_accepts_legacy_top_level_strength(self) -> None:
        normalized = self.package.normalize(
            {
                "op": "adjust_whites_blacks",
                "strength": 0.5,
                "params": {},
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertAlmostEqual(normalized["whites_amount"], 0.4)
        self.assertAlmostEqual(normalized["blacks_amount"], 0.3)

    def test_execute_adjusts_global_tonal_endpoints(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_whites_blacks",
                "region": "whole_image",
                "params": {"whites_amount": 0.6, "blacks_amount": 0.45},
            },
            OperationContext(image_path=self.gradient_path),
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertGreater(output.getpixel((15, 8)), Image.open(self.gradient_path).convert("L").getpixel((15, 8)))
        self.assertLess(output.getpixel((0, 8)), Image.open(self.gradient_path).convert("L").getpixel((0, 8)))

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_whites_blacks",
                "region": "main_subject",
                "params": {"whites_amount": -0.45, "blacks_amount": 0.5, "feather_radius": 0},
            },
            OperationContext(image_path=self.masked_image_path, masks={"main_subject": self.mask_path}),
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((1, 4)), 190)
        self.assertEqual(output.getpixel((6, 4)), 110)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_whites_blacks",
                "region": "main_subject",
                "params": {"whites_amount": 0.4, "blacks_amount": 0.2},
            },
            OperationContext(image_path=self.masked_image_path),
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
