"""Unit tests for the dehaze package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustDehazePackage, OperationContext


class AdjustDehazePackageTest(unittest.TestCase):
    """Validate dehaze adjustment behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustDehazePackage()

        self.hazy_path = str(Path(self.tmpdir.name) / "hazy.png")
        self.masked_path = str(Path(self.tmpdir.name) / "masked.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        hazy = Image.new("RGB", (16, 16), (150, 155, 162))
        for x in range(16):
            for y in range(16):
                if y > 8:
                    hazy.putpixel((x, y), (118, 125, 133))
        hazy.save(self.hazy_path)

        masked = Image.new("RGB", (8, 8), (148, 150, 155))
        for x in range(4):
            for y in range(8):
                masked.putpixel((x, y), (120, 128, 136))
        masked.save(self.masked_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_returns_dehaze_payload(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_dehaze", "params": {"amount": 0.4}},
            OperationContext(image_path=self.hazy_path),
        )
        self.assertAlmostEqual(normalized["amount"], 0.4)
        self.assertAlmostEqual(normalized["color_protection"], 0.3)

    def test_execute_adjusts_global_haze(self) -> None:
        result = self.package.execute(
            {"op": "adjust_dehaze", "region": "whole_image", "params": {"amount": 0.65}},
            OperationContext(image_path=self.hazy_path),
        )
        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((8, 12)), Image.open(self.hazy_path).convert("L").getpixel((8, 12)))

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_dehaze",
                "region": "background",
                "params": {"amount": 0.5, "feather_radius": 0},
            },
            OperationContext(image_path=self.masked_path, masks={"background": self.mask_path}),
        )
        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((1, 4)), 128)
        self.assertEqual(output.getpixel((6, 4)), 150)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {"op": "adjust_dehaze", "region": "background", "params": {"amount": 0.5}},
            OperationContext(image_path=self.masked_path),
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
