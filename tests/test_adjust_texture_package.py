"""Unit tests for the texture package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.tools.packages import AdjustTexturePackage, OperationContext


class AdjustTexturePackageTest(unittest.TestCase):
    """Validate texture adjustment behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustTexturePackage()

        self.detail_path = str(Path(self.tmpdir.name) / "detail.png")
        self.masked_path = str(Path(self.tmpdir.name) / "masked.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        detail = Image.new("RGB", (16, 16), (122, 122, 122))
        for x in range(16):
            for y in range(16):
                if (x % 3) == 0:
                    detail.putpixel((x, y), (140, 140, 140))
        detail.save(self.detail_path)

        masked = Image.new("RGB", (8, 8), (122, 122, 122))
        for x in range(4):
            for y in range(8):
                masked.putpixel((x, y), (145, 145, 145))
        masked.save(self.masked_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_returns_texture_payload(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_texture", "params": {"amount": 0.5}},
            OperationContext(image_path=self.detail_path),
        )
        self.assertAlmostEqual(normalized["amount"], 0.5)
        self.assertAlmostEqual(normalized["noise_protection"], 0.4)

    def test_execute_adjusts_global_texture(self) -> None:
        result = self.package.execute(
            {"op": "adjust_texture", "region": "whole_image", "params": {"amount": 0.7}},
            OperationContext(image_path=self.detail_path),
        )
        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((0, 0)), Image.open(self.detail_path).convert("L").getpixel((0, 0)))

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_texture",
                "region": "main_subject",
                "params": {"amount": 0.6, "feather_radius": 0},
            },
            OperationContext(image_path=self.masked_path, masks={"main_subject": self.mask_path}),
        )
        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("L")
        self.assertNotEqual(output.getpixel((1, 4)), 145)
        self.assertEqual(output.getpixel((6, 4)), 122)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {"op": "adjust_texture", "region": "main_subject", "params": {"amount": 0.6}},
            OperationContext(image_path=self.masked_path),
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
