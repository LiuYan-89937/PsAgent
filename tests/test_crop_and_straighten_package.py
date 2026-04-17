"""Unit tests for the crop and straighten package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.tools.packages import CropAndStraightenPackage, OperationContext


class CropAndStraightenPackageTest(unittest.TestCase):
    """Validate the minimal crop/straighten implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = CropAndStraightenPackage()

        self.image_path = str(Path(self.tmpdir.name) / "input.png")

        image = Image.new("RGB", (160, 100), (210, 210, 210))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 140, 80), outline=(40, 40, 40), width=4)
        draw.line((15, 70, 145, 50), fill=(180, 80, 60), width=5)
        draw.ellipse((60, 28, 100, 68), outline=(40, 90, 180), width=4)
        image.save(self.image_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_crop_ratio(self) -> None:
        normalized = self.package.normalize(
            {"op": "crop_and_straighten", "strength": 0.5, "params": {"straighten_bias": 0.25}},
            OperationContext(image_path=self.image_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["crop_ratio"], 0.0)
        self.assertGreater(normalized["straighten_angle"], 0.0)
        self.assertAlmostEqual(normalized["max_crop_ratio"], 0.16)

    def test_execute_crops_image_size(self) -> None:
        result = self.package.execute(
            {
                "op": "crop_and_straighten",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        with Image.open(self.image_path) as original, Image.open(result.output_image or "") as output:
            self.assertLess(output.size[0], original.size[0])
            self.assertLess(output.size[1], original.size[1])

    def test_execute_supports_straighten_bias(self) -> None:
        result = self.package.execute(
            {
                "op": "crop_and_straighten",
                "region": "whole_image",
                "strength": 0.35,
                "params": {"straighten_bias": 0.6, "max_straighten_angle": 6.0},
            },
            OperationContext(image_path=self.image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        with Image.open(result.output_image or "") as output:
            self.assertGreaterEqual(output.size[0], 32)
            self.assertGreaterEqual(output.size[1], 32)

    def test_execute_runs_globally_for_non_whole_region_without_mask(self) -> None:
        result = self.package.execute(
            {
                "op": "crop_and_straighten",
                "region": "main_subject",
                "strength": 0.5,
                "params": {},
            },
            OperationContext(image_path=self.image_path),
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
