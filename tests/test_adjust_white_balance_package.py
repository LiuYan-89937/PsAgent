"""Unit tests for the white-balance package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.packages import AdjustWhiteBalancePackage, OperationContext


def _mean_rgb(image_path: str) -> tuple[float, float, float]:
    """Return the mean RGB values for assertions."""

    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    means = image_np.mean(axis=(0, 1))
    return float(means[0]), float(means[1]), float(means[2])


class AdjustWhiteBalancePackageTest(unittest.TestCase):
    """Validate the minimal white-balance implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustWhiteBalancePackage()

        self.cool_image_path = str(Path(self.tmpdir.name) / "cool.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked_input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        cool = Image.new("RGB", (16, 16), (118, 132, 154))
        cool.save(self.cool_image_path)

        masked = Image.new("RGB", (8, 8), (118, 132, 154))
        for x in range(4, 8):
            for y in range(8):
                masked.putpixel((x, y), (128, 128, 128))
        masked.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_temperature_shift(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_white_balance", "strength": 0.5, "params": {"tint_bias": 0.25}},
            OperationContext(image_path=self.cool_image_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["temperature_shift"], 0.0)
        self.assertGreater(normalized["tint_shift"], 0.0)
        self.assertAlmostEqual(normalized["temperature_scale"], 12.0)

    def test_execute_warms_global_image(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_white_balance",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.cool_image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        original_r, _, original_b = _mean_rgb(self.cool_image_path)
        output_r, _, output_b = _mean_rgb(result.output_image or "")
        self.assertGreater(output_r - output_b, original_r - original_b)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_white_balance",
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

        output = Image.open(result.output_image or "").convert("RGB")
        self.assertNotEqual(output.getpixel((1, 4)), (118, 132, 154))
        self.assertEqual(output.getpixel((6, 4)), (128, 128, 128))

    def test_execute_returns_fallback_result_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_white_balance",
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
