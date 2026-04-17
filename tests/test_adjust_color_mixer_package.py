"""Unit tests for the color mixer package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.packages import AdjustColorMixerPackage, OperationContext


def _mean_rgb(image_path: str) -> tuple[float, float, float]:
    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    means = image_np.mean(axis=(0, 1))
    return float(means[0]), float(means[1]), float(means[2])


class AdjustColorMixerPackageTest(unittest.TestCase):
    """Validate Color Mixer / HSL behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustColorMixerPackage()

        self.orange_path = str(Path(self.tmpdir.name) / "orange.png")
        self.masked_path = str(Path(self.tmpdir.name) / "masked.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        orange = Image.new("RGB", (16, 16), (214, 134, 68))
        orange.save(self.orange_path)

        masked = Image.new("RGB", (8, 8), (214, 134, 68))
        for x in range(4, 8):
            for y in range(8):
                masked.putpixel((x, y), (90, 140, 210))
        masked.save(self.masked_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_builds_channel_settings(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_color_mixer", "params": {"orange_saturation": 0.5, "orange_luminance": 0.2}},
            OperationContext(image_path=self.orange_path),
        )
        self.assertIn("orange", normalized["channel_settings"])
        self.assertGreater(normalized["channel_settings"]["orange"]["saturation_shift"], 0.0)

    def test_execute_adjusts_global_target_color(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_color_mixer",
                "region": "whole_image",
                "params": {"orange_saturation": 0.75, "orange_luminance": 0.35},
            },
            OperationContext(image_path=self.orange_path),
        )

        self.assertTrue(result.ok)
        original = _mean_rgb(self.orange_path)
        output = _mean_rgb(result.output_image or "")
        self.assertNotEqual(output, original)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_color_mixer",
                "region": "main_subject",
                "params": {"orange_saturation": 0.6, "feather_radius": 0},
            },
            OperationContext(image_path=self.masked_path, masks={"main_subject": self.mask_path}),
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("RGB")
        self.assertNotEqual(output.getpixel((1, 4)), (214, 134, 68))
        self.assertEqual(output.getpixel((6, 4)), (90, 140, 210))

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {"op": "adjust_color_mixer", "region": "main_subject", "params": {"orange_saturation": 0.6}},
            OperationContext(image_path=self.masked_path),
        )
        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
