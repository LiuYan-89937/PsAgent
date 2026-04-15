"""Unit tests for the vibrance/saturation package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.tools.packages import AdjustVibranceSaturationPackage, OperationContext


def _mean_saturation(image_path: str) -> float:
    """Return the mean HSV saturation for assertions."""

    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    return float(hsv[:, :, 1].mean())


class AdjustVibranceSaturationPackageTest(unittest.TestCase):
    """Validate the minimal vibrance/saturation implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustVibranceSaturationPackage()

        self.muted_image_path = str(Path(self.tmpdir.name) / "muted.png")
        self.masked_image_path = str(Path(self.tmpdir.name) / "masked_input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        muted = Image.new("RGB", (16, 16), (170, 145, 135))
        muted.save(self.muted_image_path)

        masked = Image.new("RGB", (8, 8), (165, 140, 132))
        for x in range(4, 8):
            for y in range(8):
                masked.putpixel((x, y), (120, 120, 120))
        masked.save(self.masked_image_path)

        mask = Image.new("L", (8, 8), 0)
        for x in range(4):
            for y in range(8):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_vibrance_and_saturation(self) -> None:
        normalized = self.package.normalize(
            {"op": "adjust_vibrance_saturation", "strength": 0.5, "params": {}},
            OperationContext(image_path=self.muted_image_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["vibrance_amount"], 0.0)
        self.assertGreater(normalized["saturation_amount"], 0.0)
        self.assertGreater(normalized["vibrance_amount"], normalized["saturation_amount"])

    def test_execute_increases_global_saturation(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_vibrance_saturation",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.muted_image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        original_sat = _mean_saturation(self.muted_image_path)
        output_sat = _mean_saturation(result.output_image or "")
        self.assertGreater(output_sat, original_sat)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_vibrance_saturation",
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
        # 左侧在 mask 内应发生颜色变化；右侧保持原值不变。
        self.assertNotEqual(output.getpixel((1, 4)), (165, 140, 132))
        self.assertEqual(output.getpixel((6, 4)), (120, 120, 120))

    def test_execute_returns_fallback_result_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "adjust_vibrance_saturation",
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
