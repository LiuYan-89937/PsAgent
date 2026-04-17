"""Unit tests for the sharpen package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.tools.packages import OperationContext, SharpenPackage


def _edge_strength(image_path: str) -> float:
    """Return mean Laplacian magnitude for assertions."""

    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.abs(lap).mean())


class SharpenPackageTest(unittest.TestCase):
    """Validate the minimal sharpen implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = SharpenPackage()

        self.soft_image_path = str(Path(self.tmpdir.name) / "soft.png")
        self.masked_soft_image_path = str(Path(self.tmpdir.name) / "masked_soft.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        base = Image.new("RGB", (32, 32), (128, 128, 128))
        for x in range(8, 24):
            for y in range(8, 24):
                base.putpixel((x, y), (180, 180, 180))
        soft = base.filter(ImageFilter.GaussianBlur(radius=2.2))
        soft.save(self.soft_image_path)

        masked = Image.new("RGB", (16, 16), (128, 128, 128))
        for x in range(0, 8):
            for y in range(4, 12):
                masked.putpixel((x, y), (180, 180, 180))
        masked = masked.filter(ImageFilter.GaussianBlur(radius=1.8))
        masked.save(self.masked_soft_image_path)

        mask = Image.new("L", (16, 16), 0)
        for x in range(8):
            for y in range(16):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_sharpen_params(self) -> None:
        normalized = self.package.normalize(
            {"op": "sharpen", "strength": 0.5, "params": {}},
            OperationContext(image_path=self.soft_image_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["amount"], 0.0)
        self.assertGreater(normalized["radius"], 0.0)
        self.assertGreaterEqual(normalized["threshold"], 0.0)

    def test_execute_increases_global_edge_strength(self) -> None:
        result = self.package.execute(
            {
                "op": "sharpen",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.soft_image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        original_edge = _edge_strength(self.soft_image_path)
        output_edge = _edge_strength(result.output_image or "")
        self.assertGreater(output_edge, original_edge)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "sharpen",
                "region": "main_subject",
                "strength": 0.8,
                "params": {"feather_radius": 0},
            },
            OperationContext(
                image_path=self.masked_soft_image_path,
                masks={"main_subject": self.mask_path},
            ),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        input_np = np.asarray(Image.open(self.masked_soft_image_path).convert("RGB"), dtype=np.float32)
        output_np = np.asarray(Image.open(result.output_image or "").convert("RGB"), dtype=np.float32)

        left_diff = np.abs(output_np[:, :8, :] - input_np[:, :8, :]).mean()
        right_diff = np.abs(output_np[:, 8:, :] - input_np[:, 8:, :]).mean()

        self.assertGreater(left_diff, 0.03)
        self.assertLess(right_diff, 0.8)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "sharpen",
                "region": "main_subject",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.masked_soft_image_path),
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
