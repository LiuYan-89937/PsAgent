"""Unit tests for the native adjust_exposure @tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools import require_tool


def _mean_luminance(image_path: str) -> float:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    return float(np.dot(image[..., :3], [0.299, 0.587, 0.114]).mean())


class NativeExposureToolTest(unittest.TestCase):
    """Verify whole-image, masked, and default execution behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        Image.new("RGB", (64, 64), (70, 72, 75)).save(self.image_path)

        mask = Image.new("L", (64, 64), 0)
        for x in range(32):
            for y in range(64):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

        self.tool = require_tool("adjust_exposure")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_whole_image_execution_increases_brightness(self) -> None:
        before = _mean_luminance(self.image_path)
        result = self.tool.invoke({"image_path": self.image_path, "strength": 0.5})

        self.assertTrue(result["ok"])
        self.assertGreater(_mean_luminance(result["output_image"]), before)

    def test_masked_execution_only_changes_masked_region(self) -> None:
        result = self.tool.invoke(
            {
                "image_path": self.image_path,
                "mask_path": self.mask_path,
                "strength": 0.6,
                "feather_radius": 0.0,
            }
        )

        original = np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.float32)
        adjusted = np.asarray(Image.open(result["output_image"]).convert("RGB"), dtype=np.float32)
        delta = np.abs(adjusted - original).mean(axis=2)
        left_delta = float(delta[:, :32].mean())
        right_delta = float(delta[:, 32:].mean())

        self.assertGreater(left_delta, right_delta * 4)

    def test_missing_non_critical_params_use_defaults(self) -> None:
        result = self.tool.invoke({"image_path": self.image_path})

        self.assertTrue(result["ok"])
        self.assertIn("strength", result["applied_params"])
        self.assertTrue(Path(result["output_image"]).exists())


if __name__ == "__main__":
    unittest.main()
