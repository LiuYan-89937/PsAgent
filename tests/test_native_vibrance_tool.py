"""Unit tests for the native adjust_vibrance_saturation @tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.tools import require_tool


def _muted_color_gradient(width: int = 96, height: int = 64) -> Image.Image:
    hue = np.tile(np.linspace(0, 179, width, dtype=np.uint8), (height, 1))
    saturation = np.full((height, width), 42, dtype=np.uint8)
    value = np.full((height, width), 180, dtype=np.uint8)
    hsv = np.stack([hue, saturation, value], axis=2)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray(rgb, mode="RGB")


def _mean_saturation(image_path: str) -> float:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean())


class NativeVibranceToolTest(unittest.TestCase):
    """Verify color-space enhancement and default execution behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        _muted_color_gradient().save(self.image_path)

        mask = Image.new("L", (96, 64), 0)
        for x in range(48):
            for y in range(64):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

        self.tool = require_tool("adjust_vibrance_saturation")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_whole_image_execution_increases_saturation_without_collapsing_colors(self) -> None:
        before = _mean_saturation(self.image_path)
        result = self.tool.invoke({"image_path": self.image_path, "strength": 0.5})

        self.assertTrue(result["ok"])
        self.assertGreater(_mean_saturation(result["output_image"]), before)

        adjusted = np.asarray(Image.open(result["output_image"]).convert("RGB"), dtype=np.uint8)
        unique_colors = np.unique(adjusted.reshape(-1, 3), axis=0)
        self.assertGreater(len(unique_colors), 24)

    def test_masked_execution_only_pushes_color_inside_mask(self) -> None:
        result = self.tool.invoke(
            {
                "image_path": self.image_path,
                "mask_path": self.mask_path,
                "strength": 0.55,
                "feather_radius": 0.0,
            }
        )

        original = np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.float32)
        adjusted = np.asarray(Image.open(result["output_image"]).convert("RGB"), dtype=np.float32)
        delta = np.abs(adjusted - original).mean(axis=2)
        left_delta = float(delta[:, :48].mean())
        right_delta = float(delta[:, 48:].mean())

        self.assertGreater(left_delta, right_delta * 4)

    def test_missing_non_critical_params_use_defaults(self) -> None:
        result = self.tool.invoke({"image_path": self.image_path})

        self.assertTrue(result["ok"])
        self.assertIn("vibrance_scale", result["applied_params"])
        self.assertTrue(Path(result["output_image"]).exists())


if __name__ == "__main__":
    unittest.main()
