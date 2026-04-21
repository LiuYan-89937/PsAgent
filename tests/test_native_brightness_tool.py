"""Unit tests for the native adjust_brightness @tool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools import require_tool


def _luminance_stats(image_path: str) -> tuple[float, float, float]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    luminance = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return float(luminance.mean()), float(luminance.std()), float(np.quantile(luminance, 0.05))


def _layered_scene(width: int = 96, height: int = 64) -> Image.Image:
    # 做一张带明显黑位、中间调和高亮区的测试图，
    # 用来防止“提亮后整体发白发雾”。
    ramp = np.tile(np.linspace(20, 220, width, dtype=np.float32), (height, 1))
    rgb = np.stack([ramp, ramp * 0.95, ramp * 0.9], axis=2)
    rgb[:, :18] = np.array([12, 14, 16], dtype=np.float32)
    rgb[:, 70:90] = np.array([210, 196, 184], dtype=np.float32)
    rgb[18:46, 24:58] += np.array([18, 10, 4], dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


class NativeBrightnessToolTest(unittest.TestCase):
    """Verify whole-image, masked, and anti-haze behavior for adjust_brightness."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        _layered_scene().save(self.image_path)

        mask = Image.new("L", (96, 64), 0)
        for x in range(48):
            for y in range(64):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

        self.tool = require_tool("adjust_brightness")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_whole_image_execution_increases_brightness_without_washing_out_contrast(self) -> None:
        before_mean, before_std, before_p05 = _luminance_stats(self.image_path)
        result = self.tool.invoke(
            {
                "image_path": self.image_path,
                "brightness_offset": 0.18,
                "highlight_protection": 0.32,
            }
        )

        self.assertTrue(result["ok"])
        after_mean, after_std, after_p05 = _luminance_stats(result["output_image"])
        self.assertGreater(after_mean, before_mean)
        # 提亮后允许层次略降，但不能塌得像一层白雾。
        self.assertGreater(after_std, before_std * 0.85)
        # 黑位可以略提，但不能整体漂成灰雾。
        self.assertLess(after_p05, before_p05 + 18.0)

    def test_masked_execution_only_changes_masked_region(self) -> None:
        result = self.tool.invoke(
            {
                "image_path": self.image_path,
                "mask_path": self.mask_path,
                "brightness_offset": 0.2,
                "highlight_protection": 0.3,
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
        self.assertIn("brightness_offset", result["applied_params"])
        self.assertTrue(Path(result["output_image"]).exists())

    def test_high_brightness_offset_within_new_range_executes_stably(self) -> None:
        result = self.tool.invoke(
            {
                "image_path": self.image_path,
                "brightness_offset": 0.5,
                "highlight_protection": 0.35,
            }
        )

        self.assertTrue(result["ok"])
        mean_l, std_l, p05_l = _luminance_stats(result["output_image"])
        self.assertGreater(mean_l, 80.0)
        self.assertGreater(std_l, 20.0)
        self.assertLess(p05_l, 80.0)


if __name__ == "__main__":
    unittest.main()
