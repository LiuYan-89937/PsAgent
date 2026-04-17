"""Unit tests for the denoise package."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.packages import DenoisePackage, OperationContext


def _mean_channel_std(image_path: str) -> float:
    """Return average per-channel standard deviation for assertions."""

    image_np = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    return float(image_np.std(axis=(0, 1)).mean())


class DenoisePackageTest(unittest.TestCase):
    """Validate the minimal denoise implementation."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = DenoisePackage()

        self.noisy_image_path = str(Path(self.tmpdir.name) / "noisy.png")
        self.masked_noisy_image_path = str(Path(self.tmpdir.name) / "masked_noisy.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        rng = np.random.default_rng(7)

        base = np.full((32, 32, 3), 128, dtype=np.float32)
        noisy = np.clip(base + rng.normal(0, 22, size=(32, 32, 3)), 0, 255).astype(np.uint8)
        Image.fromarray(noisy, mode="RGB").save(self.noisy_image_path)

        masked_base = np.full((16, 16, 3), 128, dtype=np.float32)
        left_noisy = np.clip(masked_base[:, :8, :] + rng.normal(0, 24, size=(16, 8, 3)), 0, 255)
        right_clean = masked_base[:, 8:, :]
        masked_noisy = np.concatenate([left_noisy, right_clean], axis=1).astype(np.uint8)
        Image.fromarray(masked_noisy, mode="RGB").save(self.masked_noisy_image_path)

        mask = Image.new("L", (16, 16), 0)
        for x in range(8):
            for y in range(16):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_normalize_maps_strength_to_denoise_strengths(self) -> None:
        normalized = self.package.normalize(
            {"op": "denoise", "strength": 0.5, "params": {}},
            OperationContext(image_path=self.noisy_image_path),
        )

        self.assertAlmostEqual(normalized["strength"], 0.5)
        self.assertGreater(normalized["luma_strength"], 0.0)
        self.assertGreater(normalized["chroma_strength"], 0.0)
        self.assertAlmostEqual(normalized["luma_scale"], 12.0)

    def test_execute_reduces_global_noise(self) -> None:
        result = self.package.execute(
            {
                "op": "denoise",
                "region": "whole_image",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.noisy_image_path),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        original_std = _mean_channel_std(self.noisy_image_path)
        output_std = _mean_channel_std(result.output_image or "")
        self.assertLess(output_std, original_std)

    def test_execute_uses_mask_for_local_region(self) -> None:
        result = self.package.execute(
            {
                "op": "denoise",
                "region": "main_subject",
                "strength": 0.8,
                "params": {"feather_radius": 0},
            },
            OperationContext(
                image_path=self.masked_noisy_image_path,
                masks={"main_subject": self.mask_path},
            ),
        )

        self.assertTrue(result.ok)
        self.assertIsNotNone(result.output_image)

        input_np = np.asarray(Image.open(self.masked_noisy_image_path).convert("RGB"), dtype=np.float32)
        output_np = np.asarray(Image.open(result.output_image or "").convert("RGB"), dtype=np.float32)

        left_input_std = float(input_np[:, :8, :].std())
        left_output_std = float(output_np[:, :8, :].std())
        right_diff = np.abs(output_np[:, 8:, :] - input_np[:, 8:, :]).mean()

        self.assertLess(left_output_std, left_input_std)
        self.assertLess(right_diff, 1.0)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        result = self.package.execute(
            {
                "op": "denoise",
                "region": "main_subject",
                "strength": 0.8,
                "params": {},
            },
            OperationContext(image_path=self.masked_noisy_image_path),
        )

        self.assertTrue(result.ok)
        self.assertFalse(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
