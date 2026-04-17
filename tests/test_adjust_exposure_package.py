"""Tests for the first real tool package using the user's real portrait image."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageOps, ImageStat

from app.tools.packages import AdjustExposurePackage, OperationContext
from app.tools.segmentation_tools import (
    ALIYUN_ACCESS_KEY_ID_ENV,
    ALIYUN_ACCESS_KEY_SECRET_ENV,
    AliyunImageSegError,
    generate_realtime_subject_mask,
)


TESTS_DIR = Path(__file__).resolve().parent


def _find_real_test_images() -> list[Path]:
    """Pick all real images from the tests directory."""

    images: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        images.extend(sorted(TESTS_DIR.glob(pattern)))
    if images:
        return images
    raise FileNotFoundError("No real test images were found in the tests directory.")


REAL_IMAGE_PATHS = _find_real_test_images()


class AdjustExposurePackageTest(unittest.TestCase):
    """Validate normalization and execution behavior on a real portrait image."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.package = AdjustExposurePackage()

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _require_live_aliyun_setup(self) -> None:
        """Skip live mask tests when the user has not configured Aliyun credentials yet."""

        missing = [
            name
            for name in (ALIYUN_ACCESS_KEY_ID_ENV, ALIYUN_ACCESS_KEY_SECRET_ENV)
            if not os.getenv(name)
        ]
        if missing:
            joined = ", ".join(missing)
            self.skipTest(f"Missing Aliyun credentials: {joined}")

    def test_normalize_converts_strength_to_multiplier(self) -> None:
        for image_path in REAL_IMAGE_PATHS:
            with self.subTest(image=image_path.name):
                normalized = self.package.normalize(
                    {"op": "adjust_exposure", "strength": 0.5, "params": {}},
                    OperationContext(image_path=str(image_path)),
                )

                self.assertAlmostEqual(normalized["strength"], 0.5)
                self.assertGreater(normalized["exposure_multiplier"], 1.0)
                self.assertAlmostEqual(normalized["max_stops"], 1.5)

    def test_execute_brightens_real_image_whole_image(self) -> None:
        for image_path in REAL_IMAGE_PATHS:
            with self.subTest(image=image_path.name):
                result = self.package.execute(
                    {"op": "adjust_exposure", "region": "whole_image", "strength": 0.65, "params": {}},
                    OperationContext(image_path=str(image_path)),
                )

                self.assertTrue(result.ok)
                self.assertIsNotNone(result.output_image)

                original = Image.open(image_path).convert("L")
                output = Image.open(result.output_image).convert("L")
                original_mean = ImageStat.Stat(original).mean[0]
                output_mean = ImageStat.Stat(output).mean[0]

                self.assertGreater(output_mean, original_mean)

    def test_execute_runs_globally_when_mask_missing(self) -> None:
        for image_path in REAL_IMAGE_PATHS:
            with self.subTest(image=image_path.name):
                result = self.package.execute(
                    {"op": "adjust_exposure", "region": "person", "strength": -0.45, "params": {}},
                    OperationContext(image_path=str(image_path)),
                )

                self.assertTrue(result.ok)
                self.assertFalse(result.fallback_used)

    def test_generate_realtime_subject_mask_with_aliyun(self) -> None:
        self._require_live_aliyun_setup()

        for image_path in REAL_IMAGE_PATHS:
            with self.subTest(image=image_path.name):
                segmentation = generate_realtime_subject_mask(
                    str(image_path),
                    output_dir=str(Path(self.tmpdir.name) / image_path.stem),
                )

                mask = Image.open(segmentation.binary_mask_path).convert("L")
                image = Image.open(image_path)
                unique_values = set(mask.getdata())
                coverage = ImageStat.Stat(mask).mean[0] / 255.0

                self.assertEqual(mask.size, image.size)
                self.assertTrue(unique_values.issubset({0, 255}))
                self.assertIn(0, unique_values)
                self.assertIn(255, unique_values)
                self.assertGreater(coverage, 0.01)
                self.assertLess(coverage, 0.995)

    def test_execute_uses_realtime_aliyun_mask_for_local_region(self) -> None:
        self._require_live_aliyun_setup()

        for image_path in REAL_IMAGE_PATHS:
            with self.subTest(image=image_path.name):
                try:
                    segmentation = generate_realtime_subject_mask(
                        str(image_path),
                        output_dir=str(Path(self.tmpdir.name) / image_path.stem),
                    )
                except AliyunImageSegError as error:
                    self.fail(f"Aliyun realtime mask generation failed for {image_path.name}: {error}")

                result = self.package.execute(
                    {"op": "adjust_exposure", "region": "person", "strength": -0.6, "params": {}},
                    OperationContext(
                        image_path=str(image_path),
                        masks={"person": segmentation.binary_mask_path},
                    ),
                )

                self.assertTrue(result.ok)
                self.assertIsNotNone(result.output_image)

                original_gray = Image.open(image_path).convert("L")
                output_gray = Image.open(result.output_image).convert("L")
                person_mask = Image.open(segmentation.binary_mask_path).convert("L")
                background_mask = ImageOps.invert(person_mask)

                original_person_mean = ImageStat.Stat(original_gray, person_mask).mean[0]
                output_person_mean = ImageStat.Stat(output_gray, person_mask).mean[0]
                original_background_mean = ImageStat.Stat(original_gray, background_mask).mean[0]
                output_background_mean = ImageStat.Stat(output_gray, background_mask).mean[0]

                # 局部曝光测试看两个指标：
                # 1. person mask 内的均值应该明显下降；
                # 2. 背景区域应尽量保持不变，允许极小的插值/保存误差。
                self.assertLess(output_person_mean, original_person_mean)
                self.assertAlmostEqual(output_background_mean, original_background_mean, delta=1.0)


if __name__ == "__main__":
    unittest.main()
