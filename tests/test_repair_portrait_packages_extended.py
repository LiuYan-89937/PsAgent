"""Tests for repair and portrait extended packages."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.tools.packages import (
    BlemishRemovePackage,
    CloneStampPackage,
    EyeBrightenPackage,
    HairEnhancePackage,
    LipEnhancePackage,
    PointColorPackage,
    ReflectionReducePackage,
    RemoveHealPackage,
    SkinSmoothPackage,
    SkinTextureReducePackage,
    SpotHealPackage,
    TeethWhitenPackage,
    UnderEyeBrightenPackage,
    OperationContext,
)


class RepairPortraitPackagesExtendedTest(unittest.TestCase):
    """Verify new repair and portrait tools normalize and execute."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "portrait.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")

        image = Image.new("RGB", (32, 32), (214, 176, 154))
        for x in range(10, 14):
            for y in range(10, 14):
                image.putpixel((x, y), (35, 18, 18))
        image.save(self.image_path)

        mask = Image.new("L", (32, 32), 0)
        for x in range(6, 22):
            for y in range(6, 22):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

        self.context = OperationContext(image_path=self.image_path, masks={"focus": self.mask_path})

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_remove_heal_changes_masked_defect_region(self) -> None:
        package = RemoveHealPackage()
        result = package.execute(
            {"op": "remove_heal", "region": "focus", "params": {"strength": 0.5}},
            self.context,
        )

        self.assertTrue(result.ok)
        output = Image.open(result.output_image or "").convert("RGB")
        self.assertNotEqual(output.getpixel((11, 11)), (35, 18, 18))
        self.assertEqual(output.getpixel((28, 28)), (214, 176, 154))

    def test_point_color_normalizes_aliases_and_executes(self) -> None:
        package = PointColorPackage()
        normalized = package.normalize(
            {
                "op": "point_color",
                "region": "whole_image",
                "params": {
                    "strength": 0.25,
                    "color-name": "skin",
                    "selection-width": 24,
                    "brightness_shift": 0.18,
                },
            },
            self.context,
        )
        result = package.execute(
            {
                "op": "point_color",
                "region": "whole_image",
                "params": {"strength": 0.25, "target_color": "skin", "luminance_shift": 0.18},
            },
            self.context,
        )

        self.assertEqual(normalized["target_color"], "skin")
        self.assertEqual(normalized["range_width"], 24)
        self.assertAlmostEqual(normalized["luminance_shift"], 0.18)
        self.assertTrue(result.ok)

    def test_portrait_local_tools_execute_with_shared_mask(self) -> None:
        tools = [
            (BlemishRemovePackage(), {"strength": 0.25}),
            (SpotHealPackage(), {"strength": 0.25}),
            (CloneStampPackage(), {"strength": 0.25}),
            (SkinSmoothPackage(), {"strength": 0.22}),
            (SkinTextureReducePackage(), {"strength": 0.18}),
            (UnderEyeBrightenPackage(), {"strength": 0.18}),
            (TeethWhitenPackage(), {"strength": 0.2}),
            (EyeBrightenPackage(), {"strength": 0.18}),
            (HairEnhancePackage(), {"strength": 0.18}),
            (LipEnhancePackage(), {"strength": 0.16}),
            (ReflectionReducePackage(), {"strength": 0.14}),
        ]

        for package, params in tools:
            with self.subTest(package=package.name):
                result = package.execute(
                    {"op": package.name, "region": "focus", "params": params},
                    self.context,
                )
                self.assertTrue(result.ok, msg=f"{package.name} failed with: {result.error}")
                self.assertTrue(Path(result.output_image or "").exists())

    def test_skin_smooth_local_mode_changes_masked_area_more_than_outside(self) -> None:
        package = SkinSmoothPackage()
        result = package.execute(
            {"op": "skin_smooth", "region": "focus", "params": {"strength": 0.3}},
            self.context,
        )

        self.assertTrue(result.ok)
        original = np.asarray(Image.open(self.image_path).convert("RGB"), dtype=np.float32)
        output = np.asarray(Image.open(result.output_image or "").convert("RGB"), dtype=np.float32)
        center_delta = np.abs(output[11, 11] - original[11, 11]).mean()
        outside_delta = np.abs(output[28, 28] - original[28, 28]).mean()
        self.assertGreater(center_delta, outside_delta)


if __name__ == "__main__":
    unittest.main()
