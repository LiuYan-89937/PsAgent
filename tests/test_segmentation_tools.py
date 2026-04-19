"""Unit tests for provider-aware segmentation helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.tools.segmentation_tools import (
    PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV,
    PSAGENT_SEGMENTATION_PROVIDER_ENV,
    AliyunSegmentationResult,
    FalImageSegError,
    FalSegmentationResult,
    ensure_region_mask,
    generate_fal_sam3_mask,
    normalize_segmentation_prompt_label,
    resolve_region_mask,
)
from app.tools.packages import AdjustExposurePackage, OperationContext


class SegmentationToolsTest(unittest.TestCase):
    """Verify provider resolution and fallback behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")
        Image.new("RGB", (32, 32), (80, 90, 100)).save(self.image_path)
        Image.new("L", (32, 32), 255).save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _result(self, provider: str) -> FalSegmentationResult:
        return FalSegmentationResult(
            provider=provider,
            binary_mask_path=self.mask_path,
            original_image_path=self.image_path,
            api_chain=(provider,),
        )

    def test_ensure_region_mask_uses_fal_when_prompt_present_in_auto_mode(self) -> None:
        with patch.dict(
            os.environ,
            {PSAGENT_SEGMENTATION_PROVIDER_ENV: "auto"},
            clear=False,
        ), patch(
            "app.tools.segmentation_tools.generate_fal_sam3_mask",
            return_value=self._result("fal_sam3"),
        ) as mocked_fal:
            mask_path = ensure_region_mask(
                self.image_path,
                "person",
                prompt="person",
                output_dir=self.tmpdir.name,
            )

        self.assertEqual(mask_path, self.mask_path)
        mocked_fal.assert_called_once()
        self.assertEqual(mocked_fal.call_args.kwargs["prompt"], "person")

    def test_ensure_region_mask_falls_back_to_aliyun_for_default_region(self) -> None:
        with patch.dict(
            os.environ,
            {
                PSAGENT_SEGMENTATION_PROVIDER_ENV: "fal_sam3",
                PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV: "aliyun",
            },
            clear=False,
        ), patch(
            "app.tools.segmentation_tools.generate_fal_sam3_mask",
            side_effect=FalImageSegError("fal failed"),
        ) as mocked_fal, patch(
            "app.tools.segmentation_tools.generate_realtime_subject_mask",
            return_value=AliyunSegmentationResult(
                provider="aliyun",
                binary_mask_path=self.mask_path,
                original_image_path=self.image_path,
                api_chain=("aliyun",),
            ),
        ) as mocked_aliyun:
            mask_path = ensure_region_mask(
                self.image_path,
                "person",
                output_dir=self.tmpdir.name,
            )

        self.assertEqual(mask_path, self.mask_path)
        mocked_fal.assert_called_once()
        mocked_aliyun.assert_called_once()

    def test_ensure_region_mask_does_not_silently_fallback_for_explicit_prompt(self) -> None:
        with patch.dict(
            os.environ,
            {
                PSAGENT_SEGMENTATION_PROVIDER_ENV: "fal_sam3",
                PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV: "aliyun",
            },
            clear=False,
        ), patch(
            "app.tools.segmentation_tools.generate_fal_sam3_mask",
            side_effect=FalImageSegError("fal failed"),
        ) as mocked_fal, patch(
            "app.tools.segmentation_tools.generate_realtime_subject_mask",
        ) as mocked_aliyun:
            with self.assertRaises(FalImageSegError):
                ensure_region_mask(
                    self.image_path,
                    "person",
                    prompt="face",
                    output_dir=self.tmpdir.name,
                )

        mocked_fal.assert_called_once()
        mocked_aliyun.assert_not_called()

    def test_ensure_region_mask_background_with_fal_uses_direct_background_prompt_first(self) -> None:
        with patch(
            "app.tools.segmentation_tools.generate_fal_sam3_mask",
            return_value=self._result("fal_sam3"),
        ) as mocked_fal:
            mask_path = ensure_region_mask(
                self.image_path,
                "background",
                provider="fal_sam3",
                output_dir=self.tmpdir.name,
            )

        self.assertEqual(mask_path, self.mask_path)
        self.assertEqual(mocked_fal.call_args.kwargs["prompt"], "background")
        self.assertFalse(mocked_fal.call_args.kwargs["revert_mask"])

    def test_resolve_region_mask_background_retries_with_inverse_foreground_attempts(self) -> None:
        def side_effect(*args, **kwargs):
            prompt = kwargs.get("prompt")
            if prompt == "background":
                raise FalImageSegError("fal segmentation response did not include an output image URL.")
            return self._result("fal_sam3")

        with patch(
            "app.tools.segmentation_tools._ensure_fal_region_mask",
            side_effect=side_effect,
        ) as mocked_ensure:
            result = resolve_region_mask(
                self.image_path,
                "background",
                provider="fal_sam3",
                prompt="background",
                output_dir=self.tmpdir.name,
            )

        self.assertEqual(result.binary_mask_path, self.mask_path)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.attempt_index, 1)
        self.assertEqual(result.attempt_strategy, "invert_person")
        self.assertEqual(result.requested_prompt, "background")
        self.assertEqual(result.effective_prompt, "person")
        self.assertTrue(result.revert_mask)
        self.assertEqual(len(result.attempts), 2)
        first_call = mocked_ensure.call_args_list[0].kwargs
        second_call = mocked_ensure.call_args_list[1].kwargs
        self.assertEqual(first_call["prompt"], "background")
        self.assertFalse(first_call["revert_mask"])
        self.assertEqual(second_call["prompt"], "person")
        self.assertTrue(second_call["revert_mask"])

    def test_package_schema_includes_mask_prompt_fields(self) -> None:
        schema = AdjustExposurePackage().get_params_schema()
        self.assertIn("mask_provider", schema["properties"])
        self.assertIn("mask_prompt", schema["properties"])
        self.assertIn("mask_negative_prompt", schema["properties"])

    def test_package_validation_allows_text_guided_mask_even_when_region_is_whole_image(self) -> None:
        package = AdjustExposurePackage()
        parsed = package.parse_params(
            {
                "op": "adjust_exposure",
                "region": "whole_image",
                "params": {
                    "strength": 0.2,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "face skin",
                    },
                }
            )

        self.assertIsNotNone(parsed)

    def test_package_validation_rejects_aliyun_mask_prompt(self) -> None:
        package = AdjustExposurePackage()
        with self.assertRaises(ValueError):
            package.parse_params(
                {
                    "op": "adjust_exposure",
                    "region": "main_subject",
                    "params": {
                        "strength": 0.2,
                        "mask_provider": "aliyun",
                        "mask_prompt": "face skin",
                    },
                }
            )

    def test_package_validation_rejects_unknown_extra_param(self) -> None:
        package = AdjustExposurePackage()
        with self.assertRaises(ValueError):
            package.parse_params(
                {
                    "op": "adjust_exposure",
                    "region": "main_subject",
                    "params": {
                        "strength": 0.2,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "face skin",
                        "unexpected_param": 123,
                    },
                }
            )

    def test_generate_fal_sam3_mask_uses_sam3_single_mask_arguments(self) -> None:
        color_mask_path = str(Path(self.tmpdir.name) / "sam3_color_mask.png")
        color_mask = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        for x in range(8, 24):
            for y in range(8, 24):
                color_mask.putpixel((x, y), (255, 0, 0, 255))
        color_mask.save(color_mask_path)

        class FakeClient:
            def __init__(self) -> None:
                self.upload_calls = []
                self.subscribe_calls = []

            def upload(self, data, content_type, file_name, repository):
                self.upload_calls.append(
                    {
                        "content_type": content_type,
                        "file_name": file_name,
                        "repository": repository,
                        "size": len(data),
                    }
                )
                return "https://example.com/input.png"

            def subscribe(self, model_name, arguments, start_timeout, client_timeout):
                self.subscribe_calls.append(
                    {
                        "model_name": model_name,
                        "arguments": arguments,
                        "start_timeout": start_timeout,
                        "client_timeout": client_timeout,
                    }
                )
                return {"masks": [{"url": "https://example.com/mask.png"}]}

        fake_client = FakeClient()

        with patch("app.tools.segmentation_tools._create_fal_client", return_value=fake_client), patch(
            "app.tools.segmentation_tools._download_remote_image",
            return_value=color_mask_path,
        ):
            result = generate_fal_sam3_mask(
                self.image_path,
                prompt="face skin",
                output_dir=self.tmpdir.name,
            )

        self.assertEqual(result.provider, "fal_sam3")
        self.assertEqual(fake_client.subscribe_calls[0]["model_name"], "fal-ai/sam-3/image")
        self.assertFalse(fake_client.subscribe_calls[0]["arguments"]["apply_mask"])
        self.assertFalse(fake_client.subscribe_calls[0]["arguments"]["return_multiple_masks"])
        self.assertEqual(fake_client.subscribe_calls[0]["arguments"]["max_masks"], 1)
        self.assertEqual(result.negative_prompt, None)
        binary_mask = Image.open(result.binary_mask_path).convert("L")
        self.assertEqual(binary_mask.getpixel((16, 16)), 255)
        self.assertEqual(binary_mask.getpixel((2, 2)), 0)

    def test_normalize_segmentation_prompt_label_returns_short_english_tokens(self) -> None:
        self.assertEqual(normalize_segmentation_prompt_label("背景的树林和草地绿色植被", region="background"), "trees")
        self.assertEqual(normalize_segmentation_prompt_label("人物脸部和肩颈皮肤", region="person"), "face")
        self.assertEqual(normalize_segmentation_prompt_label("白裙区域", region="person"), "dress")


if __name__ == "__main__":
    unittest.main()
