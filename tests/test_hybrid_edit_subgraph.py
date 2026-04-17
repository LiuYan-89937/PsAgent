"""Unit tests for the hybrid_edit subgraph."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.subgraphs.hybrid_edit import execute_hybrid
from app.tools.segmentation_tools import SegmentationResult


class HybridEditSubgraphTest(unittest.TestCase):
    """Verify mask-backed local execution flow."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "hybrid.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")
        Image.new("RGB", (32, 32), (80, 90, 100)).save(self.image_path)
        mask = Image.new("L", (32, 32), 0)
        for x in range(16):
            for y in range(32):
                mask.putpixel((x, y), 255)
        mask.save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_execute_hybrid_without_mask_params_runs_as_global_adjustment(self) -> None:
        state = {
            "input_images": [self.image_path],
            "thread_id": "t1",
            "edit_plan": {
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "main_subject",
                        "params": {"strength": 0.3, "feather_radius": 0},
                    }
                ]
            },
            "image_analysis": {},
            "retrieved_prefs": [],
            "masks": {},
        }

        with patch(
            "app.graph.subgraphs.hybrid_edit.resolve_region_mask",
        ) as mocked_mask:
            result = execute_hybrid(state)

        mocked_mask.assert_not_called()
        self.assertTrue(bool(result["selected_output"]))
        self.assertEqual(len(result["execution_trace"]), 1)
        self.assertEqual(len(result["segmentation_trace"]), 0)
        self.assertTrue(result["execution_trace"][0]["ok"])

    def test_execute_hybrid_passes_mask_generation_options(self) -> None:
        state = {
            "input_images": [self.image_path],
            "thread_id": "t1",
            "edit_plan": {
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "main_subject",
                        "params": {
                            "strength": 0.25,
                            "feather_radius": 0,
                            "mask_provider": "fal_sam3",
                            "mask_prompt": "person",
                            "mask_negative_prompt": "background",
                            "mask_semantic_type": True,
                            "mask_fill_holes": True,
                        },
                    }
                ]
            },
            "image_analysis": {},
            "retrieved_prefs": [],
            "masks": {},
        }

        with patch(
            "app.graph.subgraphs.hybrid_edit.resolve_region_mask",
            return_value=SegmentationResult(
                provider="fal_sam3",
                binary_mask_path=self.mask_path,
                original_image_path=self.image_path,
                api_chain=("fal_client.upload", "fal-ai/sam-3/image"),
                region="main_subject",
                target_label="person face skin",
                prompt="person face skin",
                negative_prompt="hair background",
                semantic_type=True,
                requested_provider="fal_sam3",
            ),
        ) as mocked_mask:
            result = execute_hybrid(state)

        self.assertTrue(bool(result["selected_output"]))
        mocked_mask.assert_called_once()
        _, kwargs = mocked_mask.call_args
        self.assertEqual(kwargs["provider"], "fal_sam3")
        self.assertEqual(kwargs["prompt"], "person")
        self.assertEqual(kwargs["negative_prompt"], "background")
        self.assertTrue(kwargs["semantic_type"])
        self.assertTrue(kwargs["fill_holes"])
        self.assertEqual(result["segmentation_trace"][0]["provider"], "fal_sam3")
        self.assertEqual(result["segmentation_trace"][0]["target_label"], "person face skin")


if __name__ == "__main__":
    unittest.main()
