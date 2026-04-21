"""Unit tests for stage runner behavior with native tools and ToolNode."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.stage_pipeline import execute_stage_plan, prepare_stage_context
from app.graph.state import MaskCatalog


class StageRunnerWithNativeToolsTest(unittest.TestCase):
    """Verify masked execution and mask reuse in the stage runner."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")
        Image.new("RGB", (64, 64), (80, 90, 100)).save(self.image_path)
        Image.new("L", (64, 64), 255).save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _base_state(self) -> dict:
        return {
            "mode": "explicit",
            "request_text": "局部提亮背景并增强层次",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {
                "domain": "general",
                "issues": ["underexposed"],
                "main_issues": ["underexposed"],
                "metrics": {
                    "brightness_mean": 90,
                    "brightness_std": 12,
                    "shadow_ratio": 0.12,
                    "highlight_ratio": 0.05,
                },
            },
            "edit_profile": {
                "main_subject_type": "object",
                "subject_count": "single",
                "technical_issues": [],
                "global_tone_issues": ["underexposed"],
                "local_balance_needed": True,
                "subject_refine_needed": False,
                "finish_needed": True,
                "subject_capabilities": {},
            },
            "input_images": [self.image_path],
            "tool_catalog": [],
            "phases": {},
            "mask_catalog": MaskCatalog().model_dump(mode="json"),
        }

    def test_stage_runner_reuses_mask_catalog_for_same_signature(self) -> None:
        state = self._base_state()
        state.update(prepare_stage_context(state, stage_key="local_balance"))
        state["stage_plan"] = {
            "mode": "explicit",
            "domain": "general",
            "executor": "deterministic",
            "preserve": [],
            "steps": [
                {
                    "op": "adjust_exposure",
                    "region": "背景偏暗区域",
                    "params": {
                        "strength": 0.4,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "background",
                        "mask_semantic_type": True,
                    },
                    "priority": 0,
                },
                {
                    "op": "adjust_contrast",
                    "region": "背景高反差区域",
                    "params": {
                        "strength": 0.32,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "background",
                        "mask_semantic_type": True,
                    },
                    "priority": 1,
                },
            ],
            "step_budget": 3,
            "summary": "局部平衡。",
            "should_write_memory": False,
            "memory_candidates": [],
            "needs_confirmation": False,
        }

        fake_segmentation = type(
            "SegResult",
            (),
            {
                "provider": "fal_sam3",
                "binary_mask_path": self.mask_path,
                "segmentation_rgba_path": None,
                "requested_provider": "fal_sam3",
                "target_label": "background",
                "prompt": "background",
                "negative_prompt": None,
                "semantic_type": True,
                "request_id": None,
                "api_chain": [],
                "attempt_index": None,
                "attempt_strategy": None,
                "requested_prompt": None,
                "effective_prompt": None,
                "revert_mask": None,
                "attempts": [],
                "fallback_used": False,
            },
        )()

        with patch("app.graph.nodes.stage_pipeline.resolve_region_mask", return_value=fake_segmentation) as mocked_segmentation:
            result = execute_stage_plan(state, stage_key="local_balance")

        self.assertEqual(mocked_segmentation.call_count, 1)
        self.assertEqual(len(result["segmentation_trace"]), 1)
        self.assertEqual(len(result["execution_trace"]), 2)
        self.assertEqual(len(MaskCatalog.model_validate(result["mask_catalog"]).items), 1)

    def test_stage_runner_skips_required_mask_tool_without_local_target(self) -> None:
        state = self._base_state()
        state["edit_profile"]["subject_refine_needed"] = True
        state.update(prepare_stage_context(state, stage_key="subject_refine"))
        state["stage_plan"] = {
            "mode": "explicit",
            "domain": "portrait",
            "executor": "deterministic",
            "preserve": [],
            "steps": [
                {
                    "op": "adjust_teeth_whiten",
                    "region": "whole_image",
                    "params": {
                        "yellow_reduce": 0.25,
                    },
                    "priority": 0,
                }
            ],
            "step_budget": 3,
            "summary": "主体优化。",
            "should_write_memory": False,
            "memory_candidates": [],
            "needs_confirmation": False,
        }

        with patch("app.graph.nodes.stage_pipeline.resolve_region_mask") as mocked_segmentation:
            result = execute_stage_plan(state, stage_key="subject_refine")

        self.assertEqual(mocked_segmentation.call_count, 0)
        self.assertEqual(len(result["execution_trace"]), 1)
        trace_item = result["execution_trace"][0]
        self.assertFalse(trace_item.ok)
        self.assertIn("requires a mask", (trace_item.error or "").lower())


if __name__ == "__main__":
    unittest.main()
