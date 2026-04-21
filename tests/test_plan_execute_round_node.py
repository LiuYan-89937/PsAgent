"""Unit tests for the shared stage-pipeline nodes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.stage_pipeline import (
    build_stage_plan,
    execute_stage_plan,
    prepare_stage_context,
    stage_guard,
    summarize_stage,
)
from app.graph.state import MaskCatalog, PhaseArtifact, PlannerExecutionPlan, SegmentationTraceItem


class StagePipelineNodeTest(unittest.TestCase):
    """Verify stage preparation, execution, and summary behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")
        Image.new("RGB", (32, 32), (80, 90, 100)).save(self.image_path)
        Image.new("L", (32, 32), 255).save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _base_state(self) -> dict:
        return {
            "mode": "explicit",
            "request_text": "轻微提亮并优化局部关系",
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

    def test_prepare_stage_context_populates_policy_and_context(self) -> None:
        result = prepare_stage_context(self._base_state(), stage_key="global_base")

        self.assertEqual(result["current_stage"], "global_base")
        self.assertEqual(result["stage_policy"]["key"], "global_base")
        self.assertEqual(result["stage_context"]["current_image_path"], self.image_path)
        self.assertIn("global_base", result["phases"])

    def test_build_stage_plan_stores_stage_plan(self) -> None:
        state = self._base_state()
        state.update(prepare_stage_context(state, stage_key="global_base"))

        with (
            patch("app.graph.nodes.stage_pipeline.planner_execution_model_available", return_value=True),
            patch(
                "app.graph.nodes.stage_pipeline.generate_stage_execution_plan_with_qwen",
                return_value=PlannerExecutionPlan(
                    mode="explicit",
                    domain="general",
                    executor="deterministic",
                    preserve=[],
                    steps=[{"op": "adjust_exposure", "region": "whole_image", "params": {"strength": 0.2}, "priority": 0}],
                    step_budget=4,
                    summary="建立全图基线。",
                    should_write_memory=False,
                    memory_candidates=[],
                    needs_confirmation=False,
                ),
            ),
        ):
            result = build_stage_plan(state, stage_key="global_base")

        self.assertIn("global_base", result["phases"])
        self.assertIsInstance(result["phases"]["global_base"], PhaseArtifact)
        self.assertEqual(result["phases"]["global_base"].plan.summary, "建立全图基线。")

    def test_build_stage_plan_falls_back_to_rule_plan_when_planner_times_out(self) -> None:
        state = self._base_state()
        state.update(prepare_stage_context(state, stage_key="global_base"))

        with (
            patch("app.graph.nodes.stage_pipeline.planner_execution_model_available", return_value=True),
            patch(
                "app.graph.nodes.stage_pipeline.generate_stage_execution_plan_with_qwen",
                side_effect=TimeoutError("The read operation timed out"),
            ),
        ):
            result = build_stage_plan(state, stage_key="global_base")

        self.assertIn("global_base", result["phases"])
        self.assertIsInstance(result["phases"]["global_base"], PhaseArtifact)
        self.assertTrue(result["phases"]["global_base"].plan.steps)
        self.assertTrue(result["fallback_trace"])
        self.assertEqual(result["fallback_trace"][-1]["strategy"], "rule_based_plan")

    def test_execute_stage_plan_runs_whole_image_step(self) -> None:
        state = self._base_state()
        state.update(prepare_stage_context(state, stage_key="global_base"))
        state["stage_plan"] = {
            "mode": "explicit",
            "domain": "general",
            "executor": "deterministic",
            "preserve": [],
            "steps": [
                {"op": "adjust_exposure", "region": "whole_image", "params": {"strength": 0.2}, "priority": 0}
            ],
            "step_budget": 4,
            "summary": "建立全图基线。",
            "should_write_memory": False,
            "memory_candidates": [],
            "needs_confirmation": False,
        }

        result = execute_stage_plan(state, stage_key="global_base")
        self.assertTrue(bool(result["selected_output"]))
        self.assertEqual(len(result["execution_trace"]), 1)
        self.assertEqual(result["phases"]["global_base"].execution_trace[0].op, "adjust_exposure")

    def test_execute_stage_plan_reuses_mask_catalog_by_signature(self) -> None:
        state = self._base_state()
        state["edit_profile"]["subject_refine_needed"] = True
        state["edit_profile"]["main_subject_type"] = "human"
        state["edit_profile"]["subject_capabilities"] = {"face_visible": True}
        state.update(prepare_stage_context(state, stage_key="subject_refine"))
        state["stage_plan"] = {
            "mode": "explicit",
            "domain": "general",
            "executor": "deterministic",
            "preserve": [],
            "steps": [
                {
                    "op": "adjust_skin_brightness",
                    "region": "主体面部区域",
                    "params": {
                        "brightness_shift": 0.12,
                        "saturation_shift": -0.04,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "face",
                        "mask_semantic_type": True,
                    },
                    "priority": 0,
                },
                {
                    "op": "adjust_face_color_cleanup",
                    "region": "人物脸部皮肤",
                    "params": {
                        "yellow_reduce": 0.08,
                        "magenta_balance": 0.04,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "face",
                        "mask_semantic_type": True,
                    },
                    "priority": 1,
                },
            ],
            "step_budget": 3,
            "summary": "优化主体细节。",
            "should_write_memory": False,
            "memory_candidates": [],
            "needs_confirmation": False,
        }

        with patch(
            "app.graph.nodes.stage_pipeline.resolve_region_mask",
            return_value=type(
                "SegResult",
                (),
                {
                    "provider": "fal_sam3",
                    "binary_mask_path": self.mask_path,
                    "segmentation_rgba_path": None,
                    "requested_provider": "fal_sam3",
                    "target_label": "face",
                    "prompt": "face",
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
            )(),
        ) as mocked_segmentation:
            result = execute_stage_plan(state, stage_key="subject_refine")

        self.assertEqual(mocked_segmentation.call_count, 1)
        self.assertEqual(len(result["phases"]["subject_refine"].segmentation_trace), 1)

    def test_execute_stage_plan_skips_local_step_when_segmentation_is_empty(self) -> None:
        state = self._base_state()
        state.update(prepare_stage_context(state, stage_key="local_balance"))
        state["stage_plan"] = {
            "mode": "explicit",
            "domain": "general",
            "executor": "hybrid",
            "preserve": [],
            "steps": [
                {
                    "op": "adjust_exposure",
                    "region": "背景偏暗区域",
                    "params": {
                        "strength": 0.2,
                        "mask_provider": "fal_sam3",
                        "mask_prompt": "background",
                        "mask_semantic_type": True,
                    },
                    "priority": 0,
                }
            ],
            "step_budget": 3,
            "summary": "局部平衡。",
            "should_write_memory": False,
            "memory_candidates": [],
            "needs_confirmation": False,
        }

        with patch(
            "app.graph.nodes.stage_pipeline.resolve_region_mask",
            side_effect=RuntimeError("fal segmentation response did not include an output image URL."),
        ):
            result = execute_stage_plan(state, stage_key="local_balance")

        self.assertEqual(len(result["phases"]["local_balance"].execution_trace), 1)
        self.assertEqual(result["phases"]["local_balance"].execution_trace[0].error, "Skipped: segmentation returned no usable mask.")

    def test_stage_guard_and_summary_write_back_phase_artifacts(self) -> None:
        state = self._base_state()
        state["phases"] = {
            "finish_output": {
                "key": "finish_output",
                "label": "最终收尾",
                "execution_trace": [{"op": "adjust_vibrance_saturation", "ok": True, "fallback_used": False}],
                "output": {"image_path": self.image_path},
            }
        }

        guarded = stage_guard(state, stage_key="finish_output")
        summarized = summarize_stage({**state, **guarded, "mode": "explicit"}, stage_key="finish_output")

        self.assertTrue(summarized["phases"]["finish_output"].eval_report.num_operations >= 1)
        self.assertEqual(summarized["phases"]["finish_output"].summary.stage, "finish_output")
        self.assertIn("adjust_vibrance_saturation", summarized["phases"]["finish_output"].summary.used_tools)


if __name__ == "__main__":
    unittest.main()
