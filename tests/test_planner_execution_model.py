"""Unit tests for stage-aware planner output normalization."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.state import StageContextEnvelope, StagePolicy
from app.services.planner_execution_model import generate_stage_execution_plan_with_qwen


class PlannerExecutionModelTest(unittest.TestCase):
    """Verify loose stage-planner JSON gets normalized into PlannerExecutionPlan."""

    def test_generate_stage_execution_plan_normalizes_legacy_tool_shape(self) -> None:
        policy = StagePolicy(
            key="global_base",
            label="全局基线",
            prompt_name="global_base.txt",
            visible_tools=[
                "adjust_exposure",
                "adjust_contrast",
                "adjust_vibrance_saturation",
            ],
            llm_enabled=True,
            step_budget=4,
            tool_repeat_limit=2,
            tone_stack_limit=4,
            mask_allowed=False,
            mask_required=False,
            context_whitelist=[],
            guard_thresholds={},
        )
        context = StageContextEnvelope(
            request_summary="提升通透度和色彩",
            current_image_path="/tmp/input.png",
            edit_profile_summary={},
            relevant_image_analysis={},
            available_masks=[],
            previous_stage_summaries=[],
            stage_constraints=[],
        )

        raw_payload = {
            "current_stage": "global_base",
            "summary": "建立全局基线",
            "steps": [
                {"tool": "adjust_exposure", "strength": 55, "max_stops": 70, "feather_radius": 20},
                {"tool": "adjust_contrast", "strength": 48, "contrast_scale": 72, "pivot": 50},
            ],
        }

        with patch("app.services.planner_execution_model.call_qwen_for_json", return_value=raw_payload):
            plan = generate_stage_execution_plan_with_qwen(
                stage_policy=policy,
                stage_context=context,
                tool_catalog=[],
                current_image_path="/tmp/input.png",
                fallback_mode="explicit",
                fallback_domain="general",
            )

        self.assertEqual(plan.mode, "explicit")
        self.assertEqual(plan.domain, "general")
        self.assertEqual(plan.executor, "deterministic")
        self.assertEqual(plan.steps[0].op, "adjust_exposure")
        self.assertIn("max_stops", plan.steps[0].params)
        self.assertEqual(plan.steps[1].op, "adjust_contrast")


if __name__ == "__main__":
    unittest.main()
