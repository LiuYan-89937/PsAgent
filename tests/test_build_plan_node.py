"""Unit tests for the build_plan graph node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.nodes.build_plan import build_plan_round_1, build_plan_round_2
from app.graph.state import EditPlan


class BuildPlanNodeTest(unittest.TestCase):
    """Verify planner fallback and model-driven paths."""

    def test_build_plan_falls_back_to_rule_planner_when_model_unavailable(self) -> None:
        state = {
            "mode": "explicit",
            "request_intent": {
                "mode": "explicit",
                "requested_packages": [
                    {
                        "op": "adjust_exposure",
                        "region": "whole_image",
                        "strength": 0.2,
                    },
                    {
                        "op": "adjust_contrast",
                        "region": "whole_image",
                        "strength": 0.15,
                    },
                ],
                "constraints": ["avoid_overediting"],
            },
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
        }

        with patch("app.graph.nodes.build_plan.planner_model_available", return_value=False):
            result = build_plan_round_1(state)

        plan = result["edit_plan"]
        self.assertEqual(plan["mode"], "explicit")
        self.assertEqual(plan["executor"], "deterministic")
        self.assertEqual([op["op"] for op in plan["operations"]], ["adjust_exposure", "adjust_contrast"])
        self.assertEqual(plan["preserve"], ["avoid_overediting"])

    def test_build_plan_uses_model_plan_when_available(self) -> None:
        model_plan = EditPlan(
            mode="explicit",
            domain="general",
            executor="deterministic",
            preserve=["avoid_overediting"],
            operations=[],
            should_write_memory=False,
            memory_candidates=[],
            needs_confirmation=False,
        )
        state = {
            "mode": "explicit",
            "request_text": "轻微提亮",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
        }

        with (
            patch("app.graph.nodes.build_plan.planner_model_available", return_value=True),
            patch("app.graph.nodes.build_plan.generate_edit_plan_with_qwen", return_value=model_plan) as mocked_call,
        ):
            result = build_plan_round_1(state)

        mocked_call.assert_called_once()
        self.assertEqual(result["edit_plan"]["mode"], "explicit")
        self.assertEqual(result["edit_plan"]["executor"], "deterministic")
        self.assertEqual(result["edit_plan"]["preserve"], ["avoid_overediting"])

    def test_build_plan_keeps_model_local_region_without_auto_injecting_mask_params(self) -> None:
        model_plan = EditPlan(
            mode="explicit",
            domain="portrait",
            executor="hybrid",
            preserve=[],
            operations=[
                {
                    "op": "adjust_exposure",
                    "region": "main_subject",
                    "params": {"strength": 0.18},
                    "priority": 0,
                }
            ],
            should_write_memory=False,
            memory_candidates=[],
            needs_confirmation=False,
        )
        state = {
            "mode": "explicit",
            "request_text": "修复人物逆光，提亮脸部",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": ["repair_backlighting"]},
            "image_analysis": {"domain": "portrait", "subjects": ["girl"]},
            "retrieved_prefs": [],
        }

        with (
            patch("app.graph.nodes.build_plan.planner_model_available", return_value=True),
            patch("app.graph.nodes.build_plan.generate_edit_plan_with_qwen", return_value=model_plan),
        ):
            result = build_plan_round_1(state)

        operation = result["edit_plan"]["operations"][0]
        self.assertEqual(operation["region"], "main_subject")
        self.assertNotIn("mask_provider", operation["params"])
        self.assertNotIn("mask_prompt", operation["params"])

    def test_build_plan_preserves_region_label_without_mask_prompt(self) -> None:
        model_plan = EditPlan(
            mode="explicit",
            domain="portrait",
            executor="hybrid",
            preserve=[],
            operations=[
                {
                    "op": "adjust_color_mixer",
                    "region": "person",
                    "params": {"orange_saturation": 0.18},
                    "priority": 0,
                }
            ],
            should_write_memory=False,
            memory_candidates=[],
            needs_confirmation=False,
        )
        state = {
            "mode": "explicit",
            "request_text": "让人物肤色更自然一点",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait", "subjects": ["girl"]},
            "retrieved_prefs": [],
        }

        with (
            patch("app.graph.nodes.build_plan.planner_model_available", return_value=True),
            patch("app.graph.nodes.build_plan.generate_edit_plan_with_qwen", return_value=model_plan),
        ):
            result = build_plan_round_1(state)

        operation = result["edit_plan"]["operations"][0]
        self.assertEqual(operation["region"], "person")
        self.assertNotIn("mask_prompt", operation["params"])

    def test_build_plan_round_2_passes_round_context_to_model(self) -> None:
        model_plan = EditPlan(
            mode="auto",
            domain="general",
            executor="hybrid",
            preserve=[],
            operations=[],
            should_write_memory=False,
            memory_candidates=[],
            needs_confirmation=False,
        )
        state = {
            "request_text": "自然一点",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/round1.png",
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "round_plans": {"round_1": {"operations": [{"op": "adjust_curves"}]}},
            "round_execution_traces": {"round_1": [{"op": "adjust_curves", "ok": True}]},
            "round_eval_reports": {"round_1": {"warnings": ["主体仍偏暗"]}},
        }

        with (
            patch("app.graph.nodes.build_plan.planner_model_available", return_value=True),
            patch("app.graph.nodes.build_plan.generate_edit_plan_with_qwen", return_value=model_plan) as mocked_call,
        ):
            build_plan_round_2(state)

        kwargs = mocked_call.call_args.kwargs
        self.assertEqual(kwargs["round_name"], "round_2")
        self.assertEqual(kwargs["image_paths"], ["/tmp/original.png", "/tmp/round1.png"])
        self.assertEqual(kwargs["previous_plan"], {"operations": [{"op": "adjust_curves"}]})

    def test_build_plan_fallback_expands_layered_explicit_request(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "夏日质感，修复逆光",
            "request_intent": {
                "mode": "explicit",
                "requested_packages": [
                    {"op": "adjust_exposure", "region": "main_subject", "strength": 0.3},
                    {"op": "adjust_white_balance", "region": "whole_image", "strength": 0.16},
                ],
                "constraints": ["repair_backlighting", "build_summer_mood", "needs_layered_refinement"],
            },
            "image_analysis": {"domain": "portrait", "issues": ["underexposed", "crushed_shadows"]},
            "retrieved_prefs": [],
        }

        with patch("app.graph.nodes.build_plan.planner_model_available", return_value=False):
            result = build_plan_round_1(state)

        plan = result["edit_plan"]
        self.assertEqual(plan["executor"], "deterministic")
        self.assertGreaterEqual(len(plan["operations"]), 5)
        op_names = [op["op"] for op in plan["operations"]]
        self.assertIn("adjust_exposure", op_names)
        self.assertIn("adjust_highlights_shadows", op_names)
        self.assertIn("adjust_white_balance", op_names)
        self.assertIn("adjust_vibrance_saturation", op_names)
        local_operations = [op for op in plan["operations"] if op["region"] != "whole_image"]
        self.assertTrue(local_operations)
        for operation in local_operations:
            self.assertNotIn("mask_provider", operation["params"])
            self.assertNotIn("mask_prompt", operation["params"])

    def test_build_plan_round_2_fallback_adds_finishing_ops_for_layered_request(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "夏日质感，修复逆光",
            "request_intent": {
                "mode": "explicit",
                "requested_packages": [],
                "constraints": ["repair_backlighting", "build_summer_mood", "needs_layered_refinement"],
            },
            "image_analysis": {"domain": "portrait"},
            "retrieved_prefs": [],
            "round_eval_reports": {
                "round_1": {
                    "summary": "主体还可以更亮一点，色调也还可以更有夏日感。",
                    "warnings": ["主体仍偏暗", "夏日氛围还不够明显"],
                    "issues": [],
                }
            },
        }

        with patch("app.graph.nodes.build_plan.planner_model_available", return_value=False):
            result = build_plan_round_2(state)

        op_names = [op["op"] for op in result["edit_plan"]["operations"]]
        self.assertIn("adjust_clarity", op_names)
        self.assertIn("adjust_color_mixer", op_names)
        self.assertIn("adjust_dehaze", op_names)
        local_operations = [op for op in result["edit_plan"]["operations"] if op["region"] != "whole_image"]
        self.assertTrue(local_operations)
        for operation in local_operations:
            self.assertNotIn("mask_provider", operation["params"])
            self.assertNotIn("mask_prompt", operation["params"])


if __name__ == "__main__":
    unittest.main()
