"""Unit tests for the evaluate_result graph node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.nodes.evaluate_result import evaluate_result, evaluate_round_1, finalize_round_1_result


class EvaluateResultNodeTest(unittest.TestCase):
    """Verify execution-only and critic-model evaluation flows."""

    def test_evaluate_result_returns_execution_report_without_model(self) -> None:
        state = {
            "execution_trace": [
                {"ok": True, "fallback_used": False},
                {"ok": False, "fallback_used": True},
            ],
            "selected_output": "/tmp/out.png",
        }

        with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
            result = evaluate_result(state)

        self.assertEqual(result["eval_report"]["num_operations"], 2)
        self.assertEqual(result["eval_report"]["success_count"], 1)
        self.assertEqual(result["eval_report"]["fallback_count"], 1)
        self.assertFalse(result["approval_required"])

    def test_evaluate_result_merges_critic_output(self) -> None:
        state = {
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "request_text": "自然一点",
            "edit_plan": {"operations": []},
            "image_analysis": {"domain": "general"},
            "execution_trace": [{"ok": True, "fallback_used": False}],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result_with_qwen",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": ["主体稍暗"],
                    "summary": "整体自然，略可提亮主体。",
                    "should_request_review": False,
                },
            ),
        ):
            result = evaluate_result(state)

        self.assertTrue(result["eval_report"]["overall_ok"])
        self.assertEqual(result["eval_report"]["warnings"], ["主体稍暗"])
        self.assertFalse(result["approval_required"])

    def test_evaluate_round_1_sets_continue_flag_from_critic(self) -> None:
        state = {
            "current_round": 1,
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "request_text": "自然一点",
            "edit_plan": {"operations": []},
            "image_analysis": {"domain": "general"},
            "execution_trace": [{"ok": True, "fallback_used": False}],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result_with_qwen",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": ["主体仍偏暗"],
                    "summary": "还可以再提一点主体亮度。",
                    "should_continue_editing": True,
                    "should_request_review": False,
                },
            ),
        ):
            result = evaluate_round_1(state)

        self.assertTrue(result["continue_to_round_2"])
        self.assertIn("round_1", result["round_eval_reports"])
        self.assertFalse(result["approval_required"])

    def test_evaluate_round_1_falls_back_when_critic_returns_empty_content(self) -> None:
        state = {
            "current_round": 1,
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "request_text": "自然一点",
            "edit_plan": {"operations": []},
            "image_analysis": {"domain": "general"},
            "execution_trace": [{"ok": True, "fallback_used": False}],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result_with_qwen",
                side_effect=RuntimeError("DashScope returned empty content."),
            ),
        ):
            result = evaluate_round_1(state)

        self.assertIn("round_1", result["round_eval_reports"])
        self.assertEqual(result["eval_report"]["num_operations"], 1)
        self.assertFalse(result["approval_required"])

    def test_finalize_round_1_result_promotes_round_report(self) -> None:
        state = {
            "round_eval_reports": {
                "round_1": {
                    "selected_output": "/tmp/edited.png",
                    "num_operations": 2,
                    "success_count": 2,
                    "failure_count": 0,
                    "fallback_count": 0,
                    "has_output": True,
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": ["还可更通透"],
                    "summary": "首轮已经可用。",
                    "should_continue_editing": False,
                    "should_request_review": False,
                }
            }
        }

        result = finalize_round_1_result(state)
        self.assertEqual(result["eval_report"]["summary"], "首轮已经可用。")
        self.assertFalse(result["approval_required"])

    def test_evaluate_round_1_forces_second_round_for_layered_request_without_critic(self) -> None:
        state = {
            "current_round": 1,
            "request_text": "夏日质感，修复逆光",
            "request_intent": {
                "mode": "explicit",
                "requested_packages": [
                    {"op": "adjust_exposure"},
                    {"op": "adjust_white_balance"},
                    {"op": "adjust_vibrance_saturation"},
                ],
                "constraints": ["repair_backlighting", "build_summer_mood", "needs_layered_refinement"],
            },
            "image_analysis": {"issues": ["underexposed", "crushed_shadows"]},
            "execution_trace": [
                {"ok": True, "fallback_used": False},
                {"ok": True, "fallback_used": False},
            ],
            "selected_output": "/tmp/edited.png",
        }

        with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
            result = evaluate_round_1(state)

        self.assertTrue(result["continue_to_round_2"])
        self.assertIn("round_1", result["round_eval_reports"])

    def test_evaluate_round_1_continues_when_round_1_had_fallback(self) -> None:
        state = {
            "current_round": 1,
            "mode": "explicit",
            "request_text": "轻微提亮人像",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait", "issues": ["underexposed"]},
            "execution_trace": [
                {"ok": True, "fallback_used": False},
                {"ok": False, "fallback_used": True},
            ],
            "fallback_trace": [
                {
                    "stage": "plan_execute_round_1",
                    "source": "planner_tool_model",
                    "location": "round_execution",
                    "strategy": "finish_current_round",
                    "message": "实时规划中断，保留当前轮已完成结果。",
                }
            ],
            "selected_output": "/tmp/edited.png",
        }

        with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
            result = evaluate_round_1(state)

        self.assertTrue(result["continue_to_round_2"])

    def test_evaluate_round_1_continues_for_auto_portrait_request(self) -> None:
        state = {
            "current_round": 1,
            "mode": "auto",
            "request_text": "把人像修得更有质感一点",
            "request_intent": {"mode": "auto", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait", "issues": ["flat_face_light"]},
            "execution_trace": [{"ok": True, "fallback_used": False}],
            "selected_output": "/tmp/edited.png",
        }

        with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
            result = evaluate_round_1(state)

        self.assertTrue(result["continue_to_round_2"])


if __name__ == "__main__":
    unittest.main()
