"""Unit tests for the final review node."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.evaluate_result import final_review


class EvaluateResultNodeTest(unittest.TestCase):
    """Verify final review evaluation flows."""

    def test_final_review_returns_execution_report_without_model(self) -> None:
        state = {
            "execution_trace": [
                {"ok": True, "fallback_used": False},
                {"ok": False, "fallback_used": True},
            ],
            "selected_output": "/tmp/out.png",
        }

        with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
            result = final_review(state)

        self.assertEqual(result["eval_report"].num_operations, 2)
        self.assertEqual(result["eval_report"].success_count, 1)
        self.assertEqual(result["eval_report"].fallback_count, 1)
        self.assertEqual(result["final_review"].fallback_count, 1)
        self.assertFalse(result["approval_required"])
        self.assertNotIn("phases", result)

    def test_final_review_merges_critic_output(self) -> None:
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
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
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
            result = final_review(state)

        self.assertTrue(result["eval_report"].overall_ok)
        self.assertEqual(result["eval_report"].warnings, ["主体稍暗"])
        self.assertFalse(result["approval_required"])
        self.assertEqual(result["final_review"].summary, "整体自然，略可提亮主体。")

    def test_final_review_flags_deterministic_milky_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir) / "original.png"
            edited = Path(tmpdir) / "edited.png"
            Image.new("RGB", (40, 40), (20, 30, 22)).save(original)
            Image.new("RGB", (40, 40), (165, 168, 162)).save(edited)
            state = {
                "input_images": [str(original)],
                "selected_output": str(edited),
                "execution_trace": [{"ok": True, "fallback_used": False}],
            }

            with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
                result = final_review(state)

        self.assertTrue(result["approval_required"])
        self.assertTrue(any("黑位" in warning or "奶白" in warning for warning in result["eval_report"].warnings))

    def test_final_review_requests_continuation_for_underdone_auto_result(self) -> None:
        state = {
            "mode": "auto",
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "execution_trace": [{"ok": True, "fallback_used": False}],
            "objective_card": {
                "summary": "自然美化",
                "mode": "auto",
                "domain": "portrait",
                "preserve": [],
                "goals": [],
                "gaps": [
                    {
                        "id": "old_gap",
                        "focus": "finish",
                        "description": "收尾",
                        "priority": 30,
                        "resolved": True,
                    }
                ],
                "constraints": [],
            },
            "rounds": [
                {"id": "round_1", "index": 1, "focus": "subject_separation", "completed": True},
                {"id": "round_2", "index": 2, "focus": "subject_cleanup", "completed": True},
                {"id": "round_3", "index": 3, "focus": "finish", "completed": True},
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "修图方向自然但幅度过于保守，面部曝光仍偏暗，建议加大人物区域提亮力度。",
                    "should_continue_editing": False,
                    "should_request_review": False,
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])
        gaps = result["objective_card"]["gaps"]
        self.assertEqual(gaps[-1]["focus"], "subject_separation")
        self.assertIn("final_review_continuation", gaps[-1]["constraints"])

    def test_final_review_escalates_to_review_when_max_rounds_exhausted(self) -> None:
        state = {
            "mode": "auto",
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "execution_trace": [{"ok": True, "fallback_used": False}],
            "objective_card": {
                "summary": "自然美化",
                "mode": "auto",
                "domain": "portrait",
                "preserve": [],
                "goals": [],
                "gaps": [],
                "constraints": [],
            },
            "rounds": [
                {"id": f"round_{index}", "index": index, "focus": "finish", "completed": True}
                for index in range(1, 7)
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "结果仍偏暗，建议加大提亮力度。",
                    "should_continue_editing": True,
                    "should_request_review": False,
                },
            ),
        ):
            result = final_review(state)

        self.assertFalse(result["needs_search_continuation"])
        self.assertTrue(result["approval_required"])
        self.assertEqual(result["approval_payload"].reason, "final_review_unresolved_after_max_rounds")
        self.assertEqual(result["approval_payload"].metadata["max_rounds"], 6)

    def test_final_review_uses_search_effort_round_budget(self) -> None:
        base_state = {
            "mode": "auto",
            "search_effort": "high",
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "execution_trace": [{"ok": True, "fallback_used": False}],
            "objective_card": {
                "summary": "自然美化",
                "mode": "auto",
                "domain": "portrait",
                "preserve": [],
                "goals": [],
                "gaps": [],
                "constraints": [],
            },
            "rounds": [
                {"id": f"round_{index}", "index": index, "focus": "finish", "completed": True}
                for index in range(1, 8)
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "结果仍偏暗，建议加大提亮力度。",
                    "should_continue_editing": True,
                    "should_request_review": False,
                },
            ),
        ):
            result = final_review(base_state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])

    def test_final_review_counts_rounds_from_human_review_cycle_offset(self) -> None:
        state = {
            "mode": "auto",
            "search_effort": "standard",
            "search_cycle_round_offset": 6,
            "input_images": ["/tmp/original.png"],
            "selected_output": "/tmp/edited.png",
            "execution_trace": [{"ok": True, "fallback_used": False}],
            "objective_card": {
                "summary": "复核后继续",
                "mode": "auto",
                "domain": "portrait",
                "preserve": [],
                "goals": [],
                "gaps": [],
                "constraints": [],
            },
            "rounds": [
                {"id": f"round_{index}", "index": index, "focus": "finish", "completed": True}
                for index in range(1, 8)
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "结果仍偏暗，建议加大提亮力度。",
                    "should_continue_editing": True,
                    "should_request_review": False,
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])


if __name__ == "__main__":
    unittest.main()
