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
                    "decision": "accept",
                    "next_focus": None,
                    "correction_objective": "",
                    "decision_reason": "结果可接受",
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
                    "summary": "模型摘要不参与续搜路由。",
                    "decision": "continue_auto",
                    "next_focus": "subject_separation",
                    "correction_objective": "加大人物区域提亮力度，改善面部暗部可读性。",
                    "decision_reason": "主体暗部仍未达到目标",
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])
        gaps = result["objective_card"]["gaps"]
        self.assertEqual(gaps[-1]["focus"], "subject_separation")
        self.assertEqual(gaps[-1]["desired_delta"], "加大人物区域提亮力度，改善面部暗部可读性。")
        self.assertIn("final_review_continuation", gaps[-1]["constraints"])

    def test_final_review_requests_continuation_when_output_matches_original(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            original = Path(tmpdir) / "original.png"
            Image.new("RGB", (40, 40), (80, 70, 60)).save(original)
            state = {
                "mode": "auto",
                "input_images": [str(original)],
                "selected_output": str(original),
                "execution_trace": [],
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
                    {"id": "round_1", "index": 1, "focus": "global_tone", "completed": True},
                ],
            }

            with patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=False):
                result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])
        self.assertTrue(any("未产生有效修改" in warning for warning in result["eval_report"].warnings))
        self.assertIn("final_review_continuation", result["objective_card"]["gaps"][-1]["constraints"])

    def test_auto_review_request_continues_before_round_cap(self) -> None:
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
                {"id": "round_1", "index": 1, "focus": "subject_separation", "completed": True},
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": False,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": ["效果不足，需要复核。"],
                    "warnings": [],
                    "summary": "效果不足，需要复核。",
                    "decision": "request_human_review",
                    "next_focus": None,
                    "correction_objective": "",
                    "decision_reason": "需要用户判断效果取舍",
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])
        self.assertIn("final_review_continuation", result["objective_card"]["gaps"][-1]["constraints"])

    def test_auto_review_request_stops_at_min_rounds_without_actionable_continuation(self) -> None:
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
                for index in range(1, 5)
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": False,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": ["建议人工复核。"],
                    "warnings": [],
                    "summary": "建议人工复核。",
                    "decision": "request_human_review",
                    "next_focus": None,
                    "correction_objective": "",
                    "decision_reason": "需要人工确认",
                },
            ),
        ):
            result = final_review(state)

        self.assertFalse(result["needs_search_continuation"])
        self.assertTrue(result["approval_required"])

    def test_auto_corrective_review_continues_after_min_rounds(self) -> None:
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
                for index in range(1, 5)
            ],
        }

        with (
            patch("app.graph.nodes.evaluate_result.critic_model_available", return_value=True),
            patch(
                "app.graph.nodes.evaluate_result.evaluate_edit_result",
                return_value={
                    "overall_ok": False,
                    "preserve_ok": True,
                    "style_ok": False,
                    "artifact_ok": True,
                    "issues": ["模型问题文案不参与续搜路由。"],
                    "warnings": [],
                    "summary": "模型摘要不参与续搜路由。",
                    "decision": "continue_auto",
                    "next_focus": "global_tone",
                    "correction_objective": "回收高光，压住背景亮度并把整体色调调整回更自然的绿色夏日氛围。",
                    "decision_reason": "问题明确且可以由自动影调轮修正",
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])
        gap = result["objective_card"]["gaps"][-1]
        self.assertEqual(gap["focus"], "global_tone")
        self.assertIn("回收高光", gap["desired_delta"])

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
                    "decision": "continue_auto",
                    "next_focus": "subject_separation",
                    "correction_objective": "继续提升人物暗部可读性。",
                    "decision_reason": "主体仍偏暗",
                },
            ),
        ):
            result = final_review(state)

        self.assertFalse(result["needs_search_continuation"])
        self.assertTrue(result["approval_required"])
        self.assertEqual(result["approval_payload"].reason, "final_review_unresolved_after_max_rounds")
        self.assertEqual(result["approval_payload"].metadata["max_rounds"], 6)
        self.assertEqual(result["final_review"].decision, "request_human_review")

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
                    "decision": "continue_auto",
                    "next_focus": "subject_separation",
                    "correction_objective": "继续提升人物暗部可读性。",
                    "decision_reason": "主体仍偏暗",
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
                    "decision": "continue_auto",
                    "next_focus": "subject_separation",
                    "correction_objective": "继续提升人物暗部可读性。",
                    "decision_reason": "主体仍偏暗",
                },
            ),
        ):
            result = final_review(state)

        self.assertTrue(result["needs_search_continuation"])
        self.assertFalse(result["approval_required"])


if __name__ == "__main__":
    unittest.main()
