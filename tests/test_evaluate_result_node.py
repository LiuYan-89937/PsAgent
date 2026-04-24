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


if __name__ == "__main__":
    unittest.main()
