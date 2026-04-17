"""Unit tests for critic model payload shaping."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.critic_model import evaluate_edit_result_with_qwen


class CriticModelTest(unittest.TestCase):
    """Verify critic model receives compact payload context."""

    def test_evaluate_edit_result_uses_compact_payload_sections(self) -> None:
        with patch(
            "app.services.critic_model.call_qwen_for_json",
            return_value={
                "overall_ok": True,
                "preserve_ok": True,
                "style_ok": True,
                "artifact_ok": True,
                "issues": [],
                "warnings": [],
                "summary": "ok",
                "should_continue_editing": False,
                "should_request_review": False,
            },
        ) as mocked_call:
            evaluate_edit_result_with_qwen(
                original_image_path="/tmp/original.png",
                edited_image_path="/tmp/edited.png",
                request_text="自然一点",
                edit_plan={
                    "mode": "explicit",
                    "operations": [
                        {
                            "op": "adjust_exposure",
                            "region": "逆光脸部区域",
                            "params": {"strength": 0.18, "mask_prompt": "young woman face and neck skin"},
                        }
                    ],
                },
                image_analysis={
                    "domain": "portrait",
                    "issues": ["underexposed"],
                    "subjects": ["young woman"],
                    "summary": "主体偏暗",
                    "metrics": {"brightness_mean": 90, "shadow_ratio": 0.2},
                },
                execution_trace=[
                    {
                        "op": "adjust_exposure",
                        "region": "逆光脸部区域",
                        "ok": True,
                        "fallback_used": False,
                        "applied_params": {
                            "params": {
                                "strength": 0.18,
                                "max_stops": 1.35,
                                "mask_prompt": "young woman face and neck skin",
                                "mask_negative_prompt": "background foliage",
                            }
                        },
                    }
                ],
            )

        payload = mocked_call.call_args.kwargs["user_payload"]
        self.assertEqual(payload["修图计划"]["operations"][0]["op"], "adjust_exposure")
        self.assertEqual(payload["图像分析"]["domain"], "portrait")
        self.assertEqual(payload["执行摘要"][0]["op"], "adjust_exposure")
        self.assertIn("params", payload["执行摘要"][0])
        self.assertNotIn("metrics", payload["修图计划"])


if __name__ == "__main__":
    unittest.main()
