"""Unit tests for the parse_request graph node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.nodes.parse_request import parse_request
from app.graph.state import RequestIntent
from app.services.parse_request_model import generate_request_intent_with_qwen


class ParseRequestNodeTest(unittest.TestCase):
    """Verify request parsing fallback and model-driven paths."""

    def test_parse_request_uses_rule_fallback_without_model(self) -> None:
        state = {
            "request_text": "把背景稍微压暗一点，并提亮主体",
            "package_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        self.assertEqual(result["mode"], "explicit")
        self.assertTrue(result["request_intent"]["requested_packages"])

    def test_parse_request_uses_model_when_available(self) -> None:
        state = {
            "request_text": "帮我自然一点",
            "package_catalog": [
                {
                    "name": "adjust_exposure",
                    "description": "Adjust exposure",
                    "supported_regions": ["whole_image", "main_subject"],
                    "mask_policy": "optional",
                    "supported_domains": ["general"],
                    "risk_level": "low",
                    "params_schema": {},
                }
            ],
        }

        with (
            patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=True),
            patch(
                "app.graph.nodes.parse_request.generate_request_intent_with_qwen",
                return_value=RequestIntent(
                    mode="auto",
                    requested_packages=[],
                    constraints=["avoid_overediting"],
                ),
            ),
        ):
            result = parse_request(state)

        self.assertEqual(result["mode"], "auto")
        self.assertEqual(result["request_intent"]["constraints"], ["avoid_overediting"])

    def test_parse_request_marks_layered_repair_and_style_constraints(self) -> None:
        state = {
            "request_text": "夏日质感，修复逆光，自然一点",
            "package_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        constraints = set(result["request_intent"]["constraints"])
        requested_ops = {item["op"] for item in result["request_intent"]["requested_packages"]}
        self.assertIn("repair_backlighting", constraints)
        self.assertIn("build_summer_mood", constraints)
        self.assertIn("needs_layered_refinement", constraints)
        self.assertIn("adjust_exposure", requested_ops)
        self.assertIn("adjust_white_balance", requested_ops)

    def test_parse_request_model_uses_compact_tool_catalog(self) -> None:
        with patch(
            "app.services.parse_request_model.call_qwen_for_json",
            return_value={
                "mode": "explicit",
                "requested_packages": [],
                "constraints": [],
            },
        ) as mocked_call:
            generate_request_intent_with_qwen(
                request_text="提亮一点",
                package_catalog=[
                    {
                        "name": "adjust_exposure",
                        "description": "Adjust exposure",
                        "supported_regions": ["whole_image", "masked_region"],
                        "mask_policy": "optional",
                        "supported_domains": ["general"],
                        "risk_level": "low",
                        "params_schema": {
                            "properties": {
                                "strength": {"type": "number", "description": "主曝光强度"}
                            }
                        },
                    }
                ],
            )

        payload = mocked_call.call_args.kwargs["user_payload"]
        tool_catalog = payload["工具目录"]
        self.assertEqual(tool_catalog[0]["name"], "adjust_exposure")
        self.assertNotIn("params_schema", tool_catalog[0])
        self.assertNotIn("params", tool_catalog[0])


if __name__ == "__main__":
    unittest.main()
