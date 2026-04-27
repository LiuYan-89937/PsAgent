"""Unit tests for the parse_request graph node."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.graph.nodes.parse_request import parse_request
from app.graph.state import RequestGoal, RequestIntent
from app.services.parse_request_model import generate_request_intent


class ParseRequestNodeTest(unittest.TestCase):
    """Verify request parsing fallback and model-driven paths."""

    def test_parse_request_uses_rule_fallback_without_model(self) -> None:
        state = {
            "request_text": "把背景稍微压暗一点，并提亮主体",
            "tool_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        self.assertEqual(result["mode"], "auto")
        self.assertFalse(result["request_intent"]["requested_tools"])
        goals = {item["kind"] for item in result["request_intent"]["goals"]}
        self.assertIn("lift_luminance", goals)
        self.assertIn("background_balance", goals)

    def test_parse_request_respects_search_mode_for_custom_prompt(self) -> None:
        state = {
            "mode": "auto",
            "request_text": "将逆光喷水人像修成明亮清透的夏日胶片风格",
            "tool_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        self.assertEqual(result["mode"], "auto")
        self.assertEqual(result["request_intent"]["mode"], "auto")
        self.assertTrue(result["request_intent"]["goals"])

    def test_parse_request_uses_model_when_available(self) -> None:
        state = {
            "request_text": "帮我自然一点",
            "tool_catalog": [
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
                "app.graph.nodes.parse_request.generate_request_intent",
                return_value=RequestIntent(
                    mode="auto",
                    requested_tools=[],
                    constraints=["avoid_overediting"],
                ),
            ),
        ):
            result = parse_request(state)

        self.assertEqual(result["mode"], "auto")
        self.assertEqual(result["request_intent"]["constraints"], ["avoid_overediting"])

    def test_parse_request_stabilizes_model_style_false_positive(self) -> None:
        state = {
            "request_text": "保留原图逆光氛围和暗背景，自然美化",
            "tool_catalog": [],
        }

        with (
            patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=True),
            patch(
                "app.graph.nodes.parse_request.generate_request_intent",
                return_value=RequestIntent(
                    mode="auto",
                    requested_tools=[],
                    constraints=[],
                    wants_style=True,
                ),
            ),
        ):
            result = parse_request(state)

        self.assertFalse(result["request_intent"]["wants_style"])
        self.assertIn("preserve_original_mood", result["request_intent"]["constraints"])

    def test_request_goal_accepts_user_source_alias(self) -> None:
        goal = RequestGoal.model_validate(
            {
                "kind": "lift_luminance",
                "focus": "global_tone",
                "target_region": "whole_image",
                "priority": 80,
                "source": "user",
            }
        )

        self.assertEqual(goal.source, "model")

    def test_parse_request_falls_back_when_model_payload_validation_fails(self) -> None:
        state = {
            "request_text": "提亮主体并保留逆光氛围",
            "tool_catalog": [],
            "fallback_trace": [],
        }

        with (
            patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=True),
            patch(
                "app.graph.nodes.parse_request.generate_request_intent",
                side_effect=ValidationError.from_exception_data(
                    "RequestIntent",
                    [
                        {
                            "type": "literal_error",
                            "loc": ("goals", 0, "source"),
                            "msg": "Input should be 'heuristic', 'model' or 'explicit_tool'",
                            "input": "user",
                            "ctx": {"expected": "'heuristic', 'model' or 'explicit_tool'"},
                        }
                    ],
                ),
            ),
        ):
            result = parse_request(state)

        self.assertEqual(result["mode"], "auto")
        self.assertTrue(result["request_intent"]["goals"])
        self.assertTrue(result["fallback_trace"])
        self.assertEqual(result["fallback_trace"][-1]["strategy"], "heuristic_request_intent")

    def test_parse_request_marks_layered_repair_and_style_constraints(self) -> None:
        state = {
            "request_text": "胶片质感，修复逆光，自然一点",
            "tool_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        constraints = set(result["request_intent"]["constraints"])
        goal_kinds = {item["kind"] for item in result["request_intent"]["goals"]}
        self.assertIn("repair_backlighting", constraints)
        self.assertIn("needs_layered_refinement", constraints)
        self.assertTrue(result["request_intent"]["wants_style"])
        self.assertIn("lift_luminance", goal_kinds)

    def test_parse_request_does_not_treat_preserved_mood_as_style(self) -> None:
        state = {
            "request_text": "保留原图逆光氛围和暗背景，自然美化，不要过度",
            "tool_catalog": [],
        }

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request(state)

        constraints = set(result["request_intent"]["constraints"])
        self.assertIn("preserve_original_mood", constraints)
        self.assertIn("avoid_overediting", constraints)
        self.assertFalse(result["request_intent"]["wants_style"])

    def test_parse_request_model_uses_compact_tool_catalog(self) -> None:
        with patch(
            "app.services.parse_request_model.invoke_json",
            return_value={
                "mode": "auto",
                "requested_tools": [],
                "constraints": [],
            },
        ) as mocked_call:
            generate_request_intent(
                request_text="提亮一点",
                tool_catalog=[
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
