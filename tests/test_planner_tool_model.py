"""Unit tests for planner tool-name resolution helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.planner_tool_model import (
    build_operation_from_tool_call,
    build_planner_tools,
    call_planner_tool_turn,
    extract_single_tool_call,
    resolve_planner_tool_name,
)


class PlannerToolModelTest(unittest.TestCase):
    """Verify exact and fallback tool-name resolution."""

    def test_resolve_planner_tool_name_keeps_exact_match(self) -> None:
        resolved, meta = resolve_planner_tool_name("sharpen", {"region": "whole_image"})
        self.assertEqual(resolved, "sharpen")
        self.assertEqual(meta["strategy"], "exact")

    def test_resolve_planner_tool_name_handles_generated_adjust_alias(self) -> None:
        resolved, meta = resolve_planner_tool_name("adjust_sharpen", {"region": "whole_image"})
        self.assertEqual(resolved, "sharpen")
        self.assertIn(meta["strategy"], {"alias", "similarity"})

    def test_resolve_planner_tool_name_uses_similarity_for_wrong_tool_name(self) -> None:
        resolved, meta = resolve_planner_tool_name(
            "tool_sharpen",
            {"region": "whole_image", "strength": 0.2},
        )
        self.assertEqual(resolved, "sharpen")
        self.assertEqual(meta["strategy"], "similarity")

    def test_call_planner_tool_turn_attaches_image_on_first_step(self) -> None:
        with patch("app.services.planner_tool_model.call_qwen_for_tool_message", return_value={"tool_calls": []}) as mocked_call:
            call_planner_tool_turn(
                request_text="提亮一点",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                retrieved_prefs=[],
                current_image_path="/tmp/input.png",
                round_name="round_1",
                current_step=1,
                round_operations=[],
                latest_result=None,
            )

        self.assertEqual(mocked_call.call_args.kwargs["image_paths"], ["/tmp/input.png"])

    def test_call_planner_tool_turn_skips_image_on_simple_follow_up_step(self) -> None:
        with patch("app.services.planner_tool_model.call_qwen_for_tool_message", return_value={"tool_calls": []}) as mocked_call:
            call_planner_tool_turn(
                request_text="提亮一点",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                retrieved_prefs=[],
                current_image_path="/tmp/input.png",
                round_name="round_1",
                current_step=2,
                round_operations=[{"op": "adjust_exposure", "region": "whole_image"}],
                latest_result={"op": "adjust_exposure", "region": "whole_image", "ok": True, "fallback_used": False},
            )

        self.assertIsNone(mocked_call.call_args.kwargs["image_paths"])

    def test_call_planner_tool_turn_skips_image_on_follow_up_even_after_fallback(self) -> None:
        with patch("app.services.planner_tool_model.call_qwen_for_tool_message", return_value={"tool_calls": []}) as mocked_call:
            call_planner_tool_turn(
                request_text="提亮一点",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                retrieved_prefs=[],
                current_image_path="/tmp/input.png",
                round_name="round_1",
                current_step=2,
                round_operations=[{"op": "adjust_exposure", "region": "face"}],
                latest_result={"op": "adjust_exposure", "region": "face", "ok": False, "fallback_used": True},
            )

        self.assertIsNone(mocked_call.call_args.kwargs["image_paths"])

    def test_call_planner_tool_turn_uses_required_without_thinking_by_default(self) -> None:
        with patch("app.services.planner_tool_model.call_qwen_for_tool_message", return_value={"tool_calls": []}) as mocked_call:
            call_planner_tool_turn(
                request_text="提亮一点",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                retrieved_prefs=[],
                current_image_path="/tmp/input.png",
                round_name="round_1",
                current_step=1,
                round_operations=[],
                latest_result=None,
                planner_thinking_mode=False,
            )

        self.assertEqual(mocked_call.call_args.kwargs["tool_choice"], "required")
        self.assertFalse(mocked_call.call_args.kwargs["enable_thinking"])

    def test_call_planner_tool_turn_uses_auto_with_thinking_mode(self) -> None:
        with patch("app.services.planner_tool_model.call_qwen_for_tool_message", return_value={"tool_calls": []}) as mocked_call:
            call_planner_tool_turn(
                request_text="提亮一点",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                retrieved_prefs=[],
                current_image_path="/tmp/input.png",
                round_name="round_1",
                current_step=1,
                round_operations=[],
                latest_result=None,
                planner_thinking_mode=True,
            )

        self.assertEqual(mocked_call.call_args.kwargs["tool_choice"], "auto")
        self.assertTrue(mocked_call.call_args.kwargs["enable_thinking"])

    def test_build_planner_tools_uses_integer_slider_schema_for_numeric_params(self) -> None:
        tools = build_planner_tools()
        exposure_tool = next(item for item in tools if item["function"]["name"] == "adjust_exposure")
        properties = exposure_tool["function"]["parameters"]["properties"]

        self.assertEqual(properties["strength"]["type"], "integer")
        self.assertEqual(properties["strength"]["minimum"], 0)
        self.assertEqual(properties["strength"]["maximum"], 100)
        self.assertIn("仅填 0-100 整数", properties["strength"]["description"])

    def test_extract_single_tool_call_repairs_invalid_json_number(self) -> None:
        tool_name, arguments = extract_single_tool_call(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "adjust_exposure",
                            "arguments": '{"strength": 00.25, "max_stops": 50, "region": "whole_image"}',
                        }
                    }
                ]
            }
        )

        self.assertEqual(tool_name, "adjust_exposure")
        self.assertAlmostEqual(arguments["strength"], 0.25)

    def test_build_operation_from_tool_call_decodes_integer_slider_values(self) -> None:
        operation = build_operation_from_tool_call(
            "adjust_exposure",
            {
                "region": "whole_image",
                "strength": 75,
                "max_stops": 50,
                "feather_radius": 25,
                "mask_semantic_type": "True",
            },
        )

        self.assertEqual(operation["region"], "whole_image")
        self.assertAlmostEqual(operation["params"]["strength"], 0.5)
        self.assertAlmostEqual(operation["params"]["max_stops"], 1.625)
        self.assertAlmostEqual(operation["params"]["feather_radius"], 16.0)
        self.assertTrue(operation["params"]["mask_semantic_type"])


if __name__ == "__main__":
    unittest.main()
