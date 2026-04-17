"""Unit tests for planner payload normalization."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.services.planner_model import _normalize_plan_payload, generate_edit_plan_with_qwen


class PlannerModelNormalizationTest(unittest.TestCase):
    """Verify raw planner payload normalization."""

    def test_normalize_accepts_top_level_list(self) -> None:
        plan = _normalize_plan_payload(
            [
                {
                    "op": "adjust_exposure",
                    "region": "main_subject",
                    "params": {"strength": 0.15},
                },
                {
                    "op": "adjust_contrast",
                    "region": "main_subject",
                    "params": {"strength": 0.1},
                },
            ],
            mode="explicit",
            domain="general",
        )

        self.assertEqual(plan.mode, "explicit")
        self.assertEqual(plan.executor, "hybrid")
        self.assertEqual([item.op for item in plan.operations], ["adjust_exposure", "adjust_contrast"])

    def test_normalize_accepts_plan_key(self) -> None:
        plan = _normalize_plan_payload(
            {
                "plan": [
                    {
                        "op": "adjust_exposure",
                        "region": "main_subject",
                        "params": {"strength": 0.15},
                    }
                ]
            },
            mode="explicit",
            domain="general",
        )

        self.assertEqual(plan.mode, "explicit")
        self.assertEqual(plan.executor, "hybrid")
        self.assertEqual(plan.operations[0].op, "adjust_exposure")

    def test_normalize_preserves_top_level_strength_without_forcing_it_into_params(self) -> None:
        plan = _normalize_plan_payload(
            [
                {
                    "op": "adjust_clarity",
                    "region": "face and skin area",
                    "strength": 0.2,
                    "params": {"radius_scale": 1.1},
                }
            ],
            mode="explicit",
            domain="portrait",
        )

        self.assertEqual(plan.operations[0].strength, 0.2)
        self.assertNotIn("strength", plan.operations[0].params)

    def test_normalize_moves_known_top_level_package_params_into_params(self) -> None:
        plan = _normalize_plan_payload(
            [
                {
                    "op": "adjust_color_mixer",
                    "region": "bottle and water spray",
                    "orange_saturation": 0.25,
                    "yellow_luminance": 0.1,
                }
            ],
            mode="explicit",
            domain="general",
        )

        self.assertEqual(plan.operations[0].params["orange_saturation"], 0.25)
        self.assertEqual(plan.operations[0].params["yellow_luminance"], 0.1)

    def test_generate_edit_plan_uses_compact_model_catalog(self) -> None:
        package_catalog = [
            {
                "name": "adjust_exposure",
                "description": "Adjust exposure",
                "supported_regions": ["whole_image", "masked_region"],
                "mask_policy": "optional",
                "supported_domains": ["general"],
                "risk_level": "low",
                "params_schema": {
                    "properties": {
                        "strength": {
                            "type": "number",
                            "description": "主曝光强度",
                            "minimum": -1.0,
                            "maximum": 1.0,
                        }
                    }
                },
            }
        ]

        with patch(
            "app.services.planner_model.call_qwen_for_json",
            return_value={
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "whole_image",
                        "params": {"strength": 0.2},
                    }
                ]
            },
        ) as mocked_call:
            generate_edit_plan_with_qwen(
                request_text="轻微提亮",
                request_intent={"mode": "explicit", "requested_packages": [], "constraints": []},
                image_analysis={"domain": "general"},
                package_catalog=package_catalog,
                retrieved_prefs=[],
            )

        payload = mocked_call.call_args.kwargs["user_payload"]
        tool_catalog = payload["工具目录"]
        self.assertEqual(tool_catalog[0]["name"], "adjust_exposure")
        self.assertIn("params", tool_catalog[0])
        self.assertNotIn("params_schema", tool_catalog[0])
        self.assertIn("局部分割公共参数", payload)
        self.assertNotIn("mask_prompt", {item["name"] for item in tool_catalog[0]["params"]})

    def test_generate_edit_plan_uses_compact_planner_context_sections(self) -> None:
        with patch(
            "app.services.planner_model.call_qwen_for_json",
            return_value={
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "whole_image",
                        "params": {"strength": 0.2},
                    }
                ]
            },
        ) as mocked_call:
            generate_edit_plan_with_qwen(
                request_text="轻微提亮",
                request_intent={
                    "mode": "explicit",
                    "requested_packages": [{"op": "adjust_exposure", "region": "whole_image", "strength": 0.2, "params": {}}],
                    "constraints": ["avoid_overediting"],
                },
                image_analysis={
                    "domain": "general",
                    "issues": ["underexposed"],
                    "summary": "主体偏暗",
                    "metrics": {"brightness_mean": 90, "shadow_ratio": 0.2},
                },
                package_catalog=[
                    {
                        "name": "adjust_exposure",
                        "description": "Adjust exposure",
                        "supported_regions": ["whole_image", "masked_region"],
                        "mask_policy": "optional",
                        "supported_domains": ["general"],
                        "risk_level": "low",
                        "params_schema": {"properties": {}},
                    }
                ],
                retrieved_prefs=[{"key": "style", "value": "natural", "confidence": 0.8}],
                previous_plan={"operations": [{"op": "adjust_exposure", "region": "whole_image", "params": {"strength": 0.1}}]},
                previous_execution_trace=[{"op": "adjust_exposure", "region": "whole_image", "ok": True, "fallback_used": False}],
                previous_eval_report={"summary": "还可以更亮一点", "warnings": ["主体略暗"], "should_continue_editing": True},
            )

        payload = mocked_call.call_args.kwargs["user_payload"]
        self.assertIn("mode", payload["需求意图"])
        self.assertIn("summary", payload["图像分析"])
        self.assertIn("operations", payload["上一轮计划"])
        self.assertIn("key", payload["长期偏好"][0])


if __name__ == "__main__":
    unittest.main()
