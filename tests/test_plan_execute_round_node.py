"""Unit tests for realtime planner tool-calling round nodes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.plan_execute_round import plan_execute_round_1
from app.tools.segmentation_tools import FalImageSegError, SegmentationResult


class PlanExecuteRoundNodeTest(unittest.TestCase):
    """Verify realtime planner-driven round execution."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        self.mask_path = str(Path(self.tmpdir.name) / "mask.png")
        Image.new("RGB", (32, 32), (80, 90, 100)).save(self.image_path)
        Image.new("L", (32, 32), 255).save(self.mask_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_plan_execute_round_runs_whole_image_tool_and_finishes(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "轻微提亮",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_exposure",
                                    "arguments": '{"region":"whole_image","strength":0.2}',
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"首轮基础校正完成。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertTrue(bool(result["selected_output"]))
        self.assertEqual(len(result["execution_trace"]), 1)
        self.assertEqual(result["edit_plan"]["operations"][0]["op"], "adjust_exposure")
        self.assertEqual(result["edit_plan"]["executor"], "deterministic")
        self.assertIn("round_1", result["round_plans"])
        self.assertEqual(result["round_plans"]["round_1"]["planner_summary"], "首轮基础校正完成。")

    def test_plan_execute_round_runs_local_tool_with_segmentation(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "提亮脸部",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_exposure",
                                    "arguments": (
                                        '{"region":"脸部皮肤区域","strength":0.2,'
                                        '"mask_provider":"fal_sam3",'
                                        '"mask_prompt":"young woman face skin",'
                                        '"mask_negative_prompt":"hair, background",'
                                        '"mask_semantic_type":true}'
                                    ),
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"局部提亮完成。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
            patch(
                "app.graph.nodes.plan_execute_round.resolve_region_mask",
                return_value=SegmentationResult(
                    provider="fal_sam3",
                    binary_mask_path=self.mask_path,
                    original_image_path=self.image_path,
                    api_chain=("fal_client.upload", "fal-ai/sam-3/image"),
                    region="脸部皮肤区域",
                    target_label="young woman face skin",
                    prompt="young woman face skin",
                    negative_prompt="hair, background",
                    semantic_type=True,
                    requested_provider="fal_sam3",
                ),
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(len(result["segmentation_trace"]), 1)
        self.assertEqual(result["segmentation_trace"][0]["target_label"], "young woman face skin")
        self.assertEqual(result["edit_plan"]["executor"], "hybrid")
        self.assertEqual(result["edit_plan"]["operations"][0]["region"], "脸部皮肤区域")

    def test_plan_execute_round_falls_back_when_no_tool_call_is_returned(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "轻微提亮",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                return_value={"content": "没有工具调用"},
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertTrue(bool(result["selected_output"]))
        self.assertTrue(result["execution_trace"])
        self.assertTrue(result["fallback_trace"])

    def test_plan_execute_round_accepts_long_finish_summary(self) -> None:
        long_summary = "完成本轮调整。" * 120
        state = {
            "mode": "explicit",
            "request_text": "轻微提亮",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_exposure",
                                    "arguments": '{"region":"whole_image","strength":0.2}',
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary": %s}' % __import__("json").dumps(long_summary, ensure_ascii=False),
                                }
                            }
                        ]
                    },
                ],
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(result["round_plans"]["round_1"]["planner_summary"], long_summary)

    def test_plan_execute_round_allows_finish_without_any_operation(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "已经很好了",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                return_value={
                    "tool_calls": [
                        {
                            "function": {
                                "name": "finish_round",
                                "arguments": '{"summary":"当前轮无需继续调整。"}',
                            }
                        }
                    ]
                },
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(result["edit_plan"]["operations"], [])
        self.assertEqual(result["execution_trace"], [])
        self.assertEqual(result["round_plans"]["round_1"]["planner_summary"], "当前轮无需继续调整。")

    def test_plan_execute_round_resolves_wrong_tool_name_before_execution(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "稍微锐化一下",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_sharpen",
                                    "arguments": '{"region":"whole_image","strength":0.2}',
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"锐化完成。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(result["edit_plan"]["operations"][0]["op"], "sharpen")

    def test_plan_execute_round_expands_macro_and_marks_round_as_hybrid(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "去掉背景里的干扰物",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "general"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "cleanup_distracting_objects",
                                    "arguments": '{"region":"whole_image","strength":0.3}',
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"干扰物清理完成。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
            patch(
                "app.graph.nodes.plan_execute_round.resolve_region_mask",
                return_value=SegmentationResult(
                    provider="fal_sam3",
                    binary_mask_path=self.mask_path,
                    original_image_path=self.image_path,
                    api_chain=("fal_client.upload", "fal-ai/sam-3/image"),
                    region="distracting object area",
                    target_label="the distracting object or clutter that should be removed",
                    prompt="the distracting object or clutter that should be removed",
                    negative_prompt="main subject, important foreground",
                    semantic_type=True,
                    requested_provider="fal_sam3",
                ),
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(result["edit_plan"]["operations"][0]["op"], "cleanup_distracting_objects")
        self.assertEqual(result["edit_plan"]["executor"], "hybrid")
        self.assertEqual(len(result["execution_trace"]), 1)
        self.assertEqual(result["execution_trace"][0]["op"], "remove_heal")
        self.assertEqual(len(result["segmentation_trace"]), 1)

    def test_plan_execute_round_skips_local_operation_when_segmentation_returns_empty_result(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "提亮脸部",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_exposure",
                                    "arguments": (
                                        '{"region":"脸部皮肤区域","strength":0.2,'
                                        '"mask_provider":"fal_sam3",'
                                        '"mask_prompt":"young woman face skin",'
                                        '"mask_negative_prompt":"hair, background",'
                                        '"mask_semantic_type":true}'
                                    ),
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"局部步骤已跳过，继续完成本轮。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
            patch(
                "app.graph.nodes.plan_execute_round.resolve_region_mask",
                side_effect=FalImageSegError("fal segmentation response did not include an output image URL."),
            ),
        ):
            result = plan_execute_round_1(state)

        self.assertTrue(bool(result["selected_output"]))
        self.assertEqual(result["edit_plan"]["executor"], "hybrid")
        self.assertEqual(result["edit_plan"]["operations"][0]["region"], "脸部皮肤区域")
        self.assertEqual(len(result["execution_trace"]), 1)
        self.assertEqual(result["execution_trace"][0]["region"], "脸部皮肤区域")
        self.assertTrue(result["execution_trace"][0]["fallback_used"])
        self.assertEqual(len(result["segmentation_trace"]), 1)
        self.assertFalse(result["segmentation_trace"][0]["ok"])
        self.assertTrue(result["segmentation_trace"][0]["fallback_used"])

    def test_plan_execute_round_reuses_mask_for_same_region_and_prompt_within_round(self) -> None:
        state = {
            "mode": "explicit",
            "request_text": "提亮人物上半身并恢复层次",
            "request_intent": {"mode": "explicit", "requested_packages": [], "constraints": []},
            "image_analysis": {"domain": "portrait"},
            "retrieved_prefs": [],
            "input_images": [self.image_path],
        }

        with (
            patch("app.graph.nodes.plan_execute_round.planner_tool_model_available", return_value=True),
            patch(
                "app.graph.nodes.plan_execute_round.call_planner_tool_turn",
                side_effect=[
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_exposure",
                                    "arguments": (
                                        '{"region":"人物上半身","strength":0.3,'
                                        '"mask_provider":"fal_sam3",'
                                        '"mask_prompt":"upper body",'
                                        '"mask_negative_prompt":"hair, background",'
                                        '"mask_semantic_type":true}'
                                    ),
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "adjust_highlights_shadows",
                                    "arguments": (
                                        '{"region":"人物上半身","strength":0.24,'
                                        '"mask_provider":"fal_sam3",'
                                        '"mask_prompt":"upper body",'
                                        '"mask_negative_prompt":"hair, background",'
                                        '"mask_semantic_type":true}'
                                    ),
                                }
                            }
                        ]
                    },
                    {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "finish_round",
                                    "arguments": '{"summary":"主体提亮完成。"}',
                                }
                            }
                        ]
                    },
                ],
            ),
            patch(
                "app.graph.nodes.plan_execute_round.resolve_region_mask",
                return_value=SegmentationResult(
                    provider="fal_sam3",
                    binary_mask_path=self.mask_path,
                    original_image_path=self.image_path,
                    api_chain=("fal_client.upload", "fal-ai/sam-3/image"),
                    region="人物上半身",
                    target_label="upper body",
                    prompt="upper body",
                    negative_prompt="hair, background",
                    semantic_type=True,
                    requested_provider="fal_sam3",
                ),
            ) as mocked_mask,
        ):
            result = plan_execute_round_1(state)

        self.assertEqual(mocked_mask.call_count, 1)
        self.assertEqual(len(result["segmentation_trace"]), 1)
        self.assertEqual(len(result["execution_trace"]), 2)


if __name__ == "__main__":
    unittest.main()
