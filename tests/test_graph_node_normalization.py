"""Normalization-focused tests for graph nodes."""

from __future__ import annotations

import unittest

from app.graph.nodes.human_review import human_review
from app.graph.nodes.load_context import load_context
from app.graph.nodes.parse_request import parse_request
from app.graph.nodes.route_executor import route_executor
from app.graph.nodes.update_memory import update_memory


class GraphNodeNormalizationTest(unittest.TestCase):
    """Verify node outputs are normalized through shared schemas."""

    def test_load_context_normalizes_package_catalog_and_trace(self) -> None:
        result = load_context(
            {
                "request_text": "轻微提亮",
                "execution_trace": [{"ok": True, "stage": "seed"}],
                "memory_write_candidates": [],
            }
        )

        self.assertTrue(result["package_catalog"])
        self.assertEqual(result["execution_trace"][0]["stage"], "seed")
        self.assertTrue(all("name" in item for item in result["package_catalog"]))

    def test_parse_request_returns_valid_request_intent(self) -> None:
        from unittest.mock import patch

        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request({"request_text": "把背景稍微压暗一点并提亮主体"})

        self.assertEqual(result["mode"], "explicit")
        self.assertEqual(result["request_intent"]["mode"], "explicit")
        self.assertTrue(result["request_intent"]["requested_packages"])

    def test_route_executor_normalizes_plan(self) -> None:
        result = route_executor(
            {
                "edit_plan": {
                    "mode": "explicit",
                    "domain": "general",
                    "preserve": [],
                    "operations": [
                        {
                            "op": "adjust_exposure",
                            "region": "main_subject",
                            "params": {"strength": 0.2},
                        }
                    ],
                }
            }
        )

        self.assertEqual(result["edit_plan"]["executor"], "deterministic")

    def test_human_review_normalizes_payload(self) -> None:
        result = human_review({"approval_payload": {"reason": "mask edge", "summary": "需要人工确认"}})
        self.assertEqual(result["approval_payload"]["reason"], "mask edge")

    def test_update_memory_normalizes_candidates(self) -> None:
        result = update_memory(
            {
                "memory_write_candidates": [
                    {
                        "domain": "general",
                        "key": "tone_preference",
                        "value": "natural",
                    }
                ]
            }
        )
        self.assertEqual(result["memory_write_candidates"][0]["key"], "tone_preference")


if __name__ == "__main__":
    unittest.main()
