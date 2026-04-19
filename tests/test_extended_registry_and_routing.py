"""Tests for extended tool registration, request parsing, and routing behavior."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.nodes.parse_request import parse_request
from app.graph.state import EditOperation, RequestPackageHint
from app.tools import PACKAGE_STATUS_LABELS
from app.tools.packages import build_default_package_registry
from app.tools.packages.macros import operations_require_hybrid


class ExtendedRegistryAndRoutingTest(unittest.TestCase):
    """Verify new tools are visible across registry, state validation, and routing."""

    def test_registry_exports_extended_primitives_and_macros(self) -> None:
        registry = build_default_package_registry()
        package_names = {package.name for package in registry.list()}

        for tool_name in (
            "remove_heal",
            "skin_smooth",
            "point_color",
            "lens_correction",
            "background_blur",
            "portrait_retouch",
            "summer_airy_look",
        ):
            self.assertIn(tool_name, package_names)

    def test_state_models_accept_extended_tool_names(self) -> None:
        request = RequestPackageHint(op="point_color", region="whole_image", params={"strength": 0.2})
        operation = EditOperation(op="portrait_retouch", region="whole_image", params={"strength": 0.3})

        self.assertEqual(request.op, "point_color")
        self.assertEqual(operation.op, "portrait_retouch")

    def test_parse_request_fallback_uses_extended_keywords(self) -> None:
        with patch("app.graph.nodes.parse_request.parse_request_model_available", return_value=False):
            result = parse_request({"request_text": "帮我去痘并做一点自然美白"})
        requested_ops = {item["op"] for item in result["request_intent"]["requested_packages"]}

        self.assertIn("blemish_remove", requested_ops)
        self.assertIn("portrait_natural_whitening", requested_ops)

    def test_macro_with_masked_substeps_requires_hybrid_execution(self) -> None:
        self.assertTrue(
            operations_require_hybrid(
                [
                    {
                        "op": "portrait_retouch",
                        "region": "whole_image",
                        "params": {"strength": 0.4},
                    }
                ]
            )
        )

    def test_runtime_labels_cover_new_tools(self) -> None:
        self.assertEqual(PACKAGE_STATUS_LABELS["remove_heal"], "正在智能修复")
        self.assertEqual(PACKAGE_STATUS_LABELS["portrait_retouch"], "正在进行人像精修")


if __name__ == "__main__":
    unittest.main()
