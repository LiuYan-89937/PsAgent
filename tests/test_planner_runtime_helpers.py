"""Unit tests for shared planner runtime helpers."""

from __future__ import annotations

import unittest

from app.services.planner_runtime_helpers import (
    build_operation_from_tool_call,
    resolve_planner_tool_name,
)


class PlannerRuntimeHelpersTest(unittest.TestCase):
    """Verify tool-name resolution and planner-param decoding."""

    def test_resolve_planner_tool_name_keeps_exact_match(self) -> None:
        resolved, meta = resolve_planner_tool_name("adjust_contrast", {"region": "whole_image"})
        self.assertEqual(resolved, "adjust_contrast")
        self.assertEqual(meta["strategy"], "exact")

    def test_resolve_planner_tool_name_handles_generated_adjust_alias(self) -> None:
        resolved, meta = resolve_planner_tool_name("exposure", {"region": "whole_image"})
        self.assertEqual(resolved, "adjust_exposure")
        self.assertIn(meta["strategy"], {"alias", "similarity"})

    def test_resolve_planner_tool_name_uses_similarity_for_wrong_tool_name(self) -> None:
        resolved, meta = resolve_planner_tool_name(
            "tool_exposure",
            {"region": "whole_image", "strength": 20},
        )
        self.assertEqual(resolved, "adjust_exposure")
        self.assertEqual(meta["strategy"], "similarity")

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
        self.assertAlmostEqual(operation["params"]["strength"], 0.75)
        self.assertAlmostEqual(operation["params"]["max_stops"], 1.75)
        self.assertAlmostEqual(operation["params"]["feather_radius"], 16.0)
        self.assertTrue(operation["params"]["mask_semantic_type"])

    def test_build_operation_from_tool_call_normalizes_mask_prompt(self) -> None:
        operation = build_operation_from_tool_call(
            "adjust_exposure",
            {
                "region": "背景绿植区域",
                "mask_prompt": "background green foliage and trees",
                "strength": 60,
            },
        )

        self.assertEqual(operation["params"]["mask_prompt"], "trees")


if __name__ == "__main__":
    unittest.main()
