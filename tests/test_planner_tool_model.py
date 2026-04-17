"""Unit tests for planner tool-name resolution helpers."""

from __future__ import annotations

import unittest

from app.services.planner_tool_model import resolve_planner_tool_name


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


if __name__ == "__main__":
    unittest.main()
