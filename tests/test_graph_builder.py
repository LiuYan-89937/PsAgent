"""Smoke tests for graph construction."""

from __future__ import annotations

import unittest

from app.graph.builder import build_graph, route_after_final_review, route_after_human_review


class GraphBuilderTest(unittest.TestCase):
    """Verify that the graph can be constructed with explicit IO schemas."""

    def test_build_graph_compiles(self) -> None:
        graph = build_graph()
        self.assertIsNotNone(graph)

    def test_final_review_routes_to_continuation_before_memory(self) -> None:
        self.assertEqual(route_after_final_review({"needs_search_continuation": True}), "continue")
        self.assertEqual(route_after_final_review({"approval_required": True, "needs_search_continuation": True}), "continue")
        self.assertEqual(route_after_final_review({"approval_required": True}), "review")
        self.assertEqual(route_after_final_review({}), "ok")

    def test_human_review_routes_to_continuation_when_requested(self) -> None:
        self.assertEqual(route_after_human_review({"needs_search_continuation": True}), "continue")
        self.assertEqual(route_after_human_review({}), "ok")


if __name__ == "__main__":
    unittest.main()
