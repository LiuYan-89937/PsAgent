"""Smoke tests for graph construction."""

from __future__ import annotations

import unittest

from app.graph.builder import build_graph


class GraphBuilderTest(unittest.TestCase):
    """Verify that the graph can be constructed with explicit IO schemas."""

    def test_build_graph_compiles(self) -> None:
        graph = build_graph()
        self.assertIsNotNone(graph)


if __name__ == "__main__":
    unittest.main()
