"""Unit tests for bootstrap request normalization node."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.bootstrap_request import AUTO_BEAUTIFY_FALLBACK_INSTRUCTION, bootstrap_request


class BootstrapRequestNodeTest(unittest.TestCase):
    """Verify graph-level request bootstrap behavior."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "input.png")
        Image.new("RGB", (32, 32), (90, 100, 110)).save(self.image_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_bootstrap_keeps_explicit_request_text(self) -> None:
        result = bootstrap_request({"mode": "explicit", "request_text": "提亮一点", "input_images": [self.image_path]})

        self.assertEqual(result["request_text"], "提亮一点")

    def test_bootstrap_generates_auto_instruction(self) -> None:
        with (
            patch("app.graph.nodes.bootstrap_request.auto_instruction_model_available", return_value=True),
            patch(
                "app.graph.nodes.bootstrap_request.generate_auto_beautify_instruction_with_qwen",
                return_value="把画面修得更通透明亮",
            ),
        ):
            result = bootstrap_request({"mode": "auto", "request_text": "", "input_images": [self.image_path]})

        self.assertEqual(result["request_text"], "把画面修得更通透明亮")

    def test_bootstrap_falls_back_to_generic_instruction(self) -> None:
        with (
            patch("app.graph.nodes.bootstrap_request.auto_instruction_model_available", return_value=True),
            patch(
                "app.graph.nodes.bootstrap_request.generate_auto_beautify_instruction_with_qwen",
                side_effect=RuntimeError("empty content"),
            ),
        ):
            result = bootstrap_request({"mode": "auto", "request_text": "", "input_images": [self.image_path]})

        self.assertEqual(result["request_text"], AUTO_BEAUTIFY_FALLBACK_INSTRUCTION)
        self.assertTrue(result["fallback_trace"])


if __name__ == "__main__":
    unittest.main()
