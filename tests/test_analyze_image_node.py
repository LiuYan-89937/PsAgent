"""Unit tests for the analyze_image graph node."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.nodes.analyze_image import analyze_image


class AnalyzeImageNodeTest(unittest.TestCase):
    """Verify rule-based and model-augmented image analysis."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tmpdir.name) / "analyze.png")
        Image.new("RGB", (64, 96), (90, 100, 110)).save(self.image_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_analyze_image_returns_basic_metrics_without_model(self) -> None:
        with patch("app.graph.nodes.analyze_image.analyze_image_model_available", return_value=False):
            result = analyze_image({"input_images": [self.image_path], "request_text": "自然一点"})

        analysis = result["image_analysis"]
        self.assertEqual(analysis["source_image"], self.image_path)
        self.assertIn("metrics", analysis)
        self.assertEqual(analysis["orientation"], "portrait")

    def test_analyze_image_merges_model_output_when_available(self) -> None:
        with (
            patch("app.graph.nodes.analyze_image.analyze_image_model_available", return_value=True),
            patch(
                "app.graph.nodes.analyze_image.generate_image_analysis_with_qwen",
                return_value={
                    "domain": "portrait",
                    "scene_tags": ["indoor"],
                    "issues": ["mixed_color_temperature"],
                    "subjects": ["person"],
                    "segmentation_hints": ["person"],
                    "summary": "室内人像，轻微偏色。",
                },
            ),
        ):
            result = analyze_image({"input_images": [self.image_path], "request_text": "自然一点"})

        analysis = result["image_analysis"]
        self.assertEqual(analysis["domain"], "portrait")
        self.assertEqual(analysis["subjects"], ["person"])
        self.assertEqual(analysis["segmentation_hints"], ["person"])
        self.assertIn("model_analysis", analysis)


if __name__ == "__main__":
    unittest.main()
