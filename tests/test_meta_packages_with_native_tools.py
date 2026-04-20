"""Unit tests for /meta/packages backed by the native tool registry."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import create_app


class MetaPackagesWithNativeToolsTest(unittest.TestCase):
    """Verify metadata route output after the native tool migration."""

    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_meta_packages_returns_only_native_tools(self) -> None:
        response = self.client.get("/meta/packages")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        tool_names = {item["name"] for item in payload["items"]}
        self.assertEqual(
            tool_names,
            {"adjust_exposure", "adjust_contrast", "adjust_vibrance_saturation"},
        )

        exposure = next(item for item in payload["items"] if item["name"] == "adjust_exposure")
        self.assertEqual(exposure["label"], "曝光")
        self.assertEqual(exposure["family"], "tone")
        self.assertIn("planner_schema", exposure)
        self.assertIn("mask_prompt", exposure["planner_schema"]["properties"])
        self.assertNotIn("image_path", exposure["planner_schema"]["properties"])
        self.assertNotIn("mask_path", exposure["planner_schema"]["properties"])


if __name__ == "__main__":
    unittest.main()
