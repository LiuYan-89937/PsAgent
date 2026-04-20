"""Unit tests for the native @tool registry."""

from __future__ import annotations

import unittest

from app.tools.tool_registry import build_default_tool_registry


class NativeToolRegistryTest(unittest.TestCase):
    """Verify the native registry only exposes the new baseline tools."""

    def test_registry_registers_only_three_native_tools(self) -> None:
        registry = build_default_tool_registry()
        tool_names = [registered.spec.name for registered in registry.list()]

        self.assertEqual(
            tool_names,
            ["adjust_exposure", "adjust_contrast", "adjust_vibrance_saturation"],
        )

    def test_exported_catalog_contains_tool_spec_and_planner_schema(self) -> None:
        registry = build_default_tool_registry()
        catalog = registry.export_catalog()

        exposure = next(item for item in catalog if item["name"] == "adjust_exposure")
        self.assertEqual(exposure["label"], "曝光")
        self.assertEqual(exposure["family"], "tone")
        self.assertTrue(exposure["supports_mask"])
        self.assertTrue(exposure["supports_whole_image"])
        self.assertEqual(exposure["primary_param"], "strength")
        self.assertIn("strength", exposure["planner_schema"]["properties"])
        self.assertIn("mask_prompt", exposure["planner_schema"]["properties"])
        self.assertNotIn("image_path", exposure["planner_schema"]["properties"])
        self.assertNotIn("mask_path", exposure["planner_schema"]["properties"])


if __name__ == "__main__":
    unittest.main()
