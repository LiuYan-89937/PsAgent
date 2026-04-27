"""Unit tests for the native @tool registry."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from app.tools import NATIVE_TOOLS, TOOL_SPECS, export_tool_catalog


class NativeToolRegistryTest(unittest.TestCase):
    """Verify the native registry matches the documented native tool catalog."""

    def _documented_tool_names(self) -> list[str]:
        doc_path = Path("/Users/liuyan/Desktop/PsAgent/docs/适合接入当前且易实现的纯工具列表.md")
        content = doc_path.read_text(encoding="utf-8")
        names = sorted(set(re.findall(r"(?:adjust|apply)_[a-z_]+", content)))
        return names

    def test_registry_matches_documented_native_tools(self) -> None:
        tool_names = [spec.name for spec in TOOL_SPECS]

        self.assertEqual(set(tool_names), set(self._documented_tool_names()))
        self.assertEqual(len(tool_names), len(self._documented_tool_names()))

    def test_exported_catalog_contains_tool_spec_and_planner_schema(self) -> None:
        self.assertEqual([tool.name for tool in NATIVE_TOOLS], [spec.name for spec in TOOL_SPECS])
        catalog = export_tool_catalog()

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

        skin_smooth = next(item for item in catalog if item["name"] == "adjust_skin_smooth")
        self.assertEqual(skin_smooth["family"], "portrait")
        self.assertIn("subject_cleanup", skin_smooth["focus_affinity"])

        color_lookup = next(item for item in catalog if item["name"] == "apply_color_lookup")
        self.assertEqual(color_lookup["family"], "color")
        self.assertEqual(color_lookup["mask_policy"], "optional")

        face_cleanup = next(item for item in catalog if item["name"] == "adjust_face_color_cleanup")
        self.assertTrue(face_cleanup["requires_mask"])
        self.assertEqual(face_cleanup["mask_policy"], "required")
        self.assertEqual(face_cleanup["recommended_mask_prompt"], "face")
        self.assertTrue(face_cleanup["description"].startswith("Use this tool when"))

    def test_every_registered_tool_is_exported_once(self) -> None:
        tool_names = {spec.name for spec in TOOL_SPECS}
        exported_tools = {item["name"] for item in export_tool_catalog()}

        self.assertFalse(tool_names - exported_tools)
        self.assertFalse(exported_tools - tool_names)


if __name__ == "__main__":
    unittest.main()
