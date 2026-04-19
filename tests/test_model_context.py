"""Unit tests for compact model-facing context helpers."""

from __future__ import annotations

import json
import unittest

from app.services.model_context import (
    compact_image_analysis_for_model,
    compact_package_catalog_for_model,
    compact_plan_for_model,
    compact_preferences_for_model,
    compact_request_intent_for_model,
    shared_mask_params_for_model,
)
from app.tools.packages import build_default_package_registry


class ModelContextTest(unittest.TestCase):
    """Verify compact model payload helpers."""

    def test_compact_package_catalog_reduces_payload_size(self) -> None:
        full_catalog = build_default_package_registry().export_llm_catalog()

        full_json = json.dumps(full_catalog, ensure_ascii=False)
        compact_json = json.dumps(
            compact_package_catalog_for_model(full_catalog, include_params=True),
            ensure_ascii=False,
        )

        self.assertLess(len(compact_json), len(full_json))

    def test_compact_package_catalog_preserves_core_param_info(self) -> None:
        full_catalog = build_default_package_registry().export_llm_catalog()
        compact_catalog = compact_package_catalog_for_model(full_catalog, include_params=True)

        exposure = next(item for item in compact_catalog if item["name"] == "adjust_exposure")
        strength = next(item for item in exposure["params"] if item["name"] == "strength")

        self.assertEqual(exposure["execution_modes"], ["whole_image", "masked_region"])
        self.assertEqual(strength["type"], "integer")
        self.assertIn("description", strength)
        self.assertEqual(strength["minimum"], 0)
        self.assertEqual(strength["maximum"], 100)
        self.assertIn("仅填 0-100 整数", strength["description"])

    def test_parse_request_compact_catalog_omits_param_details(self) -> None:
        full_catalog = build_default_package_registry().export_llm_catalog()
        compact_catalog = compact_package_catalog_for_model(full_catalog, include_params=False)

        exposure = next(item for item in compact_catalog if item["name"] == "adjust_exposure")
        self.assertNotIn("params", exposure)

    def test_planner_compact_catalog_omits_repeated_mask_params(self) -> None:
        full_catalog = build_default_package_registry().export_llm_catalog()
        compact_catalog = compact_package_catalog_for_model(full_catalog, include_params=True)
        exposure = next(item for item in compact_catalog if item["name"] == "adjust_exposure")
        param_names = {item["name"] for item in exposure["params"]}

        self.assertIn("strength", param_names)
        self.assertNotIn("mask_prompt", param_names)
        self.assertNotIn("mask_provider", param_names)

    def test_shared_mask_params_for_model_extracts_common_mask_schema(self) -> None:
        full_catalog = build_default_package_registry().export_llm_catalog()
        shared_params = shared_mask_params_for_model(full_catalog)
        param_names = {item["name"] for item in shared_params}

        self.assertIn("mask_provider", param_names)
        self.assertIn("mask_prompt", param_names)
        self.assertIn("mask_negative_prompt", param_names)
        expand_param = next(item for item in shared_params if item["name"] == "mask_expand")
        self.assertEqual(expand_param["type"], "integer")
        self.assertEqual(expand_param["minimum"], 0)
        self.assertEqual(expand_param["maximum"], 100)

    def test_compact_request_and_analysis_helpers_keep_decision_fields(self) -> None:
        compact_intent = compact_request_intent_for_model(
            {
                "mode": "explicit",
                "requested_packages": [
                    {"op": "adjust_exposure", "region": "逆光脸部区域", "strength": 0.2, "params": {}}
                ],
                "constraints": ["repair_backlighting"],
            }
        )
        compact_analysis = compact_image_analysis_for_model(
            {
                "domain": "portrait",
                "issues": ["underexposed"],
                "subjects": ["young woman"],
                "segmentation_hints": ["逆光脸部区域"],
                "summary": "主体偏暗",
                "metrics": {"brightness_mean": 95, "shadow_ratio": 0.2},
            }
        )
        compact_prefs = compact_preferences_for_model(
            [{"key": "style", "value": "natural", "confidence": 0.8, "source": "accepted_result"}]
        )
        compact_plan = compact_plan_for_model(
            {"mode": "explicit", "operations": [{"op": "adjust_exposure", "region": "whole_image", "params": {"strength": 0.2}}]}
        )

        self.assertEqual(compact_intent["mode"], "explicit")
        self.assertEqual(compact_analysis["domain"], "portrait")
        self.assertEqual(compact_prefs[0]["key"], "style")
        self.assertEqual(compact_plan["operations"][0]["op"], "adjust_exposure")


if __name__ == "__main__":
    unittest.main()
