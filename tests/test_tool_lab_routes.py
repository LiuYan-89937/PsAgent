"""Route tests for the tool-lab experimentation API."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from app.api.deps import get_asset_store
from app.main import create_app
from app.services.asset_store import AssetStore


class ToolLabRoutesTest(unittest.TestCase):
    """Verify deterministic tool-lab routes for mask generation and sequential execution."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.asset_store = AssetStore(root_dir=Path(self.temp_dir.name) / "assets")
        self.app = create_app()
        self.app.dependency_overrides[get_asset_store] = lambda: self.asset_store
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_png(self, color: tuple[int, int, int], *, size: tuple[int, int] = (96, 96), mode: str = "RGB") -> bytes:
        buf = io.BytesIO()
        Image.new(mode, size, color if mode == "RGB" else color[0]).save(buf, format="PNG")
        return buf.getvalue()

    def test_generate_tool_lab_mask_returns_saved_mask_asset(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("portrait.png", self._build_png((30, 40, 50)), "image/png"))],
        )
        self.assertEqual(upload.status_code, 200)
        input_asset_id = upload.json()["items"][0]["asset_id"]

        mask_path = Path(self.temp_dir.name) / "mask.png"
        preview_path = Path(self.temp_dir.name) / "preview.png"
        Image.new("L", (96, 96), 255).save(mask_path)
        Image.new("RGBA", (96, 96), (255, 0, 0, 180)).save(preview_path)

        fake_result = type(
            "SegResult",
            (),
            {
                "provider": "fal_sam3",
                "binary_mask_path": str(mask_path),
                "segmentation_rgba_path": str(preview_path),
                "requested_provider": "fal_sam3",
                "effective_prompt": "person",
                "fallback_used": False,
                "attempt_strategy": None,
                "attempt_index": None,
                "target_label": "person",
                "revert_mask": False,
            },
        )()

        with patch("app.api.routes_tool_lab.resolve_region_mask", return_value=fake_result) as mocked_resolve:
            response = self.client.post(
                "/tool-lab/masks",
                json={
                    "input_asset_id": input_asset_id,
                    "prompt": "person",
                    "provider": "fal_sam3",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["provider"], "fal_sam3")
        self.assertEqual(payload["prompt"], "person")
        self.assertIn("mask_asset", payload)
        self.assertIn("preview_asset", payload)
        mocked_resolve.assert_called_once()

    def test_run_tool_lab_executes_one_masked_tool_step(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("portrait.png", self._build_png((70, 72, 75)), "image/png"))],
        )
        self.assertEqual(upload.status_code, 200)
        input_asset_id = upload.json()["items"][0]["asset_id"]

        mask_upload = self.client.post(
            "/assets/upload",
            files=[("files", ("mask.png", self._build_png((255, 255, 255), mode="L"), "image/png"))],
        )
        self.assertEqual(mask_upload.status_code, 200)
        mask_asset_id = mask_upload.json()["items"][0]["asset_id"]

        response = self.client.post(
            "/tool-lab/run",
            json={
                "input_asset_id": input_asset_id,
                "steps": [
                    {
                        "tool_name": "adjust_exposure",
                        "params": {"strength": 0.5, "feather_radius": 0.0},
                        "mask_asset_id": mask_asset_id,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("final_output_asset", payload)
        self.assertEqual(len(payload["steps"]), 1)
        self.assertTrue(payload["steps"][0]["ok"])
        self.assertEqual(payload["steps"][0]["tool_name"], "adjust_exposure")
        self.assertIsNotNone(payload["steps"][0]["mask_asset"])

    def test_run_tool_lab_rejects_portrait_local_tool_without_mask(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("portrait.png", self._build_png((70, 72, 75)), "image/png"))],
        )
        self.assertEqual(upload.status_code, 200)
        input_asset_id = upload.json()["items"][0]["asset_id"]

        response = self.client.post(
            "/tool-lab/run",
            json={
                "input_asset_id": input_asset_id,
                "steps": [
                    {
                        "tool_name": "adjust_teeth_whiten",
                        "params": {"yellow_reduce": 0.3},
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("requires a mask", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
