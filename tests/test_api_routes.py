"""Route tests for the frontend-facing API layer."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class FakeGraph:
    """Minimal fake graph used for API route tests."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.last_payload: dict | None = None

    def invoke(self, payload: dict, config=None) -> dict:
        self.last_payload = payload
        output_path = self.output_dir / "fake_output.png"
        Image.new("RGB", (16, 16), (180, 120, 80)).save(output_path)
        request_text = payload.get("request_text") or ("auto generated instruction" if payload.get("mode") == "auto" else None)
        return {
            "candidate_outputs": [str(output_path)],
            "selected_output": str(output_path),
            "round_outputs": {"round_1": str(output_path)},
            "request_text": request_text,
            "edit_plan": {
                "mode": payload.get("mode", "explicit"),
                "domain": "general",
                "executor": "deterministic",
                "preserve": [],
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "whole_image",
                        "params": {"strength": 0.2},
                        "constraints": [],
                        "priority": 0,
                    }
                ],
                "should_write_memory": False,
                "memory_candidates": [],
                "needs_confirmation": False,
            },
            "round_plans": {"round_1": {"operations": [{"op": "adjust_exposure"}]}},
            "eval_report": {
                "selected_output": str(output_path),
                "num_operations": 1,
                "success_count": 1,
                "failure_count": 0,
                "fallback_count": 0,
                "has_output": True,
                "overall_ok": True,
                "preserve_ok": True,
                "style_ok": True,
                "artifact_ok": True,
                "issues": [],
                "warnings": [],
                "summary": "ok",
                "should_continue_editing": False,
                "should_request_review": False,
            },
            "round_eval_reports": {
                "round_1": {
                    "selected_output": str(output_path),
                    "num_operations": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "fallback_count": 0,
                    "has_output": True,
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "ok",
                    "should_continue_editing": False,
                    "should_request_review": False,
                }
            },
            "execution_trace": [
                {
                    "index": 0,
                    "stage": "round_1",
                    "op": "adjust_exposure",
                    "region": "whole_image",
                    "ok": True,
                    "fallback_used": False,
                    "error": None,
                    "output_image": str(output_path),
                    "applied_params": {"strength": 0.2},
                    "mask_path": None,
                }
            ],
            "round_execution_traces": {
                "round_1": [
                    {
                        "index": 0,
                        "stage": "round_1",
                        "op": "adjust_exposure",
                        "region": "whole_image",
                        "ok": True,
                        "fallback_used": False,
                        "error": None,
                        "output_image": str(output_path),
                        "applied_params": {"strength": 0.2},
                        "mask_path": None,
                    }
                ]
            },
            "approval_required": False,
        }

    def stream(self, payload, config=None, stream_mode=None, version=None):
        if isinstance(payload, dict):
            self.last_payload = payload
        output_path = self.output_dir / "fake_output.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists():
            Image.new("RGB", (16, 16), (180, 120, 80)).save(output_path)

        if hasattr(payload, "__class__") and payload.__class__.__name__ == "Command":
            yield ("tasks", {"name": "human_review", "input": {}, "triggers": ("resume",)})
            yield ("updates", {"human_review": {"approval_required": False}})
            yield ("tasks", {"name": "human_review", "result": {"approval_required": False}, "error": None, "interrupts": []})
            return

        yield ("tasks", {"name": "analyze_image", "input": {"x": 1}, "triggers": ("start",)})
        yield ("updates", {"analyze_image": {"domain": "general"}})
        yield ("tasks", {"name": "analyze_image", "result": {"domain": "general"}, "error": None, "interrupts": []})
        yield ("custom", {"event": "round_started", "stage": "execute_hybrid", "round": "round_1", "message": "开始执行 round_1"})
        yield ("custom", {"event": "package_started", "stage": "execute_hybrid", "op": "adjust_exposure", "region": "main_subject", "message": "正在执行 adjust_exposure"})
        yield ("custom", {"event": "package_finished", "stage": "execute_hybrid", "op": "adjust_exposure", "region": "main_subject", "ok": True, "message": "adjust_exposure 执行完成"})
        yield ("custom", {"event": "round_completed", "stage": "execute_hybrid", "round": "round_1", "message": "round_1 执行完成"})

    def get_state(self, config=None):
        output_path = str(self.output_dir / "fake_output.png")
        path_obj = Path(output_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if not path_obj.exists():
            Image.new("RGB", (16, 16), (180, 120, 80)).save(path_obj)

        class Snapshot:
            pass

        snapshot = Snapshot()
        request_text = None
        if isinstance(self.last_payload, dict):
            request_text = self.last_payload.get("request_text") or (
                "auto generated instruction" if self.last_payload.get("mode") == "auto" else None
            )
        snapshot.values = {
            "candidate_outputs": [output_path],
            "selected_output": output_path,
            "round_outputs": {"round_1": output_path},
            "request_text": request_text,
            "edit_plan": {
                "mode": "explicit",
                "domain": "general",
                "executor": "hybrid",
                "preserve": [],
                "operations": [
                    {
                        "op": "adjust_exposure",
                        "region": "main_subject",
                        "params": {"strength": 0.2},
                        "constraints": [],
                        "priority": 0,
                    }
                ],
                "should_write_memory": False,
                "memory_candidates": [],
                "needs_confirmation": False,
            },
            "round_plans": {"round_1": {"operations": [{"op": "adjust_exposure"}]}},
            "eval_report": {
                "selected_output": output_path,
                "num_operations": 1,
                "success_count": 1,
                "failure_count": 0,
                "fallback_count": 0,
                "has_output": True,
                "overall_ok": True,
                "preserve_ok": True,
                "style_ok": True,
                "artifact_ok": True,
                "issues": [],
                "warnings": [],
                "summary": "ok",
                "should_continue_editing": False,
                "should_request_review": False,
            },
            "round_eval_reports": {
                "round_1": {
                    "selected_output": output_path,
                    "num_operations": 1,
                    "success_count": 1,
                    "failure_count": 0,
                    "fallback_count": 0,
                    "has_output": True,
                    "overall_ok": True,
                    "preserve_ok": True,
                    "style_ok": True,
                    "artifact_ok": True,
                    "issues": [],
                    "warnings": [],
                    "summary": "ok",
                    "should_continue_editing": False,
                    "should_request_review": False,
                }
            },
            "execution_trace": [
                {
                    "index": 0,
                    "stage": "round_1",
                    "op": "adjust_exposure",
                    "region": "main_subject",
                    "ok": True,
                    "fallback_used": False,
                    "error": None,
                    "output_image": output_path,
                    "applied_params": {"strength": 0.2},
                    "mask_path": None,
                }
            ],
            "round_execution_traces": {
                "round_1": [
                    {
                        "index": 0,
                        "stage": "round_1",
                        "op": "adjust_exposure",
                        "region": "main_subject",
                        "ok": True,
                        "fallback_used": False,
                        "error": None,
                        "output_image": output_path,
                        "applied_params": {"strength": 0.2},
                        "mask_path": None,
                    }
                ]
            },
            "approval_required": False,
            "approval_payload": None,
        }
        return snapshot


class ApiRoutesTest(unittest.TestCase):
    """Verify API contracts needed by the frontend."""

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.api.deps import get_asset_store, get_graph_app, get_job_store, get_package_registry
        from app.main import create_app
        from app.services.asset_store import AssetStore
        from app.services.job_store import JobStore
        from app.tools.packages import build_default_package_registry

        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.asset_store = AssetStore(root_dir=self.root / "assets")
        self.job_store = JobStore()
        self.fake_graph = FakeGraph(self.root / "graph_outputs")
        self.fake_graph.output_dir.mkdir(parents=True, exist_ok=True)
        self.registry = build_default_package_registry()

        self.app = create_app()
        self.app.dependency_overrides[get_asset_store] = lambda: self.asset_store
        self.app.dependency_overrides[get_job_store] = lambda: self.job_store
        self.app.dependency_overrides[get_graph_app] = lambda: self.fake_graph
        self.app.dependency_overrides[get_package_registry] = lambda: self.registry
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _build_png(self, color: tuple[int, int, int]) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (96, 96), color).save(buf, format="PNG")
        return buf.getvalue()

    def test_health_and_package_catalog(self) -> None:
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ok"])

        catalog = self.client.get("/meta/packages")
        self.assertEqual(catalog.status_code, 200)
        self.assertTrue(catalog.json()["items"])
        package_names = {item["name"] for item in catalog.json()["items"]}
        self.assertIn("adjust_color_mixer", package_names)
        self.assertIn("adjust_dehaze", package_names)

    def test_upload_edit_and_job_polling(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("test.png", self._build_png((20, 40, 60)), "image/png"))],
        )
        self.assertEqual(upload.status_code, 200)
        items = upload.json()["items"]
        self.assertEqual(len(items), 1)
        asset_id = items[0]["asset_id"]

        content = self.client.get(f"/assets/{asset_id}/content")
        self.assertEqual(content.status_code, 200)

        edit = self.client.post(
            "/edit",
            json={
                "user_id": "u1",
                "instruction": "提亮一点",
                "input_asset_ids": [asset_id],
            },
        )
        self.assertEqual(edit.status_code, 200)
        payload = edit.json()
        self.assertEqual(payload["job"]["status"], "completed")
        self.assertIsNotNone(payload["selected_output"])
        self.assertEqual(len(payload["candidate_outputs"]), 1)
        self.assertIn("round_1", payload["round_outputs"])
        self.assertIn("round_1", payload["round_plans"])
        self.assertIn("round_1", payload["round_eval_reports"])

        job_id = payload["job"]["job_id"]
        job = self.client.get(f"/jobs/{job_id}")
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()["job"]["job_id"], job_id)
        self.assertIn("round_1", job.json()["round_outputs"])
        self.assertIn("stage_timings", job.json())

        feedback = self.client.post(
            "/feedback",
            json={
                "job_id": job_id,
                "accepted": True,
                "rating": 5,
                "feedback_text": "不错",
            },
        )
        self.assertEqual(feedback.status_code, 200)
        self.assertEqual(feedback.json()["feedback_count"], 1)

    def test_edit_without_instruction_uses_auto_beautify_instruction(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("test.png", self._build_png((20, 40, 60)), "image/png"))],
        )
        asset_id = upload.json()["items"][0]["asset_id"]

        response = self.client.post(
            "/edit",
            json={
                "user_id": "u1",
                "auto_mode": True,
                "input_asset_ids": [asset_id],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(self.fake_graph.last_payload)
        self.assertEqual(self.fake_graph.last_payload["request_text"], "")
        self.assertEqual(response.json()["job"]["request_text"], "auto generated instruction")

    def test_edit_and_stream_share_same_graph_input_flags(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("test.png", self._build_png((20, 40, 60)), "image/png"))],
        )
        asset_id = upload.json()["items"][0]["asset_id"]

        edit = self.client.post(
            "/edit",
            json={
                "user_id": "u1",
                "instruction": "提亮一点",
                "planner_thinking_mode": True,
                "input_asset_ids": [asset_id],
            },
        )
        self.assertEqual(edit.status_code, 200)
        self.assertTrue(self.fake_graph.last_payload["planner_thinking_mode"])

        with self.client.stream(
            "POST",
            "/edit/stream",
            json={
                "user_id": "u1",
                "instruction": "提亮一点",
                "planner_thinking_mode": True,
                "input_asset_ids": [asset_id],
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            _ = "".join(response.iter_text())

        self.assertTrue(self.fake_graph.last_payload["planner_thinking_mode"])

    def test_upload_rejects_invalid_image(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("broken.png", b"not-an-image", "image/png"))],
        )
        self.assertEqual(upload.status_code, 400)
        self.assertIn("有效图片", upload.json()["detail"])

    def test_stream_edit_emits_progress_and_completion(self) -> None:
        upload = self.client.post(
            "/assets/upload",
            files=[("files", ("test.png", self._build_png((20, 40, 60)), "image/png"))],
        )
        asset_id = upload.json()["items"][0]["asset_id"]

        with self.client.stream(
            "POST",
            "/edit/stream",
            json={
                "user_id": "u1",
                "instruction": "提亮一点",
                "input_asset_ids": [asset_id],
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(response.iter_text())

        self.assertIn("event: job_created", body)
        self.assertIn("event: node_started", body)
        self.assertIn("event: round_started", body)
        self.assertIn("event: package_started", body)
        self.assertIn("event: round_completed", body)
        self.assertIn("event: job_completed", body)

    def test_resume_review_placeholder(self) -> None:
        job = self.job_store.create_job(
            user_id="u1",
            thread_id="t1",
            request_text="test",
            input_asset_ids=[],
        )
        self.job_store.set_review_state(
            job.job_id,
            status="review_required",
            approval_required=True,
            current_stage="human_review",
            current_message="等待人工确认",
        )
        response = self.client.post(
            "/resume-review",
            json={"job_id": job.job_id, "approved": True, "note": "ok"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["implemented"])


if __name__ == "__main__":
    unittest.main()
