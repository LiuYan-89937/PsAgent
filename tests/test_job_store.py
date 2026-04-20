"""Unit tests for the in-memory JobStore write model."""

from __future__ import annotations

import unittest

from app.services.job_store import JobStore


class JobStoreTest(unittest.TestCase):
    """Verify explicit job write helpers and append semantics."""

    def test_append_event_updates_stage_and_message(self) -> None:
        store = JobStore()
        job = store.create_job(user_id="u1", thread_id="t1", request_text="提亮一点")

        updated = store.append_event(
            job.job_id,
            {"event": "node_started", "stage": "bootstrap_request", "message": "正在准备修图请求"},
            current_stage="bootstrap_request",
            current_message="正在准备修图请求",
        )

        self.assertEqual(len(updated.events), 1)
        self.assertEqual(updated.current_stage, "bootstrap_request")
        self.assertEqual(updated.current_message, "正在准备修图请求")

    def test_set_execution_result_updates_core_result_fields(self) -> None:
        store = JobStore()
        job = store.create_job(user_id="u1", thread_id="t1", request_text=None)

        updated = store.set_execution_result(
            job.job_id,
            status="completed",
            request_text="自动生成的提示词",
            output_asset_ids=["asset_1"],
            execution_trace=[{"op": "adjust_exposure", "ok": True}],
            phases={
                "global_base": {
                    "key": "global_base",
                    "label": "全局基线",
                    "execution_trace": [{"op": "adjust_exposure", "ok": True}],
                    "output": {"asset_id": "asset_1"},
                }
            },
            current_stage="completed",
            current_message="任务完成",
        )

        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.request_text, "自动生成的提示词")
        self.assertEqual(updated.output_asset_ids, ["asset_1"])
        self.assertEqual(updated.execution_trace[0].op, "adjust_exposure")
        self.assertIn("global_base", updated.phases)
        self.assertEqual(updated.current_stage, "completed")


if __name__ == "__main__":
    unittest.main()
