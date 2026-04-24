"""Unit tests for the in-memory JobStore write model."""

from __future__ import annotations

import unittest

from app.services.job_store import JobStore


class JobStoreTest(unittest.TestCase):
    """Verify explicit job write helpers and append semantics."""

    def test_append_event_updates_round_and_message(self) -> None:
        store = JobStore()
        job = store.create_job(user_id="u1", thread_id="t1", request_text="提亮一点")

        updated = store.append_event(
            job.job_id,
            {"event": "round_started", "round": "round_1", "focus": "finish", "message": "开始搜索"},
            current_round="round_1",
            current_focus="finish",
            current_message="正在准备修图请求",
        )

        self.assertEqual(len(updated.events), 1)
        self.assertEqual(updated.current_round, "round_1")
        self.assertEqual(updated.current_focus, "finish")
        self.assertEqual(updated.current_message, "正在准备修图请求")

    def test_set_execution_result_updates_core_result_fields(self) -> None:
        store = JobStore()
        job = store.create_job(user_id="u1", thread_id="t1", request_text=None)

        updated = store.set_execution_result(
            job.job_id,
            status="completed",
            request_text="自动生成的提示词",
            output_asset_ids=["asset_1"],
            execution_trace=[{"op": "adjust_exposure", "ok": True, "round_id": "round_1", "focus": "finish"}],
            final_execution_trace=[{"op": "adjust_exposure", "ok": True, "round_id": "round_1", "focus": "finish"}],
            objective_card={
                "summary": "自动美化",
                "mode": "auto",
                "domain": "general",
                "preserve": [],
                "goals": [],
                "gaps": [],
                "constraints": [],
            },
            rounds=[
                {
                    "id": "round_1",
                    "index": 0,
                    "focus": "finish",
                    "output_asset_id": "asset_1",
                    "candidates": [],
                    "selected_candidate_id": None,
                    "recovery_candidates": [],
                    "completed": True,
                }
            ],
            selected_candidate_id="candidate_1",
            current_round="completed",
            current_focus="finish",
            current_message="任务完成",
        )

        self.assertEqual(updated.status, "completed")
        self.assertEqual(updated.request_text, "自动生成的提示词")
        self.assertEqual(updated.output_asset_ids, ["asset_1"])
        self.assertEqual(updated.execution_trace[0].op, "adjust_exposure")
        self.assertEqual(updated.final_execution_trace[0].round_id, "round_1")
        self.assertEqual(updated.rounds[0].id, "round_1")
        self.assertEqual(updated.rounds[0].output_asset_id, "asset_1")
        self.assertEqual(updated.current_round, "completed")


if __name__ == "__main__":
    unittest.main()
