"""Tests for the round-first search agent orchestration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.graph.state import CandidateProgram, MaskCatalog, ObjectiveCard, ObjectiveGap, PlannerExecutionStep, RequestIntent
from app.services.search_agent.orchestrator import run_search_first_agent
from app.services.tool_runtime.chain_executor import ChainExecutionResult


def _objective(focus: str = "global_tone") -> ObjectiveCard:
    return ObjectiveCard(
        summary="自然美化",
        mode="auto",
        domain="general",
        preserve=[],
        goals=[],
        constraints=[],
        gaps=[
            ObjectiveGap(
                id=f"gap_{focus}",
                focus=focus,  # type: ignore[arg-type]
                description="测试缺口",
                priority=80,
                target_region="whole_image",
            )
        ],
    )


def _chain_result(
    *,
    input_path: str,
    output_path: str,
    round_id: str | None,
    focus: str,
    candidate_id: str,
    ok: bool = True,
    fallback: bool = False,
    tool_outputs: bool = True,
) -> ChainExecutionResult:
    trace = []
    outputs = []
    if tool_outputs:
        trace.append(
            {
                "index": 0,
                "round_id": round_id,
                "focus": focus,
                "candidate_id": candidate_id,
                "op": "adjust_exposure",
                "region": "whole_image",
                "ok": ok,
                "fallback_used": fallback,
                "error": None if ok else "failed",
                "output_image": output_path,
                "applied_params": {"strength": 0.2},
                "mask_path": None,
            }
        )
        if ok:
            outputs.append(output_path)
    return ChainExecutionResult(
        input_image_path=input_path,
        output_image_path=output_path,
        execution_trace=trace,
        segmentation_trace=[],
        fallback_trace=[
            {
                "index": 0,
                "round_id": round_id,
                "focus": focus,
                "candidate_id": candidate_id,
                "source": "test",
                "location": "preview",
                "strategy": "fallback",
                "message": "fallback",
                "fallback_used": True,
            }
        ]
        if fallback
        else [],
        candidate_outputs=outputs,
        mask_catalog=MaskCatalog(),
    )


class SearchAgentTest(unittest.TestCase):
    """Verify the search-first agent contract."""

    def test_auto_mode_generates_three_candidates_and_commits_only_selected(self) -> None:
        committed: list[str] = []

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            self.assertLessEqual(len(program.steps), 2)
            self.assertEqual(max_steps, 2)
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/preview_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
            )

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            committed.append(program.id)
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/full_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
            )

        with (
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit) as commit_mock,
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("global_tone"),
                request_intent=None,
                mode="auto",
            )

        self.assertEqual(preview_mock.call_count, 3)
        self.assertEqual(commit_mock.call_count, 1)
        self.assertEqual(len(result["rounds"][0].candidates), 3)
        self.assertEqual(result["rounds"][0].selected_candidate_id, committed[0])
        self.assertEqual(len(result["final_execution_trace"]), 1)
        self.assertNotIn("preview", result["candidate_outputs"][0])

    def test_fallback_commit_triggers_same_round_recovery_with_two_candidates(self) -> None:
        commit_calls: list[str] = []

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            self.assertEqual(max_steps, 2)
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/preview_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
            )

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            commit_calls.append(program.id)
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/full_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
                fallback=len(commit_calls) == 1,
            )

        with (
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("subject_cleanup"),
                request_intent=None,
                mode="auto",
            )

        round_artifact = result["rounds"][0]
        self.assertTrue(round_artifact.recovery_decision.triggered)
        self.assertEqual(len(round_artifact.recovery_candidates), 2)
        self.assertEqual(preview_mock.call_count, 5)
        self.assertEqual(len(commit_calls), 2)
        self.assertEqual(round_artifact.recovery_decision.selected_candidate_id, commit_calls[-1])

    def test_zero_step_candidate_can_be_selected_and_stop(self) -> None:
        failing = CandidateProgram(
            id="failing",
            label="失败候选",
            focus="finish",
            source="variant",
            summary="失败",
            steps=[PlannerExecutionStep(op="adjust_exposure", params={"strength": 0.2})],
        )
        also_failing = failing.model_copy(update={"id": "also_failing", "label": "另一个失败候选"})
        noop = CandidateProgram(
            id="noop",
            label="停止当前轮",
            focus="finish",
            source="noop",
            summary="停手",
            steps=[],
        )

        def fake_preview(*, input_image_path, program, round_id, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/preview_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
                ok=program.id == "noop",
                tool_outputs=bool(program.steps),
            )

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=input_image_path,
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
                tool_outputs=False,
            )

        with (
            patch("app.services.search_agent.orchestrator.generate_candidates", return_value=[failing, also_failing, noop]),
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview),
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("finish"),
                request_intent=None,
                mode="auto",
            )

        self.assertEqual(result["rounds"][0].selected_candidate_id, "noop")
        self.assertEqual(result["rounds"][0].round_review.recommended_action, "stop_round")
        self.assertEqual(result["final_execution_trace"], [])

    def test_explicit_mode_uses_direct_round_without_search_preview(self) -> None:
        request_intent = RequestIntent(
            mode="explicit",
            domain="general",
            requested_tools=[
                {
                    "op": "adjust_exposure",
                    "region": "whole_image",
                    "strength": 0.2,
                    "params": {},
                    "constraints": [],
                }
            ],
            goals=[],
            constraints=[],
            preserve=[],
        )

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/full_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
            )

        with (
            patch("app.services.search_agent.orchestrator.generate_candidates") as search_mock,
            patch("app.services.search_agent.orchestrator.execute_preview") as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit) as commit_mock,
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("global_tone"),
                request_intent=request_intent,
                mode="explicit",
            )

        search_mock.assert_not_called()
        preview_mock.assert_not_called()
        self.assertEqual(commit_mock.call_count, 1)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(len(result["rounds"][0].candidates), 1)
        self.assertEqual(result["rounds"][0].candidates[0].program.source, "direct")


if __name__ == "__main__":
    unittest.main()
