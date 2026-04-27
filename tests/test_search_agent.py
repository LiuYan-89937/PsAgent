"""Tests for the round-first search agent orchestration."""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.graph.state import CandidateProgram, MaskCatalog, ObjectiveCard, ObjectiveGap, PlannerExecutionStep, RoundGuidance
from app.graph.nodes.run_search_agent import run_search_agent
from app.services.search_agent.candidate_review_model import build_candidate_review_payload, review_candidate_batch
from app.services.search_agent.orchestrator import run_search_first_agent
from app.services.search_agent.round_guidance_model import build_round_guidance_payload, generate_round_guidance
from app.services.tool_runtime.chain_executor import ChainExecutionResult


def _objective(focus: str = "global_tone") -> ObjectiveCard:
    return ObjectiveCard(
        summary="自然美化",
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


def _program(focus: str, suffix: str, *, steps: int = 2, source: str = "model") -> CandidateProgram:
    return CandidateProgram(
        id=f"{focus}_{suffix}",
        label=f"候选 {suffix}",
        focus=focus,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        summary="模型候选",
        steps=[
            PlannerExecutionStep(op="adjust_exposure", region="whole_image", params={"strength": 0.2}, priority=index)
            for index in range(steps)
        ],
    )


def _guidance(focus: str, *, count: int = 3, steps: int = 2, candidates: list[CandidateProgram] | None = None) -> RoundGuidance:
    return RoundGuidance(
        focus=focus,  # type: ignore[arg-type]
        target_prompt=f"{focus} 本轮导向",
        visual_diagnosis="只看当前图和当前目标",
        preserve=["自然质感"],
        avoid=["过度处理"],
        candidate_programs=candidates or [_program(focus, str(index), steps=steps) for index in range(count)],
    )


class SearchAgentTest(unittest.TestCase):
    """Verify the search-first agent contract."""

    def test_search_generates_three_candidates_and_commits_only_selected(self) -> None:
        committed: list[str] = []

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            self.assertLessEqual(len(program.steps), 3)
            self.assertEqual(max_steps, 3)
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
            patch("app.services.search_agent.orchestrator.generate_round_guidance", return_value=_guidance("global_tone")) as guidance_mock,
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit) as commit_mock,
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("global_tone"),
                min_rounds=1,
                max_rounds=1,
            )

        self.assertEqual(preview_mock.call_count, 3)
        self.assertEqual(guidance_mock.call_count, 1)
        self.assertEqual(commit_mock.call_count, 1)
        self.assertEqual(len(result["rounds"][0].candidates), 3)
        self.assertEqual(result["rounds"][0].guidance.target_prompt, "global_tone 本轮导向")
        self.assertEqual(result["rounds"][0].selected_candidate_id, committed[0])
        self.assertEqual(len(result["final_execution_trace"]), 1)
        self.assertNotIn("preview", result["candidate_outputs"][0])

    def test_candidate_previews_run_concurrently(self) -> None:
        active = 0
        max_active = 0
        lock = threading.Lock()
        barrier = threading.Barrier(3)

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                barrier.wait(timeout=2)
                return _chain_result(
                    input_path=input_image_path,
                    output_path=f"/tmp/preview_{program.id}.png",
                    round_id=round_id,
                    focus=program.focus,
                    candidate_id=program.id,
                )
            finally:
                with lock:
                    active -= 1

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/full_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
            )

        with (
            patch.dict(os.environ, {"PSAGENT_PREVIEW_CONCURRENCY": "3"}),
            patch("app.services.search_agent.orchestrator.generate_round_guidance", return_value=_guidance("global_tone")),
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview),
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("global_tone"),
                min_rounds=1,
                max_rounds=1,
            )

        self.assertGreaterEqual(max_active, 2)

    def test_candidate_batch_review_selects_visual_model_choice(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.png"
            Image.new("RGB", (32, 32), (100, 100, 100)).save(input_path)
            committed: list[str] = []

            def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
                output_path = root / f"preview_{program.id}.png"
                color = (120, 120, 120) if program.id.endswith("_0") else (180, 170, 150) if program.id.endswith("_1") else (90, 90, 90)
                Image.new("RGB", (32, 32), color).save(output_path)
                return _chain_result(
                    input_path=input_image_path,
                    output_path=str(output_path),
                    round_id=round_id,
                    focus=program.focus,
                    candidate_id=program.id,
                )

            def fake_commit(*, input_image_path, program, round_id, **_kwargs):
                committed.append(program.id)
                output_path = root / f"full_{program.id}.png"
                Image.new("RGB", (32, 32), (190, 180, 160)).save(output_path)
                return _chain_result(
                    input_path=input_image_path,
                    output_path=str(output_path),
                    round_id=round_id,
                    focus=program.focus,
                    candidate_id=program.id,
                )

            with (
                patch("app.services.search_agent.orchestrator.generate_round_guidance", return_value=_guidance("global_tone")),
                patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview),
                patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
                patch("app.services.search_agent.candidate_review_model.model_available", return_value=True),
                patch(
                    "app.services.search_agent.candidate_review_model.invoke_json",
                    return_value={
                        "candidate_scores": [
                            {"candidate_id": "global_tone_0", "score": 3.1},
                            {"candidate_id": "global_tone_1", "score": 4.6},
                            {"candidate_id": "global_tone_2", "score": 2.4},
                        ],
                    },
                ) as invoke_mock,
            ):
                result = run_search_first_agent(
                    input_image_path=str(input_path),
                    objective=_objective("global_tone"),
                    min_rounds=1,
                    max_rounds=1,
                )

        self.assertEqual(committed, ["global_tone_1"])
        self.assertEqual(result["rounds"][0].selected_candidate_id, "global_tone_1")
        self.assertEqual(result["rounds"][0].candidates[0].eliminated_reason, "候选 0 小模型视觉评分 3.10。")
        self.assertEqual(invoke_mock.call_args.kwargs["model_env_name"], "OPENAI_CANDIDATE_REVIEW_MODEL")
        self.assertEqual(invoke_mock.call_args.kwargs["default_model"], "qwen3-vl-flash")

    def test_fallback_commit_triggers_same_round_recovery_with_two_candidates(self) -> None:
        commit_calls: list[str] = []

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            self.assertIn(max_steps, {2, 3})
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
            patch(
                "app.services.search_agent.orchestrator.generate_round_guidance",
                side_effect=[_guidance("subject_cleanup"), _guidance("subject_cleanup", count=2)],
            ) as guidance_mock,
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("subject_cleanup"),
                min_rounds=1,
                max_rounds=1,
            )

        round_artifact = result["rounds"][0]
        self.assertTrue(round_artifact.recovery_decision.triggered)
        self.assertEqual(len(round_artifact.recovery_candidates), 2)
        self.assertEqual(guidance_mock.call_count, 2)
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
            patch("app.services.search_agent.orchestrator.generate_round_guidance", return_value=_guidance("finish", candidates=[failing, also_failing, noop])),
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview),
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("finish"),
                min_rounds=1,
                max_rounds=1,
            )

        self.assertEqual(result["rounds"][0].selected_candidate_id, "noop")
        self.assertEqual(result["rounds"][0].round_review.recommended_action, "stop_round")
        self.assertEqual(result["final_execution_trace"], [])

    def test_guidance_failure_creates_single_stop_candidate(self) -> None:
        committed: list[str] = []

        def fake_preview(*, input_image_path, program, round_id, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=input_image_path,
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
                tool_outputs=False,
            )

        def fake_commit(*, input_image_path, program, round_id, **_kwargs):
            committed.append(program.id)
            return _chain_result(
                input_path=input_image_path,
                output_path=input_image_path,
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
                tool_outputs=False,
            )

        with (
            patch("app.services.search_agent.orchestrator.generate_round_guidance", side_effect=RuntimeError("bad guidance")),
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit),
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=_objective("global_tone"),
                min_rounds=1,
                max_rounds=1,
            )

        round_artifact = result["rounds"][0]
        self.assertEqual(preview_mock.call_count, 1)
        self.assertEqual(len(round_artifact.candidates), 1)
        self.assertEqual(round_artifact.candidates[0].program.source, "noop")
        self.assertEqual(round_artifact.guidance.visual_diagnosis, "bad guidance")
        self.assertEqual(round_artifact.selected_candidate_id, committed[0])

    def test_min_rounds_drive_refinement_rounds_before_finish(self) -> None:
        objective = _objective("global_tone")
        objective.gaps.append(
            ObjectiveGap(
                id="gap_finish",
                focus="finish",
                description="收尾",
                priority=30,
                target_region="whole_image",
            )
        )

        def fake_preview(*, input_image_path, program, round_id, max_steps, **_kwargs):
            return _chain_result(
                input_path=input_image_path,
                output_path=f"/tmp/preview_{program.id}.png",
                round_id=round_id,
                focus=program.focus,
                candidate_id=program.id,
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
            patch("app.services.search_agent.orchestrator.generate_round_guidance", side_effect=lambda **kwargs: _guidance(kwargs["focus"])) as guidance_mock,
            patch("app.services.search_agent.orchestrator.execute_preview", side_effect=fake_preview) as preview_mock,
            patch("app.services.search_agent.orchestrator.execute_chain", side_effect=fake_commit) as commit_mock,
        ):
            result = run_search_first_agent(
                input_image_path="/tmp/input.png",
                objective=objective,
                min_rounds=4,
                max_rounds=4,
            )

        self.assertEqual(len(result["rounds"]), 4)
        self.assertEqual([round_artifact.focus for round_artifact in result["rounds"]], ["global_tone", "global_tone", "global_tone", "finish"])
        self.assertEqual(preview_mock.call_count, 12)
        self.assertEqual(guidance_mock.call_count, 4)
        self.assertEqual(commit_mock.call_count, 4)
        self.assertIn("search_refinement_round", result["rounds"][1].objective_gaps[0].constraints)

    def test_run_search_agent_appends_continuation_round_to_existing_artifacts(self) -> None:
        continuation_round = {
            "id": "round_4_subject_separation",
            "index": 4,
            "focus": "subject_separation",
            "output_image_path": "/tmp/new.png",
            "candidates": [],
            "selected_candidate_id": "candidate_4",
            "recovery_candidates": [],
            "completed": True,
        }

        with patch(
            "app.graph.nodes.run_search_agent.run_search_first_agent",
            return_value={
                "selected_output": "/tmp/new.png",
                "candidate_outputs": ["/tmp/new.png"],
                "execution_trace": [
                    {
                        "round_id": "round_4_subject_separation",
                        "focus": "subject_separation",
                        "candidate_id": "candidate_4",
                        "op": "adjust_exposure",
                        "region": "person area",
                        "ok": True,
                        "fallback_used": False,
                        "output_image": "/tmp/new.png",
                    }
                ],
                "final_execution_trace": [
                    {
                        "round_id": "round_4_subject_separation",
                        "focus": "subject_separation",
                        "candidate_id": "candidate_4",
                        "op": "adjust_exposure",
                        "region": "person area",
                        "ok": True,
                        "fallback_used": False,
                        "output_image": "/tmp/new.png",
                    }
                ],
                "segmentation_trace": [],
                "fallback_trace": [],
                "rounds": [continuation_round],
                "selected_candidate_id": "candidate_4",
                "edit_plan": {
                    "mode": "auto",
                    "domain": "portrait",
                    "executor": "deterministic",
                    "preserve": [],
                    "operations": [
                        {
                            "op": "adjust_exposure",
                            "region": "person area",
                            "params": {"strength": 0.34},
                            "constraints": [],
                            "priority": 0,
                        }
                    ],
                    "should_write_memory": False,
                    "memory_candidates": [],
                    "needs_confirmation": False,
                },
                "mask_catalog": {"items": {}},
            },
        ) as mocked_run:
            result = run_search_agent(
                {
                    "mode": "auto",
                    "input_images": ["/tmp/input.png"],
                    "selected_output": "/tmp/current.png",
                    "objective_card": _objective("subject_separation").model_dump(mode="json"),
                    "rounds": [
                        {"id": "round_1", "index": 1, "focus": "finish", "completed": True},
                        {"id": "round_2", "index": 2, "focus": "finish", "completed": True},
                        {"id": "round_3", "index": 3, "focus": "finish", "completed": True},
                    ],
                    "candidate_outputs": ["/tmp/current.png"],
                    "execution_trace": [{"round_id": "round_3", "focus": "finish", "op": "adjust_vignette", "ok": True}],
                    "final_execution_trace": [{"round_id": "round_3", "focus": "finish", "op": "adjust_vignette", "ok": True}],
                }
            )

        self.assertEqual(mocked_run.call_args.kwargs["round_index_offset"], 3)
        self.assertEqual(mocked_run.call_args.kwargs["max_rounds"], 3)
        self.assertEqual(mocked_run.call_args.kwargs["min_rounds"], 1)
        self.assertEqual(len(result["rounds"]), 4)
        self.assertEqual(result["rounds"][-1].id, "round_4_subject_separation")
        self.assertEqual(len(result["final_execution_trace"]), 2)
        self.assertFalse(result["needs_search_continuation"])

    def test_run_search_agent_allows_one_human_review_round_after_auto_cap(self) -> None:
        continuation_round = {
            "id": "round_5_global_tone",
            "index": 5,
            "focus": "global_tone",
            "output_image_path": "/tmp/new.png",
            "candidates": [],
            "selected_candidate_id": "candidate_5",
            "recovery_candidates": [],
            "completed": True,
        }

        with patch(
            "app.graph.nodes.run_search_agent.run_search_first_agent",
            return_value={
                "selected_output": "/tmp/new.png",
                "candidate_outputs": ["/tmp/new.png"],
                "execution_trace": [],
                "final_execution_trace": [],
                "segmentation_trace": [],
                "fallback_trace": [],
                "rounds": [continuation_round],
                "selected_candidate_id": "candidate_5",
                "edit_plan": {
                    "mode": "auto",
                    "domain": "general",
                    "executor": "deterministic",
                    "preserve": [],
                    "operations": [],
                    "should_write_memory": False,
                    "memory_candidates": [],
                    "needs_confirmation": False,
                },
                "mask_catalog": {"items": {}},
            },
        ) as mocked_run:
            result = run_search_agent(
                {
                    "mode": "auto",
                    "human_review_continuation": True,
                    "input_images": ["/tmp/input.png"],
                    "selected_output": "/tmp/current.png",
                    "objective_card": _objective("global_tone").model_dump(mode="json"),
                    "rounds": [
                        {"id": "round_1", "index": 1, "focus": "subject_separation", "completed": True},
                        {"id": "round_2", "index": 2, "focus": "subject_cleanup", "completed": True},
                        {"id": "round_3", "index": 3, "focus": "global_tone", "completed": True},
                        {"id": "round_4", "index": 4, "focus": "finish", "completed": True},
                    ],
                }
            )

        self.assertEqual(mocked_run.call_args.kwargs["round_index_offset"], 4)
        self.assertEqual(mocked_run.call_args.kwargs["max_rounds"], 6)
        self.assertEqual(mocked_run.call_args.kwargs["min_rounds"], 4)
        self.assertEqual(result["rounds"][-1].id, "round_5_global_tone")
        self.assertFalse(result["human_review_continuation"])
        self.assertIsNone(result["approval_payload"])
        self.assertEqual(result["search_cycle_round_offset"], 4)


class RoundGuidanceModelTest(unittest.TestCase):
    """Verify the per-round guidance model contract."""

    def test_payload_is_current_round_only(self) -> None:
        payload = build_round_guidance_payload(
            objective=_objective("global_tone"),
            focus="global_tone",
            round_gaps=_objective("global_tone").gaps,
            tool_catalog=[{"name": "adjust_exposure", "description": "曝光", "supported_regions": ["whole_image"], "params_schema": {}}],
            candidate_count=3,
            max_steps=2,
        )

        self.assertIn("当前round", payload)
        self.assertEqual(payload["候选限制"]["min_steps_per_non_stop_candidate"], 2)
        self.assertNotIn("previous_rounds", payload)
        self.assertNotIn("execution_trace", payload)
        self.assertNotIn("selected_candidate_history", payload)

    def test_generate_round_guidance_normalizes_invalid_tool_to_noop(self) -> None:
        with (
            patch("app.services.search_agent.round_guidance_model.model_available", return_value=True),
            patch(
                "app.services.search_agent.round_guidance_model.invoke_json",
                return_value={
                    "target_prompt": "提亮面部但保留高反差",
                    "visual_diagnosis": "当前脸部偏暗",
                    "preserve": ["逆光发丝"],
                    "avoid": ["背景抬灰"],
                    "candidates": [
                        {"label": "非法工具", "summary": "不应执行", "steps": [{"op": "not_a_tool", "params": {}}]},
                        {
                            "label": "有效工具",
                            "summary": "可以执行",
                            "steps": [
                                {"op": "adjust_exposure", "params": {"strength": 60}},
                                {"op": "adjust_brightness", "params": {"brightness_offset": 65}},
                            ],
                        },
                    ],
                },
            ) as invoke_mock,
        ):
            guidance = generate_round_guidance(
                current_image_path="/tmp/current.png",
                objective=_objective("global_tone"),
                focus="global_tone",
                round_gaps=_objective("global_tone").gaps,
                tool_catalog=[{"name": "adjust_exposure", "description": "曝光", "supported_regions": ["whole_image"], "params_schema": {}}],
                candidate_count=3,
                max_steps=2,
            )

        self.assertEqual(invoke_mock.call_args.kwargs["image_paths"], ["/tmp/current.png"])
        self.assertEqual(len(guidance.candidate_programs), 3)
        self.assertEqual(guidance.candidate_programs[0].source, "noop")
        self.assertEqual(guidance.candidate_programs[1].source, "model")
        self.assertEqual(guidance.candidate_programs[1].steps[0].op, "adjust_exposure")
        self.assertAlmostEqual(guidance.candidate_programs[1].steps[0].params["strength"], 0.45)
        self.assertAlmostEqual(guidance.candidate_programs[1].steps[1].params["brightness_offset"], 0.18)

    def test_one_step_auto_candidate_is_rejected_to_noop(self) -> None:
        with (
            patch("app.services.search_agent.round_guidance_model.model_available", return_value=True),
            patch(
                "app.services.search_agent.round_guidance_model.invoke_json",
                return_value={
                    "target_prompt": "自然提亮",
                    "visual_diagnosis": "候选只有一步",
                    "preserve": [],
                    "avoid": [],
                    "candidates": [
                        {"label": "单步曝光", "summary": "不应作为普通 auto 候选", "steps": [{"op": "adjust_exposure", "params": {"strength": 40}}]},
                    ],
                },
            ),
        ):
            guidance = generate_round_guidance(
                current_image_path="/tmp/current.png",
                objective=_objective("global_tone"),
                focus="global_tone",
                round_gaps=_objective("global_tone").gaps,
                tool_catalog=[{"name": "adjust_exposure", "description": "曝光", "supported_regions": ["whole_image"], "params_schema": {}}],
                candidate_count=1,
                max_steps=3,
            )

        self.assertEqual(guidance.candidate_programs[0].source, "noop")
        self.assertEqual(guidance.candidate_programs[0].steps, [])


class CandidateReviewModelTest(unittest.TestCase):
    """Verify the per-round candidate review model contract."""

    def test_candidate_review_payload_is_current_round_batch(self) -> None:
        program = _program("global_tone", "0")
        execution = _chain_result(
            input_path="/tmp/current.png",
            output_path="/tmp/preview.png",
            round_id="round_1",
            focus="global_tone",
            candidate_id=program.id,
        ).to_candidate_execution()
        artifact = {
            "candidate_id": program.id,
            "label": program.label,
            "focus": program.focus,
            "program": program,
            "preview_execution": execution,
            "review": None,
        }
        from app.graph.state import SearchCandidateArtifact

        payload = build_candidate_review_payload(target="自然提亮", artifacts=[SearchCandidateArtifact.model_validate(artifact)])

        self.assertIn("图片顺序", payload)
        self.assertEqual(payload["目标"], "自然提亮")
        self.assertEqual(payload["图片顺序"][0]["role"], "current_image")
        self.assertNotIn("candidate_id", payload["图片顺序"][0])
        self.assertEqual(payload["图片顺序"][1]["role"], "candidate_preview")
        self.assertEqual(payload["图片顺序"][1]["candidate_id"], program.id)
        self.assertNotIn("label", payload["图片顺序"][1])
        self.assertNotIn("总目标摘要", payload)
        self.assertNotIn("当前round", payload)
        self.assertNotIn("候选", payload)
        self.assertNotIn("previous_rounds", payload)
        self.assertNotIn("execution_trace", payload)
        self.assertNotIn("selected_candidate_history", payload)

    def test_candidate_review_batch_uses_qwen_vl_flash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current.png"
            preview = root / "preview.png"
            Image.new("RGB", (24, 24), (120, 120, 120)).save(current)
            Image.new("RGB", (24, 24), (150, 150, 150)).save(preview)
            program = _program("global_tone", "0")
            execution = _chain_result(
                input_path=str(current),
                output_path=str(preview),
                round_id="round_1",
                focus="global_tone",
                candidate_id=program.id,
            ).to_candidate_execution()
            from app.graph.state import SearchCandidateArtifact

            artifact = SearchCandidateArtifact(
                candidate_id=program.id,
                label=program.label,
                focus="global_tone",
                program=program,
                preview_execution=execution,
            )

            with (
                patch("app.services.search_agent.candidate_review_model.model_available", return_value=True),
                patch(
                    "app.services.search_agent.candidate_review_model.invoke_json",
                    return_value={
                        "candidate_scores": [{"candidate_id": program.id, "score": 4.2}],
                    },
                ) as invoke_mock,
            ):
                batch = review_candidate_batch(
                    current_image_path=str(current),
                    objective_summary="自然提亮",
                    focus="global_tone",
                    round_gaps=_objective("global_tone").gaps,
                    guidance=_guidance("global_tone"),
                    artifacts=[artifact],
                )

        self.assertIsNotNone(batch)
        self.assertEqual(batch.selected_candidate_id, program.id)
        self.assertEqual(invoke_mock.call_args.kwargs["image_paths"], [str(current), str(preview)])
        self.assertEqual(invoke_mock.call_args.kwargs["user_payload"]["目标"], "global_tone 本轮导向")
        self.assertEqual(invoke_mock.call_args.kwargs["model_env_name"], "OPENAI_CANDIDATE_REVIEW_MODEL")
        self.assertEqual(invoke_mock.call_args.kwargs["default_model"], "qwen3-vl-flash")

    def test_candidate_review_rejects_non_score_only_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = root / "current.png"
            preview = root / "preview.png"
            Image.new("RGB", (24, 24), (120, 120, 120)).save(current)
            Image.new("RGB", (24, 24), (150, 150, 150)).save(preview)
            program = _program("global_tone", "0")
            execution = _chain_result(
                input_path=str(current),
                output_path=str(preview),
                round_id="round_1",
                focus="global_tone",
                candidate_id=program.id,
            ).to_candidate_execution()
            from app.graph.state import SearchCandidateArtifact

            artifact = SearchCandidateArtifact(
                candidate_id=program.id,
                label=program.label,
                focus="global_tone",
                program=program,
                preview_execution=execution,
            )

            with (
                patch("app.services.search_agent.candidate_review_model.model_available", return_value=True),
                patch(
                    "app.services.search_agent.candidate_review_model.invoke_json",
                    return_value={
                        "candidate_scores": [{"candidate_id": program.id, "score": 4.2, "summary": "extra"}],
                    },
                ),
            ):
                with self.assertRaises(RuntimeError):
                    review_candidate_batch(
                        current_image_path=str(current),
                        objective_summary="自然提亮",
                        focus="global_tone",
                        round_gaps=_objective("global_tone").gaps,
                        guidance=_guidance("global_tone"),
                        artifacts=[artifact],
                    )


if __name__ == "__main__":
    unittest.main()
