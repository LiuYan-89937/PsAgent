"""Round-first search orchestrator."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.graph.state import (
    CandidatePreviewExecution,
    CandidateProgram,
    EditOperation,
    EditPlan,
    FocusKey,
    MaskCatalog,
    ObjectiveCard,
    RecoveryDecision,
    RequestIntent,
    SearchCandidateArtifact,
    SearchRoundArtifact,
    SearchRunArtifact,
    coerce_execution_trace,
    coerce_segmentation_trace,
)
from app.services.search_agent.planner import (
    CANDIDATE_COUNT,
    CANDIDATE_PREVIEW_STEP_LIMIT,
    RECOVERY_CANDIDATE_COUNT,
    RECOVERY_PREVIEW_STEP_LIMIT,
    generate_candidates,
    generate_direct_candidate,
    generate_recovery_candidates,
)
from app.services.search_agent.reviewer import review_candidate, review_round
from app.services.tool_runtime import execute_chain, execute_preview


MAX_ROUNDS = 4


def _event_writer(writer):
    return writer or (lambda *_args, **_kwargs: None)


def _round_id(index: int, focus: FocusKey) -> str:
    return f"round_{index}_{focus}_{uuid4().hex[:6]}"


def _select_focus(objective: ObjectiveCard, *, completed_focuses: set[FocusKey]) -> FocusKey | None:
    for gap in sorted(objective.gaps, key=lambda item: item.priority, reverse=True):
        if not gap.resolved and gap.focus not in completed_focuses:
            return gap.focus
    return None


def _candidate_artifact(
    *,
    program: CandidateProgram,
    selected: bool,
    preview_execution: CandidatePreviewExecution | None,
    review,
    eliminated_reason: str | None = None,
) -> SearchCandidateArtifact:
    return SearchCandidateArtifact(
        candidate_id=program.id,
        label=program.label,
        focus=program.focus,
        selected=selected,
        eliminated_reason=eliminated_reason,
        program=program,
        preview_execution=preview_execution,
        review=review,
    )


def _select_best(artifacts: list[SearchCandidateArtifact]) -> SearchCandidateArtifact:
    return max(
        artifacts,
        key=lambda item: (
            item.review.score if item.review is not None else float("-inf"),
            -len(item.review.issues if item.review is not None else []),
            -(len(item.program.steps) if item.program is not None else 0),
        ),
    )


def _run_preview_candidates(
    *,
    input_image_path: str,
    programs: list[CandidateProgram],
    mask_catalog: MaskCatalog,
    round_id: str,
    writer,
    max_steps: int,
) -> list[SearchCandidateArtifact]:
    artifacts: list[SearchCandidateArtifact] = []
    for program in programs:
        writer(
            {
                "event": "candidate_preview_started",
                "round": round_id,
                "focus": program.focus,
                "message": f"开始预览候选：{program.label}",
                "payload": {"candidate_id": program.id, "max_steps": max_steps},
            }
        )
        preview_result = execute_preview(
            input_image_path=input_image_path,
            program=program,
            mask_catalog=mask_catalog,
            round_id=round_id,
            max_steps=max_steps,
        )
        preview_execution = preview_result.to_candidate_execution()
        review = review_candidate(program=program, execution=preview_execution)
        artifacts.append(
            _candidate_artifact(
                program=program,
                selected=False,
                preview_execution=preview_execution,
                review=review,
            )
        )
        writer(
            {
                "event": "candidate_preview_finished",
                "round": round_id,
                "focus": program.focus,
                "message": f"候选预览完成：{program.label}",
                "payload": {"candidate_id": program.id, "score": review.score, "recommended_action": review.recommended_action},
            }
        )
    selected = _select_best(artifacts)
    for artifact in artifacts:
        artifact.selected = artifact.candidate_id == selected.candidate_id
        if not artifact.selected:
            artifact.eliminated_reason = artifact.review.summary if artifact.review is not None else "候选得分较低。"
    return artifacts


def _commit_program(
    *,
    input_image_path: str,
    program: CandidateProgram,
    mask_catalog: MaskCatalog,
    round_id: str,
    writer,
    mode: str,
) -> tuple[CandidatePreviewExecution, MaskCatalog, list[str]]:
    result = execute_chain(
        input_image_path=input_image_path,
        program=program,
        mask_catalog=mask_catalog,
        writer=writer,
        mode=mode,
        round_id=round_id,
        focus=program.focus,
        candidate_id=program.id,
    )
    return result.to_candidate_execution(), result.mask_catalog, result.candidate_outputs


def _flatten_edit_plan(*, objective: ObjectiveCard, mode: str, committed_programs: list[CandidateProgram]) -> EditPlan:
    operations: list[EditOperation] = []
    for program in committed_programs:
        for step in program.steps:
            operations.append(EditOperation.model_validate(step.model_dump(mode="json")))
    return EditPlan(
        mode="auto" if mode == "auto" else "explicit",
        domain=objective.domain,
        executor="deterministic",
        preserve=list(objective.preserve),
        operations=operations,
        should_write_memory=False,
        memory_candidates=[],
        needs_confirmation=False,
    )


def _direct_round(
    *,
    input_image_path: str,
    objective: ObjectiveCard,
    request_intent: RequestIntent | None,
    mask_catalog: MaskCatalog,
    writer,
) -> tuple[list[SearchRoundArtifact], str, list[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], MaskCatalog, list[CandidateProgram]]:
    program = generate_direct_candidate(request_intent=request_intent, objective=objective)
    rid = _round_id(1, program.focus)
    writer({"event": "round_started", "round": rid, "focus": program.focus, "message": "开始显式直接执行轮"})
    writer(
        {
            "event": "candidate_generated",
            "round": rid,
            "focus": program.focus,
            "message": "已生成直接执行候选",
            "payload": {"candidate_id": program.id, "candidate_count": 1},
        }
    )
    writer(
        {
            "event": "candidate_selected",
            "round": rid,
            "focus": program.focus,
            "message": "显式模式选中直接执行候选",
            "payload": {"candidate_id": program.id},
        }
    )
    full_execution, mask_catalog, outputs = _commit_program(
        input_image_path=input_image_path,
        program=program,
        mask_catalog=mask_catalog,
        round_id=rid,
        writer=writer,
        mode="explicit",
    )
    review = review_candidate(program=program, execution=full_execution)
    round_review = review_round(selected_review=review, full_execution=full_execution)
    round_artifact = SearchRoundArtifact(
        id=rid,
        index=1,
        focus=program.focus,
        input_image_path=input_image_path,
        output_image_path=full_execution.output_image_path or input_image_path,
        objective_gaps=list(objective.gaps),
        candidates=[
            _candidate_artifact(program=program, selected=True, preview_execution=None, review=review)
        ],
        selected_candidate_id=program.id,
        selected_full_execution=full_execution,
        round_review=round_review,
        recovery_decision=RecoveryDecision(triggered=False, source="none"),
        completed=True,
    )
    writer({"event": "round_review_finished", "round": rid, "focus": program.focus, "message": round_review.summary})
    writer({"event": "round_completed", "round": rid, "focus": program.focus, "message": "显式直接执行轮完成"})
    return (
        [round_artifact],
        full_execution.output_image_path or input_image_path,
        outputs,
        [item.model_dump(mode="json") for item in full_execution.execution_trace],
        [item.model_dump(mode="json") for item in full_execution.segmentation_trace],
        [item.model_dump(mode="json") for item in full_execution.fallback_trace],
        mask_catalog,
        [program],
    )


def run_search_first_agent(
    *,
    input_image_path: str,
    objective: ObjectiveCard,
    request_intent: RequestIntent | None,
    mode: str,
    mask_catalog: MaskCatalog | None = None,
    writer=None,
) -> dict[str, Any]:
    """Run either explicit direct execution or auto round-first search."""

    writer = _event_writer(writer)
    runtime_catalog = mask_catalog or MaskCatalog()
    if mode != "auto":
        rounds, selected_output, outputs, execution_trace, segmentation_trace, fallback_trace, runtime_catalog, committed = _direct_round(
            input_image_path=input_image_path,
            objective=objective,
            request_intent=request_intent,
            mask_catalog=runtime_catalog,
            writer=writer,
        )
        selected_candidate_id = rounds[-1].selected_candidate_id
        search_run = SearchRunArtifact(
            objective_card=objective,
            rounds=rounds,
            selected_candidate_id=selected_candidate_id,
            final_execution_trace=coerce_execution_trace(execution_trace),
        )
        return {
            "selected_output": selected_output,
            "candidate_outputs": outputs,
            "execution_trace": coerce_execution_trace(execution_trace),
            "final_execution_trace": coerce_execution_trace(execution_trace),
            "segmentation_trace": coerce_segmentation_trace(segmentation_trace),
            "fallback_trace": fallback_trace,
            "rounds": rounds,
            "search_run": search_run,
            "selected_candidate_id": selected_candidate_id,
            "edit_plan": _flatten_edit_plan(objective=objective, mode=mode, committed_programs=committed),
            "mask_catalog": runtime_catalog.model_dump(mode="json"),
        }

    current_image = input_image_path
    completed_focuses: set[FocusKey] = set()
    rounds: list[SearchRoundArtifact] = []
    candidate_outputs: list[str] = []
    execution_trace: list[dict[str, Any]] = []
    segmentation_trace: list[dict[str, Any]] = []
    fallback_trace: list[dict[str, Any]] = []
    committed_programs: list[CandidateProgram] = []
    selected_candidate_id: str | None = None

    for round_index in range(1, MAX_ROUNDS + 1):
        focus = _select_focus(objective, completed_focuses=completed_focuses)
        if focus is None:
            break
        rid = _round_id(round_index, focus)
        round_gaps = [gap for gap in objective.gaps if gap.focus == focus and not gap.resolved]
        writer({"event": "round_started", "round": rid, "focus": focus, "message": f"开始搜索轮：{focus}"})
        programs = generate_candidates(objective=objective, focus=focus, round_index=round_index)
        for program in programs:
            writer(
                {
                    "event": "candidate_generated",
                    "round": rid,
                    "focus": focus,
                    "message": f"已生成候选：{program.label}",
                    "payload": {"candidate_id": program.id, "candidate_count": CANDIDATE_COUNT},
                }
            )
        artifacts = _run_preview_candidates(
            input_image_path=current_image,
            programs=programs,
            mask_catalog=runtime_catalog,
            round_id=rid,
            writer=writer,
            max_steps=CANDIDATE_PREVIEW_STEP_LIMIT,
        )
        selected_artifact = next(item for item in artifacts if item.selected)
        selected_program = selected_artifact.program or programs[0]
        selected_candidate_id = selected_program.id
        writer(
            {
                "event": "candidate_selected",
                "round": rid,
                "focus": focus,
                "message": f"选中候选：{selected_program.label}",
                "payload": {"candidate_id": selected_program.id, "score": selected_artifact.review.score if selected_artifact.review else None},
            }
        )
        full_execution, runtime_catalog, outputs = _commit_program(
            input_image_path=current_image,
            program=selected_program,
            mask_catalog=runtime_catalog,
            round_id=rid,
            writer=writer,
            mode="auto",
        )
        candidate_outputs.extend(outputs)
        execution_trace.extend(item.model_dump(mode="json") for item in full_execution.execution_trace)
        segmentation_trace.extend(item.model_dump(mode="json") for item in full_execution.segmentation_trace)
        fallback_trace.extend(item.model_dump(mode="json") for item in full_execution.fallback_trace)
        current_image = full_execution.output_image_path or current_image
        committed_programs.append(selected_program)
        round_review = review_round(selected_review=selected_artifact.review, full_execution=full_execution) if selected_artifact.review else None
        recovery_decision = RecoveryDecision(triggered=False, source="none")
        recovery_artifacts: list[SearchCandidateArtifact] = []

        if round_review is not None and round_review.recommended_action == "recover_same_round":
            writer(
                {
                    "event": "recovery_started",
                    "round": rid,
                    "focus": focus,
                    "message": "本轮触发 recovery search",
                    "payload": {"max_candidates": RECOVERY_CANDIDATE_COUNT, "max_steps": RECOVERY_PREVIEW_STEP_LIMIT},
                }
            )
            recovery_programs = generate_recovery_candidates(focus=focus, reason=round_review.summary)
            recovery_artifacts = _run_preview_candidates(
                input_image_path=current_image,
                programs=recovery_programs,
                mask_catalog=runtime_catalog,
                round_id=rid,
                writer=writer,
                max_steps=RECOVERY_PREVIEW_STEP_LIMIT,
            )
            selected_recovery = next(item for item in recovery_artifacts if item.selected)
            recovery_program = selected_recovery.program or recovery_programs[0]
            recovery_full, runtime_catalog, recovery_outputs = _commit_program(
                input_image_path=current_image,
                program=recovery_program,
                mask_catalog=runtime_catalog,
                round_id=rid,
                writer=writer,
                mode="auto",
            )
            selected_candidate_id = recovery_program.id
            candidate_outputs.extend(recovery_outputs)
            execution_trace.extend(item.model_dump(mode="json") for item in recovery_full.execution_trace)
            segmentation_trace.extend(item.model_dump(mode="json") for item in recovery_full.segmentation_trace)
            fallback_trace.extend(item.model_dump(mode="json") for item in recovery_full.fallback_trace)
            current_image = recovery_full.output_image_path or current_image
            committed_programs.append(recovery_program)
            recovery_decision = RecoveryDecision(
                triggered=True,
                source="round_review",
                fallback_aware=bool(round_review.warnings),
                reason=round_review.summary,
                candidate_ids=[item.candidate_id for item in recovery_artifacts],
                selected_candidate_id=recovery_program.id,
            )
            full_execution = recovery_full
            round_review = review_round(selected_review=selected_recovery.review, full_execution=recovery_full) if selected_recovery.review else round_review

        writer({"event": "round_review_finished", "round": rid, "focus": focus, "message": round_review.summary if round_review else "本轮完成。"})
        for gap in objective.gaps:
            if gap.focus == focus:
                gap.resolved = True
        completed_focuses.add(focus)
        round_artifact = SearchRoundArtifact(
            id=rid,
            index=round_index,
            focus=focus,
            input_image_path=full_execution.input_image_path,
            output_image_path=current_image,
            objective_gaps=round_gaps,
            candidates=artifacts,
            selected_candidate_id=selected_candidate_id,
            selected_full_execution=full_execution,
            round_review=round_review,
            recovery_decision=recovery_decision,
            recovery_candidates=recovery_artifacts,
            completed=True,
        )
        rounds.append(round_artifact)
        writer({"event": "round_completed", "round": rid, "focus": focus, "message": f"搜索轮完成：{focus}"})
        if focus == "finish" or (round_review is not None and round_review.recommended_action == "stop_round" and selected_program.source == "noop"):
            break

    search_run = SearchRunArtifact(
        objective_card=objective,
        rounds=rounds,
        selected_candidate_id=selected_candidate_id,
        final_execution_trace=coerce_execution_trace(execution_trace),
    )
    return {
        "selected_output": current_image,
        "candidate_outputs": candidate_outputs,
        "execution_trace": coerce_execution_trace(execution_trace),
        "final_execution_trace": coerce_execution_trace(execution_trace),
        "segmentation_trace": coerce_segmentation_trace(segmentation_trace),
        "fallback_trace": fallback_trace,
        "rounds": rounds,
        "search_run": search_run,
        "selected_candidate_id": selected_candidate_id,
        "edit_plan": _flatten_edit_plan(objective=objective, mode=mode, committed_programs=committed_programs),
        "mask_catalog": runtime_catalog.model_dump(mode="json"),
    }
