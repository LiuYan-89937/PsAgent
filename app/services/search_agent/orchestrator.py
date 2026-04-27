"""Round-first search orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ObjectiveGap,
    RecoveryDecision,
    RequestIntent,
    RoundGuidance,
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
    build_stop_candidate,
    generate_direct_candidate,
)
from app.services.search_agent.round_guidance_model import generate_round_guidance
from app.services.search_agent.reviewer import review_candidate, review_round
from app.services.search_agent.config import HARD_MAX_ROUNDS, resolve_search_round_limits
from app.services.tool_runtime import execute_chain, execute_preview


DEFAULT_MAX_ROUNDS = resolve_search_round_limits("standard").max_rounds
DEFAULT_MIN_ROUNDS = resolve_search_round_limits("standard").min_rounds
FOCUS_REFINEMENT_ORDER: tuple[FocusKey, ...] = ("global_tone", "subject_separation", "subject_cleanup")
REFINEMENT_GAP_LABELS: dict[FocusKey, str] = {
    "global_tone": "继续细化整体明暗、对比和色彩基线。",
    "subject_separation": "继续细化主体与背景分离和局部可读性。",
    "subject_cleanup": "继续细化人像主体肤色、脸部或发丝细节。",
    "finish": "进行最终收口，控制处理痕迹。",
}
REFINEMENT_TARGETS: dict[FocusKey, str] = {
    "global_tone": "whole_image",
    "subject_separation": "subject area",
    "subject_cleanup": "face and skin area",
    "finish": "whole_image",
}


@dataclass(slots=True)
class SearchRunState:
    """Mutable state accumulated while executing one search cycle."""

    current_image: str
    mask_catalog: MaskCatalog
    rounds: list[SearchRoundArtifact] = field(default_factory=list)
    candidate_outputs: list[str] = field(default_factory=list)
    execution_trace: list[dict[str, Any]] = field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = field(default_factory=list)
    committed_programs: list[CandidateProgram] = field(default_factory=list)
    selected_candidate_id: str | None = None

    def absorb_commit(self, *, execution: CandidatePreviewExecution, outputs: list[str], program: CandidateProgram) -> None:
        self.selected_candidate_id = program.id
        self.candidate_outputs.extend(outputs)
        self.execution_trace.extend(_dump_trace_items(execution.execution_trace))
        self.segmentation_trace.extend(_dump_trace_items(execution.segmentation_trace))
        self.fallback_trace.extend(_dump_trace_items(execution.fallback_trace))
        self.current_image = execution.output_image_path or self.current_image
        self.committed_programs.append(program)


def _event_writer(writer):
    return writer or (lambda *_args, **_kwargs: None)


def _round_id(index: int, focus: FocusKey) -> str:
    return f"round_{index}_{focus}_{uuid4().hex[:6]}"


def _focus_priority(objective: ObjectiveCard, focus: FocusKey) -> int:
    priorities = [gap.priority for gap in objective.gaps if gap.focus == focus]
    return max(priorities) if priorities else 0


def _refinement_focuses(objective: ObjectiveCard) -> list[FocusKey]:
    available = [focus for focus in FOCUS_REFINEMENT_ORDER if any(gap.focus == focus for gap in objective.gaps)]
    return available or ["global_tone"]


def _select_refinement_focus(objective: ObjectiveCard, *, focus_counts: dict[FocusKey, int]) -> FocusKey:
    focuses = _refinement_focuses(objective)
    return max(focuses, key=lambda focus: (-focus_counts.get(focus, 0), _focus_priority(objective, focus)))


def _select_focus(
    objective: ObjectiveCard,
    *,
    focus_counts: dict[FocusKey, int],
    local_round_index: int,
    min_rounds: int,
) -> FocusKey | None:
    unresolved_non_finish = [
        gap
        for gap in sorted(objective.gaps, key=lambda item: item.priority, reverse=True)
        if not gap.resolved and gap.focus != "finish"
    ]
    if unresolved_non_finish:
        return unresolved_non_finish[0].focus

    unresolved_finish = [gap for gap in objective.gaps if not gap.resolved and gap.focus == "finish"]
    if unresolved_finish and local_round_index >= min_rounds:
        return "finish"

    if local_round_index <= min_rounds:
        return _select_refinement_focus(objective, focus_counts=focus_counts)

    for gap in sorted(objective.gaps, key=lambda item: item.priority, reverse=True):
        if not gap.resolved:
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


def _dump_trace_items(items: list[Any]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item) for item in items]


def _build_result_payload(*, state: SearchRunState, objective: ObjectiveCard, mode: str) -> dict[str, Any]:
    search_run = SearchRunArtifact(
        objective_card=objective,
        rounds=state.rounds,
        selected_candidate_id=state.selected_candidate_id,
        final_execution_trace=coerce_execution_trace(state.execution_trace),
    )
    return {
        "selected_output": state.current_image,
        "candidate_outputs": state.candidate_outputs,
        "execution_trace": coerce_execution_trace(state.execution_trace),
        "final_execution_trace": coerce_execution_trace(state.execution_trace),
        "segmentation_trace": coerce_segmentation_trace(state.segmentation_trace),
        "fallback_trace": state.fallback_trace,
        "rounds": state.rounds,
        "search_run": search_run,
        "selected_candidate_id": state.selected_candidate_id,
        "edit_plan": _flatten_edit_plan(objective=objective, mode=mode, committed_programs=state.committed_programs),
        "mask_catalog": state.mask_catalog.model_dump(mode="json"),
    }


def _emit_generated_candidates(*, writer, round_id: str, focus: FocusKey, programs: list[CandidateProgram], candidate_count: int) -> None:
    for program in programs:
        writer(
            {
                "event": "candidate_generated",
                "round": round_id,
                "focus": focus,
                "message": f"已生成候选：{program.label}",
                "payload": {"candidate_id": program.id, "candidate_count": candidate_count},
            }
        )


def _emit_candidate_selected(*, writer, round_id: str, focus: FocusKey, artifact: SearchCandidateArtifact, program: CandidateProgram) -> None:
    writer(
        {
            "event": "candidate_selected",
            "round": round_id,
            "focus": focus,
            "message": f"选中候选：{program.label}",
            "payload": {"candidate_id": program.id, "score": artifact.review.score if artifact.review else None},
        }
    )


def _emit_round_guidance(*, writer, round_id: str, focus: FocusKey, guidance: RoundGuidance) -> None:
    writer(
        {
            "event": "round_guidance_generated",
            "round": round_id,
            "focus": focus,
            "message": guidance.target_prompt or guidance.visual_diagnosis or "已生成本轮导向提示词",
            "payload": {
                "target_prompt": guidance.target_prompt,
                "visual_diagnosis": guidance.visual_diagnosis,
                "preserve": guidance.preserve,
                "avoid": guidance.avoid,
            },
        }
    )


def _guidance_stop_candidate(*, focus: FocusKey, error: Exception, is_recovery: bool = False) -> CandidateProgram:
    return build_stop_candidate(
        focus=focus,
        label="停止恢复" if is_recovery else "停止当前轮",
        summary=f"本轮导向生成失败，停止追加工具调用：{error}",
        is_recovery=is_recovery,
    )


def _generate_guidance_or_stop(
    *,
    current_image_path: str,
    objective: ObjectiveCard,
    focus: FocusKey,
    round_gaps: list[ObjectiveGap],
    tool_catalog: list[dict[str, Any]],
    candidate_count: int,
    max_steps: int,
    is_recovery: bool,
    recovery_reason: str | None = None,
) -> RoundGuidance:
    try:
        return generate_round_guidance(
            current_image_path=current_image_path,
            objective=objective,
            focus=focus,
            round_gaps=round_gaps,
            tool_catalog=tool_catalog,
            candidate_count=candidate_count,
            max_steps=max_steps,
            is_recovery=is_recovery,
            recovery_reason=recovery_reason,
        )
    except Exception as exc:
        stop_candidate = _guidance_stop_candidate(focus=focus, error=exc, is_recovery=is_recovery)
        return RoundGuidance(
            focus=focus,
            target_prompt="本轮导向生成失败，停止追加工具调用。",
            visual_diagnosis=str(exc),
            preserve=[],
            avoid=["不要在缺少有效导向时继续自动调用工具"],
            candidate_programs=[stop_candidate],
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


def _run_recovery_if_needed(
    *,
    state: SearchRunState,
    objective: ObjectiveCard,
    focus: FocusKey,
    round_id: str,
    tool_catalog: list[dict[str, Any]],
    writer,
    round_review,
    full_execution: CandidatePreviewExecution,
) -> tuple[CandidatePreviewExecution, Any, RecoveryDecision, list[SearchCandidateArtifact]]:
    if round_review is None or round_review.recommended_action != "recover_same_round":
        return full_execution, round_review, RecoveryDecision(triggered=False, source="none"), []

    writer(
        {
            "event": "recovery_started",
            "round": round_id,
            "focus": focus,
            "message": "本轮触发 recovery search",
            "payload": {"max_candidates": RECOVERY_CANDIDATE_COUNT, "max_steps": RECOVERY_PREVIEW_STEP_LIMIT},
        }
    )
    recovery_gaps = [
        ObjectiveGap(
            id=f"recovery_{focus}_{uuid4().hex[:8]}",
            focus=focus,
            description=round_review.summary or "修正当前 round review 指出的问题。",
            priority=90,
            target_region=REFINEMENT_TARGETS[focus],
            desired_delta=round_review.summary or "",
            constraints=["same_round_recovery"],
        )
    ]
    guidance = _generate_guidance_or_stop(
        current_image_path=state.current_image,
        objective=objective,
        focus=focus,
        round_gaps=recovery_gaps,
        tool_catalog=tool_catalog,
        candidate_count=RECOVERY_CANDIDATE_COUNT,
        max_steps=RECOVERY_PREVIEW_STEP_LIMIT,
        is_recovery=True,
        recovery_reason=round_review.summary,
    )
    _emit_round_guidance(writer=writer, round_id=round_id, focus=focus, guidance=guidance)
    recovery_programs = guidance.candidate_programs
    recovery_artifacts = _run_preview_candidates(
        input_image_path=state.current_image,
        programs=recovery_programs,
        mask_catalog=state.mask_catalog,
        round_id=round_id,
        writer=writer,
        max_steps=RECOVERY_PREVIEW_STEP_LIMIT,
    )
    selected_recovery = next(item for item in recovery_artifacts if item.selected)
    recovery_program = selected_recovery.program or recovery_programs[0]
    recovery_full, state.mask_catalog, recovery_outputs = _commit_program(
        input_image_path=state.current_image,
        program=recovery_program,
        mask_catalog=state.mask_catalog,
        round_id=round_id,
        writer=writer,
        mode="auto",
    )
    state.absorb_commit(execution=recovery_full, outputs=recovery_outputs, program=recovery_program)
    recovery_decision = RecoveryDecision(
        triggered=True,
        source="round_review",
        fallback_aware=bool(round_review.warnings),
        reason=round_review.summary,
        candidate_ids=[item.candidate_id for item in recovery_artifacts],
        selected_candidate_id=recovery_program.id,
    )
    updated_review = review_round(selected_review=selected_recovery.review, full_execution=recovery_full) if selected_recovery.review else round_review
    return recovery_full, updated_review, recovery_decision, recovery_artifacts


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


def _resolve_focus_gaps(objective: ObjectiveCard, focus: FocusKey) -> None:
    for gap in objective.gaps:
        if gap.focus == focus:
            gap.resolved = True


def _round_gaps_for_focus(objective: ObjectiveCard, focus: FocusKey) -> list[ObjectiveGap]:
    gaps = [gap for gap in objective.gaps if gap.focus == focus and not gap.resolved]
    if gaps:
        return gaps
    return [
        ObjectiveGap(
            id=f"refinement_{focus}_{uuid4().hex[:8]}",
            focus=focus,
            description=REFINEMENT_GAP_LABELS[focus],
            priority=max(_focus_priority(objective, focus), 40),
            target_region=REFINEMENT_TARGETS[focus],
            constraints=["search_refinement_round"],
        )
    ]


def _run_auto_round(
    *,
    state: SearchRunState,
    objective: ObjectiveCard,
    focus: FocusKey,
    round_index: int,
    tool_catalog: list[dict[str, Any]],
    writer,
) -> bool:
    rid = _round_id(round_index, focus)
    round_gaps = _round_gaps_for_focus(objective, focus)
    writer({"event": "round_started", "round": rid, "focus": focus, "message": f"开始搜索轮：{focus}"})

    guidance = _generate_guidance_or_stop(
        current_image_path=state.current_image,
        objective=objective,
        focus=focus,
        round_gaps=round_gaps,
        tool_catalog=tool_catalog,
        candidate_count=CANDIDATE_COUNT,
        max_steps=CANDIDATE_PREVIEW_STEP_LIMIT,
        is_recovery=False,
    )
    _emit_round_guidance(writer=writer, round_id=rid, focus=focus, guidance=guidance)
    programs = guidance.candidate_programs
    _emit_generated_candidates(writer=writer, round_id=rid, focus=focus, programs=programs, candidate_count=len(programs))
    artifacts = _run_preview_candidates(
        input_image_path=state.current_image,
        programs=programs,
        mask_catalog=state.mask_catalog,
        round_id=rid,
        writer=writer,
        max_steps=CANDIDATE_PREVIEW_STEP_LIMIT,
    )
    selected_artifact = next(item for item in artifacts if item.selected)
    selected_program = selected_artifact.program or programs[0]
    _emit_candidate_selected(writer=writer, round_id=rid, focus=focus, artifact=selected_artifact, program=selected_program)

    full_execution, state.mask_catalog, outputs = _commit_program(
        input_image_path=state.current_image,
        program=selected_program,
        mask_catalog=state.mask_catalog,
        round_id=rid,
        writer=writer,
        mode="auto",
    )
    state.absorb_commit(execution=full_execution, outputs=outputs, program=selected_program)
    round_review = review_round(selected_review=selected_artifact.review, full_execution=full_execution) if selected_artifact.review else None
    full_execution, round_review, recovery_decision, recovery_artifacts = _run_recovery_if_needed(
        state=state,
        objective=objective,
        focus=focus,
        round_id=rid,
        tool_catalog=tool_catalog,
        writer=writer,
        round_review=round_review,
        full_execution=full_execution,
    )

    writer({"event": "round_review_finished", "round": rid, "focus": focus, "message": round_review.summary if round_review else "本轮完成。"})
    _resolve_focus_gaps(objective, focus)
    state.rounds.append(
        SearchRoundArtifact(
            id=rid,
            index=round_index,
            focus=focus,
            input_image_path=full_execution.input_image_path,
            output_image_path=state.current_image,
            objective_gaps=round_gaps,
            guidance=guidance,
            candidates=artifacts,
            selected_candidate_id=state.selected_candidate_id,
            selected_full_execution=full_execution,
            round_review=round_review,
            recovery_decision=recovery_decision,
            recovery_candidates=recovery_artifacts,
            completed=True,
        )
    )
    writer({"event": "round_completed", "round": rid, "focus": focus, "message": f"搜索轮完成：{focus}"})
    return focus == "finish" or (round_review is not None and round_review.recommended_action == "stop_round" and selected_program.source == "noop")


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
    tool_catalog: list[dict[str, Any]] | None = None,
    writer=None,
    round_index_offset: int = 0,
    min_rounds: int | None = None,
    max_rounds: int | None = None,
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
        return _build_result_payload(
            state=SearchRunState(
                current_image=selected_output,
                mask_catalog=runtime_catalog,
                rounds=rounds,
                candidate_outputs=outputs,
                execution_trace=execution_trace,
                segmentation_trace=segmentation_trace,
                fallback_trace=fallback_trace,
                committed_programs=committed,
                selected_candidate_id=rounds[-1].selected_candidate_id,
            ),
            objective=objective,
            mode=mode,
        )

    focus_counts: dict[FocusKey, int] = {}
    state = SearchRunState(current_image=input_image_path, mask_catalog=runtime_catalog)
    runtime_tool_catalog = list(tool_catalog or [])

    round_limit = DEFAULT_MAX_ROUNDS if max_rounds is None else max(0, min(HARD_MAX_ROUNDS, max_rounds))
    min_round_limit = DEFAULT_MIN_ROUNDS if min_rounds is None else max(0, min(round_limit, min_rounds))
    for local_round_index in range(1, round_limit + 1):
        round_index = round_index_offset + local_round_index
        focus = _select_focus(
            objective,
            focus_counts=focus_counts,
            local_round_index=local_round_index,
            min_rounds=min_round_limit,
        )
        if focus is None:
            break
        should_stop = _run_auto_round(
            state=state,
            objective=objective,
            focus=focus,
            round_index=round_index,
            tool_catalog=runtime_tool_catalog,
            writer=writer,
        )
        focus_counts[focus] = focus_counts.get(focus, 0) + 1
        if should_stop and local_round_index >= min_round_limit:
            break

    return _build_result_payload(state=state, objective=objective, mode=mode)
