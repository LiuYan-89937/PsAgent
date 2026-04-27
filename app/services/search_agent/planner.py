"""Candidate validation and direct execution helpers for round-first search."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.graph.state import CandidateProgram, FocusKey, ObjectiveCard, PlannerExecutionStep, RequestIntent
from app.tools import WHOLE_IMAGE_REGION


CANDIDATE_COUNT = 3
RECOVERY_CANDIDATE_COUNT = 2
CANDIDATE_PREVIEW_STEP_LIMIT = 2
RECOVERY_PREVIEW_STEP_LIMIT = 2


def _step(op: str, *, region: str = WHOLE_IMAGE_REGION, params: dict | None = None, priority: int = 0) -> PlannerExecutionStep:
    return PlannerExecutionStep(op=op, region=region, params=dict(params or {}), priority=priority)


def _candidate(
    *,
    focus: FocusKey,
    label: str,
    summary: str,
    steps: list[PlannerExecutionStep],
    source: str,
    is_recovery: bool = False,
) -> CandidateProgram:
    return CandidateProgram(
        id=f"{focus}_{'recovery_' if is_recovery else ''}{uuid4().hex[:8]}",
        label=label,
        focus=focus,
        source=source,  # type: ignore[arg-type]
        summary=summary,
        steps=steps[: RECOVERY_PREVIEW_STEP_LIMIT if is_recovery else CANDIDATE_PREVIEW_STEP_LIMIT],
        is_recovery=is_recovery,
    )


def build_stop_candidate(
    *,
    focus: FocusKey,
    label: str = "停止当前轮",
    summary: str = "本轮不追加工具调用，等待后续评估或人工复核。",
    is_recovery: bool = False,
) -> CandidateProgram:
    """Build a bounded 0-step candidate used when model guidance asks to stop or fails."""

    return _candidate(
        focus=focus,
        label=label,
        summary=summary,
        steps=[],
        source="noop",
        is_recovery=is_recovery,
    )


def _coerce_text(value: Any, *, fallback: str, max_length: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _coerce_step(raw_step: Any, *, priority: int) -> PlannerExecutionStep | None:
    if not isinstance(raw_step, dict):
        return None
    payload = {
        "op": raw_step.get("op"),
        "region": raw_step.get("region") or WHOLE_IMAGE_REGION,
        "params": raw_step.get("params") if isinstance(raw_step.get("params"), dict) else {},
        "constraints": raw_step.get("constraints") if isinstance(raw_step.get("constraints"), list) else [],
        "priority": raw_step.get("priority", priority),
    }
    if raw_step.get("strength") is not None:
        payload["strength"] = raw_step.get("strength")
    try:
        return PlannerExecutionStep.model_validate(payload)
    except ValidationError:
        return None


def _coerce_candidate(
    raw_candidate: Any,
    *,
    focus: FocusKey,
    index: int,
    max_steps: int,
    is_recovery: bool,
) -> CandidateProgram | None:
    if isinstance(raw_candidate, CandidateProgram):
        candidate = raw_candidate.model_copy(deep=True)
        candidate.id = f"{focus}_{'recovery_' if is_recovery else ''}{uuid4().hex[:8]}"
        candidate.focus = focus
        candidate.is_recovery = is_recovery
        candidate.steps = candidate.steps[:max_steps]
        candidate.source = "noop" if not candidate.steps else "model"
        return candidate
    if not isinstance(raw_candidate, dict):
        return None

    raw_steps = raw_candidate.get("steps")
    steps = [
        step
        for step in (
            _coerce_step(raw_step, priority=step_index)
            for step_index, raw_step in enumerate(raw_steps if isinstance(raw_steps, list) else [])
        )
        if step is not None
    ][:max_steps]

    label = _coerce_text(raw_candidate.get("label"), fallback=f"候选 {index + 1}")
    summary = _coerce_text(raw_candidate.get("summary"), fallback="模型生成的候选工具链。", max_length=180)
    return _candidate(
        focus=focus,
        label=label,
        summary=summary,
        steps=steps,
        source="model" if steps else "noop",
        is_recovery=is_recovery,
    )


def normalize_model_candidates(
    raw_candidates: Any,
    *,
    focus: FocusKey,
    candidate_count: int,
    max_steps: int,
    is_recovery: bool = False,
) -> list[CandidateProgram]:
    """Normalize model candidate payloads into safe CandidateProgram objects."""

    candidates: list[CandidateProgram] = []
    iterable = raw_candidates if isinstance(raw_candidates, list) else []
    for index, raw_candidate in enumerate(iterable):
        candidate = _coerce_candidate(
            raw_candidate,
            focus=focus,
            index=index,
            max_steps=max_steps,
            is_recovery=is_recovery,
        )
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= candidate_count:
            break

    while len(candidates) < candidate_count:
        candidates.append(
            build_stop_candidate(
                focus=focus,
                label="停止当前轮" if not is_recovery else "停止恢复",
                summary="模型未提供足够可安全执行的候选，本候选保持当前结果。",
                is_recovery=is_recovery,
            )
        )
    return candidates


def generate_direct_candidate(*, request_intent: RequestIntent | None, objective: ObjectiveCard) -> CandidateProgram:
    """Build one direct explicit-mode candidate without search."""

    steps: list[PlannerExecutionStep] = []
    if request_intent is not None:
        for index, hint in enumerate(request_intent.requested_tools[:4]):
            params = dict(hint.params or {})
            if hint.strength is not None and params.get("strength") is None:
                params["strength"] = hint.strength
            steps.append(_step(hint.op, region=hint.region, params=params, priority=index))
        if not steps:
            for index, goal in enumerate(request_intent.goals[:2]):
                kind = goal.kind.lower()
                params = {"strength": min(max(abs(goal.intensity or 0.3), 0.05), 0.8)}
                if "bright" in kind or "luminance" in kind or "exposure" in kind:
                    steps.append(_step("adjust_exposure", region=goal.target_region, params=params, priority=index))
                elif "contrast" in kind or "tonal" in kind:
                    steps.append(_step("adjust_contrast", region=goal.target_region, params=params, priority=index))
                elif "color" in kind or "saturation" in kind:
                    steps.append(_step("adjust_vibrance_saturation", region=goal.target_region, params=params, priority=index))
    if not steps:
        steps.append(_step("adjust_exposure", params={"strength": 0.2}, priority=0))

    focus = objective.gaps[0].focus if objective.gaps else "global_tone"
    return _candidate(
        focus=focus,
        label="直接执行",
        summary="显式模式按用户请求直接执行工具链。",
        steps=steps[:4],
        source="direct",
    )
