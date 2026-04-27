"""Human review checkpoint node."""

from __future__ import annotations

from uuid import uuid4

from langgraph.types import interrupt
from pydantic import ValidationError

from app.graph.state import (
    ApprovalPayload,
    EditState,
    FocusKey,
    ObjectiveCard,
    ObjectiveGap,
    coerce_eval_report,
    coerce_objective_card,
)
from app.services.search_agent.config import normalize_search_effort


ACCEPT_FINAL_NOTES = {"ok", "okay", "pass", "approve", "approved", "通过", "确认", "可以", "没问题"}


def _clean_note(value: object) -> str:
    return str(value or "").strip()


def _is_accept_final_note(note: str) -> bool:
    normalized = note.strip().lower()
    return bool(normalized) and normalized in ACCEPT_FINAL_NOTES


def _continuation_instruction(*, note: str, payload: ApprovalPayload) -> str:
    """Choose the text that should steer a human-requested follow-up round."""

    if note and _is_accept_final_note(note):
        return ""
    if note:
        return note
    for value in (payload.suggested_action, payload.summary):
        text = _clean_note(value)
        if text:
            return text
    return ""


def _structured_followup_focus(state: EditState) -> FocusKey:
    """Use the critic's structured focus when available; never infer it from text."""

    for key in ("final_review", "eval_report"):
        try:
            report = coerce_eval_report(state.get(key))
        except ValidationError:
            report = None
        if report is not None and report.next_focus:
            return report.next_focus
    return "global_tone"


def _target_region(focus: FocusKey) -> str:
    if focus == "subject_separation":
        return "person and background area"
    if focus == "subject_cleanup":
        return "face and skin area"
    if focus == "finish":
        return "detail area"
    return "whole_image"


def _append_human_continuation_gap(state: EditState, instruction: str) -> ObjectiveCard:
    objective = coerce_objective_card(state.get("objective_card")) or ObjectiveCard(
        summary=str(state.get("request_text") or "人工复核后的继续调整"),
        mode="auto",
        domain="general",
    )
    focus = _structured_followup_focus(state)
    gap = ObjectiveGap(
        id=f"human_review_{focus}_{uuid4().hex[:8]}",
        focus=focus,
        description=instruction,
        priority=100,
        target_region=_target_region(focus),
        desired_delta=instruction,
        constraints=["human_review_continuation"],
    )
    gaps = list(objective.gaps)
    gaps.append(gap)
    return objective.model_copy(update={"gaps": gaps})


def human_review(state: EditState) -> dict:
    """Pause or resume high-risk edits."""

    payload = state.get("approval_payload") or {}
    validated = ApprovalPayload.model_validate(payload)
    try:
        decision = interrupt(
            {
                "type": "human_review",
                **validated.model_dump(mode="json"),
            },
        )
    except RuntimeError:
        # 单元测试或离线归一化场景下没有 LangGraph runnable context，
        # 这里退化成“只做 payload 归一化”，不真正发起中断。
        return {
            "approval_required": True,
            "approval_payload": validated.model_dump(mode="json"),
        }

    approved = False
    note = None
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        note = _clean_note(decision.get("note"))
        requested_effort = normalize_search_effort(decision.get("search_effort") or state.get("search_effort"))
    else:
        approved = bool(decision)
        requested_effort = normalize_search_effort(state.get("search_effort"))

    metadata = dict(validated.metadata)
    metadata["review_result"] = {
        "approved": approved,
        "note": note,
    }
    update = {
        "approval_required": False,
        "needs_search_continuation": False,
        "search_continuation_reason": None,
        "human_review_continuation": False,
        "search_effort": requested_effort,
        "approval_payload": ApprovalPayload(
            reason=validated.reason,
            summary=validated.summary,
            suggested_action=validated.suggested_action,
            metadata=metadata,
        ).model_dump(mode="json"),
    }
    if not approved:
        update["selected_output"] = None
    else:
        instruction = _continuation_instruction(note=note or "", payload=validated)
        if instruction:
            objective = _append_human_continuation_gap(state, instruction)
            update.update(
                {
                    "objective_card": objective.model_dump(mode="json"),
                    "needs_search_continuation": True,
                    "search_continuation_reason": instruction,
                    "human_review_continuation": True,
                    "search_cycle_round_offset": len(list(state.get("rounds") or [])),
                }
            )
    return update
