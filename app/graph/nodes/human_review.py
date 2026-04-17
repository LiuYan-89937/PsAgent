"""Human review checkpoint node."""

from __future__ import annotations

from langgraph.types import interrupt

from app.graph.state import ApprovalPayload, EditState


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
        note = decision.get("note")
    else:
        approved = bool(decision)

    metadata = dict(validated.metadata)
    metadata["review_result"] = {
        "approved": approved,
        "note": note,
    }
    update = {
        "approval_required": False,
        "approval_payload": ApprovalPayload(
            reason=validated.reason,
            summary=validated.summary,
            suggested_action=validated.suggested_action,
            metadata=metadata,
        ).model_dump(mode="json"),
    }
    if not approved:
        update["selected_output"] = None
    return update
