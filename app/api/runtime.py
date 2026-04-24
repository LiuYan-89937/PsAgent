"""Runtime helpers for API-driven graph execution."""

from __future__ import annotations

import json
import traceback
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

from langgraph.types import Command

from app.graph.state import ErrorDetail, JobEvent, coerce_job_events
from app.services.job_store import JobStore


NODE_STATUS_LABELS = {
    "bootstrap_request": "正在准备修图请求",
    "load_context": "正在加载上下文",
    "analyze_image": "正在分析图片",
    "parse_request": "正在理解用户需求",
    "build_objective": "正在建立搜索目标",
    "run_search_agent": "正在搜索候选方案",
    "final_review": "正在评估最终结果",
    "human_review": "等待人工确认",
    "update_memory": "正在更新记忆",
}

FOCUS_LABELS = {
    "global_tone": "整体影调",
    "subject_separation": "主体分离",
    "subject_cleanup": "主体清理",
    "finish": "最终收口",
}


def build_error_detail(
    exc: Exception,
    *,
    node: str | None = None,
    op: str | None = None,
    region: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ErrorDetail:
    """Build a frontend-friendly structured error payload."""

    detail = ErrorDetail(
        type=exc.__class__.__name__,
        message=str(exc),
        node=node,
        op=op,
        region=region,
        traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )
    if extra:
        payload = detail.model_dump(mode="json")
        payload.update(extra)
        return ErrorDetail.model_validate(payload)
    return detail


def make_event(event: str, **payload: Any) -> JobEvent:
    """Create a normalized job event payload."""

    return JobEvent.model_validate({"event": event, **payload})


def _stamp_event(event: JobEvent | dict[str, Any]) -> JobEvent:
    normalized = coerce_job_events([event])[0]
    if normalized.occurred_at:
        return normalized
    payload = normalized.model_dump(mode="json")
    payload["occurred_at"] = datetime.now(timezone.utc).isoformat()
    return JobEvent.model_validate(payload)


def format_sse(event: str, data: JobEvent | dict[str, Any]) -> str:
    """Format a single SSE payload."""

    payload = coerce_job_events([data])[0].model_dump(mode="json")
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def append_job_event(job_store: JobStore, job_id: str, event: JobEvent | dict[str, Any]) -> dict[str, Any]:
    """Persist a job event, update current round/message, and return the stamped event."""

    stamped_event = _stamp_event(event)
    job_store.append_event(
        job_id,
        stamped_event,
        current_round=stamped_event.round,
        current_focus=stamped_event.focus,
        current_message=stamped_event.message,
    )
    return stamped_event.model_dump(mode="json")


def compute_round_timings(events: list[JobEvent | dict[str, Any]]) -> list[dict[str, Any]]:
    """Build timing summaries from round lifecycle events."""

    open_rounds: dict[str, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []
    for normalized in coerce_job_events(events):
        event_type = str(normalized.event or "")
        round_id = normalized.round
        occurred_at = normalized.occurred_at
        if not isinstance(round_id, str) or not isinstance(occurred_at, str):
            continue
        try:
            timestamp = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if event_type == "round_started":
            open_rounds[round_id] = {
                "round": round_id,
                "focus": normalized.focus,
                "started_at": occurred_at,
                "started_dt": timestamp,
            }
            continue
        if event_type not in {"round_completed", "round_failed"}:
            continue
        started = open_rounds.pop(round_id, None)
        if started is None:
            continue
        duration_ms = max(int((timestamp - started["started_dt"]).total_seconds() * 1000), 0)
        focus = started.get("focus") or normalized.focus
        timings.append(
            {
                "round": round_id,
                "focus": focus,
                "label": FOCUS_LABELS.get(str(focus), round_id),
                "started_at": started["started_at"],
                "ended_at": occurred_at,
                "duration_ms": duration_ms,
                "duration_seconds": round(duration_ms / 1000.0, 3),
                "status": "failed" if event_type == "round_failed" else "completed",
            }
        )
    return timings


def iter_graph_events(
    *,
    graph,
    graph_input: dict[str, Any] | Command,
    config: dict[str, Any],
    job_store: JobStore,
    job_id: str,
) -> Generator[dict[str, Any], None, None]:
    """Stream normalized graph events for frontend consumption."""

    stream = graph.stream(
        graph_input,
        config=config,
        stream_mode=["tasks", "updates", "custom"],
        version="v2",
    )
    for mode, payload in stream:
        if mode == "tasks":
            name = payload.get("name")
            if "input" in payload and "result" not in payload and "error" not in payload:
                event = make_event(
                    "node_started",
                    node=name,
                    message=NODE_STATUS_LABELS.get(name, f"正在执行 {name}"),
                )
                yield append_job_event(job_store, job_id, event)
            elif payload.get("interrupts"):
                interrupt_payload = payload["interrupts"][0]
                event = make_event(
                    "interrupt",
                    node=name,
                    interrupt_id=interrupt_payload.get("id"),
                    payload=interrupt_payload.get("value"),
                    message=NODE_STATUS_LABELS.get(name, "等待人工确认"),
                )
                yield append_job_event(job_store, job_id, event)
            elif payload.get("error") is not None:
                error_obj = payload.get("error")
                event = make_event(
                    "node_failed",
                    node=name,
                    message=f"{NODE_STATUS_LABELS.get(name, name)}失败",
                    error=str(error_obj),
                    error_detail={
                        "type": type(error_obj).__name__,
                        "message": str(error_obj),
                        "node": name,
                    },
                )
                yield append_job_event(job_store, job_id, event)
            else:
                event = make_event(
                    "node_finished",
                    node=name,
                    ok=payload.get("error") is None,
                    message=f"{NODE_STATUS_LABELS.get(name, name)}完成",
                )
                yield append_job_event(job_store, job_id, event)

        elif mode == "custom":
            event = payload if isinstance(payload, dict) else make_event("custom", payload=payload)
            yield append_job_event(job_store, job_id, event)

        elif mode == "updates" and "__interrupt__" in payload:
            interrupt_obj = payload["__interrupt__"][0]
            event = make_event(
                "interrupt",
                node="human_review",
                interrupt_id=getattr(interrupt_obj, "id", None),
                payload=getattr(interrupt_obj, "value", None),
                message=NODE_STATUS_LABELS["human_review"],
            )
            yield append_job_event(job_store, job_id, event)


def build_graph_config(thread_id: str) -> dict[str, Any]:
    """Build the per-thread graph config."""

    return {"configurable": {"thread_id": thread_id}}


def read_final_state(graph, config: dict[str, Any]) -> dict[str, Any]:
    """Read the current state snapshot after execution or interruption."""

    snapshot = graph.get_state(config)
    return dict(snapshot.values or {})


def collect_terminal_status(final_state: dict[str, Any]) -> str:
    """Infer job status from the final state snapshot."""

    if final_state.get("approval_required"):
        return "review_required"
    if final_state.get("selected_output") or final_state.get("candidate_outputs"):
        return "completed"
    return "failed"
