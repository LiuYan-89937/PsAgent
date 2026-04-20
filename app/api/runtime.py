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
from app.services.stage_policy import STAGE_LABELS, STAGE_ORDER
from app.tools import TOOL_STATUS_LABELS


NODE_STATUS_LABELS = {
    "bootstrap_request": "正在准备修图请求",
    "load_context": "正在加载上下文",
    "analyze_image": "正在分析图片",
    "parse_request": "正在理解用户需求",
    "build_edit_profile": "正在建立修图画像",
    "technical_prep_subgraph": "正在执行技术预处理",
    "global_base_subgraph": "正在执行全局基线",
    "local_balance_subgraph": "正在执行局部平衡",
    "subject_refine_subgraph": "正在执行主体优化",
    "finish_output_subgraph": "正在执行最终收尾",
    "final_review": "正在评估最终结果",
    "execute_generative": "正在执行生成式编辑",
    "evaluate_result": "正在评估结果",
    "human_review": "等待人工确认",
    "update_memory": "正在更新记忆",
}

for _stage_key in STAGE_ORDER:
    _label = STAGE_LABELS[_stage_key]
    NODE_STATUS_LABELS.setdefault(f"{_stage_key}_prepare_stage_context", f"正在准备{_label}上下文")
    NODE_STATUS_LABELS.setdefault(f"{_stage_key}_build_stage_plan", f"正在规划{_label}")
    NODE_STATUS_LABELS.setdefault(f"{_stage_key}_execute_stage_plan", f"正在执行{_label}")
    NODE_STATUS_LABELS.setdefault(f"{_stage_key}_stage_guard", f"正在检查{_label}")
    NODE_STATUS_LABELS.setdefault(f"{_stage_key}_summarize_stage", f"正在总结{_label}")

def build_error_detail(
    exc: Exception,
    *,
    stage: str | None = None,
    node: str | None = None,
    op: str | None = None,
    region: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ErrorDetail:
    """Build a frontend-friendly structured error payload."""

    detail = ErrorDetail(
        type=exc.__class__.__name__,
        message=str(exc),
        stage=stage,
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
    """Attach an occurrence timestamp to an event when missing."""

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
    """Persist a job event, update current stage/message, and return the stamped event."""

    stamped_event = _stamp_event(event)
    job_store.append_event(
        job_id,
        stamped_event,
        current_stage=stamped_event.stage,
        current_message=stamped_event.message,
    )
    return stamped_event.model_dump(mode="json")


def compute_stage_timings(events: list[JobEvent | dict[str, Any]]) -> list[dict[str, Any]]:
    """Build stage timing summaries from persisted node lifecycle events."""

    open_stages: dict[str, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []

    for normalized in coerce_job_events(events):
        event_type = str(normalized.event or "")
        stage = normalized.stage or normalized.node
        occurred_at = normalized.occurred_at
        if not isinstance(stage, str) or not isinstance(occurred_at, str):
            continue
        try:
            timestamp = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError:
            continue

        if event_type == "node_started":
            open_stages[stage] = {
                "stage": stage,
                "started_at": occurred_at,
                "started_dt": timestamp,
            }
            continue

        if event_type not in {"node_finished", "node_failed"}:
            continue

        started = open_stages.pop(stage, None)
        if started is None:
            continue

        duration_ms = max(int((timestamp - started["started_dt"]).total_seconds() * 1000), 0)
        timings.append(
            {
                "stage": stage,
                "label": NODE_STATUS_LABELS.get(stage, stage),
                "started_at": started["started_at"],
                "ended_at": occurred_at,
                "duration_ms": duration_ms,
                "duration_seconds": round(duration_ms / 1000.0, 3),
                "status": "failed" if event_type == "node_failed" else "completed",
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
                    stage=name,
                    node=name,
                    message=NODE_STATUS_LABELS.get(name, f"正在执行 {name}"),
                )
                event = append_job_event(job_store, job_id, event)
                yield event
            elif payload.get("interrupts"):
                interrupt_payload = payload["interrupts"][0]
                event = make_event(
                    "interrupt",
                    stage=name,
                    node=name,
                    interrupt_id=interrupt_payload.get("id"),
                    payload=interrupt_payload.get("value"),
                    message=NODE_STATUS_LABELS.get(name, "等待人工确认"),
                )
                event = append_job_event(job_store, job_id, event)
                yield event
            elif payload.get("error") is not None:
                error_obj = payload.get("error")
                event = make_event(
                    "node_failed",
                    stage=name,
                    node=name,
                    message=f"{NODE_STATUS_LABELS.get(name, name)}失败",
                    error=str(error_obj),
                    error_detail={
                        "type": type(error_obj).__name__,
                        "message": str(error_obj),
                        "stage": name,
                        "node": name,
                    },
                )
                event = append_job_event(job_store, job_id, event)
                yield event
            else:
                event = make_event(
                    "node_finished",
                    stage=name,
                    node=name,
                    ok=payload.get("error") is None,
                    message=f"{NODE_STATUS_LABELS.get(name, name)}完成",
                )
                event = append_job_event(job_store, job_id, event)
                yield event

        elif mode == "custom":
            event = payload if isinstance(payload, dict) else make_event("custom", payload=payload)
            event = append_job_event(job_store, job_id, event)
            yield event

        elif mode == "updates" and "__interrupt__" in payload:
            interrupt_obj = payload["__interrupt__"][0]
            event = make_event(
                "interrupt",
                stage="human_review",
                node="human_review",
                interrupt_id=getattr(interrupt_obj, "id", None),
                payload=getattr(interrupt_obj, "value", None),
                message=NODE_STATUS_LABELS["human_review"],
            )
            event = append_job_event(job_store, job_id, event)
            yield event


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
