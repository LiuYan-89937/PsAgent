"""Runtime helpers for API-driven graph execution."""

from __future__ import annotations

import json
import traceback
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any

from langgraph.types import Command

from app.services.job_store import JobStore
from app.tools import PACKAGE_STATUS_LABELS


NODE_STATUS_LABELS = {
    "load_context": "正在加载上下文",
    "analyze_image": "正在分析图片",
    "parse_request": "正在理解用户需求",
    "plan_execute_round_1": "正在规划并执行第一轮",
    "plan_execute_round_2": "正在规划并执行第二轮",
    "build_plan": "正在生成修图计划",
    "build_plan_round_1": "正在生成第一轮修图计划",
    "build_plan_round_2": "正在生成第二轮修图计划",
    "route_executor": "正在选择执行器",
    "route_executor_round_1": "正在选择第一轮执行器",
    "route_executor_round_2": "正在选择第二轮执行器",
    "execute_deterministic": "正在执行全局调整",
    "execute_hybrid": "正在执行局部调整",
    "execute_generative": "正在执行生成式编辑",
    "execute_round_1_deterministic": "正在执行第一轮全局调整",
    "execute_round_1_hybrid": "正在执行第一轮局部调整",
    "execute_round_1_generative": "正在执行第一轮生成式编辑",
    "execute_round_2_deterministic": "正在执行第二轮全局调整",
    "execute_round_2_hybrid": "正在执行第二轮局部调整",
    "execute_round_2_generative": "正在执行第二轮生成式编辑",
    "evaluate_result": "正在评估结果",
    "evaluate_round_1": "正在评估第一轮结果",
    "evaluate_result_final": "正在评估最终结果",
    "finalize_round_1_result": "正在确认首轮结果",
    "human_review": "等待人工确认",
    "update_memory": "正在更新记忆",
}

def build_error_detail(
    exc: Exception,
    *,
    stage: str | None = None,
    node: str | None = None,
    op: str | None = None,
    region: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a frontend-friendly structured error payload."""

    detail = {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "stage": stage,
        "node": node,
        "op": op,
        "region": region,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    if extra:
        detail.update(extra)
    return detail


def make_event(event: str, **payload: Any) -> dict[str, Any]:
    """Create a normalized job event payload."""

    return {"event": event, **payload}


def _stamp_event(event: dict[str, Any]) -> dict[str, Any]:
    """Attach an occurrence timestamp to an event when missing."""

    if event.get("occurred_at"):
        return event
    return {
        **event,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
    }


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format a single SSE payload."""

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def append_job_event(job_store: JobStore, job_id: str, event: dict[str, Any]) -> dict[str, Any]:
    """Persist a job event, update current stage/message, and return the stamped event."""

    stamped_event = _stamp_event(event)
    job_store.append_event(
        job_id,
        stamped_event,
        current_stage=stamped_event.get("stage"),
        current_message=stamped_event.get("message"),
    )
    return stamped_event


def compute_stage_timings(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build stage timing summaries from persisted node lifecycle events."""

    open_stages: dict[str, dict[str, Any]] = {}
    timings: list[dict[str, Any]] = []

    for event in events:
        event_type = str(event.get("event") or "")
        stage = event.get("stage") or event.get("node")
        occurred_at = event.get("occurred_at")
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
