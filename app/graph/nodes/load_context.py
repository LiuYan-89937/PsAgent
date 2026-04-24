"""Load short-term context and long-term preferences."""

from __future__ import annotations

from app.graph.state import (
    ApprovalPayload,
    EditState,
    coerce_approval_payload,
    coerce_execution_trace,
    coerce_fallback_trace,
    coerce_mask_catalog,
    coerce_memory_write_candidates,
    coerce_objective_card,
    coerce_preferences,
    coerce_request_intent,
    coerce_segmentation_trace,
    coerce_search_rounds,
    coerce_search_run,
    coerce_tool_catalog,
)
from app.tools import export_tool_catalog


def load_context(state: EditState) -> dict:
    """Populate graph context with defaults needed by downstream nodes.

    当前阶段先不接真实的长期记忆存储，因此这个节点主要做三件事：
    1. 给线程状态补齐常用默认字段；
    2. 读取并缓存原生工具目录，供 planner 直接消费；
    3. 保证后面的节点不需要反复处理 None / 缺字段。
    """

    tool_catalog = coerce_tool_catalog(state.get("tool_catalog", export_tool_catalog()))
    request_intent = coerce_request_intent(state.get("request_intent"))
    objective_card = coerce_objective_card(state.get("objective_card"))
    search_run = coerce_search_run(state.get("search_run"))
    execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("execution_trace", []))]
    final_execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("final_execution_trace", []))]
    segmentation_trace = [item.model_dump(mode="json") for item in coerce_segmentation_trace(state.get("segmentation_trace", []))]
    fallback_trace = [item.model_dump(mode="json") for item in coerce_fallback_trace(state.get("fallback_trace", []))]
    memory_write_candidates = [item.model_dump(mode="json") for item in coerce_memory_write_candidates(state.get("memory_write_candidates", []))]
    approval_payload = coerce_approval_payload(state.get("approval_payload"))
    retrieved_prefs = coerce_preferences(state.get("retrieved_prefs", []))
    mask_catalog = coerce_mask_catalog(state.get("mask_catalog"))

    return {
        "request_text": state.get("request_text"),
        "request_intent": request_intent.model_dump(mode="json") if request_intent is not None else None,
        "tool_catalog": [item.model_dump(mode="json") for item in tool_catalog],
        "retrieved_prefs": [item.model_dump(mode="json") for item in retrieved_prefs],
        "objective_card": objective_card.model_dump(mode="json") if objective_card is not None else None,
        "search_run": search_run.model_dump(mode="json") if search_run is not None else None,
        "rounds": [item.model_dump(mode="json") for item in coerce_search_rounds(state.get("rounds", []))],
        "current_round": state.get("current_round"),
        "current_focus": state.get("current_focus"),
        "selected_candidate_id": state.get("selected_candidate_id"),
        "mask_catalog": mask_catalog.model_dump(mode="json"),
        "candidate_outputs": state.get("candidate_outputs", []),
        "execution_trace": execution_trace,
        "final_execution_trace": final_execution_trace,
        "segmentation_trace": segmentation_trace,
        "fallback_trace": fallback_trace,
        "memory_write_candidates": memory_write_candidates,
        "approval_payload": approval_payload.model_dump(mode="json") if approval_payload is not None else None,
    }
