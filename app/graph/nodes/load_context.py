"""Load short-term context and long-term preferences."""

from __future__ import annotations

from app.graph.state import (
    ApprovalPayload,
    EditState,
    ExecutionTraceItem,
    MemoryWriteCandidate,
    PackageCatalogItem,
    RequestIntent,
    SegmentationTraceItem,
)
from app.tools.packages import build_default_package_registry


def load_context(state: EditState) -> dict:
    """Populate graph context with defaults needed by downstream nodes.

    当前阶段先不接真实的长期记忆存储，因此这个节点主要做三件事：
    1. 给线程状态补齐常用默认字段；
    2. 读取并缓存工具包目录，供 planner 直接消费；
    3. 保证后面的节点不需要反复处理 None / 缺字段。
    """

    registry = build_default_package_registry()
    package_catalog = [
        PackageCatalogItem.model_validate(item).model_dump(mode="json")
        for item in state.get("package_catalog", registry.export_llm_catalog())
    ]
    request_intent = state.get("request_intent")
    if request_intent is not None:
        request_intent = RequestIntent.model_validate(request_intent).model_dump(mode="json")
    execution_trace = [
        ExecutionTraceItem.model_validate(item).model_dump(mode="json")
        for item in state.get("execution_trace", [])
    ]
    segmentation_trace = [
        SegmentationTraceItem.model_validate(item).model_dump(mode="json")
        for item in state.get("segmentation_trace", [])
    ]
    memory_write_candidates = [
        MemoryWriteCandidate.model_validate(item).model_dump(mode="json")
        for item in state.get("memory_write_candidates", [])
    ]
    approval_payload = state.get("approval_payload")
    if approval_payload is not None:
        approval_payload = ApprovalPayload.model_validate(approval_payload).model_dump(mode="json")

    return {
        "request_text": state.get("request_text"),
        "request_intent": request_intent,
        "package_catalog": package_catalog,
        "retrieved_prefs": state.get("retrieved_prefs", []),
        "masks": state.get("masks", {}),
        "candidate_outputs": state.get("candidate_outputs", []),
        "execution_trace": execution_trace,
        "segmentation_trace": segmentation_trace,
        "memory_write_candidates": memory_write_candidates,
        "approval_payload": approval_payload,
    }
