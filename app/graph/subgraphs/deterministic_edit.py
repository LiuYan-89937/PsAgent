"""Deterministic editing subgraph."""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer

from app.graph.state import EditPlan, EditState, ExecutionTraceItem
from app.tools.packages import OperationContext, build_default_package_registry
from app.tools.packages.macros import expand_macro_operations, operations_require_hybrid


def _build_operation_context(state: EditState, image_path: str) -> OperationContext:
    """Create the shared operation context for package execution."""

    return OperationContext(
        image_path=image_path,
        image_analysis=state.get("image_analysis") or {},
        retrieved_prefs=state.get("retrieved_prefs") or [],
        masks=state.get("masks") or {},
        thread_id=state.get("thread_id"),
        audit={},
    )


def _normalize_runtime_plan(state: EditState) -> dict[str, Any]:
    """Normalize a runtime plan before executor consumption.

    执行器允许接收“只包含 operations 的最小 plan”，因为部分测试和中间节点
    还会用这种结构调用执行器。
    """

    raw_plan = dict(state.get("edit_plan") or {})
    raw_plan.setdefault("mode", str(state.get("mode") or "auto"))
    raw_plan.setdefault("domain", str((state.get("image_analysis") or {}).get("domain", "general")))
    raw_plan["executor"] = "hybrid" if operations_require_hybrid(raw_plan.get("operations", [])) else "deterministic"
    return EditPlan.model_validate(raw_plan).model_dump(mode="json")


def _safe_stream_writer():
    """Return a stream writer when inside LangGraph runtime, otherwise a no-op."""

    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda *_args, **_kwargs: None


def execute_deterministic(state: EditState) -> dict:
    """Run deterministic image operations sequentially.

    当前阶段只先把确定性主链跑通：
    1. 按 `edit_plan.operations` 顺序取包；
    2. 每个包基于上一个输出继续处理；
    3. 统一记录执行轨迹，为 evaluate_result 提供事实输入。
    """

    input_images = state.get("input_images") or []
    if not input_images:
        return {
            "candidate_outputs": state.get("candidate_outputs", []),
            "execution_trace": [
                {
                    "stage": "execute_deterministic",
                    "ok": False,
                    "error": "No input image available.",
                },
            ],
            "selected_output": None,
        }

    registry = build_default_package_registry()
    current_round = int(state.get("current_round") or 1)
    round_key = f"round_{current_round}"
    current_image = str(state.get("selected_output") or input_images[0]) if current_round > 1 else input_images[0]
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = list(state.get("execution_trace") or [])
    round_execution_trace: list[dict[str, Any]] = []
    round_outputs = dict(state.get("round_outputs") or {})
    round_execution_traces = dict(state.get("round_execution_traces") or {})
    plan = _normalize_runtime_plan(state)
    expanded_operations = expand_macro_operations(plan.get("operations", []))
    writer = _safe_stream_writer()

    writer(
        {
            "event": "round_started",
            "stage": "execute_deterministic",
            "round": round_key,
            "message": f"开始执行 {round_key}",
        },
    )

    for index, operation in enumerate(expanded_operations):
        package = registry.require(operation["op"])
        writer(
            {
                "event": "package_started",
                "stage": "execute_deterministic",
                "round": round_key,
                "op": operation["op"],
                "region": operation.get("region", "whole_image"),
                "message": f"正在执行 {operation['op']}",
            },
        )
        context = _build_operation_context(state, current_image)
        result = package.execute(operation, context)

        trace_item = ExecutionTraceItem(
            index=index,
            stage=round_key,
            op=operation["op"],
            region=operation.get("region", "whole_image"),
            ok=result.ok,
            fallback_used=result.fallback_used,
            error=result.error,
            output_image=result.output_image,
            applied_params=result.applied_params,
        )
        round_trace_item = trace_item.model_dump(mode="json")
        execution_trace.append(round_trace_item)
        round_execution_trace.append(round_trace_item)

        if result.ok and result.output_image:
            current_image = result.output_image
            candidate_outputs.append(result.output_image)

        writer(
            {
                "event": "package_finished" if result.ok else "package_failed",
                "stage": "execute_deterministic",
                "round": round_key,
                "op": operation["op"],
                "region": operation.get("region", "whole_image"),
                "ok": result.ok,
                "message": f"{operation['op']} 执行完成" if result.ok else f"{operation['op']} 执行失败",
                "error": result.error,
                "warnings": result.warnings,
                "artifacts": result.artifacts,
            },
        )

    round_outputs[round_key] = current_image
    round_execution_traces[round_key] = round_execution_trace
    writer(
        {
            "event": "round_completed",
            "stage": "execute_deterministic",
            "round": round_key,
            "message": f"{round_key} 执行完成",
        },
    )

    return {
        "candidate_outputs": candidate_outputs,
        "execution_trace": execution_trace,
        "selected_output": current_image,
        "round_outputs": round_outputs,
        "round_execution_traces": round_execution_traces,
    }
