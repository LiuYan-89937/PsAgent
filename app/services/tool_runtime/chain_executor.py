"""Sequential candidate-chain execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.graph.state import (
    CandidatePreviewExecution,
    CandidateProgram,
    FocusKey,
    MaskCatalog,
    PlannerExecutionStep,
    coerce_execution_trace,
    coerce_segmentation_trace,
)
from app.services.tool_runtime.single_tool_executor import execute_single_tool_call


@dataclass(slots=True)
class ChainExecutionResult:
    """Runtime result for a candidate chain."""

    input_image_path: str
    output_image_path: str
    execution_trace: list[dict[str, Any]]
    segmentation_trace: list[dict[str, Any]]
    fallback_trace: list[dict[str, Any]]
    candidate_outputs: list[str]
    mask_catalog: MaskCatalog

    def to_candidate_execution(self) -> CandidatePreviewExecution:
        return CandidatePreviewExecution(
            input_image_path=self.input_image_path,
            output_image_path=self.output_image_path,
            execution_trace=coerce_execution_trace(self.execution_trace),
            segmentation_trace=coerce_segmentation_trace(self.segmentation_trace),
            fallback_trace=self.fallback_trace,
        )


@dataclass(slots=True)
class ToolLabRuntimeStep:
    """One ToolLab step result with before/after paths."""

    index: int
    tool_name: str
    input_image_path: str
    output_image_path: str | None
    mask_path: str | None
    ok: bool
    applied_params: dict[str, Any]
    warnings: list[str]
    artifacts: dict[str, Any]
    fallback_used: bool
    error: str | None


def _step_payload(step: PlannerExecutionStep | dict[str, Any]) -> dict[str, Any]:
    return step.model_dump(mode="json") if hasattr(step, "model_dump") else dict(step)


def execute_chain(
    *,
    input_image_path: str,
    program: CandidateProgram,
    mask_catalog: MaskCatalog | None = None,
    writer=None,
    mode: str = "auto",
    round_id: str | None = None,
    focus: FocusKey | None = None,
    candidate_id: str | None = None,
    max_steps: int | None = None,
) -> ChainExecutionResult:
    """Execute a candidate program sequentially."""

    writer = writer or (lambda *_args, **_kwargs: None)
    current_image = input_image_path
    execution_trace: list[dict[str, Any]] = []
    segmentation_trace: list[dict[str, Any]] = []
    fallback_trace: list[dict[str, Any]] = []
    candidate_outputs: list[str] = []
    runtime_catalog = mask_catalog.model_copy(deep=True) if mask_catalog is not None else MaskCatalog()
    steps = list(program.steps[:max_steps] if max_steps is not None else program.steps)

    for step in steps:
        current_image, _, runtime_catalog = execute_single_tool_call(
            current_image=current_image,
            operation=_step_payload(step),
            execution_trace=execution_trace,
            segmentation_trace=segmentation_trace,
            fallback_trace=fallback_trace,
            candidate_outputs=candidate_outputs,
            mask_catalog=runtime_catalog,
            writer=writer,
            round_id=round_id,
            focus=focus or program.focus,
            candidate_id=candidate_id or program.id,
            mode=mode,
        )

    return ChainExecutionResult(
        input_image_path=input_image_path,
        output_image_path=current_image,
        execution_trace=execution_trace,
        segmentation_trace=segmentation_trace,
        fallback_trace=fallback_trace,
        candidate_outputs=candidate_outputs,
        mask_catalog=runtime_catalog,
    )


def execute_tool_lab_chain(
    *,
    input_image_path: str,
    steps: list[dict[str, Any]],
    writer=None,
) -> tuple[str, list[ToolLabRuntimeStep]]:
    """Run ToolLab's direct chain while still using the shared single-tool runtime."""

    writer = writer or (lambda *_args, **_kwargs: None)
    current_image = input_image_path
    mask_catalog = MaskCatalog()
    results: list[ToolLabRuntimeStep] = []
    for index, step in enumerate(steps):
        before = current_image
        execution_trace: list[dict[str, Any]] = []
        segmentation_trace: list[dict[str, Any]] = []
        fallback_trace: list[dict[str, Any]] = []
        candidate_outputs: list[str] = []
        current_image, _, mask_catalog = execute_single_tool_call(
            current_image=current_image,
            operation=step,
            execution_trace=execution_trace,
            segmentation_trace=segmentation_trace,
            fallback_trace=fallback_trace,
            candidate_outputs=candidate_outputs,
            mask_catalog=mask_catalog,
            writer=writer,
            round_id="tool_lab",
            focus="finish",
            candidate_id="tool_lab",
            mode="explicit",
        )
        trace_item = execution_trace[-1] if execution_trace else {}
        ok = bool(trace_item.get("ok"))
        output_path = str(trace_item.get("output_image") or current_image) if ok else None
        results.append(
            ToolLabRuntimeStep(
                index=index,
                tool_name=str(step.get("op") or step.get("tool_name") or ""),
                input_image_path=before,
                output_image_path=output_path,
                mask_path=str(trace_item.get("mask_path")) if trace_item.get("mask_path") else None,
                ok=ok,
                applied_params=dict(trace_item.get("applied_params") or {}),
                warnings=list(trace_item.get("warnings") or []),
                artifacts=dict(trace_item.get("artifacts") or {}),
                fallback_used=bool(trace_item.get("fallback_used")),
                error=trace_item.get("error") if isinstance(trace_item.get("error"), str) else None,
            )
        )
        if not ok:
            break
    return current_image, results
