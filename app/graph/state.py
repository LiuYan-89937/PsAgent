"""Core state and schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from app.tools import validate_tool_name


class GraphInputState(TypedDict, total=False):
    """Graph entry schema.

    这层约束的是 Graph 边界输入，不直接等于内部完整状态。
    """

    user_id: str
    thread_id: str
    input_images: list[str]
    request_text: str
    mode: str
    planner_thinking_mode: bool
    messages: list[Any]


class GraphOutputState(TypedDict, total=False):
    """Graph exit schema."""

    selected_output: str | None
    candidate_outputs: list[str]
    edit_plan: dict[str, Any]
    eval_report: dict[str, Any]
    execution_trace: list[dict[str, Any]]
    segmentation_trace: list[dict[str, Any]]
    fallback_trace: list[dict[str, Any]]
    round_outputs: dict[str, Any]
    round_plans: dict[str, Any]
    round_eval_reports: dict[str, Any]
    round_execution_traces: dict[str, Any]
    round_segmentation_traces: dict[str, Any]
    approval_required: bool


class ImageQualityMetrics(BaseModel):
    """Deterministic image statistics used as analyzer hints."""

    brightness_mean: float
    brightness_std: float
    shadow_ratio: float
    highlight_ratio: float


class AnalyzeImageResult(BaseModel):
    """Validated image-analysis payload returned by `analyze_image`."""

    source_image: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    orientation: Literal["portrait", "landscape"] | None = None
    domain: Literal["portrait", "landscape", "food", "document", "general"]
    scene_tags: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    segmentation_hints: list[str] = Field(default_factory=list)
    summary: str = ""
    metrics: ImageQualityMetrics | None = None
    model_analysis: dict[str, Any] | None = None


class PackageCatalogItem(BaseModel):
    """Planner-facing package catalog entry."""

    name: str
    description: str
    supported_regions: list[str] = Field(default_factory=list)
    mask_policy: Literal["none", "optional", "required"]
    supported_domains: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]
    params_schema: dict[str, Any] = Field(default_factory=dict)


class RequestPackageHint(BaseModel):
    """A coarse package request parsed from user instruction."""

    op: str
    region: str = "whole_image"
    strength: float | None = Field(default=None, ge=-1.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("op")
    @classmethod
    def _validate_op(cls, value: str) -> str:
        return validate_tool_name(value)


class RequestIntent(BaseModel):
    """Normalized request-intent payload passed from parse_request to planner."""

    mode: Literal["explicit", "auto"]
    requested_packages: list[RequestPackageHint] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    """Validated critic-model output."""

    overall_ok: bool
    preserve_ok: bool
    style_ok: bool
    artifact_ok: bool
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    should_continue_editing: bool = False
    should_request_review: bool = False


class ExecutionTraceItem(BaseModel):
    """Normalized trace item for package execution."""

    index: int | None = None
    stage: str | None = None
    op: str | None = None
    region: str | None = None
    ok: bool
    fallback_used: bool = False
    error: str | None = None
    output_image: str | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    mask_path: str | None = None


class SegmentationTraceItem(BaseModel):
    """Normalized trace item for a single segmentation request."""

    index: int | None = None
    stage: str | None = None
    source_op: str | None = None
    region: str | None = None
    provider: str | None = None
    requested_provider: str | None = None
    target_label: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    semantic_type: bool | None = None
    ok: bool
    fallback_used: bool = False
    error: str | None = None
    mask_path: str | None = None
    request_id: str | None = None
    api_chain: list[str] = Field(default_factory=list)
    attempt_index: int | None = None
    attempt_strategy: str | None = None
    requested_prompt: str | None = None
    effective_prompt: str | None = None
    revert_mask: bool | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)


class FallbackTraceItem(BaseModel):
    """Normalized trace item for a non-fatal fallback decision."""

    index: int | None = None
    stage: str | None = None
    source: str | None = None
    location: str | None = None
    strategy: str | None = None
    message: str = ""
    error: str | None = None
    fallback_used: bool = True


class MemoryWriteCandidate(BaseModel):
    """Normalized long-term memory write candidate."""

    domain: Literal["portrait", "landscape", "food", "document", "general"] = "general"
    key: str
    value: Any
    source: Literal["explicit", "accepted_result", "repeated_behavior", "negative_feedback"] = "accepted_result"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ApprovalPayload(BaseModel):
    """Normalized payload for human review."""

    reason: str = ""
    summary: str = ""
    suggested_action: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Unified evaluation report combining execution facts and critic output."""

    selected_output: str | None = None
    num_operations: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    has_output: bool = False
    overall_ok: bool | None = None
    preserve_ok: bool | None = None
    style_ok: bool | None = None
    artifact_ok: bool | None = None
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    should_continue_editing: bool = False
    should_request_review: bool = False


class EditOperation(BaseModel):
    """A single edit operation in the planner output."""

    # op 对齐工具包唯一标识；region 为 whole_image 或动态局部区域标签；
    # params 是后续推荐给 planner 使用的主填参位置；
    # strength 先保留为兼容字段，避免打断当前工具包测试与旧调用方式。
    op: str
    region: str = "whole_image"
    strength: float | None = Field(default=None, ge=-1.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    priority: int = 0

    @field_validator("op")
    @classmethod
    def _validate_op(cls, value: str) -> str:
        return validate_tool_name(value)


class EditPlan(BaseModel):
    """Structured edit plan produced by the planner."""

    # mode 表示显式修图还是自动修图；
    # executor 决定走哪类执行器；
    # preserve 用于声明必须保留的约束，例如身份、自然感等。
    mode: Literal["explicit", "auto"]
    domain: Literal["portrait", "landscape", "food", "document", "general"]
    executor: Literal["deterministic", "generative", "hybrid"]
    preserve: list[str] = Field(default_factory=list)
    operations: list[EditOperation] = Field(default_factory=list)
    should_write_memory: bool = False
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    needs_confirmation: bool = False


class PreferenceMemory(BaseModel):
    """User preference entry persisted in long-term memory."""

    # 一条长期偏好记录，既可以来自用户显式表达，也可以来自行为证据。
    user_id: str
    domain: Literal["portrait", "landscape", "food", "document", "general"]
    key: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal[
        "explicit",
        "accepted_result",
        "repeated_behavior",
        "negative_feedback",
    ]
    evidence_count: int = 1
    last_updated_at: datetime


class RequestContextState(TypedDict, total=False):
    """Request-scoped inputs and normalized intent."""

    messages: Annotated[list, add_messages]
    user_id: str
    thread_id: str
    input_images: list[str]
    mode: str
    request_text: str | None
    planner_thinking_mode: bool
    request_intent: dict[str, Any] | None


class PlanningArtifactsState(TypedDict, total=False):
    """Planner-visible catalog and plan artifacts."""

    package_catalog: list[dict[str, Any]]
    image_analysis: dict[str, Any] | None
    retrieved_prefs: list[dict[str, Any]]
    edit_plan: dict[str, Any] | None
    current_round: int
    continue_to_round_2: bool
    round_plans: dict[str, Any]
    round_eval_reports: dict[str, Any]


class ExecutionArtifactsState(TypedDict, total=False):
    """Execution-time masks, traces, and round outputs."""

    masks: dict[str, str]
    candidate_outputs: list[str]
    execution_trace: list[dict[str, Any]]
    segmentation_trace: list[dict[str, Any]]
    fallback_trace: list[dict[str, Any]]
    round_outputs: dict[str, Any]
    round_execution_traces: dict[str, Any]
    round_segmentation_traces: dict[str, Any]


class ReviewArtifactsState(TypedDict, total=False):
    """Evaluation, memory, and human review state."""

    eval_report: dict[str, Any] | None
    selected_output: str | None
    memory_write_candidates: list[dict[str, Any]]
    approval_required: bool
    approval_payload: dict[str, Any] | None


class EditState(
    RequestContextState,
    PlanningArtifactsState,
    ExecutionArtifactsState,
    ReviewArtifactsState,
    total=False,
):
    """LangGraph state for an edit thread."""


def coerce_request_intent(value: RequestIntent | dict[str, Any] | None) -> RequestIntent | None:
    """Normalize a request-intent payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, RequestIntent) else RequestIntent.model_validate(value)


def coerce_image_analysis(value: AnalyzeImageResult | dict[str, Any] | None) -> AnalyzeImageResult | None:
    """Normalize an image-analysis payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, AnalyzeImageResult) else AnalyzeImageResult.model_validate(value)


def coerce_edit_plan(value: EditPlan | dict[str, Any] | None) -> EditPlan | None:
    """Normalize an edit-plan payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, EditPlan) else EditPlan.model_validate(value)


def coerce_eval_report(value: EvaluationReport | dict[str, Any] | None) -> EvaluationReport | None:
    """Normalize an evaluation report into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, EvaluationReport) else EvaluationReport.model_validate(value)


def coerce_approval_payload(value: ApprovalPayload | dict[str, Any] | None) -> ApprovalPayload | None:
    """Normalize an approval payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, ApprovalPayload) else ApprovalPayload.model_validate(value)


def coerce_execution_trace(values: list[ExecutionTraceItem | dict[str, Any]] | None) -> list[ExecutionTraceItem]:
    """Normalize execution trace items into typed objects."""

    return [item if isinstance(item, ExecutionTraceItem) else ExecutionTraceItem.model_validate(item) for item in values or []]


def coerce_segmentation_trace(values: list[SegmentationTraceItem | dict[str, Any]] | None) -> list[SegmentationTraceItem]:
    """Normalize segmentation trace items into typed objects."""

    return [item if isinstance(item, SegmentationTraceItem) else SegmentationTraceItem.model_validate(item) for item in values or []]


def coerce_fallback_trace(values: list[FallbackTraceItem | dict[str, Any]] | None) -> list[FallbackTraceItem]:
    """Normalize fallback trace items into typed objects."""

    return [item if isinstance(item, FallbackTraceItem) else FallbackTraceItem.model_validate(item) for item in values or []]


def coerce_memory_write_candidates(
    values: list[MemoryWriteCandidate | dict[str, Any]] | None,
) -> list[MemoryWriteCandidate]:
    """Normalize memory write candidates into typed objects."""

    return [item if isinstance(item, MemoryWriteCandidate) else MemoryWriteCandidate.model_validate(item) for item in values or []]
