"""Core state and schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    edit_plan: EditPlan
    eval_report: EvaluationReport
    execution_trace: list[ExecutionTraceItem]
    segmentation_trace: list[SegmentationTraceItem]
    fallback_trace: list[FallbackTraceItem]
    phases: dict[str, PhaseArtifact]
    approval_required: bool


StageKey = Literal[
    "technical_prep",
    "global_base",
    "local_balance",
    "subject_refine",
    "finish_output",
]


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
    main_issues: list[str] = Field(default_factory=list)
    primary_subject: str | None = None
    main_subject_type: Literal["human", "object", "scene", "mixed", "unknown"] | None = None
    subject_count: Literal["single", "multiple", "unknown"] | None = None
    subject_capabilities: dict[str, bool] = Field(default_factory=dict)
    has_portrait: bool | None = None
    needs_local_editing: bool | None = None
    has_background_distraction: bool | None = None
    summary: str = ""
    metrics: ImageQualityMetrics | None = None
    model_analysis: dict[str, Any] | None = None


class ToolCatalogItem(BaseModel):
    """Planner-facing catalog entry sourced from the native tool registry."""

    name: str
    label: str | None = None
    description: str
    family: str | None = None
    stage_affinity: list[str] = Field(default_factory=list)
    supports_mask: bool | None = None
    supports_whole_image: bool | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    planner_schema: dict[str, Any] = Field(default_factory=dict)
    primary_param: str | None = None
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
    goal_summary: str = ""
    wants_repair: bool = False
    wants_style: bool = False
    requires_local_editing: bool = False


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
    preview_path: str | None = None
    mask_asset_id: str | None = None
    preview_asset_id: str | None = None
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


class ErrorDetail(BaseModel):
    """Structured error detail payload used by API and job persistence."""

    model_config = ConfigDict(extra="allow")

    type: str = ""
    message: str = ""
    stage: str | None = None
    node: str | None = None
    op: str | None = None
    region: str | None = None
    traceback: str | None = None


class JobEvent(BaseModel):
    """Structured job event used by persistence and API streaming."""

    model_config = ConfigDict(extra="allow")

    event: str
    occurred_at: str | None = None
    stage: str | None = None
    round: str | None = None
    node: str | None = None
    op: str | None = None
    region: str | None = None
    provider: str | None = None
    requested_provider: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    target_label: str | None = None
    message: str | None = None
    job_id: str | None = None
    ok: bool | None = None
    error: str | None = None
    error_detail: ErrorDetail | None = None
    payload: dict[str, Any] | None = None
    approval_payload: ApprovalPayload | None = None
    interrupt_id: str | None = None


class FeedbackItem(BaseModel):
    """Stored user feedback entry."""

    accepted: bool
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_text: str | None = None
    manual_adjustments: dict[str, Any] = Field(default_factory=dict)


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


class SubjectCapabilities(BaseModel):
    """Capabilities describing which subject-specific refinements are safe to expose."""

    face_visible: bool = False
    skin_visible: bool = False
    hair_visible: bool = False
    eyes_visible: bool = False
    teeth_visible: bool = False
    lips_visible: bool = False


class EditProfile(BaseModel):
    """Profile used to decide stage activation, subject tools, and context strategy."""

    main_subject_type: Literal["human", "object", "scene", "mixed", "unknown"] = "unknown"
    subject_count: Literal["single", "multiple", "unknown"] = "unknown"
    technical_issues: list[str] = Field(default_factory=list)
    global_tone_issues: list[str] = Field(default_factory=list)
    local_balance_needed: bool = False
    subject_refine_needed: bool = False
    finish_needed: bool = True
    subject_capabilities: SubjectCapabilities = Field(default_factory=SubjectCapabilities)


class MaskCatalogItem(BaseModel):
    """Reusable mask record keyed by a deterministic mask signature."""

    signature: str
    provider: str
    mask_prompt: str
    normalized_mask_prompt: str
    semantic_type: bool = False
    revert_mask: bool = False
    mask_path: str | None = None
    preview_path: str | None = None
    mask_asset_id: str | None = None
    preview_asset_id: str | None = None
    source_stage: StageKey | None = None
    source_op: str | None = None
    region_labels: list[str] = Field(default_factory=list)
    reuse_count: int = 0


class MaskCatalog(BaseModel):
    """Shared reusable mask store across stages."""

    items: dict[str, MaskCatalogItem] = Field(default_factory=dict)


class StagePolicy(BaseModel):
    """Stage-level configuration for tool visibility, context, and guard behavior."""

    key: StageKey
    label: str
    prompt_name: str
    visible_tools: list[str] = Field(default_factory=list)
    llm_enabled: bool = True
    step_budget: int = Field(default=2, ge=0, le=8)
    tool_repeat_limit: int = Field(default=2, ge=1, le=8)
    tone_stack_limit: int | None = Field(default=None, ge=1, le=8)
    mask_allowed: bool = False
    mask_required: bool = False
    context_whitelist: list[str] = Field(default_factory=list)
    guard_thresholds: dict[str, float] = Field(default_factory=dict)


class StageContextEnvelope(BaseModel):
    """Minimal stage-specific context passed to the planner."""

    request_summary: str = ""
    current_image_path: str | None = None
    edit_profile_summary: dict[str, Any] = Field(default_factory=dict)
    relevant_image_analysis: dict[str, Any] = Field(default_factory=dict)
    available_masks: list[dict[str, Any]] = Field(default_factory=list)
    previous_stage_summaries: list[dict[str, Any]] = Field(default_factory=list)
    stage_constraints: list[str] = Field(default_factory=list)


class StageSummary(BaseModel):
    """Compact summary passed between adjacent stages."""

    stage: StageKey
    summary: str = ""
    used_tools: list[str] = Field(default_factory=list)
    key_changes: list[str] = Field(default_factory=list)
    remaining_issues: list[str] = Field(default_factory=list)


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


class PlannerExecutionStep(EditOperation):
    """A bounded execution step produced by the single-shot planner."""


class PlannerExecutionPlan(BaseModel):
    """Single-shot execution plan for one phase."""

    mode: Literal["explicit", "auto"]
    domain: Literal["portrait", "landscape", "food", "document", "general"]
    executor: Literal["deterministic", "generative", "hybrid"]
    preserve: list[str] = Field(default_factory=list)
    steps: list[PlannerExecutionStep] = Field(default_factory=list)
    step_budget: int = Field(default=4, ge=1, le=8)
    summary: str = ""
    should_write_memory: bool = False
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    needs_confirmation: bool = False


class PhaseOutputArtifact(BaseModel):
    """Selected output for one execution phase."""

    image_path: str | None = None
    asset_id: str | None = None


class PhaseArtifact(BaseModel):
    """All planner, execution, segmentation, and evaluation artifacts for one phase."""

    key: StageKey
    label: str = ""
    plan: PlannerExecutionPlan | None = None
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    eval_report: EvaluationReport | None = None
    output: PhaseOutputArtifact | None = None
    summary: StageSummary | None = None
    skipped: bool = False
    skip_reason: str | None = None
    trigger_reasons: list[str] = Field(default_factory=list)
    stopped_early: bool = False


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
    request_intent: RequestIntent | None


class PlanningArtifactsState(TypedDict, total=False):
    """Planner-visible catalog and plan artifacts."""

    tool_catalog: list[ToolCatalogItem]
    image_analysis: AnalyzeImageResult | None
    retrieved_prefs: list[PreferenceMemory]
    edit_profile: EditProfile | None
    edit_plan: EditPlan | None
    current_stage: StageKey | None
    stage_policy: StagePolicy | None
    stage_context: StageContextEnvelope | None
    stage_plan: PlannerExecutionPlan | None
    mask_catalog: MaskCatalog | None
    phases: dict[str, PhaseArtifact]


class ExecutionArtifactsState(TypedDict, total=False):
    """Execution-time masks, traces, and round outputs."""

    candidate_outputs: list[str]
    execution_trace: list[ExecutionTraceItem]
    segmentation_trace: list[SegmentationTraceItem]
    fallback_trace: list[FallbackTraceItem]


class ReviewArtifactsState(TypedDict, total=False):
    """Evaluation, memory, and human review state."""

    eval_report: EvaluationReport | None
    selected_output: str | None
    memory_write_candidates: list[MemoryWriteCandidate]
    approval_required: bool
    approval_payload: ApprovalPayload | None


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


def coerce_planner_execution_plan(
    value: PlannerExecutionPlan | dict[str, Any] | None,
) -> PlannerExecutionPlan | None:
    """Normalize a single-shot planner execution plan into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, PlannerExecutionPlan) else PlannerExecutionPlan.model_validate(value)


def coerce_eval_report(value: EvaluationReport | dict[str, Any] | None) -> EvaluationReport | None:
    """Normalize an evaluation report into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, EvaluationReport) else EvaluationReport.model_validate(value)


def coerce_edit_profile(value: EditProfile | dict[str, Any] | None) -> EditProfile | None:
    """Normalize an edit profile into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, EditProfile) else EditProfile.model_validate(value)


def coerce_approval_payload(value: ApprovalPayload | dict[str, Any] | None) -> ApprovalPayload | None:
    """Normalize an approval payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, ApprovalPayload) else ApprovalPayload.model_validate(value)


def coerce_mask_catalog(value: MaskCatalog | dict[str, Any] | None) -> MaskCatalog:
    """Normalize the shared mask catalog into a typed object."""

    if value is None:
        return MaskCatalog()
    return value if isinstance(value, MaskCatalog) else MaskCatalog.model_validate(value)


def coerce_stage_policy(value: StagePolicy | dict[str, Any] | None) -> StagePolicy | None:
    """Normalize a stage policy into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, StagePolicy) else StagePolicy.model_validate(value)


def coerce_stage_context(value: StageContextEnvelope | dict[str, Any] | None) -> StageContextEnvelope | None:
    """Normalize a stage context envelope into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, StageContextEnvelope) else StageContextEnvelope.model_validate(value)


def coerce_stage_summary(value: StageSummary | dict[str, Any] | None) -> StageSummary | None:
    """Normalize a stage summary into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, StageSummary) else StageSummary.model_validate(value)


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


def coerce_tool_catalog(
    values: list[ToolCatalogItem | dict[str, Any]] | None,
) -> list[ToolCatalogItem]:
    """Normalize tool catalog items into typed objects."""

    return [item if isinstance(item, ToolCatalogItem) else ToolCatalogItem.model_validate(item) for item in values or []]


def coerce_preferences(
    values: list[PreferenceMemory | dict[str, Any]] | None,
) -> list[PreferenceMemory]:
    """Normalize retrieved preferences into typed objects."""

    return [item if isinstance(item, PreferenceMemory) else PreferenceMemory.model_validate(item) for item in values or []]


def coerce_phase_artifacts(
    values: dict[str, PhaseArtifact | dict[str, Any]] | None,
) -> dict[str, PhaseArtifact]:
    """Normalize grouped phase artifacts into typed objects."""

    return {
        key: value if isinstance(value, PhaseArtifact) else PhaseArtifact.model_validate(value)
        for key, value in (values or {}).items()
    }


def coerce_job_events(values: list[JobEvent | dict[str, Any]] | None) -> list[JobEvent]:
    """Normalize job events into typed objects."""

    return [item if isinstance(item, JobEvent) else JobEvent.model_validate(item) for item in values or []]


def coerce_feedback_items(values: list[FeedbackItem | dict[str, Any]] | None) -> list[FeedbackItem]:
    """Normalize feedback items into typed objects."""

    return [item if isinstance(item, FeedbackItem) else FeedbackItem.model_validate(item) for item in values or []]


def coerce_error_detail(value: ErrorDetail | dict[str, Any] | None) -> ErrorDetail | None:
    """Normalize an error detail payload into a typed object."""

    if value is None:
        return None
    return value if isinstance(value, ErrorDetail) else ErrorDetail.model_validate(value)
