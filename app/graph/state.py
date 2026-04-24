"""Core graph state and round-first search schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.tools import validate_tool_name


Domain = Literal["portrait", "landscape", "food", "document", "general"]
EditMode = Literal["explicit", "auto"]
ExecutorKind = Literal["deterministic", "generative", "hybrid"]
FocusKey = Literal["global_tone", "subject_separation", "subject_cleanup", "finish"]
CandidateSource = Literal["model", "rule", "variant", "noop", "direct", "recovery"]
RoundAction = Literal["keep", "recover_same_round", "stop_round"]


class GraphInputState(TypedDict, total=False):
    """Graph entry schema."""

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
    objective_card: ObjectiveCard
    rounds: list[SearchRoundArtifact]
    selected_candidate_id: str | None
    final_review: EvaluationReport | None
    final_execution_trace: list[ExecutionTraceItem]
    approval_required: bool


class ImageQualityMetrics(BaseModel):
    """Deterministic image statistics used as analyzer hints."""

    brightness_mean: float
    brightness_std: float
    shadow_ratio: float
    highlight_ratio: float
    midtone_ratio: float = 0.0
    saturation_mean: float = 0.0
    saturation_std: float = 0.0
    local_contrast_mean: float = 0.0
    dynamic_range: float = 0.0
    color_cast_rgb: dict[str, float] = Field(default_factory=dict)
    exposure_histogram: list[float] = Field(default_factory=list)
    subject_luminance_mean: float | None = None
    background_luminance_mean: float | None = None
    skin_luminance_mean: float | None = None


class AnalyzeImageResult(BaseModel):
    """Validated image-analysis payload returned by `analyze_image`."""

    source_image: str | None = None
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    orientation: Literal["portrait", "landscape"] | None = None
    domain: Domain
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
    focus_affinity: list[str] = Field(default_factory=list)
    supports_mask: bool | None = None
    requires_mask: bool | None = None
    supports_whole_image: bool | None = None
    recommended_mask_prompt: str | None = None
    recommended_mask_prompts: list[str] = Field(default_factory=list)
    selection_guidance: str = ""
    conflict_tools: list[str] = Field(default_factory=list)
    default_params: dict[str, Any] = Field(default_factory=dict)
    planner_schema: dict[str, Any] = Field(default_factory=dict)
    primary_param: str | None = None
    supported_regions: list[str] = Field(default_factory=list)
    mask_policy: Literal["none", "optional", "required"]
    supported_domains: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"]
    params_schema: dict[str, Any] = Field(default_factory=dict)


class RequestToolHint(BaseModel):
    """A coarse tool request parsed from user instruction."""

    op: str
    region: str = "whole_image"
    strength: float | None = Field(default=None, ge=-1.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)

    @field_validator("op")
    @classmethod
    def _validate_op(cls, value: str) -> str:
        return validate_tool_name(value)


class RequestGoal(BaseModel):
    """Goal-level request intent that is not tied to one concrete tool."""

    kind: str
    target_region: str = "whole_image"
    priority: int = Field(default=50, ge=0, le=100)
    intensity: float | None = Field(default=None, ge=-1.0, le=1.0)
    constraints: list[str] = Field(default_factory=list)
    source: Literal["heuristic", "model", "explicit_tool"] = "heuristic"

    @field_validator("source", mode="before")
    @classmethod
    def _normalize_source(cls, value: Any) -> Any:
        if value is None:
            return "heuristic"
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "user": "model",
            "request": "model",
            "llm": "model",
            "ai": "model",
            "parser": "model",
            "rule": "heuristic",
            "rules": "heuristic",
            "explicit": "explicit_tool",
            "tool": "explicit_tool",
            "requested_tool": "explicit_tool",
        }
        return aliases.get(normalized, normalized)


class RequestIntent(BaseModel):
    """Normalized request-intent payload passed from parse_request to the search agent."""

    mode: EditMode
    goals: list[RequestGoal] = Field(default_factory=list)
    requested_tools: list[RequestToolHint] = Field(default_factory=list)
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
    """Normalized trace item for one tool execution."""

    index: int | None = None
    round_id: str | None = None
    focus: FocusKey | None = None
    candidate_id: str | None = None
    op: str | None = None
    region: str | None = None
    ok: bool
    fallback_used: bool = False
    error: str | None = None
    output_image: str | None = None
    output_asset_id: str | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    mask_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class SegmentationTraceItem(BaseModel):
    """Normalized trace item for a single segmentation request."""

    index: int | None = None
    round_id: str | None = None
    focus: FocusKey | None = None
    candidate_id: str | None = None
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
    quality_score: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    rejected: bool = False


class FallbackTraceItem(BaseModel):
    """Normalized trace item for a non-fatal fallback decision."""

    index: int | None = None
    round_id: str | None = None
    focus: FocusKey | None = None
    candidate_id: str | None = None
    source: str | None = None
    location: str | None = None
    strategy: str | None = None
    message: str = ""
    error: str | None = None
    fallback_used: bool = True


class MemoryWriteCandidate(BaseModel):
    """Normalized long-term memory write candidate."""

    domain: Domain = "general"
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
    node: str | None = None
    op: str | None = None
    region: str | None = None
    traceback: str | None = None


class JobEvent(BaseModel):
    """Structured job event used by persistence and API streaming."""

    model_config = ConfigDict(extra="allow")

    event: str
    occurred_at: str | None = None
    round: str | None = None
    focus: FocusKey | None = None
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


class EditOperation(BaseModel):
    """A single edit operation."""

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


class PlannerExecutionStep(EditOperation):
    """A bounded deterministic tool step."""


class EditPlan(BaseModel):
    """Flattened plan summary produced from selected committed candidates."""

    mode: EditMode
    domain: Domain
    executor: ExecutorKind
    preserve: list[str] = Field(default_factory=list)
    operations: list[EditOperation] = Field(default_factory=list)
    should_write_memory: bool = False
    memory_candidates: list[dict[str, Any]] = Field(default_factory=list)
    needs_confirmation: bool = False


class ObjectiveGap(BaseModel):
    """One unresolved search target."""

    id: str
    focus: FocusKey
    description: str
    priority: int = Field(default=50, ge=0, le=100)
    target_region: str = "whole_image"
    desired_delta: str = ""
    constraints: list[str] = Field(default_factory=list)
    resolved: bool = False


class ObjectiveCard(BaseModel):
    """Round-search objective distilled from request intent and image facts."""

    summary: str = ""
    mode: EditMode = "auto"
    domain: Domain = "general"
    preserve: list[str] = Field(default_factory=list)
    goals: list[RequestGoal] = Field(default_factory=list)
    gaps: list[ObjectiveGap] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class CandidateProgram(BaseModel):
    """A candidate tool chain proposed for one round."""

    id: str
    label: str = ""
    focus: FocusKey
    source: CandidateSource = "variant"
    summary: str = ""
    steps: list[PlannerExecutionStep] = Field(default_factory=list)
    is_recovery: bool = False


class CandidatePreviewExecution(BaseModel):
    """Preview execution facts for one candidate or committed full-res execution."""

    input_image_path: str | None = None
    output_image_path: str | None = None
    output_asset_id: str | None = None
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    fallback_trace: list[FallbackTraceItem] = Field(default_factory=list)


class CandidateReview(BaseModel):
    """Structured review result for one candidate."""

    overall_ok: bool = True
    preserve_ok: bool = True
    style_ok: bool = True
    artifact_ok: bool = True
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    recommended_action: RoundAction = "keep"
    score: float = 0.0


class RoundReview(BaseModel):
    """Review result for a committed round."""

    overall_ok: bool = True
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""
    recommended_action: RoundAction = "keep"
    score: float = 0.0


class RecoveryDecision(BaseModel):
    """Decision metadata for one same-round recovery search."""

    triggered: bool = False
    source: Literal["candidate_review", "round_review", "deterministic", "none"] = "none"
    fallback_aware: bool = False
    reason: str = ""
    candidate_ids: list[str] = Field(default_factory=list)
    selected_candidate_id: str | None = None


class SearchCandidateArtifact(BaseModel):
    """Candidate artifact stored under a search round."""

    candidate_id: str
    label: str = ""
    focus: FocusKey
    selected: bool = False
    eliminated_reason: str | None = None
    program: CandidateProgram | None = None
    preview_execution: CandidatePreviewExecution | None = None
    review: CandidateReview | None = None


class SearchRoundArtifact(BaseModel):
    """All artifacts for one search round."""

    id: str
    index: int
    focus: FocusKey
    input_image_path: str | None = None
    output_image_path: str | None = None
    output_asset_id: str | None = None
    objective_gaps: list[ObjectiveGap] = Field(default_factory=list)
    candidates: list[SearchCandidateArtifact] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    selected_full_execution: CandidatePreviewExecution | None = None
    round_review: RoundReview | None = None
    recovery_decision: RecoveryDecision | None = None
    recovery_candidates: list[SearchCandidateArtifact] = Field(default_factory=list)
    completed: bool = False


class SearchRunArtifact(BaseModel):
    """Complete round-first search run."""

    objective_card: ObjectiveCard | None = None
    rounds: list[SearchRoundArtifact] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    final_execution_trace: list[ExecutionTraceItem] = Field(default_factory=list)


class MaskQuality(BaseModel):
    """Deterministic quality facts for a generated or reused mask."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    area_ratio: float = 0.0
    bbox: dict[str, int] = Field(default_factory=dict)
    connected_components: int = 0
    edge_density: float = 0.0
    flags: list[str] = Field(default_factory=list)
    rejected: bool = False


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
    source_focus: FocusKey | None = None
    source_op: str | None = None
    region_labels: list[str] = Field(default_factory=list)
    reuse_count: int = 0
    quality: MaskQuality | None = None
    quality_score: float | None = None
    quality_flags: list[str] = Field(default_factory=list)
    rejected: bool = False


class MaskCatalog(BaseModel):
    """Shared reusable mask store across full-resolution rounds."""

    items: dict[str, MaskCatalogItem] = Field(default_factory=dict)


class PreferenceMemory(BaseModel):
    """User preference entry persisted in long-term memory."""

    user_id: str
    domain: Domain
    key: str
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["explicit", "accepted_result", "repeated_behavior", "negative_feedback"]
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
    """Planner-visible catalog and search artifacts."""

    tool_catalog: list[ToolCatalogItem]
    image_analysis: AnalyzeImageResult | None
    retrieved_prefs: list[PreferenceMemory]
    objective_card: ObjectiveCard | None
    search_run: SearchRunArtifact | None
    rounds: list[SearchRoundArtifact]
    selected_candidate_id: str | None
    current_round: str | None
    current_focus: FocusKey | None
    edit_plan: EditPlan | None
    mask_catalog: MaskCatalog | None


class ExecutionArtifactsState(TypedDict, total=False):
    """Execution-time masks, traces, and outputs."""

    candidate_outputs: list[str]
    execution_trace: list[ExecutionTraceItem]
    final_execution_trace: list[ExecutionTraceItem]
    segmentation_trace: list[SegmentationTraceItem]
    fallback_trace: list[FallbackTraceItem]


class ReviewArtifactsState(TypedDict, total=False):
    """Evaluation, memory, and human review state."""

    eval_report: EvaluationReport | None
    final_review: EvaluationReport | None
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
    if value is None:
        return None
    return value if isinstance(value, RequestIntent) else RequestIntent.model_validate(value)


def coerce_image_analysis(value: AnalyzeImageResult | dict[str, Any] | None) -> AnalyzeImageResult | None:
    if value is None:
        return None
    return value if isinstance(value, AnalyzeImageResult) else AnalyzeImageResult.model_validate(value)


def coerce_edit_plan(value: EditPlan | dict[str, Any] | None) -> EditPlan | None:
    if value is None:
        return None
    return value if isinstance(value, EditPlan) else EditPlan.model_validate(value)


def coerce_eval_report(value: EvaluationReport | dict[str, Any] | None) -> EvaluationReport | None:
    if value is None:
        return None
    return value if isinstance(value, EvaluationReport) else EvaluationReport.model_validate(value)


def coerce_objective_card(value: ObjectiveCard | dict[str, Any] | None) -> ObjectiveCard | None:
    if value is None:
        return None
    return value if isinstance(value, ObjectiveCard) else ObjectiveCard.model_validate(value)


def coerce_mask_catalog(value: MaskCatalog | dict[str, Any] | None) -> MaskCatalog:
    if value is None:
        return MaskCatalog()
    return value if isinstance(value, MaskCatalog) else MaskCatalog.model_validate(value)


def coerce_search_rounds(values: list[SearchRoundArtifact | dict[str, Any]] | None) -> list[SearchRoundArtifact]:
    return [item if isinstance(item, SearchRoundArtifact) else SearchRoundArtifact.model_validate(item) for item in values or []]


def coerce_search_run(value: SearchRunArtifact | dict[str, Any] | None) -> SearchRunArtifact | None:
    if value is None:
        return None
    return value if isinstance(value, SearchRunArtifact) else SearchRunArtifact.model_validate(value)


def coerce_execution_trace(values: list[ExecutionTraceItem | dict[str, Any]] | None) -> list[ExecutionTraceItem]:
    return [item if isinstance(item, ExecutionTraceItem) else ExecutionTraceItem.model_validate(item) for item in values or []]


def coerce_segmentation_trace(values: list[SegmentationTraceItem | dict[str, Any]] | None) -> list[SegmentationTraceItem]:
    return [item if isinstance(item, SegmentationTraceItem) else SegmentationTraceItem.model_validate(item) for item in values or []]


def coerce_fallback_trace(values: list[FallbackTraceItem | dict[str, Any]] | None) -> list[FallbackTraceItem]:
    return [item if isinstance(item, FallbackTraceItem) else FallbackTraceItem.model_validate(item) for item in values or []]


def coerce_memory_write_candidates(values: list[MemoryWriteCandidate | dict[str, Any]] | None) -> list[MemoryWriteCandidate]:
    return [item if isinstance(item, MemoryWriteCandidate) else MemoryWriteCandidate.model_validate(item) for item in values or []]


def coerce_tool_catalog(values: list[ToolCatalogItem | dict[str, Any]] | None) -> list[ToolCatalogItem]:
    return [item if isinstance(item, ToolCatalogItem) else ToolCatalogItem.model_validate(item) for item in values or []]


def coerce_preferences(values: list[PreferenceMemory | dict[str, Any]] | None) -> list[PreferenceMemory]:
    return [item if isinstance(item, PreferenceMemory) else PreferenceMemory.model_validate(item) for item in values or []]


def coerce_approval_payload(value: ApprovalPayload | dict[str, Any] | None) -> ApprovalPayload | None:
    if value is None:
        return None
    return value if isinstance(value, ApprovalPayload) else ApprovalPayload.model_validate(value)


def coerce_job_events(values: list[JobEvent | dict[str, Any]] | None) -> list[JobEvent]:
    return [item if isinstance(item, JobEvent) else JobEvent.model_validate(item) for item in values or []]


def coerce_feedback_items(values: list[FeedbackItem | dict[str, Any]] | None) -> list[FeedbackItem]:
    return [item if isinstance(item, FeedbackItem) else FeedbackItem.model_validate(item) for item in values or []]


def coerce_error_detail(value: ErrorDetail | dict[str, Any] | None) -> ErrorDetail | None:
    if value is None:
        return None
    return value if isinstance(value, ErrorDetail) else ErrorDetail.model_validate(value)
