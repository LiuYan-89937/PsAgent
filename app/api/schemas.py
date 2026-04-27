"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.graph.state import (
    EditPlan,
    ErrorDetail,
    EvaluationReport,
    FallbackTraceItem,
    FeedbackItem,
    FocusKey,
    JobEvent,
    ObjectiveCard,
    RecoveryDecision,
    RoundGuidance,
    RoundReview,
    SearchCandidateArtifact,
    SearchRoundArtifact,
    SegmentationTraceItem,
)


JobStatus = Literal["pending", "running", "completed", "failed", "review_required"]
SearchEffort = Literal["standard", "high", "ultra"]


class AssetResponse(BaseModel):
    """Frontend-facing asset payload."""

    asset_id: str
    filename: str
    media_type: str | None = None
    size_bytes: int | None = None
    created_at: datetime
    content_url: str


class UploadAssetsResponse(BaseModel):
    """Asset upload response."""

    items: list[AssetResponse] = Field(default_factory=list)


class EditRequest(BaseModel):
    """Edit entry request."""

    user_id: str
    thread_id: str | None = None
    instruction: str | None = None
    planner_thinking_mode: bool = False
    search_effort: SearchEffort = "standard"
    input_asset_ids: list[str] = Field(default_factory=list)
    input_image_paths: list[str] = Field(default_factory=list)


class JobSummaryResponse(BaseModel):
    """Reusable job summary."""

    job_id: str
    status: JobStatus
    user_id: str
    thread_id: str
    created_at: datetime
    updated_at: datetime
    approval_required: bool = False
    request_text: str | None = None
    current_round: str | None = None
    current_focus: FocusKey | None = None
    current_message: str | None = None
    error: str | None = None
    error_detail: ErrorDetail | None = None


class RoundTimingResponse(BaseModel):
    """Frontend-facing round timing summary."""

    round: str
    focus: FocusKey | None = None
    label: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    duration_seconds: float
    status: Literal["completed", "failed"]


class ExecutionTraceResponse(BaseModel):
    """Frontend-facing execution trace item."""

    model_config = ConfigDict(extra="allow")

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
    output_asset: AssetResponse | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    mask_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class CandidateExecutionResponse(BaseModel):
    """Frontend-facing execution result for one candidate."""

    input_image_path: str | None = None
    output_image_path: str | None = None
    output_asset_id: str | None = None
    output_asset: AssetResponse | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = Field(default_factory=list)


class SearchCandidateResponse(BaseModel):
    """Frontend-facing candidate search artifact."""

    candidate_id: str
    label: str = ""
    focus: FocusKey
    selected: bool = False
    eliminated_reason: str | None = None
    program: dict[str, Any] | None = None
    preview_execution: CandidateExecutionResponse | None = None
    review: dict[str, Any] | None = None


class SearchRoundResponse(BaseModel):
    """Frontend-facing round artifact."""

    id: str
    index: int
    focus: FocusKey
    input_image_path: str | None = None
    output_image_path: str | None = None
    output_asset_id: str | None = None
    output_asset: AssetResponse | None = None
    objective_gaps: list[dict[str, Any]] = Field(default_factory=list)
    guidance: RoundGuidance | None = None
    candidates: list[SearchCandidateResponse] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    selected_full_execution: CandidateExecutionResponse | None = None
    round_review: RoundReview | None = None
    recovery_decision: RecoveryDecision | None = None
    recovery_candidates: list[SearchCandidateResponse] = Field(default_factory=list)
    completed: bool = False


class EditResponse(BaseModel):
    """Edit route response."""

    job: JobSummaryResponse
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: EditPlan | None = None
    eval_report: EvaluationReport | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = Field(default_factory=list)
    objective_card: ObjectiveCard | None = None
    rounds: list[SearchRoundResponse] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    final_review: EvaluationReport | None = None
    final_execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    events: list[JobEvent] = Field(default_factory=list)
    round_timings: list[RoundTimingResponse] = Field(default_factory=list)


class JobDetailResponse(BaseModel):
    """Detailed job response for frontend polling."""

    job: JobSummaryResponse
    input_assets: list[AssetResponse] = Field(default_factory=list)
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: EditPlan | None = None
    eval_report: EvaluationReport | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = Field(default_factory=list)
    objective_card: ObjectiveCard | None = None
    rounds: list[SearchRoundResponse] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    final_review: EvaluationReport | None = None
    final_execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    events: list[JobEvent] = Field(default_factory=list)
    round_timings: list[RoundTimingResponse] = Field(default_factory=list)
    feedback: list[FeedbackItem] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Feedback submission request."""

    job_id: str
    accepted: bool
    rating: int | None = Field(default=None, ge=1, le=5)
    feedback_text: str | None = None
    manual_adjustments: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Feedback acknowledgement."""

    job_id: str
    saved: bool = True
    feedback_count: int = 0


class ResumeReviewRequest(BaseModel):
    """Resume-review request."""

    job_id: str
    approved: bool
    note: str | None = None
    search_effort: SearchEffort | None = None


class ResumeReviewResponse(BaseModel):
    """Resume-review response."""

    job_id: str
    accepted: bool
    implemented: bool = True
    status: JobStatus
    message: str


class ToolCatalogItemResponse(BaseModel):
    """Planner-facing tool catalog item."""

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


class ToolCatalogResponse(BaseModel):
    """Planner-facing tool catalog response."""

    items: list[ToolCatalogItemResponse] = Field(default_factory=list)


class ToolLabMaskRequest(BaseModel):
    """Request payload for generating a debug/test mask from one uploaded image."""

    input_asset_id: str
    prompt: str = Field(min_length=1, max_length=120)
    provider: Literal["auto", "aliyun", "fal_sam3"] = "fal_sam3"


class ToolLabMaskResponse(BaseModel):
    """Response payload for one generated tool-lab mask."""

    mask_asset: AssetResponse
    preview_asset: AssetResponse | None = None
    provider: str
    requested_provider: str | None = None
    prompt: str
    effective_prompt: str | None = None
    fallback_used: bool = False
    attempt_strategy: str | None = None
    attempt_index: int | None = None
    target_label: str | None = None
    revert_mask: bool | None = None


class ToolLabStepRequest(BaseModel):
    """One sequential deterministic tool step configured from the frontend."""

    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    mask_asset_id: str | None = None


class ToolLabStepResultResponse(BaseModel):
    """One executed tool-lab step with before/after assets for comparison."""

    index: int
    tool_name: str
    ok: bool
    input_asset: AssetResponse
    output_asset: AssetResponse | None = None
    mask_asset: AssetResponse | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    error: str | None = None


class ToolLabRunRequest(BaseModel):
    """Request payload for sequentially running a custom tool chain."""

    input_asset_id: str
    steps: list[ToolLabStepRequest] = Field(default_factory=list)


class ToolLabRunResponse(BaseModel):
    """Response payload for one completed tool-lab execution."""

    input_asset: AssetResponse
    final_output_asset: AssetResponse
    steps: list[ToolLabStepResultResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health response."""

    ok: bool = True
