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
    JobEvent,
    PlannerExecutionPlan,
    SegmentationTraceItem,
    StageSummary,
)


JobStatus = Literal["pending", "running", "completed", "failed", "review_required"]


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
    """Edit entry request.

    前端可以先走两种输入模式：
    1. `input_asset_ids`
    2. `input_image_paths`
    当前优先推荐前端使用上传后的 `asset_id`。
    """

    user_id: str
    thread_id: str | None = None
    instruction: str | None = None
    auto_mode: bool = False
    planner_thinking_mode: bool = False
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
    current_stage: str | None = None
    current_message: str | None = None
    error: str | None = None
    error_detail: ErrorDetail | None = None


class StageTimingResponse(BaseModel):
    """Frontend-facing stage timing summary."""

    stage: str
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
    stage: str | None = None
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


class PhaseResponse(BaseModel):
    """Frontend-facing grouped phase payload."""

    plan: PlannerExecutionPlan | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    eval_report: EvaluationReport | None = None
    output: AssetResponse | None = None
    summary: StageSummary | None = None
    skipped: bool = False
    skip_reason: str | None = None
    trigger_reasons: list[str] = Field(default_factory=list)
    stopped_early: bool = False


class EditResponse(BaseModel):
    """Edit route response."""

    job: JobSummaryResponse
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: EditPlan | None = None
    eval_report: EvaluationReport | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    fallback_trace: list[FallbackTraceItem] = Field(default_factory=list)
    phases: dict[str, PhaseResponse] = Field(default_factory=dict)
    events: list[JobEvent] = Field(default_factory=list)
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)


class JobDetailResponse(BaseModel):
    """Detailed job response for frontend polling."""

    job: JobSummaryResponse
    input_assets: list[AssetResponse] = Field(default_factory=list)
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: EditPlan | None = None
    eval_report: EvaluationReport | None = None
    execution_trace: list[ExecutionTraceResponse] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    fallback_trace: list[FallbackTraceItem] = Field(default_factory=list)
    phases: dict[str, PhaseResponse] = Field(default_factory=dict)
    events: list[JobEvent] = Field(default_factory=list)
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)
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


class ResumeReviewResponse(BaseModel):
    """Resume-review response.

    当前只是把接口契约和状态承载先定下来，真正的 interrupt/resume
    还要等审核链路正式接入。
    """

    job_id: str
    accepted: bool
    implemented: bool = True
    status: JobStatus
    message: str


class ToolCatalogResponse(BaseModel):
    """Planner-facing tool catalog response."""

    items: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health payload."""

    ok: bool = True
    service: str = "PsAgent"
