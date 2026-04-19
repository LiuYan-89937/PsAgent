"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    error_detail: dict[str, Any] | None = None


class StageTimingResponse(BaseModel):
    """Frontend-facing stage timing summary."""

    stage: str
    label: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    duration_seconds: float
    status: Literal["completed", "failed"]


class EditResponse(BaseModel):
    """Edit route response."""

    job: JobSummaryResponse
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: dict[str, Any] | None = None
    eval_report: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = Field(default_factory=list)
    round_outputs: dict[str, AssetResponse | None] = Field(default_factory=dict)
    round_plans: dict[str, Any] = Field(default_factory=dict)
    round_eval_reports: dict[str, Any] = Field(default_factory=dict)
    round_execution_traces: dict[str, Any] = Field(default_factory=dict)
    round_segmentation_traces: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)


class JobDetailResponse(BaseModel):
    """Detailed job response for frontend polling."""

    job: JobSummaryResponse
    input_assets: list[AssetResponse] = Field(default_factory=list)
    selected_output: AssetResponse | None = None
    candidate_outputs: list[AssetResponse] = Field(default_factory=list)
    edit_plan: dict[str, Any] | None = None
    eval_report: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    fallback_trace: list[dict[str, Any]] = Field(default_factory=list)
    round_outputs: dict[str, AssetResponse | None] = Field(default_factory=dict)
    round_plans: dict[str, Any] = Field(default_factory=dict)
    round_eval_reports: dict[str, Any] = Field(default_factory=dict)
    round_execution_traces: dict[str, Any] = Field(default_factory=dict)
    round_segmentation_traces: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    stage_timings: list[StageTimingResponse] = Field(default_factory=list)
    feedback: list[dict[str, Any]] = Field(default_factory=list)


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


class PackageCatalogResponse(BaseModel):
    """Planner-facing package catalog response."""

    items: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Service health payload."""

    ok: bool = True
    service: str = "PsAgent"
