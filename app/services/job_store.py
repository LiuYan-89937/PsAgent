"""Job persistence service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field
from app.graph.state import (
    ApprovalPayload,
    EditPlan,
    EvaluationReport,
    ExecutionTraceItem,
    FallbackTraceItem,
    SegmentationTraceItem,
    coerce_approval_payload,
    coerce_edit_plan,
    coerce_eval_report,
    coerce_execution_trace,
    coerce_fallback_trace,
    coerce_segmentation_trace,
)


JobStatus = Literal["pending", "running", "completed", "failed", "review_required"]
_UNSET = object()


class JobRecord(BaseModel):
    """Stored edit job record."""

    job_id: str
    status: JobStatus
    user_id: str
    thread_id: str
    request_text: str | None = None
    input_asset_ids: list[str] = Field(default_factory=list)
    output_asset_ids: list[str] = Field(default_factory=list)
    round_output_asset_ids: dict[str, str] = Field(default_factory=dict)
    edit_plan: EditPlan | None = None
    eval_report: EvaluationReport | None = None
    execution_trace: list[ExecutionTraceItem] = Field(default_factory=list)
    segmentation_trace: list[SegmentationTraceItem] = Field(default_factory=list)
    fallback_trace: list[FallbackTraceItem] = Field(default_factory=list)
    round_plans: dict[str, Any] = Field(default_factory=dict)
    round_eval_reports: dict[str, Any] = Field(default_factory=dict)
    round_execution_traces: dict[str, Any] = Field(default_factory=dict)
    round_segmentation_traces: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    approval_required: bool = False
    approval_payload: ApprovalPayload | None = None
    current_stage: str | None = None
    current_message: str | None = None
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    error_detail: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime


class JobStore:
    """In-memory job and result persistence abstraction."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}

    def create_job(
        self,
        *,
        user_id: str,
        thread_id: str,
        request_text: str | None = None,
        input_asset_ids: list[str] | None = None,
    ) -> JobRecord:
        """Create a new job record."""

        now = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=f"job_{uuid4().hex}",
            status="pending",
            user_id=user_id,
            thread_id=thread_id,
            request_text=request_text,
            input_asset_ids=list(input_asset_ids or []),
            created_at=now,
            updated_at=now,
        )
        self._jobs[record.job_id] = record
        return record

    @staticmethod
    def _touch(record: JobRecord) -> JobRecord:
        """Refresh the update timestamp for a mutable in-memory record."""

        record.updated_at = datetime.now(timezone.utc)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        """Return a stored job if present."""

        return self._jobs.get(job_id)

    def require(self, job_id: str) -> JobRecord:
        """Return a stored job or raise if missing."""

        record = self.get(job_id)
        if record is None:
            raise KeyError(f"Unknown job: {job_id}")
        return record

    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
        error_detail: dict[str, Any] | None = None,
        current_stage: str | None = None,
        current_message: str | None = None,
        approval_required: bool | None = None,
        request_text: str | None | object = _UNSET,
    ) -> JobRecord:
        """Update the mutable status fields for a job without rebuilding the whole record."""

        record = self.require(job_id)
        record.status = status
        record.error = error
        record.error_detail = error_detail
        if current_stage is not None:
            record.current_stage = current_stage
        if current_message is not None:
            record.current_message = current_message
        if approval_required is not None:
            record.approval_required = approval_required
        if request_text is not _UNSET:
            record.request_text = request_text if isinstance(request_text, str) or request_text is None else record.request_text
        return self._touch(record)

    def set_execution_result(
        self,
        job_id: str,
        *,
        output_asset_ids: list[str] | object = _UNSET,
        round_output_asset_ids: dict[str, str] | object = _UNSET,
        edit_plan: dict[str, Any] | None | object = _UNSET,
        eval_report: dict[str, Any] | None | object = _UNSET,
        execution_trace: list[dict[str, Any]] | object = _UNSET,
        segmentation_trace: list[dict[str, Any]] | object = _UNSET,
        fallback_trace: list[dict[str, Any]] | object = _UNSET,
        round_plans: dict[str, Any] | object = _UNSET,
        round_eval_reports: dict[str, Any] | object = _UNSET,
        round_execution_traces: dict[str, Any] | object = _UNSET,
        round_segmentation_traces: dict[str, Any] | object = _UNSET,
        approval_required: bool | None = None,
        approval_payload: ApprovalPayload | dict[str, Any] | None | object = _UNSET,
        request_text: str | None | object = _UNSET,
        current_stage: str | None = None,
        current_message: str | None = None,
        status: JobStatus | None = None,
        error: str | None | object = _UNSET,
        error_detail: dict[str, Any] | None | object = _UNSET,
    ) -> JobRecord:
        """Persist execution outputs and traces with incremental field assignment."""

        record = self.require(job_id)
        if output_asset_ids is not _UNSET:
            record.output_asset_ids = list(output_asset_ids)
        if round_output_asset_ids is not _UNSET:
            record.round_output_asset_ids = dict(round_output_asset_ids)
        if edit_plan is not _UNSET:
            record.edit_plan = coerce_edit_plan(edit_plan if isinstance(edit_plan, (dict, EditPlan)) or edit_plan is None else None)
        if eval_report is not _UNSET:
            record.eval_report = coerce_eval_report(eval_report if isinstance(eval_report, (dict, EvaluationReport)) or eval_report is None else None)
        if execution_trace is not _UNSET:
            record.execution_trace = coerce_execution_trace(execution_trace if isinstance(execution_trace, list) else [])
        if segmentation_trace is not _UNSET:
            record.segmentation_trace = coerce_segmentation_trace(segmentation_trace if isinstance(segmentation_trace, list) else [])
        if fallback_trace is not _UNSET:
            record.fallback_trace = coerce_fallback_trace(fallback_trace if isinstance(fallback_trace, list) else [])
        if round_plans is not _UNSET:
            record.round_plans = dict(round_plans)
        if round_eval_reports is not _UNSET:
            record.round_eval_reports = dict(round_eval_reports)
        if round_execution_traces is not _UNSET:
            record.round_execution_traces = dict(round_execution_traces)
        if round_segmentation_traces is not _UNSET:
            record.round_segmentation_traces = dict(round_segmentation_traces)
        if approval_required is not None:
            record.approval_required = approval_required
        if approval_payload is not _UNSET:
            record.approval_payload = coerce_approval_payload(
                approval_payload if isinstance(approval_payload, (dict, ApprovalPayload)) or approval_payload is None else None
            )
        if request_text is not _UNSET:
            record.request_text = request_text if isinstance(request_text, str) or request_text is None else record.request_text
        if current_stage is not None:
            record.current_stage = current_stage
        if current_message is not None:
            record.current_message = current_message
        if status is not None:
            record.status = status
        if error is not _UNSET:
            record.error = error if isinstance(error, str) or error is None else record.error
        if error_detail is not _UNSET:
            record.error_detail = error_detail if isinstance(error_detail, dict) or error_detail is None else record.error_detail
        return self._touch(record)

    def set_review_state(
        self,
        job_id: str,
        *,
        status: JobStatus,
        approval_required: bool,
        current_stage: str,
        current_message: str,
        error: str | None = None,
        error_detail: dict[str, Any] | None = None,
    ) -> JobRecord:
        """Update the mutable review-related status fields for a job."""

        return self.set_status(
            job_id,
            status,
            approval_required=approval_required,
            current_stage=current_stage,
            current_message=current_message,
            error=error,
            error_detail=error_detail,
        )

    def append_feedback(self, job_id: str, item: dict[str, Any]) -> JobRecord:
        """Append a feedback item to the job."""

        current = self.require(job_id)
        current.feedback.append(item)
        return self._touch(current)

    def append_event(
        self,
        job_id: str,
        item: dict[str, Any],
        *,
        current_stage: str | None = None,
        current_message: str | None = None,
    ) -> JobRecord:
        """Append a progress event to the job."""

        current = self.require(job_id)
        current.events.append(item)
        if current_stage is not None:
            current.current_stage = current_stage
        if current_message is not None:
            current.current_message = current_message
        return self._touch(current)

    def list(self) -> list[JobRecord]:
        """List all jobs."""

        return list(self._jobs.values())
