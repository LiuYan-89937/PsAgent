"""Job persistence service."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


JobStatus = Literal["pending", "running", "completed", "failed", "review_required"]


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
    edit_plan: dict[str, Any] | None = None
    eval_report: dict[str, Any] | None = None
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    segmentation_trace: list[dict[str, Any]] = Field(default_factory=list)
    round_plans: dict[str, Any] = Field(default_factory=dict)
    round_eval_reports: dict[str, Any] = Field(default_factory=dict)
    round_execution_traces: dict[str, Any] = Field(default_factory=dict)
    round_segmentation_traces: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    approval_required: bool = False
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

    def get(self, job_id: str) -> JobRecord | None:
        """Return a stored job if present."""

        return self._jobs.get(job_id)

    def require(self, job_id: str) -> JobRecord:
        """Return a stored job or raise if missing."""

        record = self.get(job_id)
        if record is None:
            raise KeyError(f"Unknown job: {job_id}")
        return record

    def update(self, job_id: str, **fields: Any) -> JobRecord:
        """Patch a job record."""

        current = self.require(job_id)
        payload = current.model_dump(mode="json")
        payload.update(fields)
        payload["updated_at"] = datetime.now(timezone.utc)
        updated = JobRecord.model_validate(payload)
        self._jobs[job_id] = updated
        return updated

    def append_feedback(self, job_id: str, item: dict[str, Any]) -> JobRecord:
        """Append a feedback item to the job."""

        current = self.require(job_id)
        feedback = list(current.feedback)
        feedback.append(item)
        return self.update(job_id, feedback=feedback)

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
        events = list(current.events)
        events.append(item)
        return self.update(
            job_id,
            events=events,
            current_stage=current_stage if current_stage is not None else current.current_stage,
            current_message=current_message if current_message is not None else current.current_message,
        )

    def list(self) -> list[JobRecord]:
        """List all jobs."""

        return list(self._jobs.values())
