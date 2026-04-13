"""Core state and schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class EditOperation(BaseModel):
    """A single edit operation in the planner output."""

    op: Literal[
        "global_exposure",
        "local_exposure",
        "contrast",
        "white_balance",
        "denoise",
        "sharpen",
        "crop",
        "background_blur",
        "remove_object",
        "replace_background",
        "inpaint_region",
    ]
    region: str | None = None
    strength: float = Field(ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class EditPlan(BaseModel):
    """Structured edit plan produced by the planner."""

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


class EditState(TypedDict, total=False):
    """LangGraph state for an edit thread."""

    messages: Annotated[list, add_messages]
    user_id: str
    thread_id: str
    input_images: list[str]
    mode: str
    image_analysis: dict[str, Any] | None
    retrieved_prefs: list[dict[str, Any]]
    edit_plan: dict[str, Any] | None
    masks: list[str]
    candidate_outputs: list[str]
    eval_report: dict[str, Any] | None
    selected_output: str | None
    memory_write_candidates: list[dict[str, Any]]
    approval_required: bool
    approval_payload: dict[str, Any] | None
