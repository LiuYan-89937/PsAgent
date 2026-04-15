"""Core state and schema definitions."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class EditOperation(BaseModel):
    """A single edit operation in the planner output."""

    # op 对齐工具包唯一标识；region 指向 whole_image 或局部区域；
    # strength 使用抽象强度，后续由参数归一化层转换成内部参数。
    op: Literal[
        "adjust_exposure",
        "adjust_highlights_shadows",
        "adjust_contrast",
        "adjust_white_balance",
        "adjust_vibrance_saturation",
        "crop_and_straighten",
        "denoise",
        "sharpen",
    ]
    region: str | None = None
    strength: float = Field(ge=-1.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


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


class EditState(TypedDict, total=False):
    """LangGraph state for an edit thread."""

    # messages: 对话消息历史
    # user_id/thread_id: 用户与线程标识
    # input_images: 当前输入图片
    messages: Annotated[list, add_messages]
    user_id: str
    thread_id: str
    input_images: list[str]

    # 理解与规划阶段产物
    mode: str
    image_analysis: dict[str, Any] | None
    retrieved_prefs: list[dict[str, Any]]
    edit_plan: dict[str, Any] | None

    # 执行阶段产物
    masks: list[str]
    candidate_outputs: list[str]

    # 评估与结果阶段产物
    eval_report: dict[str, Any] | None
    selected_output: str | None

    # 长期记忆写回候选
    memory_write_candidates: list[dict[str, Any]]

    # 人工审核控制字段
    approval_required: bool
    approval_payload: dict[str, Any] | None
