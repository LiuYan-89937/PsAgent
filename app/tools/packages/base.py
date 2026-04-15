"""Shared package models and abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, Field


MaskPolicy = Literal["none", "optional", "required"]
RiskLevel = Literal["low", "medium", "high"]


class PackageSpec(BaseModel):
    """Static capability declaration for a tool package."""

    # name: 工具包唯一标识，必须和 planner 输出的 op 对齐
    # supported_regions: 允许这个包作用的区域集合
    # mask_policy: none/optional/required，决定执行前是否要准备 mask
    name: str
    description: str
    supported_regions: list[str] = Field(default_factory=list)
    mask_policy: MaskPolicy = "none"
    supported_domains: list[str] = Field(default_factory=list)
    risk_level: RiskLevel = "low"
    default_params: dict[str, Any] = Field(default_factory=dict)


class OperationContext(BaseModel):
    """Shared execution context for all tool packages."""

    # 这里放所有包共用的执行上下文，避免每个包各自发明参数签名。
    image_path: str | None = None
    image_analysis: dict[str, Any] = Field(default_factory=dict)
    retrieved_prefs: list[dict[str, Any]] = Field(default_factory=list)
    masks: dict[str, str] = Field(default_factory=dict)
    thread_id: str | None = None
    audit: dict[str, Any] = Field(default_factory=dict)


class PackageResult(BaseModel):
    """Standardized package execution result."""

    # 所有工具包统一返回这个结构，方便后续评估和审计层消费。
    ok: bool
    package: str
    output_image: str | None = None
    applied_params: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    error: str | None = None


class ToolPackage(ABC):
    """Abstract package interface for planner-selected edit operations."""

    # spec 只负责声明能力；真正执行逻辑放在下面的方法里。
    spec: PackageSpec

    @property
    def name(self) -> str:
        """Expose the unique package name."""

        return self.spec.name

    @abstractmethod
    def get_llm_schema(self) -> dict[str, Any]:
        """Return the simplified schema exposed to the planner."""

    @abstractmethod
    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        """Validate a normalized operation before execution."""

    @abstractmethod
    def resolve_requirements(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        """Declare execution requirements such as masks or region inputs."""

    @abstractmethod
    def normalize(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> dict[str, Any]:
        """Normalize abstract planner params into stable internal params."""

    @abstractmethod
    def execute(
        self,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        """Execute the package against the provided context."""

    def fallback(
        self,
        error: Exception,
        operation: dict[str, Any],
        context: OperationContext,
    ) -> PackageResult:
        """Return a default fallback result for unimplemented or failed execution."""

        # 骨架阶段统一走这里，后面每个包可以覆盖自己的降级逻辑。
        return PackageResult(
            ok=False,
            package=self.name,
            applied_params={"operation": operation},
            fallback_used=True,
            error=str(error),
        )
