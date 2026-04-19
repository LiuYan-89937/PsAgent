"""Shared package models and abstract base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MaskPolicy = Literal["none", "optional", "required"]
RiskLevel = Literal["low", "medium", "high"]
MaskProvider = Literal["aliyun", "fal_sam3"]
RegionExecutionMode = Literal["whole_image", "masked_region"]

WHOLE_IMAGE_REGION: RegionExecutionMode = "whole_image"
MASKED_REGION_MODE: RegionExecutionMode = "masked_region"

MASK_PARAM_TO_RUNTIME_KEY = {
    "mask_provider": "provider",
    "mask_prompt": "prompt",
    "mask_negative_prompt": "negative_prompt",
    "mask_semantic_type": "semantic_type",
    "mask_fill_holes": "fill_holes",
    "mask_expand": "expand_mask",
    "mask_blur": "blur_mask",
    "mask_use_grounding_dino": "use_grounding_dino",
    "mask_revert": "revert_mask",
    "mask_start_timeout_seconds": "start_timeout_seconds",
    "mask_client_timeout_seconds": "client_timeout_seconds",
}
MASK_PARAM_KEYS = frozenset(MASK_PARAM_TO_RUNTIME_KEY)


def _schema_non_null_variants(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-null schema variants for a property."""

    variants = spec.get("anyOf") if isinstance(spec.get("anyOf"), list) else [spec]
    non_null = [
        item
        for item in variants
        if isinstance(item, dict) and item.get("type") != "null"
    ]
    return non_null or [spec]


def _schema_bound(spec: dict[str, Any], key: str) -> Any:
    """Read the first bound-like key from non-null variants."""

    for variant in _schema_non_null_variants(spec):
        if key in variant:
            return variant[key]
    return spec.get(key)


def _format_schema_value(value: Any) -> str:
    """Format schema numeric values compactly for human/model descriptions."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _schema_range_hint(spec: dict[str, Any]) -> str | None:
    """Build a concise range hint from JSON schema bounds."""

    type_names = {
        variant.get("type")
        for variant in _schema_non_null_variants(spec)
        if isinstance(variant, dict) and isinstance(variant.get("type"), str)
    }
    if not type_names:
        type_names = {spec.get("type")} if isinstance(spec.get("type"), str) else set()

    minimum = _schema_bound(spec, "minimum")
    maximum = _schema_bound(spec, "maximum")
    exclusive_minimum = _schema_bound(spec, "exclusiveMinimum")
    exclusive_maximum = _schema_bound(spec, "exclusiveMaximum")
    min_length = _schema_bound(spec, "minLength")
    max_length = _schema_bound(spec, "maxLength")

    if {"number", "integer"} & type_names:
        if minimum is not None and maximum is not None:
            return f"范围 {_format_schema_value(minimum)}~{_format_schema_value(maximum)}"
        if exclusive_minimum is not None and exclusive_maximum is not None:
            return f"范围 ({_format_schema_value(exclusive_minimum)},{_format_schema_value(exclusive_maximum)})"
        if minimum is not None:
            return f">= {_format_schema_value(minimum)}"
        if maximum is not None:
            return f"<= {_format_schema_value(maximum)}"
        if exclusive_minimum is not None:
            return f"> {_format_schema_value(exclusive_minimum)}"
        if exclusive_maximum is not None:
            return f"< {_format_schema_value(exclusive_maximum)}"

    if "string" in type_names:
        if min_length is not None and max_length is not None:
            return f"长度 {min_length}~{max_length}"
        if min_length is not None:
            return f"长度 >= {min_length}"
        if max_length is not None:
            return f"长度 <= {max_length}"

    return None


def _compact_schema_description(description: str | None, spec: dict[str, Any]) -> str | None:
    """Normalize a property description into a shorter planner-facing form."""

    text = " ".join(str(description or "").strip().split())
    if not text:
        return None

    replacements = (
        ("局部模式下的", "局部"),
        ("用于", ""),
        ("适合", ""),
        ("控制", ""),
        ("整体", ""),
        ("区域", ""),
        ("的力度", ""),
        ("的强度", "强度"),
        ("的偏移", "偏移"),
        ("的调整", "调整"),
        ("的增减", "增减"),
    )
    for source, target in replacements:
        text = text.replace(source, target)

    while "  " in text:
        text = text.replace("  ", " ")
    text = text.strip(" ;；。")

    range_hint = _schema_range_hint(spec)
    if range_hint and range_hint not in text:
        text = f"{text}；{range_hint}"

    return text


class MaskParams(BaseModel):
    """Validated optional mask-generation params shared by local edit packages."""

    model_config = ConfigDict(extra="forbid")

    mask_provider: MaskProvider | None = Field(
        default=None,
        description="Optional segmentation backend override. Prefer fal_sam3 for text-guided fine masks.",
    )
    mask_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Dynamic natural-language segmentation target generated by the planner, for example face skin, white dress, or bottle only.",
    )
    mask_negative_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Optional exclusion prompt used with text-guided segmentation to avoid unrelated areas.",
    )
    mask_semantic_type: bool | None = Field(
        default=None,
        description="Hint that the segmentation target is semantic rather than purely visual.",
    )
    mask_fill_holes: bool | None = Field(
        default=None,
        description="Whether to fill holes inside the generated mask.",
    )
    mask_expand: int | None = Field(
        default=None,
        ge=0,
        le=64,
        description="Optional mask expansion radius in pixels.",
    )
    mask_blur: bool | None = Field(
        default=None,
        description="Whether to blur the raw mask before binarization.",
    )
    mask_use_grounding_dino: bool | None = Field(
        default=None,
        description="Whether to enable GroundingDINO assistance for text-guided segmentation when supported.",
    )
    mask_revert: bool | None = Field(
        default=None,
        description="Whether to invert the generated mask result.",
    )
    mask_start_timeout_seconds: float | None = Field(
        default=None,
        ge=10.0,
        le=600.0,
        description="Provider start timeout in seconds.",
    )
    mask_client_timeout_seconds: float | None = Field(
        default=None,
        ge=10.0,
        le=900.0,
        description="Provider client timeout in seconds.",
    )

    @field_validator("mask_prompt", "mask_negative_prompt", mode="before")
    @classmethod
    def _normalize_prompt_text(cls, value: Any) -> Any:
        """Trim and normalize whitespace for free-text mask prompts."""

        if value is None or not isinstance(value, str):
            return value
        normalized = " ".join(value.strip().split())
        return normalized or None

    @model_validator(mode="after")
    def _validate_mask_constraints(self) -> "MaskParams":
        """Validate provider-specific constraints for optional mask params."""

        if self.mask_negative_prompt and not self.mask_prompt:
            raise ValueError("mask_negative_prompt requires mask_prompt.")
        if self.mask_use_grounding_dino and not self.mask_prompt:
            raise ValueError("mask_use_grounding_dino requires mask_prompt.")

        if self.mask_provider == "aliyun":
            unsupported = [
                field_name
                for field_name in (
                    "mask_prompt",
                    "mask_negative_prompt",
                    "mask_semantic_type",
                    "mask_fill_holes",
                    "mask_expand",
                    "mask_blur",
                    "mask_use_grounding_dino",
                    "mask_revert",
                )
                if getattr(self, field_name) not in (None, False, 0)
            ]
            if unsupported:
                joined = ", ".join(unsupported)
                raise ValueError(f"Aliyun segmentation does not support: {joined}")

        return self

    def to_runtime_options(self) -> dict[str, Any]:
        """Convert validated mask params into segmentation runtime kwargs."""

        payload = self.model_dump(exclude_none=True)
        return {
            runtime_key: payload[param_key]
            for param_key, runtime_key in MASK_PARAM_TO_RUNTIME_KEY.items()
            if param_key in payload
        }


class PackageParamsModel(BaseModel):
    """Strict base model for planner-fillable package params."""

    model_config = ConfigDict(extra="forbid")


def extract_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Pick only shared mask params from a merged package params payload."""

    return {
        key: value
        for key, value in params.items()
        if key in MASK_PARAM_KEYS and value is not None and value != ""
    }


def strip_mask_params(params: dict[str, Any]) -> dict[str, Any]:
    """Remove shared mask params from a merged package params payload."""

    return {key: value for key, value in params.items() if key not in MASK_PARAM_KEYS}


class PackageSpec(BaseModel):
    """Static capability declaration for a tool package."""

    # name: 工具包唯一标识，必须和 planner 输出的 op 对齐
    # supported_regions: 工具包支持的执行模式集合，只区分 whole_image / masked_region
    # mask_policy: none/optional/required，决定执行前是否要准备 mask
    name: str
    description: str
    supported_regions: list[RegionExecutionMode] = Field(default_factory=list)
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
    params_model: type[BaseModel] | None = None
    param_aliases: dict[str, tuple[str, ...]] = {}

    @property
    def name(self) -> str:
        """Expose the unique package name."""

        return self.spec.name

    def get_llm_schema(self) -> dict[str, Any]:
        """Return the planner-facing schema for this package."""

        return self.build_llm_schema()

    def build_llm_schema(self) -> dict[str, Any]:
        """Build the package catalog item exposed to the planner."""

        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "supported_regions": self.spec.supported_regions,
            "mask_policy": self.spec.mask_policy,
            "supported_domains": self.spec.supported_domains,
            "risk_level": self.spec.risk_level,
            "params_schema": self.get_params_schema(),
        }

    def is_whole_image_region(self, region: str | None) -> bool:
        """Return whether the operation targets the whole image."""

        normalized = str(region or WHOLE_IMAGE_REGION).strip() or WHOLE_IMAGE_REGION
        return normalized == WHOLE_IMAGE_REGION

    def supports_region(self, region: str | None) -> bool:
        """Return whether this package supports the provided execution mode label."""

        required_mode: RegionExecutionMode = (
            WHOLE_IMAGE_REGION if self.is_whole_image_region(region) else MASKED_REGION_MODE
        )
        return required_mode in self.spec.supported_regions

    def operation_requires_mask(
        self,
        operation: dict[str, Any],
        context: OperationContext | None = None,
        *,
        merged_params: dict[str, Any] | None = None,
    ) -> bool:
        """Return whether this operation should execute against a mask."""

        if self.parse_mask_params(operation, merged_params=merged_params) is not None:
            return True

        if context is None:
            return False

        region = operation.get("region")
        if isinstance(region, str) and region and context.masks.get(region):
            return True
        return False

    def supports_operation(self, operation: dict[str, Any], context: OperationContext | None = None) -> bool:
        """Return whether this package supports the operation's current execution mode."""

        required_mode: RegionExecutionMode = (
            MASKED_REGION_MODE if self.operation_requires_mask(operation, context) else WHOLE_IMAGE_REGION
        )
        return required_mode in self.spec.supported_regions

    def get_params_schema(self) -> dict[str, Any]:
        """Return the JSON schema for planner-fillable params."""

        schema = (
            self.params_model.model_json_schema()
            if self.params_model is not None
            else {
                "type": "object",
                "properties": {},
            }
        )
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        if self.spec.mask_policy != "none":
            mask_schema = MaskParams.model_json_schema()
            schema["properties"].update(mask_schema.get("properties", {}))
            if "title" in mask_schema and "title" not in schema:
                schema["title"] = mask_schema["title"]

        properties = schema.get("properties")
        if isinstance(properties, dict):
            for property_name, property_spec in properties.items():
                if not isinstance(property_spec, dict):
                    continue
                compact_description = _compact_schema_description(
                    property_spec.get("description"),
                    property_spec,
                )
                if compact_description:
                    property_spec["description"] = compact_description
        schema["additionalProperties"] = False
        return schema

    def get_operation_params(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Return the merged package params.

        The current runtime still accepts some legacy or loosely structured
        top-level params. This helper folds known top-level params into
        `params` before strict model validation.
        """

        params = self.normalize_external_params(dict(operation.get("params", {})))
        if self.params_model is None:
            return params

        model_fields = getattr(self.params_model, "model_fields", {})
        for field_name in model_fields:
            if field_name in params:
                continue
            if field_name in operation:
                params[field_name] = operation[field_name]

        legacy_strength = operation.get("strength")
        if legacy_strength is not None:
            for key, value in self.coerce_legacy_strength_params(legacy_strength, params=params).items():
                params.setdefault(key, value)
        return params

    @staticmethod
    def normalize_param_key(key: str) -> str:
        """Normalize a free-form external param key into a stable identifier."""

        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(key).strip().lower())
        normalized = re.sub(r"_+", "_", normalized)
        return normalized.strip("_")

    def normalize_external_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Fold external alias keys into canonical package param names."""

        normalized: dict[str, Any] = {}
        for key, value in params.items():
            normalized_key = self.normalize_param_key(key)
            normalized[normalized_key] = value

        for canonical_key, aliases in self.param_aliases.items():
            if normalized.get(canonical_key) not in (None, ""):
                continue
            for alias in aliases:
                alias_key = self.normalize_param_key(alias)
                if normalized.get(alias_key) not in (None, ""):
                    normalized[canonical_key] = normalized[alias_key]
                    break

        if self.params_model is not None:
            model_fields = getattr(self.params_model, "model_fields", {})
            for field_name in model_fields:
                normalized_key = self.normalize_param_key(field_name)
                if normalized_key in normalized and field_name not in normalized:
                    normalized[field_name] = normalized[normalized_key]
                if normalized_key != field_name:
                    normalized.pop(normalized_key, None)

        for mask_key in MASK_PARAM_KEYS:
            normalized_key = self.normalize_param_key(mask_key)
            if normalized_key in normalized and mask_key not in normalized:
                normalized[mask_key] = normalized[normalized_key]
            if normalized_key != mask_key:
                normalized.pop(normalized_key, None)

        for canonical_key, aliases in self.param_aliases.items():
            for alias in aliases:
                alias_key = self.normalize_param_key(alias)
                if alias_key != canonical_key:
                    normalized.pop(alias_key, None)

        return normalized

    def coerce_legacy_strength_params(
        self,
        legacy_strength: Any,
        *,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Map a legacy top-level strength into package params when possible."""

        if self.params_model is None:
            return {}

        model_fields = getattr(self.params_model, "model_fields", {})
        if "strength" in model_fields and "strength" not in params:
            return {"strength": legacy_strength}
        if "amount" in model_fields and "amount" not in params:
            return {"amount": legacy_strength}
        return {}

    def parse_params(self, operation: dict[str, Any]) -> BaseModel | None:
        """Validate merged params against the package params model."""

        merged_params = self.get_operation_params(operation)
        self.parse_mask_params(operation, merged_params=merged_params)

        if self.params_model is None:
            if strip_mask_params(merged_params):
                raise ValueError(f"{self.name} does not accept custom params.")
            return None
        return self.params_model.model_validate(strip_mask_params(merged_params))

    def parse_mask_params(
        self,
        operation: dict[str, Any],
        *,
        merged_params: dict[str, Any] | None = None,
    ) -> MaskParams | None:
        """Validate optional shared mask params for local-region operations."""

        merged = merged_params or self.get_operation_params(operation)
        mask_payload = extract_mask_params(merged)
        if not mask_payload:
            return None

        return MaskParams.model_validate(mask_payload)

    def get_mask_runtime_options(self, operation: dict[str, Any]) -> dict[str, Any]:
        """Return validated mask runtime kwargs for segmentation backends."""

        parsed = self.parse_mask_params(operation)
        if parsed is None:
            return {}
        return parsed.to_runtime_options()

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
