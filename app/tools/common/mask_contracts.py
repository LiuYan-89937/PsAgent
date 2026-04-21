"""Shared mask parameter contracts and schema helpers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


MaskProvider = Literal["aliyun", "fal_sam3"]

MASK_PARAM_TO_RUNTIME_KEY = {
    # 左边是 planner / API / catalog 侧的参数名，
    # 右边是 segmentation provider 调用时真正用的 runtime kwargs。
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


class MaskParams(BaseModel):
    """Shared planner-facing mask parameters handled by the stage runner."""

    # 这份 contract 的作用是：
    # 让所有工具共用一套 mask_* 语义，而不是每个工具各自发明一套。
    model_config = ConfigDict(extra="forbid")

    mask_provider: MaskProvider | None = Field(
        default=None,
        description="Optional segmentation backend override.",
    )
    mask_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Single visible English subject term for segmentation, e.g. face, hair, dress, background.",
    )
    mask_negative_prompt: str | None = Field(
        default=None,
        min_length=2,
        max_length=160,
        description="Optional exclusion prompt for text-guided segmentation.",
    )
    mask_semantic_type: bool | None = Field(
        default=None,
        description="Hint that the target is semantic rather than purely visual.",
    )
    mask_fill_holes: bool | None = Field(
        default=None,
        description="Whether to fill holes in the generated mask.",
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
        description="Whether to enable GroundingDINO assistance when supported.",
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
        """Trim and normalize mask prompt whitespace."""

        # prompt 文本在进入缓存签名和 provider 调用前先做一层简单归一化，
        # 避免因为空格差异导致复用失效。
        if value is None or not isinstance(value, str):
            return value
        normalized = " ".join(value.strip().split())
        return normalized or None

    @model_validator(mode="after")
    def _validate_mask_constraints(self) -> "MaskParams":
        """Validate provider-specific mask constraints."""

        # 这里把 provider 能力边界前置住，
        # 避免到了分割层才因为参数不支持而出很晚的错。
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

        # stage runner 最终调用 segmentation provider 时，
        # 用的就是这层转换后的 runtime 参数名。
        payload = self.model_dump(exclude_none=True)
        return {
            runtime_key: payload[param_key]
            for param_key, runtime_key in MASK_PARAM_TO_RUNTIME_KEY.items()
            if param_key in payload
        }


MASK_PARAMS_SCHEMA = MaskParams.model_json_schema()
