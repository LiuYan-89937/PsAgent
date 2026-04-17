"""Reusable deterministic package base classes for newly added tools."""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from typing import Any, Callable

from app.tools.packages.base import OperationContext, PackageResult, ToolPackage


class DeterministicImageOpPackage(ToolPackage):
    """Base class for packages backed by a single deterministic image-op function."""

    image_op: Callable[..., str] | None = None
    output_prefix: str = "psagent_extended_"

    def validate(self, operation: dict[str, Any], context: OperationContext) -> None:
        region = operation.get("region") or "whole_image"
        self.parse_params(operation)
        if not self.supports_operation(operation, context):
            raise ValueError(f"Unsupported region for {self.name}: {region}")
        if not context.image_path:
            raise ValueError(f"image_path is required for {self.name}")

    def resolve_requirements(self, operation: dict[str, Any], context: OperationContext) -> dict[str, Any]:
        region = operation.get("region") or "whole_image"
        requires_mask = self.spec.mask_policy != "none" and self.operation_requires_mask(operation, context)
        return {
            "requires_mask": requires_mask,
            "required_region": region if requires_mask else None,
        }

    def resolve_mask_path(self, requirements: dict[str, Any], context: OperationContext) -> str | None:
        """Resolve an optional execution mask from context."""

        if not requirements["requires_mask"]:
            return None
        required_region = requirements["required_region"]
        mask_path = context.masks.get(required_region) if required_region else None
        if not mask_path:
            raise ValueError(f"Mask is required for region: {required_region}")
        return mask_path

    @abstractmethod
    def build_image_op_kwargs(
        self,
        normalized: dict[str, Any],
        context: OperationContext,
        *,
        mask_path: str | None,
    ) -> dict[str, Any]:
        """Build keyword arguments passed to the backing deterministic op."""

    def execute(self, operation: dict[str, Any], context: OperationContext) -> PackageResult:
        try:
            self.validate(operation, context)
            normalized = self.normalize(operation, context)
            requirements = self.resolve_requirements(operation, context)
            mask_path = self.resolve_mask_path(requirements, context)

            if self.image_op is None:
                raise ValueError(f"{self.name} does not define an image operation.")

            output_path = tempfile.mktemp(prefix=self.output_prefix, suffix=".png")
            saved_path = self.image_op(
                context.image_path or "",
                output_path,
                **self.build_image_op_kwargs(normalized, context, mask_path=mask_path),
            )
            return PackageResult(
                ok=True,
                package=self.name,
                output_image=saved_path,
                applied_params=normalized,
                artifacts={
                    "input_image": context.image_path,
                    "mask_path": mask_path,
                    "requirements": requirements,
                },
            )
        except Exception as error:  # pragma: no cover - generic fallback path
            return self.fallback(error, operation, context)
