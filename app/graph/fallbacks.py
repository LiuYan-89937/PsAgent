"""Helpers for recording non-fatal fallback decisions."""

from __future__ import annotations

from typing import Any

from app.graph.state import FallbackTraceItem


def append_fallback_trace(
    existing: list[dict[str, Any]] | None,
    *,
    stage: str,
    source: str,
    location: str,
    strategy: str,
    message: str,
    error: str | None = None,
) -> list[dict[str, Any]]:
    """Append one normalized fallback trace item."""

    items = list(existing or [])
    items.append(
        FallbackTraceItem(
            index=len(items),
            stage=stage,
            source=source,
            location=location,
            strategy=strategy,
            message=message,
            error=error,
        ).model_dump(mode="json")
    )
    return items
