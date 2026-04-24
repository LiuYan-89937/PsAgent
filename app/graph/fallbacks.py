"""Helpers for recording non-fatal fallback decisions."""

from __future__ import annotations

from typing import Any

from app.graph.state import FallbackTraceItem


def append_fallback_trace(
    existing: list[dict[str, Any]] | None,
    *,
    round_id: str | None = None,
    focus: str | None = None,
    candidate_id: str | None = None,
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
            round_id=round_id,
            focus=focus,
            candidate_id=candidate_id,
            source=source,
            location=location,
            strategy=strategy,
            message=message,
            error=error,
        ).model_dump(mode="json")
    )
    return items
