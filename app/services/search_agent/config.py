"""Search-effort configuration for round-first orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SearchEffort = Literal["standard", "high", "ultra"]


@dataclass(frozen=True)
class SearchRoundLimits:
    """Round-budget range for one autonomous search cycle."""

    min_rounds: int
    max_rounds: int


DEFAULT_SEARCH_EFFORT: SearchEffort = "standard"
HARD_MAX_ROUNDS = 12

SEARCH_ROUND_LIMITS: dict[SearchEffort, SearchRoundLimits] = {
    "standard": SearchRoundLimits(min_rounds=4, max_rounds=6),
    "high": SearchRoundLimits(min_rounds=6, max_rounds=8),
    "ultra": SearchRoundLimits(min_rounds=8, max_rounds=12),
}


def normalize_search_effort(value: object) -> SearchEffort:
    """Normalize untrusted frontend/API values to a supported effort."""

    if value in SEARCH_ROUND_LIMITS:
        return value  # type: ignore[return-value]
    return DEFAULT_SEARCH_EFFORT


def resolve_search_round_limits(value: object) -> SearchRoundLimits:
    """Resolve the configured round range for the current search cycle."""

    return SEARCH_ROUND_LIMITS[normalize_search_effort(value)]
