"""Shared planner runtime helpers used by the single-shot planner."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from app.services.planner_param_codec import decode_planner_operation_params
from app.tools.tool_registry import ToolRegistry, build_default_tool_registry
from app.tools.tool_specs import WHOLE_IMAGE_REGION


def _normalize_tool_name(value: str) -> str:
    """Normalize a planner-returned tool name into a stable comparison key."""

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _generated_aliases(tool_name: str) -> set[str]:
    """Generate zero-cost aliases from a registered tool name."""

    aliases = {_normalize_tool_name(tool_name)}
    if tool_name.startswith("adjust_"):
        aliases.add(_normalize_tool_name(tool_name.removeprefix("adjust_")))
    else:
        aliases.add(_normalize_tool_name(f"adjust_{tool_name}"))
    return aliases


def _char_trigram_vector(text: str) -> Counter[str]:
    """Build a light-weight character trigram vector."""

    normalized = f"  {_normalize_tool_name(text)}  "
    return Counter(normalized[index : index + 3] for index in range(max(len(normalized) - 2, 0)))


def _cosine_similarity(left: Counter[str], right: Counter[str]) -> float:
    """Return cosine similarity for sparse trigram vectors."""

    keys = set(left) | set(right)
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tool_candidate_text(registered_tool) -> str:
    """Build a compact retrieval document for one tool."""

    spec = registered_tool.spec
    fields = [
        spec.name,
        spec.label,
        spec.description,
        spec.family,
        " ".join(spec.stage_affinity),
        "masked_region" if spec.supports_mask else "",
        "whole_image" if spec.supports_whole_image else "",
        " ".join(sorted(_generated_aliases(spec.name))),
    ]
    return " ".join(part for part in fields if part)


def resolve_planner_tool_name(
    raw_tool_name: str,
    arguments: dict[str, Any],
    registry: ToolRegistry | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve a planner-returned tool name with exact-first fallback to light similarity."""

    active_registry = registry or build_default_tool_registry()
    normalized_name = _normalize_tool_name(raw_tool_name)
    if active_registry.get(raw_tool_name) is not None:
        return raw_tool_name, {"strategy": "exact", "score": 1.0}
    if active_registry.get(normalized_name) is not None:
        return normalized_name, {"strategy": "normalized", "score": 1.0}

    for registered_tool in active_registry.list():
        if normalized_name in _generated_aliases(registered_tool.spec.name):
            return registered_tool.spec.name, {"strategy": "alias", "score": 1.0}

    raw_query_vector = _char_trigram_vector(raw_tool_name)
    query_fragments = [raw_tool_name]
    region = arguments.get("region")
    if isinstance(region, str) and region:
        query_fragments.append(region)
    for key in sorted(arguments):
        if key != "region":
            query_fragments.append(key)
    query_vector = _char_trigram_vector(" ".join(query_fragments))

    scored_candidates: list[tuple[str, float]] = []
    for registered_tool in active_registry.list():
        candidate_vector = _char_trigram_vector(_tool_candidate_text(registered_tool))
        candidate_score = max(
            _cosine_similarity(raw_query_vector, candidate_vector),
            _cosine_similarity(query_vector, candidate_vector),
        )
        scored_candidates.append((registered_tool.spec.name, candidate_score))

    scored_candidates.sort(key=lambda item: item[1], reverse=True)
    if not scored_candidates:
        raise RuntimeError(f"Planner returned unknown tool: {raw_tool_name}")

    best_name, best_score = scored_candidates[0]
    second_score = scored_candidates[1][1] if len(scored_candidates) > 1 else 0.0
    if best_score < 0.42 or best_score - second_score < 0.05:
        top_candidates = [
            {"name": name, "score": round(score, 4)}
            for name, score in scored_candidates[:3]
        ]
        raise RuntimeError(
            f"Planner returned unknown tool '{raw_tool_name}', and similarity match was inconclusive: {top_candidates}"
        )

    return best_name, {
        "strategy": "similarity",
        "score": round(best_score, 4),
        "candidates": [
            {"name": name, "score": round(score, 4)}
            for name, score in scored_candidates[:3]
        ],
    }


def build_operation_from_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Convert planner-facing params into a runtime operation dict."""

    region, params, strength = decode_planner_operation_params(tool_name, arguments)
    return {
        "op": tool_name,
        "region": region or WHOLE_IMAGE_REGION,
        "strength": strength if isinstance(strength, (int, float)) else None,
        "params": params,
        "constraints": [],
    }
