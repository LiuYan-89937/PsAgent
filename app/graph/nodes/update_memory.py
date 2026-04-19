"""Update long-term memory from feedback and accepted results."""

from pydantic import ValidationError

from app.graph.fallbacks import append_fallback_trace
from app.graph.state import EditState, MemoryWriteCandidate


def update_memory(state: EditState) -> dict:
    """Persist memory candidates for future sessions."""

    candidates: list[dict] = []
    fallback_trace = list(state.get("fallback_trace") or [])

    for item in state.get("memory_write_candidates", []):
        try:
            candidates.append(MemoryWriteCandidate.model_validate(item).model_dump(mode="json"))
        except ValidationError as error:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage="update_memory",
                source="memory_candidates",
                location="memory_write_candidates",
                strategy="drop_invalid_memory_candidate",
                message="检测到无效记忆候选，已自动跳过。",
                error=str(error),
            )

    return {
        "memory_write_candidates": candidates,
        "fallback_trace": fallback_trace,
    }
