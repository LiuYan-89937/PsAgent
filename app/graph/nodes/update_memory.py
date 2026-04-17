"""Update long-term memory from feedback and accepted results."""

from app.graph.state import EditState, MemoryWriteCandidate


def update_memory(state: EditState) -> dict:
    """Persist memory candidates for future sessions."""

    candidates = [
        MemoryWriteCandidate.model_validate(item).model_dump(mode="json")
        for item in state.get("memory_write_candidates", [])
    ]
    return {
        "memory_write_candidates": candidates,
    }
