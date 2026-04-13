"""Update long-term memory from feedback and accepted results."""

from app.graph.state import EditState


def update_memory(state: EditState) -> dict:
    """Persist memory candidates for future sessions."""

    return {
        "memory_write_candidates": state.get("memory_write_candidates", []),
    }
