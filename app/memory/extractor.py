"""Preference extraction helpers."""

from app.graph.state import EditState


def extract_memory_candidates(state: EditState) -> list[dict]:
    """Extract stable memory signals from the current interaction."""

    return state.get("memory_write_candidates", [])
