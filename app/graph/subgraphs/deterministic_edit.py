"""Deterministic editing subgraph."""

from app.graph.state import EditState


def execute_deterministic(state: EditState) -> dict:
    """Run deterministic image operations."""

    return {
        "candidate_outputs": state.get("candidate_outputs", []),
    }
