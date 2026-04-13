"""Hybrid editing subgraph."""

from app.graph.state import EditState


def execute_hybrid(state: EditState) -> dict:
    """Run hybrid deterministic and generative editing."""

    return {
        "candidate_outputs": state.get("candidate_outputs", []),
    }
