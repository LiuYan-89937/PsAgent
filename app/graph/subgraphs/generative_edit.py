"""Generative editing subgraph."""

from app.graph.state import EditState


def execute_generative(state: EditState) -> dict:
    """Run generative image editing."""

    return {
        "candidate_outputs": state.get("candidate_outputs", []),
    }
