"""Load short-term context and long-term preferences."""

from app.graph.state import EditState


def load_context(state: EditState) -> dict:
    """Populate thread context and retrieved preferences."""

    return {
        "retrieved_prefs": state.get("retrieved_prefs", []),
    }
