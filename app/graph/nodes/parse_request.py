"""Parse user instruction and choose explicit or auto mode."""

from app.graph.state import EditState


def parse_request(state: EditState) -> dict:
    """Determine request mode from the current input."""

    return {
        "mode": state.get("mode", "explicit"),
    }
