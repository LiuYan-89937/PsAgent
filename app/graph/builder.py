"""LangGraph builder for the photo-editing agent."""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.bootstrap_request import bootstrap_request
from app.graph.nodes.analyze_image import analyze_image
from app.graph.nodes.build_objective import build_objective
from app.graph.nodes.evaluate_result import final_review
from app.graph.nodes.human_review import human_review
from app.graph.nodes.load_context import load_context
from app.graph.nodes.parse_request import parse_request
from app.graph.nodes.run_search_agent import run_search_agent
from app.graph.nodes.update_memory import update_memory
from app.graph.state import EditState, GraphInputState, GraphOutputState


def need_review(state: EditState) -> str:
    """Route to review when the current result requires confirmation."""

    return "review" if state.get("approval_required") else "ok"


def build_graph(checkpointer=None, store=None):
    """Create the application graph."""

    builder = StateGraph(
        EditState,
        input_schema=GraphInputState,
        output_schema=GraphOutputState,
    )

    builder.add_node("bootstrap_request", bootstrap_request)
    builder.add_node("load_context", load_context)
    builder.add_node("analyze_image", analyze_image)
    builder.add_node("parse_request", parse_request)
    builder.add_node("build_objective", build_objective)
    builder.add_node("run_search_agent", run_search_agent)
    builder.add_node("final_review", final_review)

    builder.add_node("human_review", human_review)
    builder.add_node("update_memory", update_memory)

    builder.add_edge(START, "bootstrap_request")
    builder.add_edge("bootstrap_request", "load_context")
    builder.add_edge("load_context", "analyze_image")
    builder.add_edge("analyze_image", "parse_request")
    builder.add_edge("parse_request", "build_objective")
    builder.add_edge("build_objective", "run_search_agent")
    builder.add_edge("run_search_agent", "final_review")

    builder.add_conditional_edges(
        "final_review",
        need_review,
        {
            "review": "human_review",
            "ok": "update_memory",
        },
    )

    builder.add_edge("human_review", "update_memory")
    builder.add_edge("update_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)
