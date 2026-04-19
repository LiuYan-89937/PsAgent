"""LangGraph builder for the photo-editing agent."""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.bootstrap_request import bootstrap_request
from app.graph.nodes.analyze_image import analyze_image
from app.graph.nodes.evaluate_result import evaluate_result, evaluate_round_1, finalize_round_1_result
from app.graph.nodes.human_review import human_review
from app.graph.nodes.load_context import load_context
from app.graph.nodes.parse_request import parse_request
from app.graph.nodes.plan_execute_round import plan_execute_round_1, plan_execute_round_2
from app.graph.nodes.update_memory import update_memory
from app.graph.state import EditState, GraphInputState, GraphOutputState


def should_continue_round_2(state: EditState) -> str:
    """Decide whether the graph should enter the second round."""

    return "round_2" if state.get("continue_to_round_2") else "finalize_round_1"


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

    builder.add_node("plan_execute_round_1", plan_execute_round_1)
    builder.add_node("evaluate_round_1", evaluate_round_1)

    builder.add_node("plan_execute_round_2", plan_execute_round_2)
    builder.add_node("evaluate_result_final", evaluate_result)
    builder.add_node("finalize_round_1_result", finalize_round_1_result)

    builder.add_node("human_review", human_review)
    builder.add_node("update_memory", update_memory)

    builder.add_edge(START, "bootstrap_request")
    builder.add_edge("bootstrap_request", "load_context")
    builder.add_edge("load_context", "analyze_image")
    builder.add_edge("analyze_image", "parse_request")
    builder.add_edge("parse_request", "plan_execute_round_1")
    builder.add_edge("plan_execute_round_1", "evaluate_round_1")

    builder.add_conditional_edges(
        "evaluate_round_1",
        should_continue_round_2,
        {
            "round_2": "plan_execute_round_2",
            "finalize_round_1": "finalize_round_1_result",
        },
    )

    builder.add_edge("plan_execute_round_2", "evaluate_result_final")

    builder.add_conditional_edges(
        "finalize_round_1_result",
        need_review,
        {
            "review": "human_review",
            "ok": "update_memory",
        },
    )
    builder.add_conditional_edges(
        "evaluate_result_final",
        need_review,
        {
            "review": "human_review",
            "ok": "update_memory",
        },
    )

    builder.add_edge("human_review", "update_memory")
    builder.add_edge("update_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)
