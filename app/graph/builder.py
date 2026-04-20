"""LangGraph builder for the photo-editing agent."""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.bootstrap_request import bootstrap_request
from app.graph.nodes.analyze_image import analyze_image
from app.graph.nodes.build_edit_profile import build_edit_profile
from app.graph.nodes.evaluate_result import final_review
from app.graph.nodes.human_review import human_review
from app.graph.nodes.load_context import load_context
from app.graph.nodes.parse_request import parse_request
from app.graph.nodes.update_memory import update_memory
from app.graph.subgraphs.stage_pipeline import build_stage_subgraph
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
    builder.add_node("build_edit_profile", build_edit_profile)
    builder.add_node("technical_prep_subgraph", build_stage_subgraph("technical_prep"))
    builder.add_node("global_base_subgraph", build_stage_subgraph("global_base"))
    builder.add_node("local_balance_subgraph", build_stage_subgraph("local_balance"))
    builder.add_node("subject_refine_subgraph", build_stage_subgraph("subject_refine"))
    builder.add_node("finish_output_subgraph", build_stage_subgraph("finish_output"))
    builder.add_node("final_review", final_review)

    builder.add_node("human_review", human_review)
    builder.add_node("update_memory", update_memory)

    builder.add_edge(START, "bootstrap_request")
    builder.add_edge("bootstrap_request", "load_context")
    builder.add_edge("load_context", "analyze_image")
    builder.add_edge("analyze_image", "parse_request")
    builder.add_edge("parse_request", "build_edit_profile")
    builder.add_edge("build_edit_profile", "technical_prep_subgraph")
    builder.add_edge("technical_prep_subgraph", "global_base_subgraph")
    builder.add_edge("global_base_subgraph", "local_balance_subgraph")
    builder.add_edge("local_balance_subgraph", "subject_refine_subgraph")
    builder.add_edge("subject_refine_subgraph", "finish_output_subgraph")
    builder.add_edge("finish_output_subgraph", "final_review")

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
