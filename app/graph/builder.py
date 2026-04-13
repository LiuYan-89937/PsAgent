"""LangGraph builder skeleton."""

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.analyze_image import analyze_image
from app.graph.nodes.build_plan import build_plan
from app.graph.nodes.evaluate_result import evaluate_result
from app.graph.nodes.human_review import human_review
from app.graph.nodes.load_context import load_context
from app.graph.nodes.parse_request import parse_request
from app.graph.nodes.route_executor import route_executor
from app.graph.nodes.update_memory import update_memory
from app.graph.state import EditState
from app.graph.subgraphs.deterministic_edit import execute_deterministic
from app.graph.subgraphs.generative_edit import execute_generative
from app.graph.subgraphs.hybrid_edit import execute_hybrid


def choose_executor(state: EditState) -> str:
    """Pick the executor branch from the current plan."""

    plan = state.get("edit_plan") or {}
    return str(plan.get("executor", "deterministic"))


def need_review(state: EditState) -> str:
    """Route to review when the current result requires confirmation."""

    return "review" if state.get("approval_required") else "ok"


def build_graph(checkpointer=None, store=None):
    """Create the application graph."""

    builder = StateGraph(EditState)

    builder.add_node("load_context", load_context)
    builder.add_node("analyze_image", analyze_image)
    builder.add_node("parse_request", parse_request)
    builder.add_node("build_plan", build_plan)
    builder.add_node("route_executor", route_executor)
    builder.add_node("execute_deterministic", execute_deterministic)
    builder.add_node("execute_generative", execute_generative)
    builder.add_node("execute_hybrid", execute_hybrid)
    builder.add_node("evaluate_result", evaluate_result)
    builder.add_node("human_review", human_review)
    builder.add_node("update_memory", update_memory)

    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "analyze_image")
    builder.add_edge("analyze_image", "parse_request")
    builder.add_edge("parse_request", "build_plan")
    builder.add_edge("build_plan", "route_executor")

    builder.add_conditional_edges(
        "route_executor",
        choose_executor,
        {
            "deterministic": "execute_deterministic",
            "generative": "execute_generative",
            "hybrid": "execute_hybrid",
        },
    )

    builder.add_edge("execute_deterministic", "evaluate_result")
    builder.add_edge("execute_generative", "evaluate_result")
    builder.add_edge("execute_hybrid", "evaluate_result")

    builder.add_conditional_edges(
        "evaluate_result",
        need_review,
        {
            "review": "human_review",
            "ok": "update_memory",
        },
    )

    builder.add_edge("human_review", "update_memory")
    builder.add_edge("update_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)
