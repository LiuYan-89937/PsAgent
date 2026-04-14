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

    # 根据规划结果里的 executor 字段，决定进入哪个执行器子图。
    plan = state.get("edit_plan") or {}
    return str(plan.get("executor", "deterministic"))


def need_review(state: EditState) -> str:
    """Route to review when the current result requires confirmation."""

    # 如果结果被标记为需要确认，则进入人工审核节点。
    return "review" if state.get("approval_required") else "ok"


def build_graph(checkpointer=None, store=None):
    """Create the application graph."""

    # 整个修图流程共享同一个 EditState。
    builder = StateGraph(EditState)

    # 主图节点：负责上下文加载、理解、规划、执行、评估和记忆更新。
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

    # 主链路：先读上下文，再分析图片、解析请求、构建计划。
    builder.add_edge(START, "load_context")
    builder.add_edge("load_context", "analyze_image")
    builder.add_edge("analyze_image", "parse_request")
    builder.add_edge("parse_request", "build_plan")
    builder.add_edge("build_plan", "route_executor")

    # 在路由节点决定后，分流到不同执行器。
    builder.add_conditional_edges(
        "route_executor",
        choose_executor,
        {
            "deterministic": "execute_deterministic",
            "generative": "execute_generative",
            "hybrid": "execute_hybrid",
        },
    )

    # 所有执行器最终都汇聚到结果评估节点。
    builder.add_edge("execute_deterministic", "evaluate_result")
    builder.add_edge("execute_generative", "evaluate_result")
    builder.add_edge("execute_hybrid", "evaluate_result")

    # 评估后决定是直接结束，还是先进入人工审核。
    builder.add_conditional_edges(
        "evaluate_result",
        need_review,
        {
            "review": "human_review",
            "ok": "update_memory",
        },
    )

    # 审核完成后统一进入记忆更新，再结束整个流程。
    builder.add_edge("human_review", "update_memory")
    builder.add_edge("update_memory", END)

    # checkpointer 管短期记忆，store 管长期记忆。
    return builder.compile(checkpointer=checkpointer, store=store)
