"""Normalize routing-related state before executor dispatch."""

from app.graph.state import EditPlan, EditState
from app.tools.packages.macros import operations_require_hybrid


def route_executor(state: EditState) -> dict:
    """Prepare execution routing state.

    这个节点当前不直接执行任何图像处理，只负责把 planner 的结果收口成
    稳定的路由字段，避免 builder 里的条件判断依赖脏数据。
    """

    plan = dict(state.get("edit_plan") or {})
    operations = plan.get("operations", [])
    plan["executor"] = "hybrid" if operations_require_hybrid(operations) else plan.get("executor", "deterministic")
    validated_plan = EditPlan.model_validate(plan)

    return {
        "edit_plan": validated_plan.model_dump(mode="json"),
        "approval_required": state.get("approval_required", False),
    }
