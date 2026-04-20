"""Stage subgraph factory for the retouch pipeline."""

from __future__ import annotations

from typing import Callable

from langgraph.graph import START, END, StateGraph

from app.graph.nodes.stage_pipeline import (
    build_stage_plan,
    execute_stage_plan,
    prepare_stage_context,
    should_skip_stage,
    stage_guard,
    summarize_stage,
)
from app.graph.state import EditState, StageKey


def _named(fn: Callable, name: str) -> Callable:
    """Assign a stable function name for LangGraph task labels."""

    setattr(fn, "__name__", name)
    return fn


def build_stage_subgraph(stage_key: StageKey):
    """Build one homomorphic stage subgraph."""

    builder = StateGraph(EditState)

    prepare_name = f"{stage_key}_prepare_stage_context"
    plan_name = f"{stage_key}_build_stage_plan"
    execute_name = f"{stage_key}_execute_stage_plan"
    guard_name = f"{stage_key}_stage_guard"
    summarize_name = f"{stage_key}_summarize_stage"

    builder.add_node(prepare_name, _named(lambda state: prepare_stage_context(state, stage_key=stage_key), prepare_name))
    builder.add_node(plan_name, _named(lambda state: build_stage_plan(state, stage_key=stage_key), plan_name))
    builder.add_node(execute_name, _named(lambda state: execute_stage_plan(state, stage_key=stage_key), execute_name))
    builder.add_node(guard_name, _named(lambda state: stage_guard(state, stage_key=stage_key), guard_name))
    builder.add_node(summarize_name, _named(lambda state: summarize_stage(state, stage_key=stage_key), summarize_name))

    builder.add_edge(START, prepare_name)
    builder.add_conditional_edges(
        prepare_name,
        _named(lambda state: should_skip_stage(state, stage_key=stage_key), f"{stage_key}_should_skip_stage"),
        {
            "skip": summarize_name,
            "run": plan_name,
        },
    )
    builder.add_edge(plan_name, execute_name)
    builder.add_edge(execute_name, guard_name)
    builder.add_edge(guard_name, summarize_name)
    builder.add_edge(summarize_name, END)

    return builder.compile()
