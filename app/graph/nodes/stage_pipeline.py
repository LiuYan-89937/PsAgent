"""Shared stage-pipeline node entrypoints."""

from __future__ import annotations

from app.graph.fallbacks import append_fallback_trace
from app.graph.stage.common import base_phase_artifact, compute_image_metrics, current_image_path, safe_stream_writer
from app.graph.stage.execution import execute_single_tool_call
from app.graph.stage.planning import (
    build_rule_stage_plan,
    merge_stage_plans_into_edit_plan,
    should_skip_stage_deterministically,
)
from app.graph.state import (
    EditState,
    EvaluationReport,
    PhaseOutputArtifact,
    PlannerExecutionPlan,
    StageKey,
    StageSummary,
    coerce_edit_profile,
    coerce_execution_trace,
    coerce_image_analysis,
    coerce_mask_catalog,
    coerce_phase_artifacts,
    coerce_request_intent,
    coerce_segmentation_trace,
    coerce_stage_context,
    coerce_stage_policy,
)
from app.services.planner_param_codec import (
    extract_runtime_mask_params,
    normalize_runtime_tool_params,
    strip_runtime_mask_params,
)
from app.services.planner_execution_model import (
    generate_stage_execution_plan_with_qwen,
    planner_execution_model_available,
)
from app.services.stage_context import build_stage_context
from app.services.stage_policy import STAGE_LABELS, STAGE_ORDER, resolve_stage_policy, summarize_policy_constraints
from app.tools.common import WHOLE_IMAGE_REGION
from app.tools.segmentation_tools import resolve_region_mask


TONE_STACK_OPS = {
    "adjust_exposure",
    "adjust_brightness",
    "adjust_contrast",
    "adjust_local_contrast",
    "adjust_highlights_shadows",
    "adjust_whites_blacks",
    "adjust_levels",
    "adjust_curves",
    "adjust_midtones",
    "adjust_temperature_tint",
    "adjust_vibrance_saturation",
    "adjust_hue_saturation",
    "adjust_color_balance",
    "adjust_color_mixer",
    "adjust_dehaze",
}


def prepare_stage_context(state: EditState, *, stage_key: StageKey) -> dict[str, object]:
    """Prepare the minimal context and effective policy for one stage."""

    # 这一层只做“当前阶段开始前”的上下文准备：
    # 取当前工作图、算有效 policy、整理可给 planner 的最小信息集。
    image_path = current_image_path(state)
    edit_profile = coerce_edit_profile(state.get("edit_profile"))
    image_analysis = coerce_image_analysis(state.get("image_analysis"))
    request_intent = coerce_request_intent(state.get("request_intent"))
    stage_policy = resolve_stage_policy(stage_key, edit_profile)
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    should_skip, skip_reason = should_skip_stage_deterministically(state, stage_key=stage_key)

    stage_context = build_stage_context(
        stage_key=stage_key,
        request_text=str(
            state.get("request_text")
            or (request_intent.goal_summary if request_intent is not None else "")
        ),
        current_image_path=image_path,
        image_analysis=image_analysis,
        edit_profile=edit_profile,
        mask_catalog=state.get("mask_catalog"),
        phases={key: value.model_dump(mode="json") for key, value in phases.items()},
        stage_constraints=[
            *(request_intent.constraints if request_intent is not None else []),
            *summarize_policy_constraints(stage_policy),
        ],
    )

    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    # skip 信息先写回 phase，后面的路由只看 phase.skipped。
    phase.skipped = should_skip
    phase.skip_reason = skip_reason
    phases[stage_key] = phase

    writer = safe_stream_writer()
    writer(
        {
            "event": "stage_started",
            "stage": stage_key,
            "message": f"开始{STAGE_LABELS[stage_key]}",
        },
    )

    return {
        "current_stage": stage_key,
        "stage_policy": stage_policy.model_dump(mode="json"),
        "stage_context": stage_context.model_dump(mode="json"),
        "stage_plan": None,
        "phases": phases,
        "mask_catalog": coerce_mask_catalog(state.get("mask_catalog")).model_dump(mode="json"),
    }


def should_skip_stage(state: EditState, *, stage_key: StageKey) -> str:
    """Route within the stage subgraph based on deterministic skip decision."""

    # LangGraph 的 conditional edge 只需要知道“skip 还是 run”，
    # 真正的原因已经提前写进 phase.skip_reason。
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    return "skip" if phase.skipped else "run"


def build_stage_plan(state: EditState, *, stage_key: StageKey) -> dict[str, object]:
    """Build a bounded stage plan with model-first and rule fallback behavior."""

    # 这一层的职责很单纯：
    # 能调模型就优先用模型规划；模型不可用就退回规则计划。
    stage_policy = coerce_stage_policy(state.get("stage_policy"))
    stage_context = coerce_stage_context(state.get("stage_context"))
    if stage_policy is None or stage_context is None:
        raise ValueError(f"Missing stage policy/context for {stage_key}")

    tool_catalog = state.get("tool_catalog") or []
    fallback_trace = list(state.get("fallback_trace") or [])
    image_path = current_image_path(state)

    writer = safe_stream_writer()
    writer(
        {
            "event": "planner_started",
            "stage": stage_key,
            "message": f"正在生成{STAGE_LABELS[stage_key]}计划",
        },
    )

    if stage_policy.llm_enabled and planner_execution_model_available():
        try:
            # 模型规划只负责产出本阶段 step，不直接执行任何工具。
            stage_plan = generate_stage_execution_plan_with_qwen(
                stage_policy=stage_policy,
                stage_context=stage_context,
                tool_catalog=tool_catalog,
                current_image_path=image_path,
                fallback_mode=str(state.get("mode") or "auto"),
                fallback_domain=str((coerce_image_analysis(state.get("image_analysis")).domain if state.get("image_analysis") else None) or "general"),
            )
        except Exception as error:
            # 规划失败不打断整条链路，而是清晰记录一次 fallback。
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="planner_model",
                location=stage_key,
                strategy="rule_based_plan",
                message="阶段规划失败，改用规则计划。",
                error=str(error),
            )
            stage_plan = build_rule_stage_plan(
                stage_key=stage_key,
                stage_policy=stage_policy,
                state=state,
                stage_context=stage_context,
            )
    else:
        # 模型不可用时直接走规则 fallback。
        stage_plan = build_rule_stage_plan(
            stage_key=stage_key,
            stage_policy=stage_policy,
            state=state,
            stage_context=stage_context,
        )

    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    phase.plan = stage_plan
    phases[stage_key] = phase
    writer(
        {
            "event": "planner_finished",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}计划已生成",
            "summary": stage_plan.summary,
            "num_steps": len(stage_plan.steps),
        },
    )
    return {
        "stage_plan": stage_plan.model_dump(mode="json"),
        "phases": phases,
        "fallback_trace": fallback_trace,
    }


def execute_stage_plan(state: EditState, *, stage_key: StageKey) -> dict[str, object]:
    """Execute one stage plan sequentially with stage-local guard semantics."""

    # 这层是阶段内真正的串行执行器：
    # 逐 step 调 execute_single_tool_call，并在阶段层处理 budget / guard。
    stage_policy = coerce_stage_policy(state.get("stage_policy"))
    stage_plan = state.get("stage_plan")
    stage_plan_obj = PlannerExecutionPlan.model_validate(stage_plan) if stage_plan is not None else None
    if stage_policy is None or stage_plan_obj is None:
        raise ValueError(f"Missing stage policy/plan for {stage_key}")

    image_path = current_image_path(state)
    candidate_outputs = list(state.get("candidate_outputs") or [])
    execution_trace = [item.model_dump(mode="json") for item in coerce_execution_trace(state.get("execution_trace") or [])]
    segmentation_trace = [item.model_dump(mode="json") for item in coerce_segmentation_trace(state.get("segmentation_trace") or [])]
    fallback_trace = list(state.get("fallback_trace") or [])
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    mask_catalog = coerce_mask_catalog(state.get("mask_catalog"))
    stage_execution_trace: list[dict[str, object]] = []
    stage_segmentation_trace: list[dict[str, object]] = []
    trigger_reasons = list(phase.trigger_reasons)
    stopped_early = False
    op_counts: dict[str, int] = {}
    tone_stack_count = 0

    writer = safe_stream_writer()
    writer(
        {
            "event": "stage_execution_started",
            "stage": stage_key,
            "message": f"开始执行{STAGE_LABELS[stage_key]}",
        },
    )

    current_image = image_path
    for step in stage_plan_obj.steps[: stage_policy.step_budget]:
        op_name = step.op
        op_counts[op_name] = op_counts.get(op_name, 0)
        # 单工具次数上限，防止 planner 在一个阶段里无限叠同一个工具。
        if op_counts[op_name] >= stage_policy.tool_repeat_limit:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message=f"{op_name} 已达到当前阶段单工具上限，提前结束阶段。",
                error=None,
            )
            trigger_reasons.append("tool_budget_exceeded")
            stopped_early = True
            break
        # tone stack 上限，防止连续色调类步骤把画面推得过头。
        if stage_policy.tone_stack_limit is not None and op_name in TONE_STACK_OPS and tone_stack_count >= stage_policy.tone_stack_limit:
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message="当前阶段 tone stack 已达到上限，提前结束阶段。",
                error=None,
            )
            trigger_reasons.append("tone_budget_exceeded")
            stopped_early = True
            break

        # 单步执行已经下沉到 stage.execution 模块；
        # 这里保留的是阶段层的串行控制和 guard。
        current_image, result_summary, mask_catalog = execute_single_tool_call(
            stage_key=stage_key,
            stage_policy=stage_policy,
            current_image=current_image,
            operation=step.model_dump(mode="json"),
            execution_trace=execution_trace,
            stage_execution_trace=stage_execution_trace,
            segmentation_trace=segmentation_trace,
            stage_segmentation_trace=stage_segmentation_trace,
            fallback_trace=fallback_trace,
            candidate_outputs=candidate_outputs,
            mask_catalog=mask_catalog,
            normalize_runtime_tool_params=normalize_runtime_tool_params,
            extract_runtime_mask_params=extract_runtime_mask_params,
            strip_runtime_mask_params=strip_runtime_mask_params,
            segmentation_resolver=resolve_region_mask,
            writer=writer,
        )
        op_counts[op_name] += 1
        if op_name in TONE_STACK_OPS:
            tone_stack_count += 1

        if stage_key == "global_base":
            # 全局基线阶段额外盯整图过曝风险，避免前几步把亮部直接推炸。
            metrics = compute_image_metrics(current_image)
            brightness_limit = stage_policy.guard_thresholds.get("brightness_mean_max")
            highlight_limit = stage_policy.guard_thresholds.get("highlight_ratio_max")
            if (
                (brightness_limit is not None and metrics["brightness_mean"] > brightness_limit)
                or (highlight_limit is not None and metrics["highlight_ratio"] > highlight_limit)
            ):
                fallback_trace = append_fallback_trace(
                    fallback_trace,
                    stage=stage_key,
                    source="execution_guard",
                    location=op_name,
                    strategy="stop_current_stage",
                    message="检测到全图过曝风险，提前结束当前阶段。",
                    error=None,
                )
                trigger_reasons.append("highlight_guard_triggered")
                stopped_early = True
                break

        if stage_key == "subject_refine" and result_summary.get("mask_path"):
            # 主体优化阶段如果是局部步骤，再单独看主体区域的亮度/高光/饱和度风险。
            metrics = compute_image_metrics(current_image, mask_path=str(result_summary["mask_path"]))
            if (
                metrics["brightness_mean"] > stage_policy.guard_thresholds.get("human_subject_brightness_mean_max", 242.0)
                or metrics["highlight_ratio"] > stage_policy.guard_thresholds.get("human_subject_highlight_ratio_max", 0.42)
                or metrics["saturation_mean"] < stage_policy.guard_thresholds.get("human_subject_saturation_mean_min", 0.04)
            ):
                fallback_trace = append_fallback_trace(
                    fallback_trace,
                    stage=stage_key,
                    source="execution_guard",
                    location=op_name,
                    strategy="stop_current_stage",
                    message="检测到主体局部异常风险，提前结束当前阶段。",
                    error=None,
                )
                trigger_reasons.append("subject_guard_triggered")
                stopped_early = True
                break

        if stage_key == "finish_output" and any(item.get("mask_path") for item in stage_execution_trace):
            # finish_output 过渡期不允许新增局部遮罩，发现后立即截停本阶段。
            fallback_trace = append_fallback_trace(
                fallback_trace,
                stage=stage_key,
                source="execution_guard",
                location=op_name,
                strategy="stop_current_stage",
                message="finish_output 不允许新增局部遮罩，提前结束当前阶段。",
                error=None,
            )
            trigger_reasons.append("mask_not_allowed")
            stopped_early = True
            break

    phase.execution_trace = coerce_execution_trace(stage_execution_trace)
    phase.segmentation_trace = coerce_segmentation_trace(stage_segmentation_trace)
    phase.output = PhaseOutputArtifact(image_path=current_image)
    phase.trigger_reasons = trigger_reasons
    phase.stopped_early = stopped_early
    phases[stage_key] = phase
    writer(
        {
            "event": "stage_execution_completed",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}执行完成",
        },
    )

    return {
        "selected_output": current_image,
        "candidate_outputs": candidate_outputs,
        "execution_trace": coerce_execution_trace(execution_trace),
        "segmentation_trace": coerce_segmentation_trace(segmentation_trace),
        "fallback_trace": fallback_trace,
        "phases": phases,
        "mask_catalog": mask_catalog.model_dump(mode="json"),
    }


def stage_guard(state: EditState, *, stage_key: StageKey) -> dict[str, object]:
    """Compute deterministic stage evaluation facts after execution or skip."""

    # stage_guard 不做美学判断，只把本阶段执行事实整理成一个统一 report。
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    if phase.skipped:
        report = EvaluationReport(
            selected_output=state.get("selected_output"),
            num_operations=0,
            success_count=0,
            failure_count=0,
            fallback_count=0,
            has_output=bool(state.get("selected_output")),
            overall_ok=True,
            artifact_ok=True,
            summary=phase.skip_reason or f"{STAGE_LABELS[stage_key]}已跳过。",
            should_continue_editing=False,
            should_request_review=False,
        )
    else:
        # 非 skip 阶段就基于 execution_trace 统计成功数、失败数和 fallback 数。
        trace_items = [item.model_dump(mode="json") for item in phase.execution_trace]
        success_count = sum(1 for item in trace_items if item.get("ok"))
        failure_count = sum(1 for item in trace_items if item.get("ok") is False)
        fallback_count = sum(1 for item in trace_items if item.get("fallback_used"))
        issues = []
        warnings = []
        if failure_count:
            issues.append("当前阶段存在失败步骤")
        if fallback_count:
            warnings.append("当前阶段触发了自动降级")
        if phase.stopped_early:
            warnings.append("当前阶段提前结束")
        report = EvaluationReport(
            selected_output=phase.output.image_path if phase.output is not None else state.get("selected_output"),
            num_operations=len(trace_items),
            success_count=success_count,
            failure_count=failure_count,
            fallback_count=fallback_count,
            has_output=bool(phase.output and phase.output.image_path),
            overall_ok=not issues,
            artifact_ok=not issues,
            issues=issues,
            warnings=warnings,
            summary=f"{STAGE_LABELS[stage_key]}已完成。",
            should_continue_editing=False,
            should_request_review=False,
        )
    phase.eval_report = report
    phases[stage_key] = phase
    return {"phases": phases}


def summarize_stage(state: EditState, *, stage_key: StageKey) -> dict[str, object]:
    """Summarize one stage and update the aggregated edit plan."""

    # summarize_stage 负责两件事：
    # 1. 给当前阶段写 summary
    # 2. 重新汇总顶层 edit_plan，方便 API/前端展示
    phases = dict(coerce_phase_artifacts(state.get("phases") or {}))
    phase = phases.get(stage_key) or base_phase_artifact(stage_key)
    request_intent = coerce_request_intent(state.get("request_intent"))
    image_analysis = coerce_image_analysis(state.get("image_analysis"))

    if phase.skipped:
        phase.summary = StageSummary(
            stage=stage_key,
            summary=phase.skip_reason or f"{STAGE_LABELS[stage_key]}已跳过。",
            used_tools=[],
            key_changes=[],
            remaining_issues=[],
        )
    else:
        used_tools = [item.op for item in phase.execution_trace if item.op]
        remaining_issues = list(
            (phase.eval_report.issues if phase.eval_report is not None else [])
            + (phase.eval_report.warnings if phase.eval_report is not None else [])
        )
        plan_summary = phase.plan.summary if phase.plan is not None else ""
        phase.summary = StageSummary(
            stage=stage_key,
            summary=plan_summary or (phase.eval_report.summary if phase.eval_report is not None else f"{STAGE_LABELS[stage_key]}已完成。"),
            used_tools=used_tools,
            key_changes=used_tools[:3],
            remaining_issues=remaining_issues[:3],
        )
    phases[stage_key] = phase

    edit_plan = merge_stage_plans_into_edit_plan(
        phases,
        mode=str(state.get("mode") or (request_intent.mode if request_intent is not None else "auto")),
        domain=str((image_analysis.domain if image_analysis is not None else None) or "general"),
    )
    writer = safe_stream_writer()
    writer(
        {
            "event": "stage_completed",
            "stage": stage_key,
            "message": f"{STAGE_LABELS[stage_key]}已完成",
        },
    )
    return {
        "current_stage": stage_key,
        "phases": phases,
        "edit_plan": edit_plan,
        "stage_context": None,
        "stage_plan": None,
        "stage_policy": None,
    }
