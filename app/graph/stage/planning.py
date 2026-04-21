"""Planning and summarization helpers for the stage pipeline."""

from __future__ import annotations

from typing import Any

from app.graph.state import (
    EditOperation,
    EditPlan,
    EditState,
    PhaseArtifact,
    PlannerExecutionPlan,
    PlannerExecutionStep,
    StageContextEnvelope,
    StageKey,
    StagePolicy,
    coerce_edit_profile,
    coerce_image_analysis,
    coerce_request_intent,
)
from app.services.stage_policy import STAGE_LABELS, STAGE_ORDER
from app.tools.common import WHOLE_IMAGE_REGION


def merge_stage_plans_into_edit_plan(
    phases: dict[str, PhaseArtifact],
    *,
    mode: str,
    domain: str,
) -> EditPlan:
    """Aggregate all stage plans into one top-level display plan."""

    # 顶层 edit_plan 只是把 5 个阶段的 plan 重新串起来，
    # 方便 API / 前端统一展示，不参与真实执行调度。
    operations: list[EditOperation] = []
    preserve: list[str] = []
    memory_candidates: list[dict[str, Any]] = []
    should_write_memory = False
    needs_confirmation = False
    next_priority = 0

    for stage_key in STAGE_ORDER:
        phase = phases.get(stage_key)
        if phase is None or phase.plan is None:
            continue
        plan = phase.plan
        # preserve / memory_candidates / should_write_memory 等字段
        # 在阶段之间做去重和汇总，避免后续展示重复信息。
        for item in plan.preserve:
            if item not in preserve:
                preserve.append(item)
        for candidate in plan.memory_candidates:
            if candidate not in memory_candidates:
                memory_candidates.append(candidate)
        should_write_memory = should_write_memory or bool(plan.should_write_memory)
        needs_confirmation = needs_confirmation or bool(plan.needs_confirmation)
        for step in plan.steps:
            # 阶段内 step 会被线性展开成一个总的 operations 列表，
            # priority 用统一递增序号重建，保持前端展示稳定。
            operations.append(
                EditOperation(
                    op=step.op,
                    region=step.region,
                    strength=step.strength,
                    params=dict(step.params),
                    constraints=list(step.constraints),
                    priority=next_priority,
                )
            )
            next_priority += 1

    return EditPlan(
        mode=mode if mode in {"explicit", "auto"} else "auto",
        domain=domain if domain in {"portrait", "landscape", "food", "document", "general"} else "general",
        executor="deterministic",
        preserve=preserve,
        operations=operations,
        should_write_memory=should_write_memory,
        memory_candidates=memory_candidates,
        needs_confirmation=needs_confirmation,
    )


def should_skip_stage_deterministically(state: EditState, *, stage_key: StageKey) -> tuple[bool, str | None]:
    """Return whether a stage should be skipped deterministically."""

    # 这层是“无需调用模型就能确定”的跳过规则。
    # 目的是避免无意义阶段进入 planner，减少噪声和耗时。
    edit_profile = coerce_edit_profile(state.get("edit_profile"))
    if edit_profile is None:
        return False, None

    request_constraints = {
        constraint
        for constraint in (coerce_request_intent(state.get("request_intent")).constraints if state.get("request_intent") else [])
    }

    if stage_key == "technical_prep" and not edit_profile.technical_issues and not (
        request_constraints & {"fix_technical_issues", "straighten", "denoise"}
    ):
        return True, "当前没有明显技术预处理需求。"
    if stage_key == "local_balance" and not edit_profile.local_balance_needed:
        return True, "当前没有明显局部平衡需求。"
    if stage_key == "subject_refine" and not edit_profile.subject_refine_needed:
        return True, "当前主体不需要额外细化。"
    return False, None


def build_rule_stage_plan(
    *,
    stage_key: StageKey,
    stage_policy: StagePolicy,
    state: EditState,
    stage_context: StageContextEnvelope,
) -> PlannerExecutionPlan:
    """Build a small deterministic fallback plan when stage LLM planning is unavailable."""

    # 这里的规则计划是“模型不可用时的保底方案”，
    # 只做保守、可解释的少量步骤，不追求完整审美能力。
    request_text = str(state.get("request_text") or "")
    analysis = coerce_image_analysis(state.get("image_analysis"))
    request_intent = coerce_request_intent(state.get("request_intent"))
    requested_ops = {
        item.op
        for item in (request_intent.requested_packages if request_intent is not None else [])
        if item.op
    }
    fallback_params_by_tool: dict[str, dict[str, Any]] = {
        "adjust_noise_reduction": {"luma_strength": 8.0, "chroma_strength": 7.0, "detail_protection": 0.42},
        "adjust_color_noise_reduction": {"chroma_strength": 11.0, "detail_protection": 0.4},
        "adjust_chromatic_aberration": {"amount": 0.5, "radial_bias": 0.55},
        "adjust_defringe": {"purple_amount": 0.38, "green_amount": 0.28, "edge_threshold": 0.12},
        "adjust_exposure": {"strength": 0.45, "max_stops": 2.0},
        "adjust_brightness": {"brightness_offset": 0.08, "highlight_protection": 0.28},
        "adjust_contrast": {"strength": 0.38},
        "adjust_dehaze": {"amount": 0.3, "highlight_protection": 0.25, "color_protection": 0.25},
        "adjust_vibrance_saturation": {"strength": 0.3},
        "adjust_temperature_tint": {"temperature_shift": 6.0, "tint_shift": 0.0, "protect_saturated": 0.4},
        "adjust_color_balance": {"midtone_yellow_blue": 0.08, "preserve_luminosity": True},
        "adjust_point_color": {
            "target_color": "skin",
            "range_width": 22.0,
            "saturation_shift": -0.08,
            "luminance_shift": 0.08,
        },
        "adjust_single_color_shift": {
            "target_hue": 50.0,
            "hue_width": 16.0,
            "saturation_shift": -0.12,
            "luminance_shift": 0.08,
            "softness": 0.5,
        },
        "adjust_neutral_clean_tone": {
            "neutral_range": 0.22,
            "yellow_blue_shift": -0.08,
            "green_magenta_shift": 0.04,
            "brightness_shift": 0.04,
            "protect_skin": 0.4,
        },
        "adjust_skin_brightness": {
            "brightness_shift": 0.12,
            "saturation_shift": -0.04,
            "highlight_protection": 0.26,
            "preserve_texture": 0.65,
        },
        "adjust_skin_smooth": {"strength": 0.28, "smooth_strength": 0.3, "detail_protection": 0.68},
        "adjust_face_color_cleanup": {
            "yellow_reduce": 0.08,
            "magenta_balance": 0.04,
            "green_reduce": 0.0,
            "shadow_desaturate": 0.06,
        },
        "adjust_color_grading": {
            "shadow_hue": 220.0,
            "shadow_saturation": 0.06,
            "highlight_hue": 40.0,
            "highlight_saturation": 0.08,
            "balance": 0.0,
            "blending": 0.28,
        },
        "apply_photo_filter": {
            "filter_hue": 34.0,
            "filter_saturation": 0.18,
            "density": 0.18,
            "preserve_luminosity": True,
        },
    }

    def make_step(op: str, *, region: str = WHOLE_IMAGE_REGION, params: dict[str, Any] | None = None) -> PlannerExecutionStep:
        # 规则 fallback 直接产出已经接近执行态的 step，
        # 后面仍会经过统一的 planner/runtime normalize。
        return PlannerExecutionStep(op=op, region=region, params=dict(params or {}), constraints=[], priority=0)

    steps: list[PlannerExecutionStep] = []
    if stage_key == "technical_prep":
        # 技术预处理阶段只做最明确的技术修正，不碰风格化或主体精修。
        if analysis and any(issue in analysis.issues for issue in ("noise", "noisy", "high_iso_noise")):
            steps.append(make_step("adjust_noise_reduction", params=fallback_params_by_tool["adjust_noise_reduction"]))
        if "adjust_color_noise_reduction" in requested_ops:
            steps.append(
                make_step(
                    "adjust_color_noise_reduction",
                    params=fallback_params_by_tool["adjust_color_noise_reduction"],
                )
            )
        if analysis and any(issue in analysis.issues for issue in ("chromatic_aberration", "color_fringe")):
            steps.append(
                make_step(
                    "adjust_chromatic_aberration",
                    params=fallback_params_by_tool["adjust_chromatic_aberration"],
                )
            )
        if analysis and any(issue in analysis.issues for issue in ("purple_fringe", "green_fringe", "fringing")):
            steps.append(make_step("adjust_defringe", params=fallback_params_by_tool["adjust_defringe"]))
    elif stage_key == "global_base":
        # 全局基线阶段优先处理最确定的全图问题：曝光、反差、去灰雾和基础色彩。
        if analysis and ("underexposed" in analysis.issues or "偏暗" in request_text):
            steps.append(make_step("adjust_exposure", params=fallback_params_by_tool["adjust_exposure"]))
            steps.append(make_step("adjust_brightness", params=fallback_params_by_tool["adjust_brightness"]))
        if analysis and any(issue in analysis.issues for issue in ("flat", "low_contrast", "hazy")):
            steps.append(make_step("adjust_contrast", params=fallback_params_by_tool["adjust_contrast"]))
        if analysis and "hazy" in analysis.issues:
            steps.append(make_step("adjust_dehaze", params=fallback_params_by_tool["adjust_dehaze"]))
        # 颜色增强只消费上游已经归一化出的工具意图，不再硬编码风格词。
        if "adjust_vibrance_saturation" in requested_ops:
            steps.append(
                make_step(
                    "adjust_vibrance_saturation",
                    params=fallback_params_by_tool["adjust_vibrance_saturation"],
                )
            )
        if "adjust_temperature_tint" in requested_ops:
            steps.append(make_step("adjust_temperature_tint", params=fallback_params_by_tool["adjust_temperature_tint"]))
        if "adjust_color_balance" in requested_ops:
            steps.append(make_step("adjust_color_balance", params=fallback_params_by_tool["adjust_color_balance"]))
    elif stage_key == "local_balance":
        # 局部平衡阶段必须先有可复用 mask，
        # 否则规则计划宁可为空，也不臆造局部区域。
        if stage_context.available_masks:
            mask = stage_context.available_masks[0]
            region_label = (mask.get("region_labels") or ["局部区域"])[0]
            mask_params = {
                "mask_provider": mask.get("provider") or "fal_sam3",
                "mask_prompt": mask.get("mask_prompt") or "subject",
                "mask_semantic_type": bool(mask.get("semantic_type")),
            }
            if "adjust_point_color" in requested_ops:
                steps.append(
                    make_step(
                        "adjust_point_color",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_point_color"], **mask_params},
                    )
                )
            elif "adjust_single_color_shift" in requested_ops:
                steps.append(
                    make_step(
                        "adjust_single_color_shift",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_single_color_shift"], **mask_params},
                    )
                )
            elif "adjust_neutral_clean_tone" in requested_ops:
                steps.append(
                    make_step(
                        "adjust_neutral_clean_tone",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_neutral_clean_tone"], **mask_params},
                    )
                )
            else:
                steps.append(
                    make_step(
                        "adjust_exposure",
                        region=region_label,
                        params={**{"strength": 0.28}, **mask_params},
                    )
                )
    elif stage_key == "subject_refine":
        # 主体优化阶段同理，优先复用上游已有 mask，
        # 避免 fallback 方案自己触发复杂分割策略。
        if stage_context.available_masks:
            mask = stage_context.available_masks[0]
            region_label = (mask.get("region_labels") or ["主体区域"])[0]
            mask_params = {
                "mask_provider": mask.get("provider") or "fal_sam3",
                "mask_prompt": mask.get("mask_prompt") or "subject",
                "mask_semantic_type": bool(mask.get("semantic_type")),
            }
            if "adjust_skin_smooth" in requested_ops:
                steps.append(
                    make_step(
                        "adjust_skin_smooth",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_skin_smooth"], **mask_params},
                    )
                )
            if "adjust_face_color_cleanup" in requested_ops or "adjust_color_cleanup" in requested_ops:
                steps.append(
                    make_step(
                        "adjust_face_color_cleanup",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_face_color_cleanup"], **mask_params},
                    )
                )
            if not steps:
                steps.append(
                    make_step(
                        "adjust_skin_brightness",
                        region=region_label,
                        params={**fallback_params_by_tool["adjust_skin_brightness"], **mask_params},
                    )
                )
    elif stage_key == "finish_output":
        # 收尾阶段只在用户显式表达风格化意图时做保守收尾。
        if "adjust_color_grading" in requested_ops:
            steps.append(make_step("adjust_color_grading", params=fallback_params_by_tool["adjust_color_grading"]))
        if "apply_photo_filter" in requested_ops:
            steps.append(make_step("apply_photo_filter", params=fallback_params_by_tool["apply_photo_filter"]))

    # 最终仍受 stage 的 step_budget 限制，保证 fallback 行为可控。
    trimmed = [
        step.model_copy(update={"priority": index})
        for index, step in enumerate(steps[: stage_policy.step_budget])
    ]
    return PlannerExecutionPlan(
        mode=str(state.get("mode") or "auto"),
        domain=str((analysis.domain if analysis is not None else None) or "general"),
        executor="deterministic",
        preserve=[],
        steps=trimmed,
        step_budget=stage_policy.step_budget,
        summary=f"{STAGE_LABELS[stage_key]}规则计划",
        should_write_memory=False,
        memory_candidates=[],
        needs_confirmation=False,
    )
