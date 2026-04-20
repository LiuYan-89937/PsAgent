"""Stage policies and stage-order helpers for the retouch pipeline."""

from __future__ import annotations

from app.graph.state import EditProfile, StageKey, StagePolicy


STAGE_ORDER: tuple[StageKey, ...] = (
    "technical_prep",
    "global_base",
    "local_balance",
    "subject_refine",
    "finish_output",
)

STAGE_LABELS: dict[StageKey, str] = {
    "technical_prep": "技术预处理",
    "global_base": "全局基线",
    "local_balance": "局部平衡",
    "subject_refine": "主体优化",
    "finish_output": "最终收尾",
}

_BASE_POLICIES: dict[StageKey, StagePolicy] = {
    "technical_prep": StagePolicy(
        key="technical_prep",
        label=STAGE_LABELS["technical_prep"],
        prompt_name="technical_prep.txt",
        visible_tools=[
            "adjust_exposure",
            "adjust_contrast",
            "adjust_vibrance_saturation",
        ],
        llm_enabled=True,
        step_budget=2,
        tool_repeat_limit=1,
        tone_stack_limit=None,
        mask_allowed=False,
        mask_required=False,
        context_whitelist=[
            "request_summary",
            "technical_issues",
            "metrics",
            "current_image_path",
        ],
        guard_thresholds={},
    ),
    "global_base": StagePolicy(
        key="global_base",
        label=STAGE_LABELS["global_base"],
        prompt_name="global_base.txt",
        visible_tools=[
            "adjust_exposure",
            "adjust_contrast",
            "adjust_vibrance_saturation",
        ],
        llm_enabled=True,
        step_budget=4,
        tool_repeat_limit=2,
        tone_stack_limit=4,
        mask_allowed=False,
        mask_required=False,
        context_whitelist=[
            "request_summary",
            "global_tone_issues",
            "previous_stage_summaries",
            "current_image_path",
        ],
        guard_thresholds={
            "brightness_mean_max": 220.0,
            "highlight_ratio_max": 0.28,
        },
    ),
    "local_balance": StagePolicy(
        key="local_balance",
        label=STAGE_LABELS["local_balance"],
        prompt_name="local_balance.txt",
        visible_tools=[
            "adjust_exposure",
            "adjust_contrast",
            "adjust_vibrance_saturation",
        ],
        llm_enabled=True,
        step_budget=3,
        tool_repeat_limit=2,
        tone_stack_limit=2,
        mask_allowed=True,
        mask_required=False,
        context_whitelist=[
            "request_summary",
            "local_balance_needed",
            "available_masks",
            "previous_stage_summaries",
            "current_image_path",
        ],
        guard_thresholds={},
    ),
    "subject_refine": StagePolicy(
        key="subject_refine",
        label=STAGE_LABELS["subject_refine"],
        prompt_name="subject_refine.txt",
        visible_tools=[
            "adjust_exposure",
            "adjust_contrast",
            "adjust_vibrance_saturation",
        ],
        llm_enabled=True,
        step_budget=3,
        tool_repeat_limit=2,
        tone_stack_limit=1,
        mask_allowed=True,
        mask_required=False,
        context_whitelist=[
            "request_summary",
            "edit_profile_summary",
            "available_masks",
            "previous_stage_summaries",
            "current_image_path",
        ],
        guard_thresholds={
            "human_subject_brightness_mean_max": 242.0,
            "human_subject_highlight_ratio_max": 0.42,
            "human_subject_saturation_mean_min": 0.04,
        },
    ),
    "finish_output": StagePolicy(
        key="finish_output",
        label=STAGE_LABELS["finish_output"],
        prompt_name="finish_output.txt",
        visible_tools=[
            "adjust_exposure",
            "adjust_contrast",
            "adjust_vibrance_saturation",
        ],
        llm_enabled=True,
        step_budget=2,
        tool_repeat_limit=1,
        tone_stack_limit=1,
        mask_allowed=False,
        mask_required=False,
        context_whitelist=[
            "request_summary",
            "metrics",
            "previous_stage_summaries",
            "current_image_path",
        ],
        guard_thresholds={},
    ),
}

def stage_sort_key(stage_key: str) -> int:
    """Return a stable sort key for stage display order."""

    try:
        return STAGE_ORDER.index(stage_key) + 1
    except ValueError:
        return len(STAGE_ORDER) + 100


def resolve_stage_policy(stage_key: StageKey, edit_profile: EditProfile | None) -> StagePolicy:
    """Return the effective stage policy for the current edit profile."""

    return _BASE_POLICIES[stage_key].model_copy(deep=True)


def summarize_policy_constraints(policy: StagePolicy) -> list[str]:
    """Build a compact human/model-friendly list of stage constraints."""

    constraints: list[str] = [
        f"step_budget={policy.step_budget}",
        f"tool_repeat_limit={policy.tool_repeat_limit}",
    ]
    if policy.tone_stack_limit is not None:
        constraints.append(f"tone_stack_limit={policy.tone_stack_limit}")
    constraints.append("mask_allowed=true" if policy.mask_allowed else "mask_allowed=false")
    constraints.append("mask_required=true" if policy.mask_required else "mask_required=false")
    return constraints
