"""Model-driven per-round guidance and candidate generation."""

from __future__ import annotations

from typing import Any

from app.graph.state import FocusKey, ObjectiveCard, ObjectiveGap, RoundGuidance
from app.services.model_context import compact_tool_catalog_for_model, shared_mask_params_for_model
from app.services.model_runtime import DEFAULT_VISION_MODEL, invoke_json, model_available
from app.services.search_agent.planner import normalize_model_candidates


def round_guidance_model_available() -> bool:
    """Return whether the round guidance model can be called."""

    return model_available()


def _dump_gaps(gaps: list[ObjectiveGap]) -> list[dict[str, Any]]:
    return [
        {
            "id": gap.id,
            "focus": gap.focus,
            "description": gap.description,
            "priority": gap.priority,
            "target_region": gap.target_region,
            "desired_delta": gap.desired_delta,
            "constraints": list(gap.constraints),
        }
        for gap in gaps
    ]


def _string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text[:120])
        if len(items) >= limit:
            break
    return items


def build_round_guidance_payload(
    *,
    objective: ObjectiveCard,
    focus: FocusKey,
    round_gaps: list[ObjectiveGap],
    tool_catalog: list[dict[str, Any]],
    candidate_count: int,
    max_steps: int,
    is_recovery: bool = False,
    recovery_reason: str | None = None,
) -> dict[str, Any]:
    """Build the exact non-historical payload sent to the guidance model."""

    return {
        "任务": "为当前图片和当前 round 目标生成细节导向提示词与候选工具链",
        "总目标摘要": objective.summary,
        "领域": objective.domain,
        "当前round": {
            "focus": focus,
            "gaps": _dump_gaps(round_gaps),
            "is_recovery": is_recovery,
            "recovery_reason": recovery_reason or "",
        },
        "保留项": list(objective.preserve),
        "约束项": list(objective.constraints),
        "候选限制": {
            "candidate_count": candidate_count,
            "max_steps_per_candidate": max_steps,
            "allow_zero_step_candidate": True,
        },
        "工具目录": compact_tool_catalog_for_model(tool_catalog, include_params=True),
        "共享遮罩参数": shared_mask_params_for_model(tool_catalog),
        "输出JSON": {
            "target_prompt": "本轮中文细节导向提示词",
            "visual_diagnosis": "基于当前图片的简短观察",
            "preserve": ["本轮需要保留的视觉要点"],
            "avoid": ["本轮需要避免的副作用"],
            "candidates": [
                {
                    "label": "候选名称",
                    "summary": "候选策略说明",
                    "focus": focus,
                    "steps": [
                        {
                            "op": "工具名，必须来自工具目录 name",
                            "region": "whole_image 或语义区域，如 face area",
                            "params": {"参数名": "参数值"},
                        }
                    ],
                }
            ],
        },
    }


def generate_round_guidance(
    *,
    current_image_path: str,
    objective: ObjectiveCard,
    focus: FocusKey,
    round_gaps: list[ObjectiveGap],
    tool_catalog: list[dict[str, Any]],
    candidate_count: int,
    max_steps: int,
    is_recovery: bool = False,
    recovery_reason: str | None = None,
) -> RoundGuidance:
    """Generate model guidance for one round and normalize its candidate programs."""

    if not round_guidance_model_available():
        raise RuntimeError("OPENAI_API_KEY is not configured for round guidance.")

    payload = build_round_guidance_payload(
        objective=objective,
        focus=focus,
        round_gaps=round_gaps,
        tool_catalog=tool_catalog,
        candidate_count=candidate_count,
        max_steps=max_steps,
        is_recovery=is_recovery,
        recovery_reason=recovery_reason,
    )
    response = invoke_json(
        prompt_name="round_guidance.txt",
        user_payload=payload,
        model_env_name="OPENAI_VISION_MODEL",
        default_model=DEFAULT_VISION_MODEL,
        image_paths=[current_image_path],
        temperature=0.2,
    )
    candidates = normalize_model_candidates(
        response.get("candidates"),
        focus=focus,
        candidate_count=candidate_count,
        max_steps=max_steps,
        is_recovery=is_recovery,
    )
    return RoundGuidance(
        focus=focus,
        target_prompt=str(response.get("target_prompt") or "").strip(),
        visual_diagnosis=str(response.get("visual_diagnosis") or "").strip(),
        preserve=_string_list(response.get("preserve")),
        avoid=_string_list(response.get("avoid")),
        candidate_programs=candidates,
    )
