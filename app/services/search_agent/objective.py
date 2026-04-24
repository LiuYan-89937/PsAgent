"""Objective-card construction for the search-first agent."""

from __future__ import annotations

from uuid import uuid4

from app.graph.state import AnalyzeImageResult, FocusKey, ObjectiveCard, ObjectiveGap, RequestGoal, RequestIntent


def _gap(
    *,
    focus: FocusKey,
    description: str,
    priority: int,
    target_region: str = "whole_image",
    desired_delta: str = "",
    constraints: list[str] | None = None,
) -> ObjectiveGap:
    return ObjectiveGap(
        id=f"{focus}_{uuid4().hex[:8]}",
        focus=focus,
        description=description,
        priority=priority,
        target_region=target_region,
        desired_delta=desired_delta,
        constraints=list(constraints or []),
    )


def _focus_for_goal(goal: RequestGoal) -> FocusKey:
    kind = goal.kind.lower()
    region = goal.target_region.lower()
    if any(token in kind for token in ("skin", "face", "hair", "teeth", "lips")):
        return "subject_cleanup"
    if "background" in kind or "subject" in region or "person" in region or "background" in region:
        return "subject_separation"
    if any(token in kind for token in ("finish", "style", "detail", "noise")):
        return "finish"
    return "global_tone"


def _issue_gaps(image_analysis: AnalyzeImageResult | None) -> list[ObjectiveGap]:
    if image_analysis is None:
        return []
    issues = set(image_analysis.main_issues or image_analysis.issues or [])
    gaps: list[ObjectiveGap] = []
    if issues & {"underexposed", "overexposed", "flat_contrast", "low_saturation", "clipped_highlights", "compressed_tonal_range"}:
        gaps.append(
            _gap(
                focus="global_tone",
                description="建立整体明暗、对比和颜色基线。",
                priority=82,
                desired_delta=", ".join(sorted(issues)),
                constraints=["preserve_original_mood"],
            )
        )
    if image_analysis.needs_local_editing or image_analysis.has_background_distraction:
        gaps.append(
            _gap(
                focus="subject_separation",
                description="改善主体与背景的分离和局部可读性。",
                priority=72,
                target_region=image_analysis.primary_subject or "subject area",
                constraints=["do_not_affect_unrelated_regions"],
            )
        )
    if image_analysis.has_portrait or image_analysis.domain == "portrait" or image_analysis.main_subject_type == "human":
        gaps.append(
            _gap(
                focus="subject_cleanup",
                description="整理人像主体的肤色、脸部或发丝细节。",
                priority=64,
                target_region="face and skin area",
                constraints=["preserve_identity", "avoid_over_smoothing"],
            )
        )
    return gaps


def build_objective_card(
    *,
    request_text: str,
    request_intent: RequestIntent | None,
    image_analysis: AnalyzeImageResult | None,
    mode: str,
) -> ObjectiveCard:
    """Build a compact objective card for round search."""

    effective_mode = "auto" if mode == "auto" else "explicit"
    domain = image_analysis.domain if image_analysis is not None else "general"
    goals = list(request_intent.goals if request_intent is not None else [])
    constraints = list(request_intent.constraints if request_intent is not None else [])
    gaps = _issue_gaps(image_analysis)

    for goal in goals:
        focus = _focus_for_goal(goal)
        gaps.append(
            _gap(
                focus=focus,
                description=goal.kind.replace("_", " "),
                priority=goal.priority,
                target_region=goal.target_region,
                desired_delta=str(goal.intensity) if goal.intensity is not None else "",
                constraints=goal.constraints,
            )
        )

    if not gaps:
        gaps.append(
            _gap(
                focus="global_tone",
                description="自然改善整体观感。",
                priority=50,
                constraints=["avoid_overediting"],
            )
        )
    if effective_mode == "auto" and not any(gap.focus == "finish" for gap in gaps):
        gaps.append(
            _gap(
                focus="finish",
                description="轻量收口，避免处理痕迹。",
                priority=30,
                constraints=["avoid_overediting"],
            )
        )

    preserve = ["identity", "natural_texture", "original_mood"]
    if "preserve_original_mood" in constraints and "original_mood" not in preserve:
        preserve.append("original_mood")

    return ObjectiveCard(
        summary=(request_intent.goal_summary if request_intent is not None and request_intent.goal_summary else request_text).strip()
        or "智能美化并提升整体观感",
        mode=effective_mode,
        domain=domain,
        preserve=preserve,
        goals=goals,
        gaps=sorted(gaps, key=lambda gap: gap.priority, reverse=True),
        constraints=constraints,
    )
