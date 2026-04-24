"""Candidate generation for round-first search."""

from __future__ import annotations

from uuid import uuid4

from app.graph.state import CandidateProgram, FocusKey, ObjectiveCard, PlannerExecutionStep, RequestIntent
from app.tools import WHOLE_IMAGE_REGION


CANDIDATE_COUNT = 3
RECOVERY_CANDIDATE_COUNT = 2
CANDIDATE_PREVIEW_STEP_LIMIT = 2
RECOVERY_PREVIEW_STEP_LIMIT = 2


def _step(op: str, *, region: str = WHOLE_IMAGE_REGION, params: dict | None = None, priority: int = 0) -> PlannerExecutionStep:
    return PlannerExecutionStep(op=op, region=region, params=dict(params or {}), priority=priority)


def _candidate(
    *,
    focus: FocusKey,
    label: str,
    summary: str,
    steps: list[PlannerExecutionStep],
    source: str = "variant",
    is_recovery: bool = False,
) -> CandidateProgram:
    return CandidateProgram(
        id=f"{focus}_{'recovery_' if is_recovery else ''}{uuid4().hex[:8]}",
        label=label,
        focus=focus,
        source=source,  # type: ignore[arg-type]
        summary=summary,
        steps=steps[: RECOVERY_PREVIEW_STEP_LIMIT if is_recovery else CANDIDATE_PREVIEW_STEP_LIMIT],
        is_recovery=is_recovery,
    )


def generate_candidates(*, objective: ObjectiveCard, focus: FocusKey, round_index: int) -> list[CandidateProgram]:
    """Generate deterministic candidate programs for one auto round."""

    if focus == "global_tone":
        candidates = [
            _candidate(
                focus=focus,
                label="亮度优先",
                summary="优先建立整体亮度和暗部可读性。",
                steps=[
                    _step("adjust_exposure", params={"strength": 0.24}, priority=0),
                    _step("adjust_highlights_shadows", params={"shadow_amount": 0.18, "highlight_amount": 0.08, "midtone_contrast": 0.12}, priority=1),
                ],
            ),
            _candidate(
                focus=focus,
                label="层次优先",
                summary="优先整理黑白场和中间调对比。",
                steps=[
                    _step("adjust_highlights_shadows", params={"shadow_amount": 0.12, "highlight_amount": 0.14, "midtone_contrast": 0.14}, priority=0),
                    _step("adjust_whites_blacks", params={"whites_amount": 0.08, "blacks_amount": -0.06}, priority=1),
                ],
            ),
            _candidate(
                focus=focus,
                label="颜色干净度",
                summary="轻推自然饱和并清理偏色。",
                steps=[
                    _step("adjust_vibrance_saturation", params={"strength": 0.18}, priority=0),
                    _step("adjust_color_balance", params={"midtone_yellow_blue": 0.04, "preserve_luminosity": True}, priority=1),
                ],
            ),
        ]
    elif focus == "subject_separation":
        candidates = [
            _candidate(
                focus=focus,
                label="主体提亮",
                summary="用人物遮罩提升主体可读性。",
                steps=[
                    _step(
                        "adjust_exposure",
                        region="person area",
                        params={"strength": 0.22, "mask_provider": "fal_sam3", "mask_prompt": "person", "mask_semantic_type": True},
                    )
                ],
            ),
            _candidate(
                focus=focus,
                label="背景压住",
                summary="轻压背景亮度，让主体更突出。",
                steps=[
                    _step(
                        "adjust_brightness",
                        region="background area",
                        params={"brightness_offset": -0.12, "highlight_protection": 0.28, "mask_provider": "fal_sam3", "mask_prompt": "background", "mask_semantic_type": True},
                    )
                ],
            ),
            _candidate(
                focus=focus,
                label="局部停手",
                summary="当前轮不做局部提交，避免错误遮罩污染结果。",
                steps=[],
                source="noop",
            ),
        ]
    elif focus == "subject_cleanup":
        candidates = [
            _candidate(
                focus=focus,
                label="肤色提纯",
                summary="轻量提亮肤色并清理脸部脏色。",
                steps=[
                    _step("adjust_skin_brightness", region="skin area", params={"brightness_shift": 0.1, "saturation_shift": -0.03, "mask_provider": "fal_sam3", "mask_prompt": "skin", "mask_semantic_type": True}),
                    _step("adjust_face_color_cleanup", region="face area", params={"yellow_reduce": 0.1, "magenta_balance": 0.03, "mask_provider": "fal_sam3", "mask_prompt": "face", "mask_semantic_type": True}),
                ],
            ),
            _candidate(
                focus=focus,
                label="质感整理",
                summary="轻柔整理皮肤纹理，保留自然细节。",
                steps=[
                    _step("adjust_skin_smooth", region="skin area", params={"strength": 0.2, "smooth_strength": 0.22, "detail_protection": 0.72, "mask_provider": "fal_sam3", "mask_prompt": "skin", "mask_semantic_type": True}),
                    _step("adjust_texture", region="skin area", params={"amount": -0.08, "mask_provider": "fal_sam3", "mask_prompt": "skin", "mask_semantic_type": True}),
                ],
            ),
            _candidate(
                focus=focus,
                label="发丝整理",
                summary="改善发丝分离度和局部完成度。",
                steps=[
                    _step("adjust_hair_enhance", region="hair area", params={"texture_boost": 0.18, "highlight_control": 0.1, "mask_provider": "fal_sam3", "mask_prompt": "hair", "mask_semantic_type": True})
                ],
            ),
        ]
    else:
        candidates = [
            _candidate(
                focus=focus,
                label="自然收口",
                summary="轻暗角把视线收回主体。",
                steps=[_step("adjust_vignette", params={"amount": 0.12, "midpoint": 0.58, "roundness": 0.5, "feather": 0.82})],
            ),
            _candidate(
                focus=focus,
                label="颜色收口",
                summary="轻量调整中性色，让结果更完整。",
                steps=[_step("adjust_selective_color", params={"target_band": "neutrals", "cyan_shift": 0.02, "yellow_shift": -0.02, "black_shift": 0.02})],
            ),
            _candidate(
                focus=focus,
                label="停止当前轮",
                summary="不再追加收尾工具，保留当前结果。",
                steps=[],
                source="noop",
            ),
        ]
    return candidates[:CANDIDATE_COUNT]


def generate_recovery_candidates(*, focus: FocusKey, reason: str) -> list[CandidateProgram]:
    """Generate bounded same-round recovery candidates."""

    if focus in {"subject_separation", "subject_cleanup"}:
        candidates = [
            _candidate(
                focus=focus,
                label="回退为全图轻修",
                summary=f"局部候选出现问题，改用全图轻量修正：{reason}",
                steps=[_step("adjust_clarity", params={"amount": 0.08})],
                source="recovery",
                is_recovery=True,
            ),
            _candidate(
                focus=focus,
                label="当前轮停手",
                summary="局部风险较高，本轮不继续补救。",
                steps=[],
                source="noop",
                is_recovery=True,
            ),
        ]
    else:
        candidates = [
            _candidate(
                focus=focus,
                label="保守补救",
                summary=f"用保守色调修正补救：{reason}",
                steps=[_step("adjust_highlights_shadows", params={"shadow_amount": 0.04, "highlight_amount": 0.02, "midtone_contrast": 0.04})],
                source="recovery",
                is_recovery=True,
            ),
            _candidate(
                focus=focus,
                label="当前轮停手",
                summary="本轮已有足够收益，不继续补救。",
                steps=[],
                source="noop",
                is_recovery=True,
            ),
        ]
    return candidates[:RECOVERY_CANDIDATE_COUNT]


def generate_direct_candidate(*, request_intent: RequestIntent | None, objective: ObjectiveCard) -> CandidateProgram:
    """Build one direct explicit-mode candidate without search."""

    steps: list[PlannerExecutionStep] = []
    if request_intent is not None:
        for index, hint in enumerate(request_intent.requested_tools[:4]):
            params = dict(hint.params or {})
            if hint.strength is not None and params.get("strength") is None:
                params["strength"] = hint.strength
            steps.append(_step(hint.op, region=hint.region, params=params, priority=index))
        if not steps:
            for index, goal in enumerate(request_intent.goals[:2]):
                kind = goal.kind.lower()
                params = {"strength": min(max(abs(goal.intensity or 0.3), 0.05), 0.8)}
                if "bright" in kind or "luminance" in kind or "exposure" in kind:
                    steps.append(_step("adjust_exposure", region=goal.target_region, params=params, priority=index))
                elif "contrast" in kind or "tonal" in kind:
                    steps.append(_step("adjust_contrast", region=goal.target_region, params=params, priority=index))
                elif "color" in kind or "saturation" in kind:
                    steps.append(_step("adjust_vibrance_saturation", region=goal.target_region, params=params, priority=index))
    if not steps:
        steps.append(_step("adjust_exposure", params={"strength": 0.2}, priority=0))

    focus = objective.gaps[0].focus if objective.gaps else "global_tone"
    return _candidate(
        focus=focus,
        label="直接执行",
        summary="显式模式按用户请求直接执行工具链。",
        steps=steps[:4],
        source="direct",
    )
