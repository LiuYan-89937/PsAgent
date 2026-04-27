"""Candidate validation helpers for round-first search."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.graph.state import CandidateProgram, FocusKey, PlannerExecutionStep
from app.services.planner_param_codec import decode_planner_argument_value
from app.tools import WHOLE_IMAGE_REGION
from app.tools import require_tool_spec


CANDIDATE_COUNT = 3
RECOVERY_CANDIDATE_COUNT = 2
CANDIDATE_PREVIEW_STEP_LIMIT = 3
RECOVERY_PREVIEW_STEP_LIMIT = 2
MIN_NON_STOP_CANDIDATE_STEPS = 2


NATURAL_PARAM_BOUNDS: dict[str, dict[str, tuple[float | None, float | None]]] = {
    "adjust_exposure": {
        "strength": (0.0, 0.45),
        "max_stops": (0.5, 2.0),
    },
    "adjust_brightness": {
        "brightness_offset": (-0.28, 0.28),
        "highlight_protection": (0.28, None),
    },
    "adjust_contrast": {
        "strength": (0.0, 0.42),
        "contrast_scale": (None, 0.9),
    },
    "adjust_local_contrast": {
        "amount": (-0.35, 0.35),
        "edge_protection": (0.35, None),
    },
    "adjust_highlights_shadows": {
        "shadow_amount": (-0.45, 0.45),
        "highlight_amount": (-0.45, 0.45),
        "detail_amount": (0.0, 0.55),
    },
    "adjust_whites_blacks": {
        "whites_amount": (-0.35, 0.3),
        "blacks_amount": (-0.3, 0.35),
    },
    "adjust_curves": {
        "shadow_lift": (-0.35, 0.35),
        "midtone_gamma": (0.7, 1.55),
        "highlight_compress": (-0.3, 0.45),
        "contrast_bias": (-0.32, 0.32),
    },
    "adjust_midtones": {
        "midtone_shift": (-0.24, 0.24),
    },
    "adjust_temperature_tint": {
        "temperature_shift": (-12.0, 12.0),
        "tint_shift": (-10.0, 10.0),
    },
    "adjust_vibrance_saturation": {
        "strength": (0.0, 0.45),
        "vibrance_scale": (0.2, 0.95),
        "saturation_scale": (0.0, 0.38),
        "protect_highlights": (0.28, None),
        "protect_skin": (0.42, None),
        "protect_shadows": (0.28, None),
    },
    "adjust_hue_saturation": {
        "saturation_shift": (-0.25, 0.25),
        "luminance_shift": (-0.2, 0.2),
        "protect_skin": (0.42, None),
    },
    "adjust_skin_brightness": {
        "brightness_shift": (-0.08, 0.24),
        "saturation_shift": (-0.18, 0.14),
        "highlight_protection": (0.35, None),
        "preserve_texture": (0.68, None),
    },
    "adjust_face_color_cleanup": {
        "yellow_reduce": (0.0, 0.28),
        "magenta_balance": (0.0, 0.22),
        "green_reduce": (0.0, 0.22),
        "shadow_desaturate": (0.0, 0.22),
    },
    "adjust_skin_tone_balance": {
        "skin_saturation_shift": (-0.18, 0.16),
        "warmth_shift": (-0.14, 0.14),
        "luminance_shift": (-0.12, 0.16),
        "protection": (0.42, None),
    },
    "adjust_skin_smooth": {
        "strength": (0.0, 0.32),
        "smooth_strength": (0.0, 0.28),
        "detail_protection": (0.68, None),
        "saturation_protection": (0.28, None),
    },
    "adjust_skin_texture_reduce": {
        "amount": (0.0, 0.3),
        "detail_preserve": (0.68, None),
        "tone_preserve": (0.82, None),
    },
    "adjust_texture": {
        "amount": (-0.32, 0.34),
        "noise_protection": (0.42, None),
    },
    "adjust_clarity": {
        "amount": (-0.32, 0.34),
        "highlight_protection": (0.32, None),
        "shadow_protection": (0.28, None),
    },
    "adjust_sharpness": {
        "amount": (0.0, 0.85),
        "highlight_protection": (0.38, None),
        "threshold": (0.015, None),
    },
    "adjust_dehaze": {
        "amount": (-0.28, 0.3),
        "luminance_protection": (0.28, None),
        "color_protection": (0.32, None),
    },
    "adjust_soft_glow": {
        "amount": (0.0, 0.24),
        "contrast_restore": (0.18, None),
        "highlight_bias": (0.0, 0.5),
    },
    "adjust_glow_highlights": {
        "amount": (0.0, 0.22),
        "threshold": (0.72, None),
        "warmth": (-0.12, 0.12),
    },
    "adjust_vignette": {
        "amount": (-0.28, 0.28),
        "feather": (0.45, None),
    },
    "apply_color_lookup": {
        "strength": (0.0, 0.28),
    },
    "apply_photo_filter": {
        "density": (0.0, 0.18),
    },
    "adjust_color_grading": {
        "shadow_saturation": (0.0, 0.22),
        "midtone_saturation": (0.0, 0.18),
        "highlight_saturation": (0.0, 0.18),
        "balance": (-0.35, 0.35),
        "blending": (0.0, 0.38),
    },
}


def _step(op: str, *, region: str = WHOLE_IMAGE_REGION, params: dict | None = None, priority: int = 0) -> PlannerExecutionStep:
    return PlannerExecutionStep(op=op, region=region, params=dict(params or {}), priority=priority)


def _candidate(
    *,
    focus: FocusKey,
    label: str,
    summary: str,
    steps: list[PlannerExecutionStep],
    source: str,
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


def build_stop_candidate(
    *,
    focus: FocusKey,
    label: str = "停止当前轮",
    summary: str = "本轮不追加工具调用，等待后续评估或人工复核。",
    is_recovery: bool = False,
) -> CandidateProgram:
    """Build a bounded 0-step candidate used when model guidance asks to stop or fails."""

    return _candidate(
        focus=focus,
        label=label,
        summary=summary,
        steps=[],
        source="noop",
        is_recovery=is_recovery,
    )


def _coerce_text(value: Any, *, fallback: str, max_length: int = 80) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    return text[:max_length]


def _decode_model_params(op_name: str, raw_params: Any) -> dict[str, Any]:
    if not isinstance(raw_params, dict):
        return {}
    schema_properties = require_tool_spec(op_name).planner_schema.get("properties") or {}
    params: dict[str, Any] = {}
    for key, value in raw_params.items():
        spec = schema_properties.get(key) if isinstance(schema_properties, dict) else None
        if isinstance(spec, dict):
            params[key] = decode_planner_argument_value(value, spec)
    return params


def _clamp_number(value: Any, *, lower: float | None, upper: float | None) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    result = float(value)
    if lower is not None:
        result = max(lower, result)
    if upper is not None:
        result = min(upper, result)
    return result


def _naturalize_step_params(step: PlannerExecutionStep) -> PlannerExecutionStep:
    bounds = NATURAL_PARAM_BOUNDS.get(step.op)
    if not bounds:
        return step
    params = dict(step.params)
    changed = False
    for key, (lower, upper) in bounds.items():
        if key not in params:
            continue
        next_value = _clamp_number(params[key], lower=lower, upper=upper)
        if next_value != params[key]:
            params[key] = next_value
            changed = True
    if not changed:
        return step
    return step.model_copy(update={"params": params})


def _coerce_step(raw_step: Any, *, priority: int) -> PlannerExecutionStep | None:
    if not isinstance(raw_step, dict):
        return None
    op_name = str(raw_step.get("op") or "")
    try:
        decoded_params = _decode_model_params(op_name, raw_step.get("params"))
    except Exception:
        return None
    payload = {
        "op": op_name,
        "region": raw_step.get("region") or WHOLE_IMAGE_REGION,
        "params": decoded_params,
        "constraints": raw_step.get("constraints") if isinstance(raw_step.get("constraints"), list) else [],
        "priority": raw_step.get("priority", priority),
    }
    if raw_step.get("strength") is not None:
        payload["strength"] = raw_step.get("strength")
    try:
        return _naturalize_step_params(PlannerExecutionStep.model_validate(payload))
    except ValidationError:
        return None


def _coerce_candidate(
    raw_candidate: Any,
    *,
    focus: FocusKey,
    index: int,
    max_steps: int,
    is_recovery: bool,
) -> CandidateProgram | None:
    if isinstance(raw_candidate, CandidateProgram):
        candidate = raw_candidate.model_copy(deep=True)
        candidate.id = f"{focus}_{'recovery_' if is_recovery else ''}{uuid4().hex[:8]}"
        candidate.focus = focus
        candidate.is_recovery = is_recovery
        candidate.steps = candidate.steps[:max_steps]
        candidate.source = "noop" if not candidate.steps else "model"
        return candidate
    if not isinstance(raw_candidate, dict):
        return None

    raw_steps = raw_candidate.get("steps")
    steps = [
        step
        for step in (
            _coerce_step(raw_step, priority=step_index)
            for step_index, raw_step in enumerate(raw_steps if isinstance(raw_steps, list) else [])
        )
        if step is not None
    ][:max_steps]
    if not is_recovery and 0 < len(steps) < MIN_NON_STOP_CANDIDATE_STEPS:
        return None

    label = _coerce_text(raw_candidate.get("label"), fallback=f"候选 {index + 1}")
    summary = _coerce_text(raw_candidate.get("summary"), fallback="模型生成的候选工具链。", max_length=180)
    return _candidate(
        focus=focus,
        label=label,
        summary=summary,
        steps=steps,
        source="model" if steps else "noop",
        is_recovery=is_recovery,
    )


def normalize_model_candidates(
    raw_candidates: Any,
    *,
    focus: FocusKey,
    candidate_count: int,
    max_steps: int,
    is_recovery: bool = False,
) -> list[CandidateProgram]:
    """Normalize model candidate payloads into safe CandidateProgram objects."""

    candidates: list[CandidateProgram] = []
    iterable = raw_candidates if isinstance(raw_candidates, list) else []
    for index, raw_candidate in enumerate(iterable):
        candidate = _coerce_candidate(
            raw_candidate,
            focus=focus,
            index=index,
            max_steps=max_steps,
            is_recovery=is_recovery,
        )
        if candidate is not None:
            candidates.append(candidate)
        if len(candidates) >= candidate_count:
            break

    while len(candidates) < candidate_count:
        candidates.append(
            build_stop_candidate(
                focus=focus,
                label="停止当前轮" if not is_recovery else "停止恢复",
                summary="模型未提供足够可安全执行的候选，本候选保持当前结果。",
                is_recovery=is_recovery,
            )
        )
    return candidates
