"""Build structured edit plans for round-based execution."""

from __future__ import annotations

from typing import Any

from app.graph.state import EditOperation, EditPlan, EditState, PackageCatalogItem, RequestIntent
from app.services.planner_model import generate_edit_plan_with_qwen, planner_model_available
from app.tools.packages import build_default_package_registry
from app.tools.packages.base import MASKED_REGION_MODE, WHOLE_IMAGE_REGION


STYLE_KEYWORDS = ("质感", "氛围", "色调", "夏日", "夏天", "通透", "空气感", "胶片", "明媚")
REPAIR_KEYWORDS = ("逆光", "背光", "修复", "提亮", "压高光", "层次", "肤色", "压暗")
PORTRAIT_KEYWORDS = ("人像", "人物", "肖像", "女生", "女孩", "男生", "男孩", "模特", "肤色", "脸", "皮肤")
FACE_SKIN_KEYWORDS = ("脸", "面部", "肤色", "皮肤", "脸部", "face", "skin", "neck")
HAIR_KEYWORDS = ("头发", "发丝", "发型", "hair")
CLOTHING_KEYWORDS = ("裙", "衣服", "服装", "连衣裙", "婚纱", "外套", "dress", "clothes", "clothing")
LEGACY_LOCAL_REGION_LABELS = frozenset({"person", "main_subject", "background", MASKED_REGION_MODE})


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether a text contains any of the provided keywords."""

    return any(keyword in text for keyword in keywords)


def _normalize_region_text(region: str | None) -> str:
    """Normalize an operation region label into a stable string."""

    normalized = str(region or WHOLE_IMAGE_REGION).strip()
    return normalized or WHOLE_IMAGE_REGION


def _coerce_requested_operations(requested_packages: list[dict[str, Any]]) -> list[EditOperation]:
    """Convert parse_request hints into validated edit operations."""

    registry = build_default_package_registry()
    operations: list[EditOperation] = []
    for priority, request in enumerate(requested_packages):
        package_name = request.get("op")
        package = registry.get(str(package_name)) if package_name else None
        params = package.get_operation_params(request) if package is not None else dict(request.get("params", {}))
        operations.append(
            EditOperation(
                op=request["op"],
                region=request.get("region", "whole_image"),
                strength=request.get("strength"),
                params=params,
                constraints=request.get("constraints", []),
                priority=priority,
            ),
        )
    return operations


def _append_operation(
    operations: list[EditOperation],
    *,
    op: str,
    region: str = "whole_image",
    params: dict[str, Any] | None = None,
    strength: float | None = None,
    constraints: list[str] | None = None,
) -> None:
    """Append a unique operation while preserving stable priorities."""

    if any(existing.op == op and existing.region == region for existing in operations):
        return
    operations.append(
        EditOperation(
            op=op,
            region=region,
            strength=strength,
            params=params or {},
            constraints=constraints or [],
            priority=len(operations),
        ),
    )


def _reseat_priorities(operations: list[EditOperation]) -> list[EditOperation]:
    """Return operations with contiguous priorities."""

    normalized: list[EditOperation] = []
    for priority, operation in enumerate(operations):
        normalized.append(
            EditOperation(
                op=operation.op,
                region=operation.region,
                strength=operation.strength,
                params=dict(operation.params),
                constraints=list(operation.constraints),
                priority=priority,
            ),
        )
    return normalized


def _request_constraints(state: EditState) -> set[str]:
    """Return high-level request constraints as a set."""

    request_intent = state.get("request_intent") or {}
    return {str(item) for item in request_intent.get("constraints", [])}


def _request_text(state: EditState) -> str:
    """Return the normalized request text."""

    return str(state.get("request_text") or "")


def _looks_like_portrait_request(state: EditState) -> bool:
    """Infer whether the current request is centered on a person/portrait subject."""

    analysis = state.get("image_analysis") or {}
    domain = str(analysis.get("domain") or "")
    subjects = " ".join(str(item) for item in analysis.get("subjects", []))
    text = " ".join(part for part in (_request_text(state), subjects) if part)
    return domain == "portrait" or _contains_any(text, PORTRAIT_KEYWORDS)


def _subject_mask_focus_from_request(state: EditState) -> tuple[str | None, str | None]:
    """Return a request-driven subject focus phrase when the user names a finer area."""

    text = _request_text(state)
    lowered = text.lower()

    if _contains_any(text, FACE_SKIN_KEYWORDS) or any(keyword in lowered for keyword in ("face", "skin", "neck")):
        return (
            "the subject's face, neck skin, and other visible skin that should receive this adjustment",
            "hair, clothing, accessories, background",
        )

    if _contains_any(text, HAIR_KEYWORDS) or "hair" in lowered:
        return (
            "the subject's hair and loose hair edges that should receive this adjustment",
            "face, skin, clothing, background",
        )

    if _contains_any(text, CLOTHING_KEYWORDS) or any(keyword in lowered for keyword in ("dress", "clothes", "clothing")):
        return (
            "the subject's clothing or dress area that should receive this adjustment",
            "face, skin, hair, background",
        )

    return None, None


def _build_dynamic_mask_defaults(
    state: EditState,
    operation: EditOperation,
) -> dict[str, Any]:
    """Generate default text-guided segmentation params for a local operation."""

    region = _normalize_region_text(operation.region)
    op = operation.op
    request_text = _request_text(state)
    constraints = _request_constraints(state)
    is_portrait = _looks_like_portrait_request(state)
    requested_focus, requested_negative = _subject_mask_focus_from_request(state)
    wants_backlight_repair = (
        "repair_backlighting" in constraints
        or _contains_any(request_text, ("逆光", "背光", "提亮主体", "压高光"))
    )
    wants_detail_work = op in {"adjust_clarity", "adjust_texture", "sharpen"}
    wants_color_work = op in {"adjust_color_mixer", "adjust_white_balance", "adjust_vibrance_saturation"}
    wants_tone_work = op in {"adjust_exposure", "adjust_highlights_shadows", "adjust_curves", "adjust_whites_blacks"}

    prompt: str
    negative_prompt: str | None

    if region == "background":
        if op == "adjust_dehaze":
            prompt = "the background haze and distant background areas behind the main subject"
        elif wants_color_work:
            prompt = "the background areas that should receive the requested color and atmosphere adjustment"
        elif wants_tone_work:
            prompt = "the background areas whose brightness and tone should be rebalanced"
        else:
            prompt = "the background area behind the main subject"
        negative_prompt = "main subject, person, face, skin, clothing, foreground objects"
    elif region == "person":
        if requested_focus:
            prompt = requested_focus
            negative_prompt = requested_negative
        elif wants_backlight_repair and wants_tone_work:
            prompt = "the person's face, neck skin, and upper body affected by backlight"
            negative_prompt = "hair, clothing, accessories, background"
        elif wants_color_work:
            prompt = "the visible person areas relevant to the requested color adjustment"
            negative_prompt = "background, unrelated objects"
        elif wants_detail_work:
            prompt = "the person's facial features, hair edges, and clothing details"
            negative_prompt = "background, distant scenery"
        else:
            prompt = "the person that should receive the local adjustment"
            negative_prompt = "background, unrelated objects"
    else:
        if requested_focus and is_portrait:
            prompt = requested_focus
            negative_prompt = requested_negative
        elif wants_backlight_repair and wants_tone_work and is_portrait:
            prompt = "the main subject's face, neck skin, and visible skin affected by backlight"
            negative_prompt = "hair, clothing, accessories, background"
        elif wants_backlight_repair and wants_tone_work:
            prompt = "the main subject area affected by the lighting imbalance"
            negative_prompt = "background and unrelated objects"
        elif wants_color_work and is_portrait:
            prompt = "the main subject's skin and other visible subject areas relevant to the color adjustment"
            negative_prompt = "background, sky, foliage, unrelated objects"
        elif wants_color_work:
            prompt = "the main subject areas relevant to the requested color and tone adjustment"
            negative_prompt = "background and unrelated objects"
        elif wants_detail_work and is_portrait:
            prompt = "the main subject's facial features, hair edges, and clothing details"
            negative_prompt = "background, sky, distant scenery"
        elif wants_detail_work:
            prompt = "the main subject's key textured details"
            negative_prompt = "background and unrelated objects"
        else:
            prompt = "the main subject area that should receive this local adjustment"
            negative_prompt = "background and unrelated objects"

    return {
        "mask_provider": "fal_sam3",
        "mask_prompt": prompt,
        "mask_negative_prompt": negative_prompt,
        "mask_semantic_type": True,
    }


def _finalize_operations(
    state: EditState,
    operations: list[EditOperation],
    *,
    registry,
) -> list[EditOperation]:
    """Inject default mask params for local edits and validate them against package schemas."""

    finalized: list[EditOperation] = []
    for priority, operation in enumerate(operations):
        package = registry.require(operation.op)
        params = dict(operation.params)
        normalized_region = _normalize_region_text(operation.region)

        if package.spec.mask_policy != "none" and params.get("mask_prompt"):
            params.setdefault("mask_provider", "fal_sam3")
            params.setdefault("mask_semantic_type", True)
            if normalized_region in LEGACY_LOCAL_REGION_LABELS:
                normalized_region = str(params["mask_prompt"])

        normalized = EditOperation(
            op=operation.op,
            region=normalized_region,
            strength=operation.strength,
            params=params,
            constraints=list(operation.constraints),
            priority=priority,
        )

        if not package.supports_operation(normalized.model_dump(mode="json")):
            raise ValueError(f"Unsupported region for {normalized.op}: {normalized.region}")
        package.parse_params(normalized.model_dump(mode="json"))
        finalized.append(normalized)

    return finalized


def _is_layered_request(state: EditState) -> bool:
    """Return whether the request likely needs multi-step refinement."""

    text = _request_text(state)
    constraints = _request_constraints(state)
    requested_packages = list((state.get("request_intent") or {}).get("requested_packages", []))
    issues = list((state.get("image_analysis") or {}).get("issues", []))

    if constraints & {"needs_layered_refinement", "repair_backlighting", "build_summer_mood", "match_reference_style"}:
        return True
    if _contains_any(text, STYLE_KEYWORDS) and _contains_any(text, REPAIR_KEYWORDS):
        return True
    if len(requested_packages) >= 3:
        return True
    if len(issues) >= 2 and (_contains_any(text, STYLE_KEYWORDS) or _contains_any(text, REPAIR_KEYWORDS)):
        return True
    return False


def _extend_with_requested_packages(
    state: EditState,
    operations: list[EditOperation],
    *,
    allowed_ops: set[str] | None = None,
) -> list[EditOperation]:
    """Append coarse requested packages that are not covered yet."""

    for request in _coerce_requested_operations((state.get("request_intent") or {}).get("requested_packages", [])):
        if allowed_ops is not None and request.op not in allowed_ops:
            continue
        if any(existing.op == request.op for existing in operations):
            continue
        operations.append(
            EditOperation(
                op=request.op,
                region=request.region,
                strength=request.strength,
                params=dict(request.params),
                constraints=list(request.constraints),
                priority=len(operations),
            ),
        )
    return _reseat_priorities(operations)


def _build_explicit_operations_round_1(state: EditState) -> list[EditOperation]:
    """Build a richer first-round plan for layered explicit requests."""

    text = _request_text(state)
    constraints = _request_constraints(state)
    issues = set((state.get("image_analysis") or {}).get("issues", []))
    operations: list[EditOperation] = []

    needs_backlight_repair = (
        "repair_backlighting" in constraints
        or _contains_any(text, ("逆光", "背光"))
        or {"underexposed", "crushed_shadows", "clipped_highlights"} & issues
    )
    needs_summer_mood = "build_summer_mood" in constraints or _contains_any(
        text,
        ("夏日", "夏天", "夏日感", "阳光感", "清透", "通透", "空气感", "明媚"),
    )
    wants_extra_structure = _contains_any(text, ("层次", "反差", "通透", "空气感"))

    if needs_backlight_repair:
        _append_operation(
            operations,
            op="adjust_exposure",
            region="main_subject",
            params={"strength": 0.28, "max_stops": 1.35, "feather_radius": 24.0},
        )
        _append_operation(
            operations,
            op="adjust_highlights_shadows",
            region="main_subject",
            params={
                "strength": 0.24,
                "tone_amount": 0.3,
                "feather_radius": 24.0,
                "midtone_contrast": 0.1,
            },
        )
        _append_operation(
            operations,
            op="adjust_curves",
            region="whole_image",
            params={
                "shadow_lift": 0.06,
                "midtone_gamma": 0.97,
                "highlight_compress": 0.08,
                "contrast_bias": 0.08,
            },
        )

    if needs_summer_mood:
        _append_operation(
            operations,
            op="adjust_white_balance",
            region="whole_image",
            params={"strength": 0.18, "tint": 0.04, "protect_saturated": 0.34},
        )
        _append_operation(
            operations,
            op="adjust_vibrance_saturation",
            region="whole_image",
            params={"strength": 0.18, "protect_highlights": 0.3, "protect_skin": 0.38},
        )

    if needs_summer_mood or wants_extra_structure:
        _append_operation(
            operations,
            op="adjust_dehaze",
            region="background",
            params={"amount": 0.12, "luminance_protection": 0.3, "feather_radius": 22.0},
        )

    if wants_extra_structure:
        _append_operation(
            operations,
            op="adjust_whites_blacks",
            region="whole_image",
            params={"whites_amount": 0.12, "blacks_amount": -0.05},
        )

    operations = _extend_with_requested_packages(state, operations)
    return operations or _build_auto_operations_round_1(state)


def _build_auto_operations_round_1(state: EditState) -> list[EditOperation]:
    """Create a conservative first-round enhancement plan from image facts."""

    analysis = state.get("image_analysis") or {}
    issues = set(analysis.get("issues", []))
    operations: list[EditOperation] = []

    if "underexposed" in issues:
        operations.append(
            EditOperation(
                op="adjust_exposure",
                params={"strength": 0.18},
                priority=len(operations),
            ),
        )
    elif "overexposed" in issues:
        operations.append(
            EditOperation(
                op="adjust_exposure",
                params={"strength": -0.16},
                priority=len(operations),
            ),
        )

    if "crushed_shadows" in issues or "clipped_highlights" in issues:
        operations.append(
            EditOperation(
                op="adjust_highlights_shadows",
                params={"strength": 0.18},
                priority=len(operations),
            ),
        )

    if "flat_contrast" in issues:
        operations.append(
            EditOperation(
                op="adjust_whites_blacks",
                params={"whites_amount": 0.16, "blacks_amount": 0.18},
                priority=len(operations),
            ),
        )
        operations.append(
            EditOperation(
                op="adjust_curves",
                params={
                    "shadow_lift": 0.08,
                    "midtone_gamma": 0.96,
                    "highlight_compress": 0.06,
                    "contrast_bias": 0.14,
                },
                priority=len(operations),
            ),
        )

    if not operations:
        operations.append(
            EditOperation(
                op="adjust_curves",
                params={
                    "shadow_lift": 0.04,
                    "midtone_gamma": 0.98,
                    "highlight_compress": 0.03,
                    "contrast_bias": 0.1,
                },
                priority=0,
            ),
        )

    return operations


def _build_explicit_operations_round_2(state: EditState) -> list[EditOperation]:
    """Build a second-round finishing plan for layered explicit requests."""

    text = _request_text(state)
    constraints = _request_constraints(state)
    round_1_report = dict((state.get("round_eval_reports") or {}).get("round_1", {}))
    critique = " ".join(
        str(part)
        for part in (
            round_1_report.get("summary"),
            " ".join(round_1_report.get("warnings", [])),
            " ".join(round_1_report.get("issues", [])),
        )
        if part
    ).lower()
    operations: list[EditOperation] = []

    needs_backlight_finish = (
        "repair_backlighting" in constraints
        or _contains_any(text, ("逆光", "背光"))
        or any(keyword in critique for keyword in ("暗", "under", "backlit", "主体仍偏暗"))
    )
    needs_summer_finish = "build_summer_mood" in constraints or _contains_any(
        text,
        ("夏日", "夏天", "夏日感", "阳光感", "清透", "通透", "空气感", "明媚"),
    )
    needs_detail_finish = _contains_any(text, ("质感", "细腻", "细节", "层次", "通透")) or any(
        keyword in critique for keyword in ("层次", "flat", "muddy", "通透", "空气感")
    )

    if needs_backlight_finish:
        _append_operation(
            operations,
            op="adjust_exposure",
            region="main_subject",
            params={"strength": 0.12, "max_stops": 1.2, "feather_radius": 24.0},
        )

    if needs_detail_finish:
        _append_operation(
            operations,
            op="adjust_clarity",
            region="main_subject",
            params={"amount": 0.16, "radius_scale": 1.1, "feather_radius": 20.0},
        )
        _append_operation(
            operations,
            op="adjust_texture",
            region="main_subject",
            params={"amount": 0.12, "detail_scale": 0.95, "feather_radius": 18.0},
        )

    if needs_summer_finish:
        _append_operation(
            operations,
            op="adjust_color_mixer",
            region="whole_image",
            params={
                "orange_saturation": 0.18,
                "yellow_luminance": 0.08,
                "blue_saturation": -0.08,
                "blue_luminance": -0.05,
                "saturation_protection": 0.34,
            },
        )
        _append_operation(
            operations,
            op="adjust_vibrance_saturation",
            region="whole_image",
            params={"strength": 0.12, "protect_highlights": 0.3, "protect_skin": 0.38},
        )
        _append_operation(
            operations,
            op="adjust_dehaze",
            region="background",
            params={"amount": 0.08, "feather_radius": 22.0},
        )

    if needs_detail_finish:
        _append_operation(
            operations,
            op="sharpen",
            region="whole_image",
            params={"strength": 0.08, "highlight_protection": 0.28},
        )

    operations = _extend_with_requested_packages(
        state,
        operations,
        allowed_ops={"crop_and_straighten", "denoise", "sharpen"},
    )
    return operations or _build_auto_operations_round_2(state)


def _build_auto_operations_round_2(state: EditState) -> list[EditOperation]:
    """Create a conservative finishing pass from first-round eval feedback."""

    report = (state.get("round_eval_reports") or {}).get("round_1", {})
    warnings = " ".join(report.get("warnings", []))
    issues = " ".join(report.get("issues", []))
    summary = str(report.get("summary") or "")
    critique = " ".join(part for part in (warnings, issues, summary) if part).lower()

    operations: list[EditOperation] = []

    if any(keyword in critique for keyword in ("暗", "dark", "提亮", "under")):
        operations.append(
            EditOperation(
                op="adjust_exposure",
                region="main_subject",
                params={"strength": 0.1, "feather_radius": 22.0},
                priority=len(operations),
            ),
        )

    if any(keyword in critique for keyword in ("灰", "flat", "层次", "muddy")):
        operations.append(
            EditOperation(
                op="adjust_clarity",
                region="main_subject",
                params={"amount": 0.18, "radius_scale": 1.1, "feather_radius": 20.0},
                priority=len(operations),
            ),
        )

    if any(keyword in critique for keyword in ("雾", "灰蒙", "dehaze", "空气感")):
        operations.append(
            EditOperation(
                op="adjust_dehaze",
                region="background",
                params={"amount": 0.16, "feather_radius": 20.0},
                priority=len(operations),
            ),
        )

    if any(keyword in critique for keyword in ("色", "偏色", "饱和", "颜色", "color")):
        operations.append(
            EditOperation(
                op="adjust_color_mixer",
                region="main_subject",
                params={"orange_saturation": 0.18, "yellow_luminance": 0.1, "feather_radius": 20.0},
                priority=len(operations),
            ),
        )

    if not operations:
        operations.append(
            EditOperation(
                op="adjust_texture",
                region="main_subject",
                params={"amount": 0.12, "detail_scale": 1.0, "feather_radius": 18.0},
                priority=0,
            ),
        )

    return operations


def _choose_executor(operations: list[EditOperation]) -> str:
    """Pick the minimal executor required by the current operation list."""

    if any(dict(operation.params).get("mask_prompt") for operation in operations):
        return "hybrid"
    return "deterministic"


def _build_plan_for_round(state: EditState, *, round_index: int) -> dict[str, Any]:
    """Build a validated plan for the requested round."""

    registry = build_default_package_registry()
    package_catalog = [
        PackageCatalogItem.model_validate(item).model_dump(mode="json")
        for item in state.get("package_catalog", registry.export_llm_catalog())
    ]
    request_intent = RequestIntent.model_validate(state.get("request_intent") or {"mode": "auto"}).model_dump(
        mode="json"
    )
    mode = str(state.get("mode") or request_intent.get("mode") or "auto")
    image_analysis = state.get("image_analysis") or {}
    input_images = state.get("input_images") or []
    round_key = f"round_{round_index}"
    existing_round_plans = dict(state.get("round_plans") or {})

    if planner_model_available():
        image_paths: list[str] = []
        if input_images:
            image_paths.append(input_images[0])
        if round_index == 2 and state.get("selected_output"):
            image_paths.append(str(state.get("selected_output")))

        previous_plan = existing_round_plans.get("round_1") if round_index == 2 else None
        previous_execution_trace = (state.get("round_execution_traces") or {}).get("round_1", []) if round_index == 2 else None
        previous_eval_report = (state.get("round_eval_reports") or {}).get("round_1", {}) if round_index == 2 else None

        plan = generate_edit_plan_with_qwen(
            request_text=str(state.get("request_text") or ""),
            request_intent=request_intent,
            image_analysis=image_analysis,
            package_catalog=package_catalog,
            retrieved_prefs=state.get("retrieved_prefs") or [],
            image_paths=image_paths,
            round_name=round_key,
            previous_plan=previous_plan,
            previous_execution_trace=previous_execution_trace,
            previous_eval_report=previous_eval_report,
        )
    else:
        if round_index == 1:
            requested_packages = request_intent.get("requested_packages", [])
            if mode == "explicit" and (_is_layered_request(state) or not requested_packages):
                operations = _build_explicit_operations_round_1(state)
            elif mode == "explicit" and requested_packages:
                operations = _coerce_requested_operations(requested_packages)
            else:
                mode = "auto"
                operations = _build_auto_operations_round_1(state)
        else:
            if mode == "explicit" and _is_layered_request(state):
                operations = _build_explicit_operations_round_2(state)
            else:
                mode = "auto"
                operations = _build_auto_operations_round_2(state)

        plan = EditPlan(
            mode=mode,
            domain=str(image_analysis.get("domain", "general")),
            executor=_choose_executor(operations),
            preserve=list(request_intent.get("constraints", [])),
            operations=operations,
            should_write_memory=False,
            memory_candidates=[],
            needs_confirmation=False,
        )

    finalized_operations = _finalize_operations(state, list(plan.operations), registry=registry)
    plan = EditPlan(
        mode=plan.mode,
        domain=plan.domain,
        executor=_choose_executor(finalized_operations),
        preserve=list(plan.preserve),
        operations=finalized_operations,
        should_write_memory=plan.should_write_memory,
        memory_candidates=list(plan.memory_candidates),
        needs_confirmation=plan.needs_confirmation,
    )

    round_plans = dict(existing_round_plans)
    round_plans[round_key] = plan.model_dump(mode="json")

    return {
        "package_catalog": package_catalog,
        "edit_plan": plan.model_dump(mode="json"),
        "round_plans": round_plans,
        "current_round": round_index,
    }


def build_plan_round_1(state: EditState) -> dict[str, Any]:
    """Build the first-round plan."""

    return _build_plan_for_round(state, round_index=1)


def build_plan_round_2(state: EditState) -> dict[str, Any]:
    """Build the second-round finishing plan."""

    return _build_plan_for_round(state, round_index=2)
