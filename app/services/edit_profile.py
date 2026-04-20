"""Edit profile inference from request intent and image analysis."""

from __future__ import annotations

from app.graph.state import AnalyzeImageResult, EditProfile, RequestIntent, SubjectCapabilities


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return whether the text contains any keyword."""

    return any(keyword in text for keyword in keywords)


def _infer_subject_count(subjects: list[str]) -> str:
    """Infer subject count from visible subject hints."""

    if not subjects:
        return "unknown"
    return "multiple" if len(subjects) > 1 else "single"


def _infer_main_subject_type(image_analysis: AnalyzeImageResult, request_intent: RequestIntent | None) -> str:
    """Infer the main subject type conservatively."""

    if image_analysis.main_subject_type:
        return image_analysis.main_subject_type
    if image_analysis.has_portrait:
        return "human"

    subject_text = " ".join(image_analysis.subjects).lower()
    if any(keyword in subject_text for keyword in ("person", "people", "face", "woman", "man", "girl", "boy")):
        return "human"
    if image_analysis.domain == "landscape":
        return "scene"
    if image_analysis.domain == "food":
        return "object"

    if request_intent is not None and request_intent.requires_local_editing:
        requested_regions = " ".join(item.region for item in request_intent.requested_packages).lower()
        if any(keyword in requested_regions for keyword in ("face", "hair", "skin", "person")):
            return "human"
    return "unknown"


def _infer_subject_capabilities(image_analysis: AnalyzeImageResult, main_subject_type: str) -> SubjectCapabilities:
    """Infer visible subject capabilities conservatively."""

    explicit = dict(image_analysis.subject_capabilities or {})
    if explicit:
        return SubjectCapabilities.model_validate(explicit)

    hints = " ".join(
        [
            *image_analysis.subjects,
            *image_analysis.segmentation_hints,
            image_analysis.primary_subject or "",
            image_analysis.summary or "",
        ]
    ).lower()
    is_human = main_subject_type == "human"
    return SubjectCapabilities(
        face_visible=is_human and _contains_any(hints, ("face", "person", "portrait", "head")),
        skin_visible=is_human and _contains_any(hints, ("skin", "face", "portrait", "neck")),
        hair_visible=is_human and _contains_any(hints, ("hair", "head", "portrait")),
        eyes_visible=is_human and _contains_any(hints, ("eyes", "eye", "face", "portrait")),
        teeth_visible=is_human and _contains_any(hints, ("teeth", "smile", "mouth")),
        lips_visible=is_human and _contains_any(hints, ("lips", "mouth", "smile")),
    )


def build_edit_profile(
    *,
    request_text: str,
    request_intent: RequestIntent | None,
    image_analysis: AnalyzeImageResult | None,
) -> EditProfile:
    """Build a stable profile for stage activation and tool exposure."""

    analysis = image_analysis or AnalyzeImageResult(domain="general")
    main_subject_type = _infer_main_subject_type(analysis, request_intent)
    technical_issues = [
        issue
        for issue in analysis.issues
        if issue
        in {
            "noise",
            "perspective",
            "crooked_horizon",
            "chromatic_aberration",
            "lens_distortion",
            "underexposed",
            "overexposed",
            "clipped_highlights",
            "crushed_shadows",
        }
    ]
    global_tone_issues = list(analysis.main_issues or analysis.issues or [])
    request_text_lower = request_text.lower()
    local_balance_needed = bool(analysis.needs_local_editing) or bool(analysis.has_background_distraction) or (
        request_intent.requires_local_editing if request_intent is not None else False
    )
    subject_refine_needed = main_subject_type in {"human", "object"} and (
        _contains_any(request_text_lower, ("主体", "细节", "质感", "清晰", "refine", "polish"))
        or main_subject_type == "human"
    )
    finish_needed = True

    return EditProfile(
        main_subject_type=main_subject_type,  # type: ignore[arg-type]
        subject_count=(analysis.subject_count or _infer_subject_count(analysis.subjects)),  # type: ignore[arg-type]
        technical_issues=technical_issues,
        global_tone_issues=global_tone_issues,
        local_balance_needed=local_balance_needed,
        subject_refine_needed=subject_refine_needed,
        finish_needed=finish_needed,
        subject_capabilities=_infer_subject_capabilities(analysis, main_subject_type),
    )
