"""Stage-context preparation helpers."""

from __future__ import annotations

from typing import Any

from app.graph.state import (
    AnalyzeImageResult,
    EditProfile,
    MaskCatalog,
    StageContextEnvelope,
    StageKey,
    StageSummary,
    coerce_mask_catalog,
    coerce_stage_summary,
)
from app.services.stage_policy import STAGE_ORDER, summarize_policy_constraints


def summarize_edit_profile_for_model(edit_profile: EditProfile | None) -> dict[str, Any]:
    """Build a compact edit-profile summary for a stage planner."""

    if edit_profile is None:
        return {}
    payload = edit_profile.model_dump(mode="json")
    capabilities = payload.pop("subject_capabilities", {})
    payload["subject_capabilities"] = {
        key: value for key, value in capabilities.items() if value
    }
    return payload


def summarize_available_masks(mask_catalog: MaskCatalog, *, current_stage: StageKey) -> list[dict[str, Any]]:
    """Build a compact list of reusable masks for stage planners."""

    items: list[dict[str, Any]] = []
    for item in mask_catalog.items.values():
        items.append(
            {
                "signature": item.signature,
                "mask_prompt": item.normalized_mask_prompt,
                "provider": item.provider,
                "semantic_type": item.semantic_type,
                "revert_mask": item.revert_mask,
                "region_labels": item.region_labels,
                "source_stage": item.source_stage,
                "reuse_count": item.reuse_count,
            }
        )
    return items


def summarize_previous_stages(
    phases: dict[str, Any],
    *,
    current_stage: StageKey,
) -> list[dict[str, Any]]:
    """Return summaries for stages that already ran before the current stage."""

    current_index = STAGE_ORDER.index(current_stage)
    summaries: list[dict[str, Any]] = []
    for stage_key in STAGE_ORDER[:current_index]:
        phase = phases.get(stage_key)
        if not isinstance(phase, dict):
            continue
        summary = coerce_stage_summary(phase.get("summary"))
        if summary is None:
            continue
        summaries.append(summary.model_dump(mode="json"))
    return summaries


def build_stage_context(
    *,
    stage_key: StageKey,
    request_text: str,
    current_image_path: str,
    image_analysis: AnalyzeImageResult | None,
    edit_profile: EditProfile | None,
    mask_catalog: MaskCatalog | dict[str, Any] | None,
    phases: dict[str, Any],
    stage_constraints: list[str],
) -> StageContextEnvelope:
    """Build the minimal context envelope for a given stage."""

    analysis = image_analysis.model_dump(mode="json") if image_analysis is not None else {}
    profile_summary = summarize_edit_profile_for_model(edit_profile)
    catalog = coerce_mask_catalog(mask_catalog)
    current_image_metrics = analysis.get("metrics") if isinstance(analysis.get("metrics"), dict) else {}

    relevant_image_analysis: dict[str, Any]
    if stage_key == "technical_prep":
        relevant_image_analysis = {
            "technical_issues": profile_summary.get("technical_issues", []),
            "metrics": current_image_metrics,
        }
    elif stage_key == "global_base":
        relevant_image_analysis = {
            "global_tone_issues": profile_summary.get("global_tone_issues", []),
            "metrics": current_image_metrics,
        }
    elif stage_key == "local_balance":
        relevant_image_analysis = {
            "local_balance_needed": profile_summary.get("local_balance_needed"),
            "issues": analysis.get("main_issues") or analysis.get("issues", []),
        }
    elif stage_key == "subject_refine":
        relevant_image_analysis = {
            "main_subject_type": profile_summary.get("main_subject_type"),
            "subject_capabilities": profile_summary.get("subject_capabilities", {}),
        }
    else:
        relevant_image_analysis = {
            "metrics": current_image_metrics,
            "global_tone_issues": profile_summary.get("global_tone_issues", []),
        }

    return StageContextEnvelope(
        request_summary=request_text,
        current_image_path=current_image_path,
        edit_profile_summary=profile_summary,
        relevant_image_analysis=relevant_image_analysis,
        available_masks=summarize_available_masks(catalog, current_stage=stage_key),
        previous_stage_summaries=summarize_previous_stages(phases, current_stage=stage_key),
        stage_constraints=list(stage_constraints),
    )
