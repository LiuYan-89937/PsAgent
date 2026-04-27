"""Batch visual candidate review for one search round."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.graph.state import CandidateReview, FocusKey, ObjectiveGap, RoundGuidance, SearchCandidateArtifact
from app.services.model_runtime import DEFAULT_CANDIDATE_REVIEW_MODEL, invoke_json, model_available


@dataclass(slots=True)
class CandidateBatchReview:
    """Model score result for all candidates in one round."""

    selected_candidate_id: str | None
    reviews: dict[str, CandidateReview]
    eliminated_reasons: dict[str, str]


class CandidateScoreItem(BaseModel):
    """Strict candidate score item returned by the visual review model."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=5.0)


class CandidateScoreResponse(BaseModel):
    """Strict score-only response returned by the visual review model."""

    model_config = ConfigDict(extra="forbid")

    candidate_scores: list[CandidateScoreItem] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_candidate_ids(self) -> "CandidateScoreResponse":
        candidate_ids = [item.candidate_id for item in self.candidate_scores]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_scores contains duplicate candidate_id values.")
        return self


def candidate_review_model_available() -> bool:
    """Return whether the candidate review model can be called."""

    return model_available()


def _execution_facts(artifact: SearchCandidateArtifact) -> dict[str, Any]:
    execution = artifact.preview_execution
    if execution is None:
        return {
            "success_count": 0,
            "failure_count": 0,
            "fallback_count": 0,
            "ops": [],
        }
    return {
        "success_count": sum(1 for item in execution.execution_trace if item.ok),
        "failure_count": sum(1 for item in execution.execution_trace if item.ok is False),
        "fallback_count": len(execution.fallback_trace) + sum(1 for item in execution.execution_trace if item.fallback_used),
        "ops": [
            {
                "op": item.op,
                "region": item.region,
                "ok": item.ok,
                "fallback_used": item.fallback_used,
                "error": item.error,
            }
            for item in execution.execution_trace
        ],
    }


def _candidate_output_path(artifact: SearchCandidateArtifact) -> str | None:
    if artifact.preview_execution is None:
        return None
    output_path = artifact.preview_execution.output_image_path
    return str(output_path) if output_path else None


def _valid_image_paths(*, current_image_path: str, artifacts: list[SearchCandidateArtifact]) -> list[str] | None:
    current_path = str(current_image_path or "").strip()
    if not current_path:
        return None
    image_paths: list[str] = [current_path]
    for artifact in artifacts:
        output_path = _candidate_output_path(artifact)
        if not output_path:
            return None
        image_paths.append(output_path)
    if any(not Path(path).exists() for path in image_paths):
        return None
    return image_paths


def build_candidate_review_payload(
    *,
    target: str,
    artifacts: list[SearchCandidateArtifact],
) -> dict[str, Any]:
    """Build the target-and-images-only payload sent to the visual review model."""

    image_order = [{"image_index": 0, "role": "current_image"}]
    for index, artifact in enumerate(artifacts):
        image_order.append(
            {
                "image_index": index + 1,
                "role": "candidate_preview",
                "candidate_id": artifact.candidate_id,
            }
        )
    return {
        "任务": "以当前图为基线，按目标为每张候选 preview 图片打分",
        "目标": str(target or "").strip(),
        "图片顺序": image_order,
        "输出要求": {
            "score_range": "0-5",
            "candidate_scores": "必须覆盖每一个 candidate_id，只返回分数",
        },
    }


def _review_target(*, objective_summary: str, guidance: RoundGuidance) -> str:
    target = str(guidance.target_prompt or "").strip()
    if target:
        return target
    return str(objective_summary or "").strip()


def _float_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return max(0.0, min(5.0, score))


def _base_review(artifact: SearchCandidateArtifact) -> CandidateReview:
    if artifact.review is not None:
        return artifact.review
    return CandidateReview(summary=f"{artifact.label} 缺少预览评审。")


def _score_review_with_execution(artifact: SearchCandidateArtifact, score: float) -> CandidateReview:
    base = _base_review(artifact)
    issues = list(base.issues)
    warnings = list(base.warnings)
    facts = _execution_facts(artifact)
    failure_count = int(facts.get("failure_count") or 0)
    fallback_count = int(facts.get("fallback_count") or 0)
    action = "keep"
    if failure_count:
        action = "recover_same_round"
        score = 0.0
    elif fallback_count:
        action = "recover_same_round"
        score = max(0.0, score - fallback_count * 0.7)
    elif artifact.program is not None and not artifact.program.steps and action == "keep":
        action = "stop_round"

    return CandidateReview(
        overall_ok=base.overall_ok and not failure_count,
        preserve_ok=base.preserve_ok,
        style_ok=base.style_ok,
        artifact_ok=base.artifact_ok and not failure_count,
        issues=issues,
        warnings=warnings,
        summary=f"{artifact.label} 小模型视觉评分 {score:.2f}。",
        recommended_action=action,  # type: ignore[arg-type]
        score=round(score, 4),
    )


def _normalize_batch_response(
    response: dict[str, Any],
    *,
    artifacts: list[SearchCandidateArtifact],
) -> CandidateBatchReview:
    artifact_ids = {artifact.candidate_id for artifact in artifacts}
    try:
        validated = CandidateScoreResponse.model_validate(response)
    except ValidationError as exc:
        raise RuntimeError("Candidate review model returned invalid score-only JSON.") from exc

    returned_ids = {item.candidate_id for item in validated.candidate_scores}
    if returned_ids != artifact_ids:
        missing = sorted(artifact_ids - returned_ids)
        unknown = sorted(returned_ids - artifact_ids)
        raise RuntimeError(f"Candidate review model returned mismatched candidate ids. missing={missing}, unknown={unknown}")

    reviews: dict[str, CandidateReview] = {}
    for item in validated.candidate_scores:
        candidate_id = item.candidate_id
        artifact = next(item for item in artifacts if item.candidate_id == candidate_id)
        reviews[candidate_id] = _score_review_with_execution(artifact, item.score)

    selected_candidate_id = max(
        artifacts,
        key=lambda item: reviews[item.candidate_id].score if item.candidate_id in reviews else float("-inf"),
    ).candidate_id
    return CandidateBatchReview(
        selected_candidate_id=selected_candidate_id,
        reviews=reviews,
        eliminated_reasons={},
    )


def review_candidate_batch(
    *,
    current_image_path: str,
    objective_summary: str,
    focus: FocusKey,
    round_gaps: list[ObjectiveGap],
    guidance: RoundGuidance,
    artifacts: list[SearchCandidateArtifact],
) -> CandidateBatchReview | None:
    """Review all candidate previews in one small multimodal model call."""

    if not artifacts or not candidate_review_model_available():
        return None
    image_paths = _valid_image_paths(current_image_path=current_image_path, artifacts=artifacts)
    if image_paths is None:
        return None

    payload = build_candidate_review_payload(target=_review_target(objective_summary=objective_summary, guidance=guidance), artifacts=artifacts)
    response = invoke_json(
        prompt_name="candidate_review.txt",
        user_payload=payload,
        model_env_name="OPENAI_CANDIDATE_REVIEW_MODEL",
        default_model=DEFAULT_CANDIDATE_REVIEW_MODEL,
        image_paths=image_paths,
        temperature=0.05,
    )
    return _normalize_batch_response(response, artifacts=artifacts)
