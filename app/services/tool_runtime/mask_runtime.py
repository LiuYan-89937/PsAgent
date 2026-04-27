"""Mask resolution and reuse helpers for the neutral tool runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.graph.state import FocusKey, MaskCatalog, MaskCatalogItem, MaskQuality
from app.services.mask_quality import evaluate_mask_quality
from app.tools.common import MaskParams
from app.tools.common.tool_utils import temp_output_path
from app.tools.segmentation_tools import normalize_segmentation_prompt_label, resolve_region_mask


def normalized_mask_signature(mask_options: dict[str, Any], *, region: str) -> tuple[str, dict[str, Any]] | None:
    """Build a reusable mask signature independent of free-form region labels."""

    prompt_source = str(mask_options.get("prompt") or region or "").strip()
    if not prompt_source:
        return None
    normalized_prompt = normalize_segmentation_prompt_label(prompt_source, region=region)
    payload = {
        "provider": str(mask_options.get("provider") or "auto"),
        "normalized_mask_prompt": normalized_prompt,
        "negative_prompt": str(mask_options.get("negative_prompt") or ""),
        "semantic_type": bool(mask_options.get("semantic_type", False)),
        "fill_holes": bool(mask_options.get("fill_holes", False)),
        "expand_mask": int(mask_options.get("expand_mask") or 0),
        "blur_mask": bool(mask_options.get("blur_mask", False)),
        "use_grounding_dino": bool(mask_options.get("use_grounding_dino", False)),
        "revert_mask": bool(mask_options.get("revert_mask", False)),
    }
    signature = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return signature, payload


def record_mask_catalog_item(
    mask_catalog: MaskCatalog,
    *,
    signature: str,
    payload: dict[str, Any],
    focus: FocusKey | None,
    op_name: str,
    region_label: str,
    mask_path: str | None,
    preview_path: str | None,
    quality: MaskQuality | None = None,
) -> MaskCatalog:
    """Insert or update a reusable mask entry."""

    items = dict(mask_catalog.items)
    existing = items.get(signature)
    if existing is None:
        items[signature] = MaskCatalogItem(
            signature=signature,
            provider=payload["provider"],
            mask_prompt=payload["normalized_mask_prompt"],
            normalized_mask_prompt=payload["normalized_mask_prompt"],
            semantic_type=bool(payload.get("semantic_type", False)),
            revert_mask=bool(payload.get("revert_mask", False)),
            mask_path=mask_path,
            preview_path=preview_path,
            source_focus=focus,
            source_op=op_name,
            region_labels=[region_label],
            quality=quality,
            quality_score=quality.score if quality is not None else None,
            quality_flags=list(quality.flags) if quality is not None else [],
            rejected=bool(quality.rejected) if quality is not None else False,
        )
    else:
        updated = existing.model_copy(deep=True)
        if region_label not in updated.region_labels:
            updated.region_labels.append(region_label)
        updated.reuse_count += 1
        if not updated.mask_path and mask_path:
            updated.mask_path = mask_path
        if not updated.preview_path and preview_path:
            updated.preview_path = preview_path
        if quality is not None:
            updated.quality = quality
            updated.quality_score = quality.score
            updated.quality_flags = list(quality.flags)
            updated.rejected = quality.rejected
        items[signature] = updated
    return MaskCatalog(items=items)


def ensure_mask_size_for_image(mask_path: str, image_path: str) -> str:
    """Return a mask path matching the target image dimensions."""

    mask_source = Path(mask_path)
    image_source = Path(image_path)
    if not mask_source.exists() or not image_source.exists():
        return mask_path

    with Image.open(image_source) as image:
        target_size = image.size
    with Image.open(mask_source) as source_mask:
        mask = source_mask.convert("L")
        if mask.size == target_size:
            return mask_path
        resized = mask.resize(target_size, Image.Resampling.BILINEAR)
        output_path = temp_output_path("psagent_cached_mask_")
        resized.save(output_path)
        return output_path


def merge_mask_catalogs(mask_catalog: MaskCatalog, *sources: MaskCatalog) -> MaskCatalog:
    """Merge per-candidate mask catalogs into one reusable run catalog."""

    items = {signature: item.model_copy(deep=True) for signature, item in mask_catalog.items.items()}
    for source in sources:
        for signature, item in source.items.items():
            incoming = item.model_copy(deep=True)
            existing = items.get(signature)
            if existing is None:
                items[signature] = incoming
                continue

            updated = existing.model_copy(deep=True)
            for region_label in incoming.region_labels:
                if region_label not in updated.region_labels:
                    updated.region_labels.append(region_label)
            updated.reuse_count = max(updated.reuse_count, incoming.reuse_count)

            should_prefer_incoming = (updated.rejected and not incoming.rejected) or not updated.mask_path
            if should_prefer_incoming:
                updated.mask_path = incoming.mask_path
                updated.preview_path = incoming.preview_path
                updated.quality = incoming.quality
                updated.quality_score = incoming.quality_score
                updated.quality_flags = list(incoming.quality_flags)
                updated.rejected = incoming.rejected
            elif not updated.preview_path and incoming.preview_path:
                updated.preview_path = incoming.preview_path

            items[signature] = updated
    return MaskCatalog(items=items)


def generate_mask(
    image_path: str,
    *,
    region: str,
    mask_params: dict[str, Any],
    output_dir: str | None = None,
):
    """Generate one segmentation mask with normalized MaskParams."""

    mask_options = MaskParams.model_validate(mask_params).to_runtime_options() if mask_params else {}
    resolved_output_dir = output_dir or str(Path(image_path).resolve().parent / "output" / f"{Path(image_path).stem}_mask")
    return resolve_region_mask(
        image_path,
        region,
        output_dir=resolved_output_dir,
        **mask_options,
    )


def evaluate_generated_mask(mask_path: str) -> MaskQuality:
    """Evaluate mask quality and normalize the result into graph state schema."""

    quality = evaluate_mask_quality(mask_path)
    return MaskQuality.model_validate(quality.model_dump(mode="json") if hasattr(quality, "model_dump") else quality)
