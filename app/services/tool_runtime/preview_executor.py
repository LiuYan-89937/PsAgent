"""Preview execution for candidate search."""

from __future__ import annotations

from PIL import Image

from app.graph.state import CandidateProgram, MaskCatalog
from app.services.tool_runtime.chain_executor import ChainExecutionResult, execute_chain
from app.tools.common.tool_utils import temp_output_path


PREVIEW_LONG_EDGE = 1280


def _preview_image_path(image_path: str) -> str:
    """Create a preview copy of the current image for candidate search."""

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    output_path = temp_output_path("psagent_preview_")
    if longest <= PREVIEW_LONG_EDGE:
        image.save(output_path)
        return output_path

    scale = PREVIEW_LONG_EDGE / float(longest)
    resized = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    resized.save(output_path)
    return output_path


def _scaled_mask_path(mask_path: str, *, target_size: tuple[int, int]) -> str:
    """Scale one cached mask down to preview size."""

    mask = Image.open(mask_path).convert("L")
    resized = mask.resize(target_size, Image.Resampling.BILINEAR)
    output_path = temp_output_path("psagent_preview_mask_")
    resized.save(output_path)
    return output_path


def _preview_mask_catalog(mask_catalog: MaskCatalog, *, preview_image_path: str) -> MaskCatalog:
    """Build a preview-sized mask catalog from full-resolution cached masks."""

    target_size = Image.open(preview_image_path).convert("RGB").size
    items = {}
    for signature, item in mask_catalog.items.items():
        if not item.mask_path:
            continue
        items[signature] = item.model_copy(
            update={
                "mask_path": _scaled_mask_path(item.mask_path, target_size=target_size),
                "preview_path": None,
            }
        )
    return MaskCatalog(items=items)


def execute_preview(
    *,
    input_image_path: str,
    program: CandidateProgram,
    mask_catalog: MaskCatalog | None = None,
    round_id: str | None = None,
    max_steps: int = 2,
) -> ChainExecutionResult:
    """Run a candidate on a scaled preview without mutating full-resolution state."""

    preview_input = _preview_image_path(input_image_path)
    preview_catalog = _preview_mask_catalog(mask_catalog or MaskCatalog(), preview_image_path=preview_input)
    return execute_chain(
        input_image_path=preview_input,
        program=program,
        mask_catalog=preview_catalog,
        writer=lambda *_args, **_kwargs: None,
        mode="auto",
        round_id=round_id,
        focus=program.focus,
        candidate_id=program.id,
        max_steps=max_steps,
    )
