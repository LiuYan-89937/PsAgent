"""Segmentation helpers with Aliyun and fal segmentation backends.

当前项目默认仍保留阿里云主体分割链路，确保原有 `person/main_subject/background`
不会回退。与此同时新增 fal 的文本引导分割，便于后续做更细的语义
区域选择，例如 face / hair / dress / water splash 等。
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlopen

try:
    import fal_client
except ImportError:  # pragma: no cover - covered indirectly by runtime error handling
    fal_client = None

from alibabacloud_imageseg20191230.client import Client as ImageSegClient
from alibabacloud_imageseg20191230.models import (
    GetAsyncJobResultRequest,
    SegmentHDCommonImageAdvanceRequest,
)
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from app.services.env import load_project_env
from PIL import Image


ALIYUN_ACCESS_KEY_ID_ENV = "ALIBABA_CLOUD_ACCESS_KEY_ID"
ALIYUN_ACCESS_KEY_SECRET_ENV = "ALIBABA_CLOUD_ACCESS_KEY_SECRET"
ALIYUN_REGION_ID_ENV = "ALIBABA_CLOUD_REGION_ID"
ALIYUN_IMAGESEG_ENDPOINT_ENV = "ALIYUN_IMAGESEG_ENDPOINT"
ALIYUN_SEGMENT_CONNECT_TIMEOUT_ENV = "ALIYUN_SEGMENT_CONNECT_TIMEOUT_MS"
ALIYUN_SEGMENT_READ_TIMEOUT_ENV = "ALIYUN_SEGMENT_READ_TIMEOUT_MS"
ALIYUN_SEGMENT_MAX_ATTEMPTS_ENV = "ALIYUN_SEGMENT_MAX_ATTEMPTS"
ALIYUN_SEGMENT_POLL_INTERVAL_ENV = "ALIYUN_SEGMENT_POLL_INTERVAL_SECONDS"

FAL_KEY_ENV = "FAL_KEY"
FAL_SAM3_MODEL_ENV = "FAL_SAM3_MODEL"
FAL_SAM3_START_TIMEOUT_ENV = "FAL_SAM3_START_TIMEOUT_SECONDS"
FAL_SAM3_CLIENT_TIMEOUT_ENV = "FAL_SAM3_CLIENT_TIMEOUT_SECONDS"
PSAGENT_SEGMENTATION_PROVIDER_ENV = "PSAGENT_SEGMENTATION_PROVIDER"
PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV = "PSAGENT_SEGMENTATION_FALLBACK_PROVIDER"

SegmentationProvider = Literal["auto", "aliyun", "fal_sam3"]


class SegmentationProviderError(RuntimeError):
    """Raised when a segmentation provider cannot satisfy the request."""


class AliyunImageSegError(SegmentationProviderError):
    """Raised when the Aliyun image segmentation pipeline cannot complete."""


class FalImageSegError(SegmentationProviderError):
    """Raised when the fal SAM 3 segmentation pipeline cannot complete."""


class FalImageSegAttemptsError(FalImageSegError):
    """Raised when fal segmentation exhausts multiple prompt strategies."""

    def __init__(self, message: str, *, attempts: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(slots=True)
class SegmentationResult:
    """Standardized result for a provider-backed segmentation request."""

    provider: str
    binary_mask_path: str
    original_image_path: str
    api_chain: tuple[str, ...]
    segmentation_rgba_path: str | None = None
    remote_mask_url: str | None = None
    prompt: str | None = None
    negative_prompt: str | None = None
    request_id: str | None = None
    raw_response: dict[str, Any] | None = None
    region: str | None = None
    target_label: str | None = None
    semantic_type: bool | None = None
    fallback_used: bool = False
    requested_provider: str | None = None
    attempt_index: int | None = None
    attempt_strategy: str | None = None
    requested_prompt: str | None = None
    effective_prompt: str | None = None
    revert_mask: bool | None = None
    attempts: tuple[dict[str, Any], ...] = ()


AliyunSegmentationResult = SegmentationResult
FalSegmentationResult = SegmentationResult


def _pick_value(obj: Any, *names: str) -> Any:
    """Read a value from an SDK object or dict using multiple candidate keys."""

    if obj is None:
        return None

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _slugify_fragment(value: str, *, fallback: str) -> str:
    """Build a filesystem-safe fragment for output directories."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned[:40] or fallback


def _as_bool(value: bool | str | None, *, default: bool) -> bool:
    """Normalize common bool-like env and parameter values."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_provider(value: str | None, *, default: SegmentationProvider = "auto") -> SegmentationProvider:
    """Normalize provider names and aliases."""

    if not value:
        return default

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "auto": "auto",
        "aliyun": "aliyun",
        "fal": "fal_sam3",
        "fal_sam3": "fal_sam3",
        "sam3": "fal_sam3",
        "sam_3": "fal_sam3",
    }
    provider = aliases.get(normalized)
    if provider is None:
        raise SegmentationProviderError(f"Unsupported segmentation provider: {value}")
    return provider


def _resolve_segmentation_provider(
    *,
    region: str,
    provider: str | None = None,
    prompt: str | None = None,
) -> SegmentationProvider:
    """Resolve the active provider for a given mask request."""

    load_project_env()
    requested = _normalize_provider(provider or os.getenv(PSAGENT_SEGMENTATION_PROVIDER_ENV), default="auto")
    if requested != "auto":
        return requested

    if prompt:
        return "fal_sam3"

    # 当前 planner 产出的稳定 region 仍以粗粒度为主，auto 下默认保留 Aliyun。
    if region in {"person", "main_subject", "background"}:
        return "aliyun"
    return "fal_sam3"


def _resolve_fallback_provider(
    *,
    active_provider: SegmentationProvider,
    prompt: str | None = None,
) -> SegmentationProvider | None:
    """Resolve a fallback provider without silently changing prompt semantics."""

    load_project_env()
    if prompt:
        # 显式文本目标如果失败，不应该悄悄降级成粗分割。
        return None

    raw_value = os.getenv(PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV)
    if not raw_value:
        return None

    fallback = _normalize_provider(raw_value, default="auto")
    if fallback == "auto":
        fallback = "aliyun" if active_provider != "aliyun" else "fal_sam3"
    if fallback == active_provider:
        return None
    return fallback


def _download_remote_image(url: str, output_path: str) -> str:
    """Download a remote provider image and persist it locally."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:  # noqa: S310 - trusted URL returned by provider APIs
        output.write_bytes(response.read())

    return str(output)


def _extract_binary_mask_from_rgba(segmented_rgba_path: str, output_path: str) -> str:
    """Convert an RGBA foreground PNG into a strict binary mask."""

    segmented = Image.open(segmented_rgba_path).convert("RGBA")
    alpha = segmented.getchannel("A")
    binary_mask = alpha.point(lambda value: 255 if value > 0 else 0).convert("L")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    binary_mask.save(output)
    return str(output)


def _extract_binary_mask_from_luma(source_path: str, output_path: str, *, threshold: int = 127) -> str:
    """Convert a luma-style mask PNG into a strict binary mask."""

    grayscale = Image.open(source_path).convert("L")
    binary_mask = grayscale.point(lambda value: 255 if value >= threshold else 0).convert("L")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    binary_mask.save(output)
    return str(output)


def _extract_binary_mask_from_colorful_mask(source_path: str, output_path: str) -> str:
    """Convert a colorful mask image into a strict binary mask.

    SAM 3 image endpoint may return a colored PNG mask rather than a grayscale
    binary map. We treat any non-transparent / non-black pixel as foreground.
    """

    image = Image.open(source_path).convert("RGBA")
    width, height = image.size
    binary_mask = Image.new("L", (width, height), 0)

    for x in range(width):
        for y in range(height):
            r, g, b, a = image.getpixel((x, y))
            is_foreground = a > 0 and (r > 8 or g > 8 or b > 8)
            if is_foreground:
                binary_mask.putpixel((x, y), 255)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    binary_mask.save(output)
    return str(output)


def _invert_binary_mask(mask_path: str, output_path: str) -> str:
    """Invert a 0/255 binary mask to generate a complementary region mask."""

    mask = Image.open(mask_path).convert("L")
    inverted = mask.point(lambda value: 0 if value > 0 else 255).convert("L")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    inverted.save(output)
    return str(output)


def _require_aliyun_credentials() -> tuple[str, str, str, str]:
    """Load Aliyun credentials and endpoint configuration from environment variables."""

    load_project_env()
    access_key_id = os.getenv(ALIYUN_ACCESS_KEY_ID_ENV)
    access_key_secret = os.getenv(ALIYUN_ACCESS_KEY_SECRET_ENV)
    region_id = os.getenv(ALIYUN_REGION_ID_ENV, "cn-shanghai")
    endpoint = os.getenv(ALIYUN_IMAGESEG_ENDPOINT_ENV, f"imageseg.{region_id}.aliyuncs.com")

    missing = [
        name
        for name, value in (
            (ALIYUN_ACCESS_KEY_ID_ENV, access_key_id),
            (ALIYUN_ACCESS_KEY_SECRET_ENV, access_key_secret),
        )
        if not value
    ]
    if missing:
        raise AliyunImageSegError(f"Missing Aliyun credentials: {', '.join(missing)}")

    return access_key_id or "", access_key_secret or "", region_id, endpoint


def _create_aliyun_imageseg_client() -> tuple[Any, Any]:
    """Create an Aliyun ImageSeg client with a stable runtime config."""

    access_key_id, access_key_secret, region_id, endpoint = _require_aliyun_credentials()
    config = Config(
        access_key_id=access_key_id,
        access_key_secret=access_key_secret,
        region_id=region_id,
        endpoint=endpoint,
    )
    runtime = RuntimeOptions(
        read_timeout=int(os.getenv(ALIYUN_SEGMENT_READ_TIMEOUT_ENV, "10000")),
        connect_timeout=int(os.getenv(ALIYUN_SEGMENT_CONNECT_TIMEOUT_ENV, "10000")),
    )
    return ImageSegClient(config), runtime


def _submit_hd_common_segmentation_job(image_path: str) -> str:
    """Submit an async HD common-image segmentation job and return the job id."""

    client, runtime = _create_aliyun_imageseg_client()
    with open(image_path, "rb") as image_stream:
        request = SegmentHDCommonImageAdvanceRequest(image_url_object=image_stream)
        response = client.segment_hdcommon_image_advance(request, runtime)

    job_id = _pick_value(response, "request_id")
    if not job_id:
        body = _pick_value(response, "body")
        job_id = _pick_value(body, "request_id", "requestId")
    if not job_id:
        raise AliyunImageSegError("SegmentHDCommonImage did not return a job id.")
    return str(job_id)


def _poll_hd_common_segmentation_result(
    job_id: str,
    *,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
) -> str:
    """Poll Aliyun async result until the segmented image URL is ready."""

    client, _ = _create_aliyun_imageseg_client()

    configured_poll_interval = float(os.getenv(ALIYUN_SEGMENT_POLL_INTERVAL_ENV, str(poll_interval_seconds)))
    configured_max_attempts = int(os.getenv(ALIYUN_SEGMENT_MAX_ATTEMPTS_ENV, str(max_attempts)))

    for _ in range(configured_max_attempts):
        response = client.get_async_job_result(GetAsyncJobResultRequest(job_id=job_id))
        body = _pick_value(response, "body")
        data = _pick_value(body, "data")

        status = _pick_value(data, "status")
        error_code = _pick_value(data, "error_code", "errorCode")
        error_message = _pick_value(data, "error_message", "errorMessage")
        result = _pick_value(data, "result")

        if error_code or error_message:
            raise AliyunImageSegError(
                f"Aliyun async job failed with {error_code or 'unknown'}: {error_message or 'unknown'}"
            )

        if status in {"PROCESS_SUCCESS", "FINISHED", "SUCCESS"} and result:
            result_payload = json.loads(result)
            image_url = _pick_value(result_payload, "imageUrl", "image_url", "ImageUrl", "ImageURL")
            if not image_url:
                raise AliyunImageSegError("Aliyun async job finished but imageUrl was missing.")
            return str(image_url)

        if status in {"PROCESS_FAILED", "FAIL", "FAILED"}:
            raise AliyunImageSegError(f"Aliyun async job ended with status: {status}")

        time.sleep(configured_poll_interval)

    raise AliyunImageSegError("Aliyun async job did not finish within the polling window.")


def generate_realtime_subject_mask(
    image_path: str,
    *,
    output_dir: str | None = None,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
) -> AliyunSegmentationResult:
    """Generate a realtime binary main-subject mask via Aliyun HD common segmentation."""

    original = Path(image_path).resolve()
    if not original.exists():
        raise FileNotFoundError(f"Image not found: {original}")

    if output_dir is None:
        output_dir = str(original.parent / "output" / f"{original.stem}_主体分割结果")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    job_id = _submit_hd_common_segmentation_job(str(original))
    rgba_url = _poll_hd_common_segmentation_result(
        job_id,
        poll_interval_seconds=poll_interval_seconds,
        max_attempts=max_attempts,
    )

    segmentation_rgba_path = _download_remote_image(rgba_url, str(output_root / "主体高清分割结果.png"))
    binary_mask_path = _extract_binary_mask_from_rgba(
        segmentation_rgba_path,
        str(output_root / "主体二值图.png"),
    )

    return SegmentationResult(
        provider="aliyun",
        segmentation_rgba_path=segmentation_rgba_path,
        binary_mask_path=binary_mask_path,
        original_image_path=str(original),
        api_chain=("SegmentHDCommonImage", "GetAsyncJobResult"),
        request_id=job_id,
        remote_mask_url=rgba_url,
    )


def _require_fal_key() -> str:
    """Load the fal API key from the project environment."""

    load_project_env()
    key = os.getenv(FAL_KEY_ENV)
    if not key:
        raise FalImageSegError(f"Missing fal API key: {FAL_KEY_ENV}")
    return key


def _create_fal_client() -> Any:
    """Create a fal SyncClient with the configured API key."""

    if fal_client is None:
        raise FalImageSegError("fal-client is not installed. Add it to requirements and reinstall dependencies.")
    return fal_client.SyncClient(
        key=_require_fal_key(),
        default_timeout=float(os.getenv(FAL_SAM3_CLIENT_TIMEOUT_ENV, "180")),
    )


def _extract_fal_output_url(response: dict[str, Any]) -> str:
    """Extract the primary mask URL from a fal segmentation response payload."""

    masks_payload = _pick_value(response, "masks")
    if isinstance(masks_payload, list):
        for item in masks_payload:
            image_url = _pick_value(item, "url")
            if image_url:
                return str(image_url)

    image_payload = _pick_value(response, "image")
    if isinstance(image_payload, dict):
        image_url = _pick_value(image_payload, "url")
        if image_url:
            return str(image_url)
    if isinstance(image_payload, str) and image_payload:
        return image_payload

    images_payload = _pick_value(response, "images")
    if isinstance(images_payload, list):
        for item in images_payload:
            image_url = _pick_value(item, "url")
            if image_url:
                return str(image_url)

    raise FalImageSegError("fal segmentation response did not include an output image URL.")


def is_recoverable_empty_segmentation_error(error: Exception) -> bool:
    """Return whether a segmentation error means no usable mask was produced."""

    if not isinstance(error, FalImageSegError):
        return False
    return "did not include an output image url" in str(error).lower()


def _default_fal_prompt_for_region(region: str) -> str:
    """Return the default SAM 3 prompt for a region label."""

    if region == "person":
        return "person"
    if region == "main_subject":
        return "main visual subject"
    if region == "background":
        return "background"
    return region


def _is_background_retry_candidate(region: str, prompt: str | None) -> bool:
    """Return whether the request looks like a semantic background selection."""

    if region == "background":
        return True
    normalized_prompt = _normalize_fal_prompt_label(prompt or "", region=region)
    return normalized_prompt in {"background", "trees", "haze"}


def _background_retry_attempts(region: str, prompt: str | None) -> list[dict[str, Any]]:
    """Build a fixed background retry sequence for text-guided segmentation."""

    direct_prompt = _normalize_fal_prompt_label(prompt or _default_fal_prompt_for_region(region), region=region)
    return [
        {
            "attempt_index": 0,
            "attempt_strategy": "direct_background_prompt",
            "requested_prompt": direct_prompt,
            "effective_prompt": direct_prompt,
            "revert_mask": False,
        },
        {
            "attempt_index": 1,
            "attempt_strategy": "invert_person",
            "requested_prompt": direct_prompt,
            "effective_prompt": "person",
            "revert_mask": True,
        },
        {
            "attempt_index": 2,
            "attempt_strategy": "invert_subject",
            "requested_prompt": direct_prompt,
            "effective_prompt": "subject",
            "revert_mask": True,
        },
    ]


def _normalize_fal_prompt_label(prompt: str, *, region: str) -> str:
    """Normalize free-form region descriptions into short English prompt labels."""

    cleaned = " ".join(prompt.strip().lower().replace("_", " ").split())
    if not cleaned:
        return _default_fal_prompt_for_region(region)

    if any(keyword in cleaned for keyword in ("under eye", "under-eye", "dark eye", "黑眼圈", "眼下")):
        return "eye"
    if any(keyword in cleaned for keyword in ("face skin", "skin", "肤色", "皮肤", "脸", "面部")):
        return "face"
    if any(keyword in cleaned for keyword in ("hair", "发丝", "头发")):
        return "hair"
    if any(keyword in cleaned for keyword in ("teeth", "牙")):
        return "teeth"
    if any(keyword in cleaned for keyword in ("eyes", "eye", "虹膜", "眼睛")):
        return "eyes"
    if any(keyword in cleaned for keyword in ("white dress", "wedding dress", "婚纱", "白裙")):
        return "dress"
    if any(keyword in cleaned for keyword in ("dress", "clothing", "clothes", "衣服", "服装")):
        return "dress"
    if any(keyword in cleaned for keyword in ("upper body", "上半身", "half body", "person", "人物", "人像")):
        return "person"
    if any(keyword in cleaned for keyword in ("main subject", "主体")):
        return "subject"
    if any(keyword in cleaned for keyword in ("background foliage", "foliage", "greenery", "绿植", "树林", "草地", "树", "树叶", "草")):
        return "trees"
    if any(keyword in cleaned for keyword in ("background haze", "灰雾", "雾")):
        return "haze"
    if any(keyword in cleaned for keyword in ("background", "背景")):
        return "background"
    if any(keyword in cleaned for keyword in ("water spray", "droplet", "water", "spray", "泡泡", "水花")):
        return "water"
    if any(keyword in cleaned for keyword in ("bottle", "瓶子")):
        return "bottle"
    if any(keyword in cleaned for keyword in ("blemish", "痘", "瑕疵")):
        return "blemish"
    if any(keyword in cleaned for keyword in ("passersby", "路人")):
        return "people"
    if any(keyword in cleaned for keyword in ("object", "clutter", "杂物")):
        return "object"
    if any(keyword in cleaned for keyword in ("detail", "细节")):
        return "subject"
    if cleaned in {"person", "subject", "body"}:
        return cleaned
    return cleaned


def normalize_segmentation_prompt_label(prompt: str, *, region: str = "main_subject") -> str:
    """Public helper to normalize a segmentation prompt into a short English label."""

    return _normalize_fal_prompt_label(prompt, region=region)


def _default_semantic_type_for_prompt(region: str, prompt: str) -> bool:
    """Enable semantic mode for prompts that usually benefit from it."""

    lowered = prompt.lower()
    return region == "background" or any(
        keyword in lowered
        for keyword in (
            "background",
            "face",
            "hair",
            "skin",
            "dress",
            "clothes",
            "body",
            "hand",
            "arm",
            "leg",
        )
    )


def generate_fal_sam3_mask(
    image_path: str,
    *,
    prompt: str,
    negative_prompt: str | None = None,
    output_dir: str | None = None,
    semantic_type: bool | None = None,
    fill_holes: bool | str | None = True,
    expand_mask: int | None = 0,
    blur_mask: bool | str | None = False,
    use_grounding_dino: bool | str | None = None,
    revert_mask: bool | str | None = False,
    start_timeout_seconds: float | None = None,
    client_timeout_seconds: float | None = None,
) -> FalSegmentationResult:
    """Generate a binary mask with fal segmentation model for a text prompt."""

    original = Path(image_path).resolve()
    if not original.exists():
        raise FileNotFoundError(f"Image not found: {original}")

    cleaned_prompt = _normalize_fal_prompt_label(prompt, region="main_subject")
    if not cleaned_prompt:
        raise FalImageSegError("fal segmentation requires a non-empty prompt.")

    if output_dir is None:
        prompt_fragment = _slugify_fragment(cleaned_prompt, fallback="mask")
        output_dir = str(original.parent / "output" / f"{original.stem}_fal_{prompt_fragment}")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        client = _create_fal_client()
        model_name = os.getenv(FAL_SAM3_MODEL_ENV, "fal-ai/sam-3/image")
        content_type, _ = mimetypes.guess_type(str(original))
        upload_url = client.upload(
            original.read_bytes(),
            content_type or "application/octet-stream",
            file_name=f"source_image{original.suffix.lower() or '.bin'}",
            repository="cdn",
        )

        semantic_mode = (
            semantic_type
            if semantic_type is not None
            else _default_semantic_type_for_prompt("main_subject", cleaned_prompt)
        )
        arguments: dict[str, Any] = {
            "image_url": upload_url,
            "prompt": cleaned_prompt,
            "apply_mask": False,
            "output_format": "png",
            "return_multiple_masks": False,
            "max_masks": 1,
        }
        # Keep these fields computed for upstream metadata and compatibility even
        # though SAM 3 image endpoint does not consume them directly.
        _ = (
            semantic_mode,
            _as_bool(fill_holes, default=True),
            _as_bool(blur_mask, default=False),
            _as_bool(revert_mask, default=False),
            int(expand_mask or 0),
            None,
            _as_bool(use_grounding_dino, default=False) if use_grounding_dino is not None else None,
        )

        response = client.subscribe(
            model_name,
            arguments=arguments,
            start_timeout=start_timeout_seconds or float(os.getenv(FAL_SAM3_START_TIMEOUT_ENV, "120")),
            client_timeout=client_timeout_seconds or float(os.getenv(FAL_SAM3_CLIENT_TIMEOUT_ENV, "180")),
        )

        remote_mask_url = _extract_fal_output_url(response)
        downloaded_mask_path = _download_remote_image(remote_mask_url, str(output_root / "sam3_mask_raw.png"))
        binary_mask_path = _extract_binary_mask_from_colorful_mask(
            downloaded_mask_path,
            str(output_root / "sam3_mask_binary.png"),
        )

        return SegmentationResult(
            provider="fal_sam3",
            binary_mask_path=binary_mask_path,
            original_image_path=str(original),
            api_chain=("fal_client.upload", model_name),
            remote_mask_url=remote_mask_url,
            prompt=cleaned_prompt,
            negative_prompt=None,
            raw_response=response if isinstance(response, dict) else None,
        )
    except FalImageSegError:
        raise
    except Exception as error:  # pragma: no cover - exercised through live integration run
        raise FalImageSegError(f"fal segmentation request failed: {error}") from error


def _ensure_aliyun_region_mask(
    image_path: str,
    region: str,
    *,
    output_dir: str | None = None,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
) -> SegmentationResult:
    """Resolve a supported region into a provider result via Aliyun."""

    if region in {"main_subject", "person"}:
        result = generate_realtime_subject_mask(
            image_path,
            output_dir=output_dir,
            poll_interval_seconds=poll_interval_seconds,
            max_attempts=max_attempts,
        )
        return replace(
            result,
            region=region,
            target_label=region,
            requested_provider="aliyun",
        )

    if region == "background":
        subject_result = generate_realtime_subject_mask(
            image_path,
            output_dir=output_dir,
            poll_interval_seconds=poll_interval_seconds,
            max_attempts=max_attempts,
        )
        background_output_dir = Path(output_dir or Path(subject_result.binary_mask_path).parent)
        background_mask_path = _invert_binary_mask(
            subject_result.binary_mask_path,
            str(background_output_dir / "背景二值图.png"),
        )
        return replace(
            subject_result,
            binary_mask_path=background_mask_path,
            region=region,
            target_label=region,
            requested_provider="aliyun",
        )

    raise AliyunImageSegError(f"Unsupported segmentation region: {region}")


def _ensure_fal_region_mask(
    image_path: str,
    region: str,
    *,
    output_dir: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    semantic_type: bool | None = None,
    fill_holes: bool | str | None = True,
    expand_mask: int | None = 0,
    blur_mask: bool | str | None = False,
    use_grounding_dino: bool | str | None = None,
    revert_mask: bool | str | None = None,
    start_timeout_seconds: float | None = None,
    client_timeout_seconds: float | None = None,
) -> SegmentationResult:
    """Resolve a supported region into a provider result via fal SAM 3."""

    effective_prompt = _normalize_fal_prompt_label(prompt or _default_fal_prompt_for_region(region), region=region)
    resolved_semantic_type = (
        semantic_type if semantic_type is not None else _default_semantic_type_for_prompt(region, effective_prompt)
    )
    effective_revert = (
        _as_bool(revert_mask, default=False)
        if revert_mask is not None
        else region == "background" and not prompt
    )
    result = generate_fal_sam3_mask(
        image_path,
        prompt=effective_prompt,
        negative_prompt=None,
        output_dir=output_dir,
        semantic_type=resolved_semantic_type,
        fill_holes=fill_holes,
        expand_mask=expand_mask,
        blur_mask=blur_mask,
        use_grounding_dino=use_grounding_dino,
        revert_mask=effective_revert,
        start_timeout_seconds=start_timeout_seconds,
        client_timeout_seconds=client_timeout_seconds,
    )
    return replace(
        result,
        region=region,
        target_label=effective_prompt,
        semantic_type=resolved_semantic_type,
        requested_provider="fal_sam3",
    )


def _ensure_fal_region_mask_with_background_retries(
    image_path: str,
    region: str,
    *,
    output_dir: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    semantic_type: bool | None = None,
    fill_holes: bool | str | None = True,
    expand_mask: int | None = 0,
    blur_mask: bool | str | None = False,
    use_grounding_dino: bool | str | None = None,
    revert_mask: bool | str | None = None,
    start_timeout_seconds: float | None = None,
    client_timeout_seconds: float | None = None,
) -> SegmentationResult:
    """Retry semantic background segmentation from inverse foreground angles."""

    attempts: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in _background_retry_attempts(region, prompt):
        try:
            result = _ensure_fal_region_mask(
                image_path,
                region,
                output_dir=output_dir,
                prompt=attempt["effective_prompt"],
                negative_prompt=negative_prompt,
                semantic_type=semantic_type,
                fill_holes=fill_holes,
                expand_mask=expand_mask,
                blur_mask=blur_mask,
                use_grounding_dino=use_grounding_dino,
                revert_mask=attempt["revert_mask"],
                start_timeout_seconds=start_timeout_seconds,
                client_timeout_seconds=client_timeout_seconds,
            )
            attempt_payload = {
                **attempt,
                "ok": True,
            }
            attempts.append(attempt_payload)
            return replace(
                result,
                fallback_used=attempt["attempt_index"] > 0,
                attempt_index=attempt["attempt_index"],
                attempt_strategy=attempt["attempt_strategy"],
                requested_prompt=attempt["requested_prompt"],
                effective_prompt=attempt["effective_prompt"],
                revert_mask=attempt["revert_mask"],
                attempts=tuple(attempts),
            )
        except SegmentationProviderError as error:
            last_error = error
            attempts.append(
                {
                    **attempt,
                    "ok": False,
                    "error": str(error),
                }
            )
            if not is_recoverable_empty_segmentation_error(error):
                break

    if isinstance(last_error, Exception):
        raise FalImageSegAttemptsError(str(last_error), attempts=attempts) from last_error
    raise FalImageSegAttemptsError("fal segmentation background retry did not produce a usable mask.", attempts=attempts)


def resolve_region_mask(
    image_path: str,
    region: str,
    *,
    output_dir: str | None = None,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
    provider: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    semantic_type: bool | None = None,
    fill_holes: bool | str | None = True,
    expand_mask: int | None = 0,
    blur_mask: bool | str | None = False,
    use_grounding_dino: bool | str | None = None,
    revert_mask: bool | str | None = None,
    start_timeout_seconds: float | None = None,
    client_timeout_seconds: float | None = None,
) -> SegmentationResult:
    """Resolve a supported region into a provider result with fallback metadata."""

    active_provider = _resolve_segmentation_provider(region=region, provider=provider, prompt=prompt)

    try:
        if active_provider == "fal_sam3":
            fal_result = (
                _ensure_fal_region_mask_with_background_retries(
                    image_path,
                    region,
                    output_dir=output_dir,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    semantic_type=semantic_type,
                    fill_holes=fill_holes,
                    expand_mask=expand_mask,
                    blur_mask=blur_mask,
                    use_grounding_dino=use_grounding_dino,
                    revert_mask=revert_mask,
                    start_timeout_seconds=start_timeout_seconds,
                    client_timeout_seconds=client_timeout_seconds,
                )
                if _is_background_retry_candidate(region, prompt)
                else _ensure_fal_region_mask(
                    image_path,
                    region,
                    output_dir=output_dir,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    semantic_type=semantic_type,
                    fill_holes=fill_holes,
                    expand_mask=expand_mask,
                    blur_mask=blur_mask,
                    use_grounding_dino=use_grounding_dino,
                    revert_mask=revert_mask,
                    start_timeout_seconds=start_timeout_seconds,
                    client_timeout_seconds=client_timeout_seconds,
                )
            )
            return replace(
                fal_result,
                requested_provider=active_provider,
                fallback_used=fal_result.fallback_used,
            )
        return replace(
            _ensure_aliyun_region_mask(
                image_path,
                region,
                output_dir=output_dir,
                poll_interval_seconds=poll_interval_seconds,
                max_attempts=max_attempts,
            ),
            requested_provider=active_provider,
            fallback_used=False,
        )
    except SegmentationProviderError:
        fallback_provider = _resolve_fallback_provider(active_provider=active_provider, prompt=prompt)
        if fallback_provider is None:
            raise

        if fallback_provider == "aliyun":
            return replace(
                _ensure_aliyun_region_mask(
                    image_path,
                    region,
                    output_dir=output_dir,
                    poll_interval_seconds=poll_interval_seconds,
                    max_attempts=max_attempts,
                ),
                requested_provider=active_provider,
                fallback_used=True,
            )
        return replace(
            _ensure_fal_region_mask(
                image_path,
                region,
                output_dir=output_dir,
                prompt=prompt,
                negative_prompt=negative_prompt,
                semantic_type=semantic_type,
                fill_holes=fill_holes,
                expand_mask=expand_mask,
                blur_mask=blur_mask,
                use_grounding_dino=use_grounding_dino,
                revert_mask=revert_mask,
                start_timeout_seconds=start_timeout_seconds,
                client_timeout_seconds=client_timeout_seconds,
            ),
            requested_provider=active_provider,
            fallback_used=True,
        )


def ensure_region_mask(
    image_path: str,
    region: str,
    *,
    output_dir: str | None = None,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
    provider: str | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    semantic_type: bool | None = None,
    fill_holes: bool | str | None = True,
    expand_mask: int | None = 0,
    blur_mask: bool | str | None = False,
    use_grounding_dino: bool | str | None = None,
    revert_mask: bool | str | None = None,
    start_timeout_seconds: float | None = None,
    client_timeout_seconds: float | None = None,
) -> str:
    """Resolve a supported region into a local binary mask path.

    现阶段支持两种 provider：
    1. `aliyun`：面向稳定粗区域 `main_subject/person/background`
    2. `fal_sam3`：面向文本引导语义区域，也兼容对稳定区域做更精细的替代分割
    """

    return resolve_region_mask(
        image_path,
        region,
        output_dir=output_dir,
        poll_interval_seconds=poll_interval_seconds,
        max_attempts=max_attempts,
        provider=provider,
        prompt=prompt,
        negative_prompt=negative_prompt,
        semantic_type=semantic_type,
        fill_holes=fill_holes,
        expand_mask=expand_mask,
        blur_mask=blur_mask,
        use_grounding_dino=use_grounding_dino,
        revert_mask=revert_mask,
        start_timeout_seconds=start_timeout_seconds,
        client_timeout_seconds=client_timeout_seconds,
    ).binary_mask_path


__all__ = [
    "ALIYUN_ACCESS_KEY_ID_ENV",
    "ALIYUN_ACCESS_KEY_SECRET_ENV",
    "ALIYUN_IMAGESEG_ENDPOINT_ENV",
    "ALIYUN_REGION_ID_ENV",
    "FAL_SAM3_CLIENT_TIMEOUT_ENV",
    "FAL_SAM3_MODEL_ENV",
    "FAL_SAM3_START_TIMEOUT_ENV",
    "FAL_KEY_ENV",
    "PSAGENT_SEGMENTATION_FALLBACK_PROVIDER_ENV",
    "PSAGENT_SEGMENTATION_PROVIDER_ENV",
    "AliyunImageSegError",
    "AliyunSegmentationResult",
    "FalImageSegError",
    "FalSegmentationResult",
    "SegmentationProviderError",
    "SegmentationResult",
    "ensure_region_mask",
    "generate_fal_sam3_mask",
    "generate_realtime_subject_mask",
    "is_recoverable_empty_segmentation_error",
    "normalize_segmentation_prompt_label",
    "resolve_region_mask",
]
