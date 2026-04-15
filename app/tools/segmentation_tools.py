"""Aliyun-based subject segmentation helpers for local region editing and tests.

当前统一使用阿里云“通用高清分割”来提取图片主体。
这样不仅适用于人像，也适用于模型、商品、静物等主视觉主体。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from alibabacloud_imageseg20191230.client import Client as ImageSegClient
from alibabacloud_imageseg20191230.models import (
    GetAsyncJobResultRequest,
    SegmentHDCommonImageAdvanceRequest,
)
from alibabacloud_tea_openapi.models import Config
from alibabacloud_tea_util.models import RuntimeOptions
from PIL import Image


ALIYUN_ACCESS_KEY_ID_ENV = "ALIBABA_CLOUD_ACCESS_KEY_ID"
ALIYUN_ACCESS_KEY_SECRET_ENV = "ALIBABA_CLOUD_ACCESS_KEY_SECRET"
ALIYUN_REGION_ID_ENV = "ALIBABA_CLOUD_REGION_ID"
ALIYUN_IMAGESEG_ENDPOINT_ENV = "ALIYUN_IMAGESEG_ENDPOINT"


class AliyunImageSegError(RuntimeError):
    """Raised when the Aliyun image segmentation pipeline cannot complete."""


@dataclass(slots=True)
class AliyunSegmentationResult:
    """Standardized result for the live Aliyun subject-segmentation pipeline."""

    provider: str
    segmentation_rgba_path: str
    binary_mask_path: str
    original_image_path: str
    api_chain: tuple[str, ...]
    job_id: str | None = None


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


def _require_aliyun_credentials() -> tuple[str, str, str, str]:
    """Load Aliyun credentials and endpoint configuration from environment variables."""

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
    runtime = RuntimeOptions(read_timeout=10000, connect_timeout=10000)
    return ImageSegClient(config), runtime


def _download_remote_image(url: str, output_path: str) -> str:
    """Download an image URL returned by Aliyun and persist it locally."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:  # noqa: S310 - trusted URL returned by Aliyun API
        output.write_bytes(response.read())

    return str(output)


def _extract_binary_mask_from_rgba(segmented_rgba_path: str, output_path: str) -> str:
    """Convert the transparent PNG returned by Aliyun into a strict binary mask."""

    segmented = Image.open(segmented_rgba_path).convert("RGBA")
    alpha = segmented.getchannel("A")

    # 通用高清分割返回的主体 PNG 通常通过 alpha 通道表达前景区域。
    # 这里直接把 alpha 通道阈值化成 0/255，方便后续局部调整与测试。
    binary_mask = alpha.point(lambda value: 255 if value > 0 else 0).convert("L")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    binary_mask.save(output)
    return str(output)


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

    for _ in range(max_attempts):
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

        time.sleep(poll_interval_seconds)

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

    return AliyunSegmentationResult(
        provider="aliyun",
        segmentation_rgba_path=segmentation_rgba_path,
        binary_mask_path=binary_mask_path,
        original_image_path=str(original),
        api_chain=("SegmentHDCommonImage", "GetAsyncJobResult"),
        job_id=job_id,
    )


def generate_realtime_person_mask(
    image_path: str,
    *,
    output_dir: str | None = None,
    poll_interval_seconds: float = 1.0,
    max_attempts: int = 20,
) -> AliyunSegmentationResult:
    """Backward-compatible wrapper.

    当前项目已经不再使用“人像分割”作为默认测试路径，这里统一改为主体高清分割。
    对于人像图，主体通常就是人物；对于模型/静物图，也能正常工作。
    """

    return generate_realtime_subject_mask(
        image_path,
        output_dir=output_dir,
        poll_interval_seconds=poll_interval_seconds,
        max_attempts=max_attempts,
    )


__all__ = [
    "ALIYUN_ACCESS_KEY_ID_ENV",
    "ALIYUN_ACCESS_KEY_SECRET_ENV",
    "ALIYUN_IMAGESEG_ENDPOINT_ENV",
    "ALIYUN_REGION_ID_ENV",
    "AliyunImageSegError",
    "AliyunSegmentationResult",
    "generate_realtime_person_mask",
    "generate_realtime_subject_mask",
]
