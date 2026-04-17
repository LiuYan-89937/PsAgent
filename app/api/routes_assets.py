"""Asset upload and download routes."""

from __future__ import annotations

import os
from io import BytesIO

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.api.deps import get_asset_store
from app.api.schemas import AssetResponse, UploadAssetsResponse
from app.services.asset_store import AssetRecord, AssetStore

router = APIRouter(prefix="/assets", tags=["assets"])


MAX_UPLOAD_BYTES = int(os.getenv("PSAGENT_MAX_UPLOAD_BYTES", str(15 * 1024 * 1024)))
MAX_IMAGE_SIDE = int(os.getenv("PSAGENT_MAX_IMAGE_SIDE", "8192"))
MIN_IMAGE_SIDE = int(os.getenv("PSAGENT_MIN_IMAGE_SIDE", "32"))
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}


def _build_asset_response(request: Request, record: AssetRecord) -> AssetResponse:
    """Convert a stored asset record into an API response."""

    return AssetResponse(
        asset_id=record.asset_id,
        filename=record.filename,
        media_type=record.media_type,
        size_bytes=record.size_bytes,
        created_at=record.created_at,
        content_url=str(request.url_for("get_asset_content", asset_id=record.asset_id)),
    )


def _validate_upload_image(*, filename: str, content: bytes, media_type: str | None) -> None:
    """Validate uploaded image bytes before persisting them.

    这里先把最常见的“前端上传后后端卡住”场景挡在入口：
    1. 不是图片；
    2. 文件过大；
    3. 分辨率异常；
    4. Pillow 无法正常解码。
    """

    if media_type and not media_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="上传文件不是图片类型。")

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图片文件过大，当前最大支持 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB。",
        )

    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "").upper()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="上传文件不是有效图片，无法解析。") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="图片文件损坏或格式不受支持。") from exc

    if image_format and image_format not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"暂不支持 {image_format} 格式图片，请上传 JPEG、PNG、WEBP、BMP 或 TIFF。",
        )

    if width < MIN_IMAGE_SIDE or height < MIN_IMAGE_SIDE:
        raise HTTPException(status_code=400, detail="图片尺寸过小，请上传更清晰的图片。")

    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        raise HTTPException(
            status_code=400,
            detail=f"图片尺寸过大，单边最大支持 {MAX_IMAGE_SIDE}px。",
        )


@router.post("/upload", response_model=UploadAssetsResponse)
async def upload_assets(
    request: Request,
    files: list[UploadFile] = File(...),
    asset_store: AssetStore = Depends(get_asset_store),
) -> UploadAssetsResponse:
    """Upload one or more input images for later editing."""

    items: list[AssetResponse] = []
    for upload in files:
        content = await upload.read()
        if not content:
            continue
        _validate_upload_image(
            filename=upload.filename or "upload.bin",
            content=content,
            media_type=upload.content_type,
        )
        record = asset_store.save_upload(
            filename=upload.filename or "upload.bin",
            content=content,
            media_type=upload.content_type,
        )
        items.append(_build_asset_response(request, record))
    return UploadAssetsResponse(items=items)


@router.get("/{asset_id}", response_model=AssetResponse)
async def get_asset(
    request: Request,
    asset_id: str,
    asset_store: AssetStore = Depends(get_asset_store),
) -> AssetResponse:
    """Return stored asset metadata."""

    record = asset_store.get(asset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return _build_asset_response(request, record)


@router.get("/{asset_id}/content", name="get_asset_content")
async def get_asset_content(
    asset_id: str,
    asset_store: AssetStore = Depends(get_asset_store),
) -> FileResponse:
    """Stream the stored asset file."""

    record = asset_store.get(asset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(
        path=record.local_path,
        media_type=record.media_type or "application/octet-stream",
        filename=record.filename,
    )
