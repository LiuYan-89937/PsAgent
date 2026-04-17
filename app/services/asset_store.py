"""Asset storage service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from uuid import uuid4

from pydantic import BaseModel


class AssetRecord(BaseModel):
    """Persisted local asset record."""

    asset_id: str
    filename: str
    media_type: str | None = None
    local_path: str
    size_bytes: int | None = None
    created_at: datetime


class AssetStore:
    """Object storage abstraction for source and output images."""

    def __init__(self, *, root_dir: Path) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._assets: dict[str, AssetRecord] = {}

    def _build_target_path(self, filename: str) -> tuple[str, Path]:
        asset_id = uuid4().hex
        suffix = Path(filename).suffix or ".bin"
        safe_name = Path(filename).stem or "asset"
        target_path = self.root_dir / f"{asset_id}_{safe_name}{suffix}"
        return asset_id, target_path

    def save_upload(
        self,
        *,
        filename: str,
        content: bytes,
        media_type: str | None = None,
    ) -> AssetRecord:
        """Persist uploaded bytes as a local asset."""

        asset_id, target_path = self._build_target_path(filename)
        target_path.write_bytes(content)
        record = AssetRecord(
            asset_id=asset_id,
            filename=filename,
            media_type=media_type,
            local_path=str(target_path),
            size_bytes=len(content),
            created_at=datetime.now(timezone.utc),
        )
        self._assets[asset_id] = record
        return record

    def save_generated(
        self,
        source_path: str,
        *,
        filename: str | None = None,
        media_type: str | None = None,
    ) -> AssetRecord:
        """Copy a generated file into the managed asset store."""

        source = Path(source_path)
        final_name = filename or source.name
        asset_id, target_path = self._build_target_path(final_name)
        copy2(source, target_path)
        record = AssetRecord(
            asset_id=asset_id,
            filename=final_name,
            media_type=media_type,
            local_path=str(target_path),
            size_bytes=target_path.stat().st_size,
            created_at=datetime.now(timezone.utc),
        )
        self._assets[asset_id] = record
        return record

    def get(self, asset_id: str) -> AssetRecord | None:
        """Return a stored asset if present."""

        return self._assets.get(asset_id)

    def require(self, asset_id: str) -> AssetRecord:
        """Return a stored asset or raise if missing."""

        record = self.get(asset_id)
        if record is None:
            raise KeyError(f"Unknown asset: {asset_id}")
        return record

    def list(self) -> list[AssetRecord]:
        """List all stored assets."""

        return list(self._assets.values())
