"""Metadata routes for frontend bootstrapping."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_package_registry
from app.api.schemas import PackageCatalogResponse
from app.tools.packages import PackageRegistry

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/packages", response_model=PackageCatalogResponse)
async def list_packages(
    registry: PackageRegistry = Depends(get_package_registry),
) -> PackageCatalogResponse:
    """Return the current planner-facing package catalog.

    这个接口主要给前端做：
    1. 控件面板初始化
    2. tooltips / 参数说明展示
    3. 前端调试当前后端实际支持的包和参数
    """

    return PackageCatalogResponse(items=registry.export_llm_catalog())
