"""Metadata routes for frontend bootstrapping."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_tool_registry
from app.api.schemas import ToolCatalogResponse
from app.tools.tool_registry import ToolRegistry

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/packages", response_model=ToolCatalogResponse)
async def list_packages(
    registry: ToolRegistry = Depends(get_tool_registry),
) -> ToolCatalogResponse:
    """Return the current planner-facing native tool catalog.

    这个接口主要给前端做：
    1. 控件面板初始化
    2. tooltips / 参数说明展示
    3. 前端调试当前后端实际支持的工具和参数
    """

    return ToolCatalogResponse(items=registry.export_catalog())
