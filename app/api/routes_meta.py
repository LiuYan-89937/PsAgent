"""Metadata routes for frontend bootstrapping."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_tool_catalog
from app.api.schemas import ToolCatalogItemResponse, ToolCatalogResponse

router = APIRouter(prefix="/meta", tags=["meta"])


@router.get("/tools", response_model=ToolCatalogResponse)
@router.get("/packages", response_model=ToolCatalogResponse)
async def list_tools(
    tool_catalog: tuple[dict, ...] = Depends(get_tool_catalog),
) -> ToolCatalogResponse:
    """Return the current planner-facing native tool catalog.

    这个接口主要给前端做：
    1. 控件面板初始化
    2. tooltips / 参数说明展示
    3. 前端调试当前后端实际支持的工具和参数
    """

    return ToolCatalogResponse(
        items=[ToolCatalogItemResponse.model_validate(item) for item in tool_catalog]
    )
