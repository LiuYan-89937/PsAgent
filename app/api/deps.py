"""Shared API dependencies."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_graph
from app.services.asset_store import AssetStore
from app.services.job_store import JobStore
from app.tools.packages import PackageRegistry, build_default_package_registry


@lru_cache(maxsize=1)
def get_checkpointer() -> InMemorySaver:
    """Return the shared in-memory checkpointer."""

    return InMemorySaver()


@lru_cache(maxsize=1)
def get_graph_app():
    """Return the compiled application graph."""

    return build_graph(checkpointer=get_checkpointer())


@lru_cache(maxsize=1)
def get_package_registry() -> PackageRegistry:
    """Return the shared package registry."""

    return build_default_package_registry()


@lru_cache(maxsize=1)
def get_asset_store() -> AssetStore:
    """Return the shared local asset store."""

    root_dir = Path.cwd() / "data" / "assets"
    return AssetStore(root_dir=root_dir)


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """Return the shared in-memory job store."""

    return JobStore()
