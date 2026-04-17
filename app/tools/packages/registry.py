"""Registry for tool-package discovery and export."""

from __future__ import annotations

from typing import Any

from app.tools.packages.base import ToolPackage


class PackageRegistry:
    """Simple in-memory registry for enabled tool packages."""

    def __init__(self) -> None:
        # 当前先用最简单的内存注册表，后面需要时再扩成按能力索引。
        self._packages: dict[str, ToolPackage] = {}

    def register(self, package: ToolPackage) -> None:
        """Register a package by its unique name."""

        # 显式注册比动态扫描更稳，也更方便调试和审计。
        self._packages[package.name] = package

    def get(self, name: str) -> ToolPackage | None:
        """Return a package if it exists in the registry."""

        return self._packages.get(name)

    def require(self, name: str) -> ToolPackage:
        """Return a package or raise if it is missing."""

        package = self.get(name)
        if package is None:
            raise KeyError(f"Unknown tool package: {name}")
        return package

    def list(self) -> list[ToolPackage]:
        """List all registered packages."""

        return list(self._packages.values())

    def filter(
        self,
        *,
        domain: str | None = None,
        region: str | None = None,
        risk_level: str | None = None,
    ) -> list[ToolPackage]:
        """Filter packages by declared capabilities."""

        # 这里的过滤逻辑直接基于 PackageSpec 声明信息完成。
        packages = self.list()
        if domain is not None:
            packages = [
                package
                for package in packages
                if not package.spec.supported_domains
                or domain in package.spec.supported_domains
            ]
        if region is not None:
            packages = [
                package
                for package in packages
                if not package.spec.supported_regions
                or package.supports_region(region)
            ]
        if risk_level is not None:
            packages = [
                package
                for package in packages
                if package.spec.risk_level == risk_level
            ]
        return packages

    def export_llm_catalog(self) -> list[dict[str, Any]]:
        """Export the simplified package schemas used by the planner."""

        # planner 只看简化后的能力清单，不直接接触包内部实现细节。
        return [package.get_llm_schema() for package in self.list()]
