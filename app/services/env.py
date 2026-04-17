"""Minimal project-level .env loader.

当前项目不额外依赖 python-dotenv，直接提供一个够用的本地实现：
1. 默认从仓库根目录读取 `.env`
2. 只在环境变量尚未存在时写入，避免覆盖外部注入值
3. 支持最常见的 `KEY=value` / 引号值 / 注释行
"""

from __future__ import annotations

import os
from pathlib import Path


_ENV_LOADED = False


def _strip_wrapping_quotes(value: str) -> str:
    """Strip matching single/double quotes around a value."""

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_project_env() -> None:
    """Load `.env` from the project root into `os.environ`.

    这个函数是幂等的，多次调用只会真正解析一次。
    """

    global _ENV_LOADED
    if _ENV_LOADED:
        return

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        _ENV_LOADED = True
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value

    _ENV_LOADED = True
