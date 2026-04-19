from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {resolved}")
    return data


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def read_text(path: str | Path) -> str:
    resolved = Path(path)
    return resolved.read_text(encoding="utf-8")


# 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
def write_json(data: dict[str, Any], path: str | Path) -> None:
    resolved = Path(path)
    # 写文件前确保父目录存在，避免首次运行时报错。
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
