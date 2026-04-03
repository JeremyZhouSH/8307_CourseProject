from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dictionary: {resolved}")
    return data


def read_text(path: str | Path) -> str:
    resolved = Path(path)
    return resolved.read_text(encoding="utf-8")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
