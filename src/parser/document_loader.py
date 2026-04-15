from __future__ import annotations

from pathlib import Path

from src.utils.io import read_text


class DocumentLoader:
    """从本地读取纯文本论文。"""

    def load(self, path: str | Path) -> str:
        return read_text(path)
