from __future__ import annotations

from pathlib import Path

from src.utils.io import read_text


class DocumentLoader:
    """Load plain-text documents from local storage."""

    def load(self, path: str | Path) -> str:
        return read_text(path)
