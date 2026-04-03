from __future__ import annotations

from dataclasses import field, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.parser.section_splitter import Section


@dataclass
class PipelineState:
    input_path: str = ""
    output_path: str = ""
    document_text: str = ""
    sections: list[Section] = field(default_factory=list)
    key_info: dict[str, list[str]] = field(default_factory=dict)
    structured_summary: dict[str, Any] = field(default_factory=dict)
    final_summary: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
