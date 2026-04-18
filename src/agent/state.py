from __future__ import annotations

from dataclasses import field, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.parser.section_splitter import Section


@dataclass
class PipelineState:
    # 输入输出路径与各阶段中间结果，便于调试与落盘。
    input_path: str = ""
    output_path: str = ""
    document_text: str = ""
    sections: list[Section] = field(default_factory=list)
    key_info: dict[str, list[str]] = field(default_factory=dict)
    structured_summary: dict[str, Any] = field(default_factory=dict)
    final_summary: str = ""
    verification: dict[str, Any] = field(default_factory=dict)
    step_history: list[dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    should_try_llm: bool = False
    llm_summary_attempted: bool = False
    output_written: bool = False
    extractor_strategy: str = "unknown"
    agent_finished: bool = False
    user_request: str = ""
    needs_clarification: bool = False
    clarification_question: str = ""
    clarification_request: dict[str, Any] = field(default_factory=dict)
    memory_hits: list[dict[str, Any]] = field(default_factory=list)
    clarification_checked: bool = False
    strategy_adapted: bool = False
