from __future__ import annotations

import re

from src.parser.section_splitter import Section


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class StructuredSummarizer:
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def summarize(
        self,
        sections: list[Section],
        key_info: dict[str, list[str]],
    ) -> dict[str, object]:
        # 每章取前若干句，作为可追溯的章节级摘要。
        section_summaries = [
            {
                "section": section.title,
                "summary": self._first_sentences(section.content),
            }
            for section in sections
        ]

        return {
            "section_summaries": section_summaries,
            "key_info": key_info,
        }

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _first_sentences(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected = [sentence.strip() for sentence in sentences if sentence.strip()][: self.max_sentences]
        return " ".join(selected)
