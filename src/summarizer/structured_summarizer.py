from __future__ import annotations

import re

from src.parser.section_splitter import Section


class StructuredSummarizer:
    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

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

    def _first_sentences(self, text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected = [sentence.strip() for sentence in sentences if sentence.strip()][: self.max_sentences]
        return " ".join(selected)
