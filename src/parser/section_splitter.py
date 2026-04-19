from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class Section:
    title: str
    content: str


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class SectionSplitter:
    # 常见英文章节名词表，用于标题识别。
    COMMON_HEADINGS = {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methods",
        "method",
        "approach",
        "experiments",
        "experiment",
        "results",
        "discussion",
        "limitations",
        "conclusion",
        "future work",
    }

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, max_sections: int = 20) -> None:
        self.max_sections = max_sections

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def split(self, text: str) -> list[Section]:
        sections: list[Section] = []
        current_title = "Document"
        buffer: list[str] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue

            if self._is_heading(line):
                # 遇到标题时先落盘上一段，再开启新段。
                self._append_section(sections, current_title, buffer)
                current_title = self._normalize_title(line)
                buffer = []
            else:
                buffer.append(line)

        self._append_section(sections, current_title, buffer)

        if not sections and text.strip():
            return [Section(title="Document", content=text.strip())]

        return sections[: self.max_sections]

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _append_section(self, sections: list[Section], title: str, lines: list[str]) -> None:
        content = "\n".join(lines).strip()
        if content:
            sections.append(Section(title=title, content=content))

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _is_heading(self, line: str) -> bool:
        # 过长文本或句末标点通常不是标题。
        if len(line) > 90:
            return False
        if line.endswith((".", ",", ";", ":")):
            return False

        no_index = re.sub(r"^\d+(?:\.\d+)*\s+", "", line).strip()
        normalized = no_index.lower()

        if normalized in self.COMMON_HEADINGS:
            return True
        if no_index.isupper() and any(ch.isalpha() for ch in no_index):
            return True
        if re.match(r"^\d+(?:\.\d+)*\s+[A-Za-z]", line):
            return True

        return False

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _normalize_title(self, line: str) -> str:
        title = re.sub(r"^\d+(?:\.\d+)*\s+", "", line).strip()
        title = re.sub(r"\s+", " ", title)

        if title.isupper():
            return title.title()
        return title
