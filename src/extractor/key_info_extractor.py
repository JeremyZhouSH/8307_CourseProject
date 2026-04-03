from __future__ import annotations

import re

from src.parser.section_splitter import Section


class KeyInfoExtractor:
    SECTION_BUCKETS = {
        "objective": {"abstract", "introduction", "background", "motivation"},
        "methods": {"method", "methods", "approach", "framework"},
        "results": {"result", "results", "experiment", "experiments", "evaluation"},
        "limitations": {"limitation", "limitations", "discussion", "future work", "conclusion"},
    }

    def extract(self, sections: list[Section]) -> dict[str, list[str]]:
        info: dict[str, list[str]] = {
            "objective": [],
            "methods": [],
            "results": [],
            "limitations": [],
        }

        for section in sections:
            bucket = self._bucket_for_title(section.title)
            sentences = self._split_sentences(section.content)
            if bucket:
                info[bucket].extend(sentences[:2])

        if not info["objective"] and sections:
            info["objective"] = self._split_sentences(sections[0].content)[:2]
        if not info["methods"] and len(sections) > 1:
            info["methods"] = self._split_sentences(sections[1].content)[:2]
        if not info["results"] and len(sections) > 2:
            info["results"] = self._split_sentences(sections[-2].content)[:2]
        if not info["limitations"] and sections:
            info["limitations"] = self._split_sentences(sections[-1].content)[:2]

        return {key: self._deduplicate(value) for key, value in info.items()}

    def _bucket_for_title(self, title: str) -> str | None:
        normalized = title.lower()
        for bucket, hints in self.SECTION_BUCKETS.items():
            if any(hint in normalized for hint in hints):
                return bucket
        return None

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered
