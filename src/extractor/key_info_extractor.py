from __future__ import annotations

import re

from src.extractor.ilp_sentence_selector import ILPSentenceSelector, SentenceCandidate
from src.extractor.role_tagger_crf import SentenceRoleTagger
from src.parser.section_splitter import Section


class KeyInfoExtractor:
    SECTION_BUCKETS = {
        "objective": {"abstract", "introduction", "background", "motivation"},
        "methods": {"method", "methods", "approach", "framework"},
        "results": {"result", "results", "experiment", "experiments", "evaluation"},
        "limitations": {"limitation", "limitations", "discussion", "future work", "conclusion"},
    }

    ROLE_KEYS: tuple[str, ...] = ("objective", "methods", "results", "limitations")

    def __init__(self, extractor_cfg: dict[str, object] | None = None) -> None:
        cfg = extractor_cfg or {}
        # strategy: ilp(只走新算法) / hybrid(优先新算法,不足则回退) / rule(旧规则)。
        self.strategy = str(cfg.get("strategy", "hybrid")).lower().strip()
        self.word_budget = max(60, int(cfg.get("word_budget", 200)))#type: ignore
        self.max_sentences_per_role = max(1, int(cfg.get("max_sentences_per_role", 2)))#type: ignore
        self.min_sentence_chars = max(10, int(cfg.get("min_sentence_chars", 30)))#type: ignore
        self.score_boost_section_match = float(cfg.get("score_boost_section_match", 0.3))#type: ignore
        self.score_boost_role_confidence = float(cfg.get("score_boost_role_confidence", 0.8)) #type: ignore

        min_role_coverage_cfg = cfg.get("min_role_coverage", {})
        if isinstance(min_role_coverage_cfg, dict):
            min_role_coverage = {
                role: int(min_role_coverage_cfg.get(role, 0)) for role in self.ROLE_KEYS
            }
        else:
            min_role_coverage = {role: 0 for role in self.ROLE_KEYS}

        role_tagger_method = str(cfg.get("role_tagger", "crf")).lower().strip()
        redundancy_penalty = float(cfg.get("redundancy_penalty", 0.35))#type: ignore
        similarity_threshold = float(cfg.get("similarity_threshold", 0.12))#type: ignore

        self.role_tagger = SentenceRoleTagger(method=role_tagger_method)
        self.selector = ILPSentenceSelector(
            word_budget=self.word_budget,
            redundancy_penalty=redundancy_penalty,
            similarity_threshold=similarity_threshold,
            min_role_coverage=min_role_coverage,
        )

    def extract(self, sections: list[Section]) -> dict[str, list[str]]:
        # 新主路径：角色标注 + ILP 选句。
        if self.strategy in {"ilp", "hybrid"}:
            extracted = self._extract_with_roles_and_ilp(sections)
            if self._has_enough_info(extracted):
                return extracted

        # 回退路径：兼容旧规则，保证在依赖缺失或文本异常时仍可用。
        if self.strategy in {"rule", "heuristic"}:
            return self._extract_rule_based(sections)

        return self._extract_rule_based(sections)

    def _extract_with_roles_and_ilp(self, sections: list[Section]) -> dict[str, list[str]]:
        tagged_sentences = self.role_tagger.tag_sections(sections)
        if not tagged_sentences:
            return self._empty_info()

        candidates: list[SentenceCandidate] = []
        for tagged in tagged_sentences:
            text = tagged.text.strip()
            if len(text) < self.min_sentence_chars:
                continue
            role = tagged.role if tagged.role in self.ROLE_KEYS else "other"
            base_score = tagged.role_scores.get(role, 0.0)
            score = self._score_sentence(
                sentence=text,
                role=role,
                section_title=tagged.section_title,
                role_scores=tagged.role_scores,
            )
            candidates.append(
                SentenceCandidate(
                    sentence_id=tagged.global_index,
                    text=text,
                    role=role,
                    score=max(base_score, score),
                    word_count=tagged.word_count,
                )
            )

        if not candidates:
            return self._empty_info()

        # 在统一预算下跨角色联合优化，而不是每个角色独立贪心。
        selected = self.selector.select(candidates)
        grouped = self._empty_info()
        for candidate in sorted(selected, key=lambda item: item.sentence_id):
            role = candidate.role if candidate.role in grouped else None
            if role is None:
                continue
            grouped[role].append(candidate.text)

        for role in self.ROLE_KEYS:
            grouped[role] = self._deduplicate(grouped[role])[: self.max_sentences_per_role]

        grouped = self._fill_missing_roles(grouped, candidates)
        # 最后再做一次去重和截断，避免补全阶段超出每类上限。
        for role in self.ROLE_KEYS:
            grouped[role] = self._deduplicate(grouped[role])[: self.max_sentences_per_role]
        return grouped

    def _fill_missing_roles(
        self,
        grouped: dict[str, list[str]],
        candidates: list[SentenceCandidate],
    ) -> dict[str, list[str]]:
        for role in self.ROLE_KEYS:
            if grouped[role]:
                continue
            role_candidates = [c for c in candidates if c.role == role]
            role_candidates.sort(key=lambda c: c.score, reverse=True)
            for candidate in role_candidates:
                if candidate.text not in grouped[role]:
                    grouped[role].append(candidate.text)
                if len(grouped[role]) >= self.max_sentences_per_role:
                    break
        return grouped

    def _score_sentence(
        self,
        sentence: str,
        role: str,
        section_title: str,
        role_scores: dict[str, float],
    ) -> float:
        score = role_scores.get(role, 0.0) * self.score_boost_role_confidence

        bucket = self._bucket_for_title(section_title)
        if bucket and bucket == role:
            score += self.score_boost_section_match

        word_count = self._word_count(sentence)
        if 8 <= word_count <= 45:
            score += 0.1
        if any(ch.isdigit() for ch in sentence):
            score += 0.05
        if role == "limitations" and any(marker in sentence.lower() for marker in ("however", "but", "limit")):
            score += 0.08
        return score

    def _extract_rule_based(self, sections: list[Section]) -> dict[str, list[str]]:
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

    def _empty_info(self) -> dict[str, list[str]]:
        return {role: [] for role in self.ROLE_KEYS}

    def _has_enough_info(self, info: dict[str, list[str]]) -> bool:
        non_empty_roles = sum(1 for role in self.ROLE_KEYS if info.get(role))
        return non_empty_roles >= 2

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

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text))
