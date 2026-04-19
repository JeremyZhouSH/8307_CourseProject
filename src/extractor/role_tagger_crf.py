from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from src.parser.section_splitter import Section

try:
    import sklearn_crfsuite  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    sklearn_crfsuite = None


# 角色标签集合：前四个用于摘要结构化输出，other 作为兜底类。
ROLES: tuple[str, ...] = ("objective", "methods", "results", "limitations", "other")


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class SentenceUnit:
    text: str
    section_title: str
    section_index: int
    sentence_index: int
    global_index: int
    total_sentences: int


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class TaggedSentence:
    text: str
    section_title: str
    role: str
    role_scores: dict[str, float]
    global_index: int
    word_count: int


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class SentenceRoleTagger:
    SECTION_BUCKETS = {
        "objective": {"abstract", "introduction", "background", "motivation"},
        "methods": {"method", "methods", "approach", "framework"},
        "results": {"result", "results", "experiment", "experiments", "evaluation"},
        "limitations": {"limitation", "limitations", "discussion", "future work", "conclusion"},
    }

    ROLE_KEYWORDS = {
        "objective": {
            "objective",
            "aim",
            "goal",
            "motivation",
            "problem",
            "propose",
            "introduce",
            "investigate",
        },
        "methods": {
            "method",
            "approach",
            "pipeline",
            "framework",
            "algorithm",
            "model",
            "train",
            "dataset",
        },
        "results": {
            "result",
            "improve",
            "outperform",
            "accuracy",
            "f1",
            "performance",
            "experiment",
            "show",
        },
        "limitations": {
            "limitation",
            "future",
            "weakness",
            "challenge",
            "cannot",
            "however",
            "although",
            "risk",
        },
    }

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, method: str = "crf") -> None:
        # 支持 crf/hmm/heuristic；非法值会在 _predict 中走 heuristic。
        self.method = method.lower().strip()

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def tag_sections(self, sections: list[Section]) -> list[TaggedSentence]:
        units = self._build_sentence_units(sections)
        if not units:
            return []

        predictions = self._predict(units)
        tagged: list[TaggedSentence] = []
        for unit, (role, scores) in zip(units, predictions):
            tagged.append(
                TaggedSentence(
                    text=unit.text,
                    section_title=unit.section_title,
                    role=role,
                    role_scores=scores,
                    global_index=unit.global_index,
                    word_count=self._word_count(unit.text),
                )
            )
        return tagged

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _predict(self, units: list[SentenceUnit]) -> list[tuple[str, dict[str, float]]]:
        # 主分发逻辑：优先 CRF，再退 HMM，再退启发式，保证流程不断。
        if self.method == "crf":
            crf_result = self._predict_with_crf(units)
            if crf_result is not None:
                return crf_result
            hmm_result = self._predict_with_hmm(units)
            if hmm_result is not None:
                return hmm_result
            return self._predict_with_heuristic(units)

        if self.method == "hmm":
            hmm_result = self._predict_with_hmm(units)
            if hmm_result is not None:
                return hmm_result
            return self._predict_with_heuristic(units)

        return self._predict_with_heuristic(units)

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _predict_with_crf(self, units: list[SentenceUnit]) -> list[tuple[str, dict[str, float]]] | None:
        if sklearn_crfsuite is None:
            return None
        if len(units) < 3:
            return None

        # 课程项目场景下没有人工标注语料，使用弱标注先拟合一个单文档 CRF。
        weak_labels = [self._weak_label(unit) for unit in units]
        if len(set(weak_labels)) < 2:
            return None

        features = [self._sentence_features(units, idx) for idx in range(len(units))]

        model = sklearn_crfsuite.CRF(
            algorithm="lbfgs",
            c1=0.1,
            c2=0.1,
            max_iterations=40,
            all_possible_transitions=True,
        )
        try:
            model.fit([features], [weak_labels])
            labels = model.predict_single(features)
            marginals = model.predict_marginals_single(features)
        except Exception:
            # CRF 训练或推断失败时回退，不中断主流程。
            return None
        predictions: list[tuple[str, dict[str, float]]] = []
        for label, marginal in zip(labels, marginals):
            scores = {role: float(marginal.get(role, 0.0)) for role in ROLES}
            predictions.append((label if label in ROLES else "other", self._normalize_scores(scores)))
        return predictions

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _predict_with_hmm(self, units: list[SentenceUnit]) -> list[tuple[str, dict[str, float]]] | None:
        if not units:
            return None

        transitions = self._transition_log_probs()
        start_log_probs = self._start_log_probs()

        # 发射概率直接复用启发式打分，转成概率分布。
        emissions = [self._heuristic_scores(unit) for unit in units]
        dp: list[dict[str, float]] = []
        backpointer: list[dict[str, str]] = []

        first_scores: dict[str, float] = {}
        first_backpointer: dict[str, str] = {}
        for role in ROLES:
            emission = max(emissions[0].get(role, 1e-9), 1e-9)
            first_scores[role] = start_log_probs[role] + math.log(emission)
            first_backpointer[role] = ""
        dp.append(first_scores)
        backpointer.append(first_backpointer)

        for index in range(1, len(units)):
            current_scores: dict[str, float] = {}
            current_backpointer: dict[str, str] = {}
            for role in ROLES:
                emission = max(emissions[index].get(role, 1e-9), 1e-9)
                emission_log = math.log(emission)
                best_prev = ROLES[0]
                best_score = float("-inf")
                for prev_role in ROLES:
                    score = dp[index - 1][prev_role] + transitions[prev_role][role] + emission_log
                    if score > best_score:
                        best_score = score
                        best_prev = prev_role
                current_scores[role] = best_score
                current_backpointer[role] = best_prev
            dp.append(current_scores)
            backpointer.append(current_backpointer)

        # Viterbi 回溯最优角色序列。
        path = [max(dp[-1], key=dp[-1].get)]
        for index in range(len(units) - 1, 0, -1):
            path.append(backpointer[index][path[-1]])
        path.reverse()

        predictions: list[tuple[str, dict[str, float]]] = []
        for role, emission in zip(path, emissions):
            predictions.append((role, emission))
        return predictions

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _predict_with_heuristic(self, units: list[SentenceUnit]) -> list[tuple[str, dict[str, float]]]:
        output: list[tuple[str, dict[str, float]]] = []
        for unit in units:
            scores = self._heuristic_scores(unit)
            label = max(scores, key=scores.get)
            output.append((label, scores))
        return output

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _sentence_features(self, units: Sequence[SentenceUnit], index: int) -> dict[str, object]:
        unit = units[index]
        lower = unit.text.lower()
        words = re.findall(r"[A-Za-z][A-Za-z\-]{1,}", lower)
        section_lower = unit.section_title.lower()

        # 以可解释特征为主：位置、长度、数字、章节提示、关键词计数等。
        feature: dict[str, object] = {
            "bias": 1.0,
            "word_count": len(words),
            "char_count": len(unit.text),
            "contains_digit": any(ch.isdigit() for ch in unit.text),
            "has_percent": "%" in unit.text,
            "sentence_pos_bin": self._position_bin(unit.global_index, unit.total_sentences),
            "section_title": section_lower,
            "section_hint": self._bucket_for_title(unit.section_title) or "none",
            "lower[:20]": lower[:20],
            "endswith_colon": unit.text.strip().endswith(":"),
        }

        for role in ("objective", "methods", "results", "limitations"):
            feature[f"kw_count_{role}"] = self._keyword_hits(lower, role)
            feature[f"section_is_{role}"] = int((self._bucket_for_title(unit.section_title) or "other") == role)

        if index > 0:
            prev = units[index - 1]
            feature["-1:section_hint"] = self._bucket_for_title(prev.section_title) or "none"
            feature["-1:word_count"] = self._word_count(prev.text)
        else:
            feature["BOS"] = True

        if index + 1 < len(units):
            nxt = units[index + 1]
            feature["+1:section_hint"] = self._bucket_for_title(nxt.section_title) or "none"
            feature["+1:word_count"] = self._word_count(nxt.text)
        else:
            feature["EOS"] = True

        return feature

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _heuristic_scores(self, unit: SentenceUnit) -> dict[str, float]:
        lower = unit.text.lower()
        section_bucket = self._bucket_for_title(unit.section_title)
        scores = {role: 0.05 for role in ROLES}

        # 章节标题匹配优先级较高。
        if section_bucket:
            scores[section_bucket] += 0.45

        for role in ("objective", "methods", "results", "limitations"):
            hits = self._keyword_hits(lower, role)
            scores[role] += min(0.4, 0.12 * hits)

        position = unit.global_index / max(1, unit.total_sentences - 1)
        # 文档前中后位置先验：前偏 objective，中偏 methods，后偏 results/limitations。
        if position <= 0.2:
            scores["objective"] += 0.15
        if 0.2 < position <= 0.6:
            scores["methods"] += 0.1
        if 0.5 <= position <= 0.9:
            scores["results"] += 0.12
        if position >= 0.75:
            scores["limitations"] += 0.1

        if self._word_count(unit.text) < 5:
            scores["other"] += 0.25

        return self._normalize_scores(scores)

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _transition_log_probs(self) -> dict[str, dict[str, float]]:
        # 转移矩阵体现论文常见叙事顺序：
        # objective -> methods -> results -> limitations。
        high = math.log(0.55)
        medium = math.log(0.2)
        low = math.log(0.08)
        tiny = math.log(0.03)

        return {
            "objective": {
                "objective": medium,
                "methods": high,
                "results": low,
                "limitations": tiny,
                "other": low,
            },
            "methods": {
                "objective": tiny,
                "methods": medium,
                "results": high,
                "limitations": low,
                "other": low,
            },
            "results": {
                "objective": tiny,
                "methods": low,
                "results": medium,
                "limitations": high,
                "other": low,
            },
            "limitations": {
                "objective": tiny,
                "methods": tiny,
                "results": low,
                "limitations": high,
                "other": medium,
            },
            "other": {
                "objective": low,
                "methods": low,
                "results": low,
                "limitations": low,
                "other": medium,
            },
        }

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _start_log_probs(self) -> dict[str, float]:
        return {
            "objective": math.log(0.45),
            "methods": math.log(0.25),
            "results": math.log(0.15),
            "limitations": math.log(0.1),
            "other": math.log(0.05),
        }

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_sentence_units(self, sections: list[Section]) -> list[SentenceUnit]:
        units: list[SentenceUnit] = []
        for section_index, section in enumerate(sections):
            sentences = self._split_sentences(section.content)
            for sentence_index, sentence in enumerate(sentences):
                cleaned = sentence.strip()
                if cleaned:
                    units.append(
                        SentenceUnit(
                            text=cleaned,
                            section_title=section.title,
                            section_index=section_index,
                            sentence_index=sentence_index,
                            global_index=len(units),
                            total_sentences=0,
                        )
                    )
        total = len(units)
        for idx, unit in enumerate(units):
            unit.global_index = idx
            unit.total_sentences = total
        return units

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _weak_label(self, unit: SentenceUnit) -> str:
        section_bucket = self._bucket_for_title(unit.section_title)
        if section_bucket:
            return section_bucket
        scores = self._heuristic_scores(unit)
        return max(scores, key=scores.get)

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [part.strip() for part in parts if part.strip()]
        if not sentences and text.strip():
            return [text.strip()]
        return sentences

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _bucket_for_title(self, title: str) -> str | None:
        normalized = title.lower()
        for bucket, hints in self.SECTION_BUCKETS.items():
            if any(hint in normalized for hint in hints):
                return bucket
        return None

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _keyword_hits(self, sentence_lower: str, role: str) -> int:
        keywords = self.ROLE_KEYWORDS.get(role, set())
        return sum(1 for keyword in keywords if keyword in sentence_lower)

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _position_bin(self, index: int, total: int) -> str:
        if total <= 1:
            return "single"
        ratio = index / (total - 1)
        if ratio < 0.2:
            return "very_early"
        if ratio < 0.4:
            return "early"
        if ratio < 0.6:
            return "middle"
        if ratio < 0.8:
            return "late"
        return "very_late"

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?", text))

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        cleaned = {role: max(value, 1e-9) for role, value in scores.items()}
        total = sum(cleaned.values())
        if total <= 0:
            return {role: 1.0 / len(ROLES) for role in ROLES}
        return {role: cleaned.get(role, 0.0) / total for role in ROLES}
