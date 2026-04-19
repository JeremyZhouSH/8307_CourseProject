from __future__ import annotations

import re
from typing import Any


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class FaithfulnessChecker:
    # 计算摘要与原文的一致性分数，并返回可解释的分项指标。
    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def check(
        self,
        final_summary: str,
        key_info: dict[str, list[str]],
        document_text: str,
    ) -> dict[str, Any]:
        summary_text = (final_summary or "").strip()
        doc_text = document_text or ""

        # 检查 key_info 中的 claim 是否能在原文直接定位，作为不支持声明列表。
        claims = [item for values in key_info.values() for item in values if isinstance(item, str)]
        unsupported = [claim for claim in claims if claim and claim not in doc_text]

        # 词汇、数字、句子可溯源三路打分，兼顾覆盖率与可解释性。
        lexical_score = self._lexical_overlap(summary_text, doc_text)
        number_score, unsupported_numbers = self._number_consistency(summary_text, doc_text)
        traceability_score = self._sentence_traceability(summary_text, doc_text)

        if not summary_text:
            score = 0.0
        else:
            # 固定权重聚合，保证每次运行结果可复现。
            score = (
                0.3 * lexical_score
                + 0.4 * number_score
                + 0.3 * traceability_score
            )

        return {
            "faithfulness_score": round(score, 3),
            "unsupported_claims": unsupported,
            "unsupported_numbers": unsupported_numbers,
            "subscores": {
                "lexical_overlap": round(lexical_score, 3),
                "number_consistency": round(number_score, 3),
                "sentence_traceability": round(traceability_score, 3),
            },
            "notes": (
                "Heuristic multi-signal check (lexical overlap + numeric consistency + sentence traceability). "
                "Use stronger verification for strict evaluation."
            ),
        }

    # 词汇重叠率：摘要词集合中有多少词可以在原文中找到。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _lexical_overlap(self, summary_text: str, doc_text: str) -> float:
        summary_tokens = self._tokenize(summary_text)
        doc_tokens = self._tokenize(doc_text)
        if not summary_tokens:
            return 0.0
        overlap = summary_tokens & doc_tokens
        return len(overlap) / len(summary_tokens)

    # 数字一致性：摘要中的数字是否在原文中出现，返回分数与缺失数字。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _number_consistency(self, summary_text: str, doc_text: str) -> tuple[float, list[str]]:
        summary_numbers = self._extract_numbers(summary_text)
        doc_numbers = set(self._extract_numbers(doc_text))
        if not summary_numbers:
            return 1.0, []
        missing = sorted({num for num in summary_numbers if num not in doc_numbers})
        matched = len(summary_numbers) - len(missing)
        return matched / len(summary_numbers), missing

    # 句子可溯源性：摘要每句与原文句子做最大 Jaccard，相当于弱对齐。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _sentence_traceability(self, summary_text: str, doc_text: str) -> float:
        summary_sentences = self._split_sentences(summary_text)
        doc_sentences = self._split_sentences(doc_text)
        if not summary_sentences or not doc_sentences:
            return 0.0

        doc_token_sets = [self._tokenize(sentence) for sentence in doc_sentences]
        best_scores: list[float] = []
        for sentence in summary_sentences:
            summary_tokens = self._tokenize(sentence)
            if not summary_tokens:
                best_scores.append(0.0)
                continue
            best = max(self._jaccard(summary_tokens, doc_tokens) for doc_tokens in doc_token_sets)
            best_scores.append(best)
        return sum(best_scores) / len(best_scores) if best_scores else 0.0

    # 简单句子切分，支持中英文常见句末符号。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    # 轻量词元化，仅保留英文词，避免引入额外依赖。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text.lower())}

    # 提取整数或小数，用于数值一致性检查。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _extract_numbers(self, text: str) -> list[str]:
        return re.findall(r"\d+(?:\.\d+)?", text)

    # 计算集合相似度，作为句子对齐分数。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _jaccard(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
