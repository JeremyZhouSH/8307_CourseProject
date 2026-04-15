from __future__ import annotations

import re
from typing import Any


class FaithfulnessChecker:
    def check(
        self,
        final_summary: str,
        key_info: dict[str, list[str]],
        document_text: str,
    ) -> dict[str, Any]:
        # 逐条检查抽取 claim 是否原文可定位（字符串级启发式）。
        claims = [item for values in key_info.values() for item in values]
        unsupported = [claim for claim in claims if claim and claim not in document_text]

        summary_tokens = self._tokenize(final_summary)
        doc_tokens = self._tokenize(document_text)

        if not summary_tokens:
            score = 0.0
        else:
            # 以词汇重叠率近似 faithfulness，适合课程项目快速对比。
            overlap = summary_tokens & doc_tokens
            score = len(overlap) / len(summary_tokens)

        return {
            "faithfulness_score": round(score, 3),
            "unsupported_claims": unsupported,
            "notes": "Heuristic lexical-overlap check; use stronger verification for strict evaluation.",
        }

    def _tokenize(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text.lower())}
