from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class MemoryRecord:
    request: str
    summary: str
    extractor_strategy: str
    faithfulness_score: float
    retry_count: int
    note: str = ""


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class AgentMemoryStore:
    """长期记忆：将历史运行记录持久化到 JSONL。"""

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, path: str | Path = "data/outputs/agent_memory.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def append(self, record: MemoryRecord) -> None:
        payload = {
            "request": record.request,
            "summary": record.summary,
            "extractor_strategy": record.extractor_strategy,
            "faithfulness_score": round(record.faithfulness_score, 3),
            "retry_count": record.retry_count,
            "note": record.note,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        query_tokens = self._tokens(query)
        rows: list[tuple[float, dict[str, Any]]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = f"{obj.get('request', '')} {obj.get('summary', '')}"
                score = self._similarity(query_tokens, self._tokens(text))
                rows.append((score, obj))
        rows.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in rows[: max(1, top_k)] if score > 0]

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def best_strategy_for(self, query: str, min_score: float = 0.7) -> str | None:
        candidates = self.retrieve(query, top_k=10)
        if not candidates:
            return None
        # 选 faithfulness 更高、重试更少的历史策略。
        ranked = sorted(
            candidates,
            key=lambda row: (
                float(row.get("faithfulness_score", 0.0)),
                -int(row.get("retry_count", 0)),
            ),
            reverse=True,
        )
        best = ranked[0]
        score = float(best.get("faithfulness_score", 0.0))
        strategy = str(best.get("extractor_strategy", "")).strip()
        if strategy and score >= min_score:
            return strategy
        return None

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _tokens(self, text: str) -> set[str]:
        return {t for t in re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text.lower())}

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = left & right
        union = left | right
        return len(overlap) / max(1, len(union))
