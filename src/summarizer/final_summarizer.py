from __future__ import annotations


class FinalSummarizer:
    def summarize(self, structured_summary: dict[str, object]) -> str:
        key_info = structured_summary.get("key_info", {})
        if not isinstance(key_info, dict):
            return "No summary available."

        objective = self._join_items(key_info.get("objective", []))
        methods = self._join_items(key_info.get("methods", []))
        results = self._join_items(key_info.get("results", []))
        limitations = self._join_items(key_info.get("limitations", []))

        sentences: list[str] = []
        # 固定模板组装最终学术摘要，便于课程展示与解释。
        if objective:
            sentences.append(f"The paper investigates the following objective: {objective}")
        if methods:
            sentences.append(f"It applies the following methodology: {methods}")
        if results:
            sentences.append(f"Key findings include: {results}")
        if limitations:
            sentences.append(f"Reported limitations are: {limitations}")

        if not sentences:
            return "No summary available."

        return " ".join(sentence if sentence.endswith(".") else sentence + "." for sentence in sentences)

    def _join_items(self, items: object) -> str:
        if not isinstance(items, list):
            return ""
        # 每类最多拼接两条，避免摘要段落过长。
        cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        return " ".join(cleaned[:2])
