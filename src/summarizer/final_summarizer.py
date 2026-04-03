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
        cleaned = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        return " ".join(cleaned[:2])
