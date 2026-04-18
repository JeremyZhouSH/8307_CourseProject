from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import pulp  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    pulp = None


@dataclass
class SentenceCandidate:
    sentence_id: int
    text: str
    role: str
    score: float
    word_count: int


class ILPSentenceSelector:
    ROLES: tuple[str, ...] = ("objective", "methods", "results", "limitations")

    def __init__(
        self,
        word_budget: int = 200,
        redundancy_penalty: float = 0.35,
        similarity_threshold: float = 0.12,
        min_role_coverage: dict[str, int] | None = None,
    ) -> None:
        # 200 词预算是课程项目默认值，可在配置中覆盖。
        self.word_budget = max(1, int(word_budget))
        self.redundancy_penalty = max(0.0, float(redundancy_penalty))
        self.similarity_threshold = max(0.0, float(similarity_threshold))
        self.min_role_coverage = {
            role: max(0, int((min_role_coverage or {}).get(role, 0))) for role in self.ROLES
        }

    def select(self, candidates: list[SentenceCandidate]) -> list[SentenceCandidate]:
        # 先过滤明显无效句子，减少后续优化规模。
        filtered = [
            candidate
            for candidate in candidates
            if candidate.word_count > 0 and candidate.word_count <= self.word_budget and candidate.text.strip()
        ]
        if not filtered:
            return []

        chosen = self._select_with_ilp(filtered)
        if chosen is None:
            chosen = self._select_with_greedy(filtered)

        return sorted(chosen, key=lambda candidate: candidate.sentence_id)

    def _select_with_ilp(self, candidates: list[SentenceCandidate]) -> list[SentenceCandidate] | None:
        if pulp is None:
            return None

        # ILP：最大化信息分数，在预算与覆盖约束下抑制冗余。
        problem = pulp.LpProblem("summary_sentence_selection", pulp.LpMaximize)
        x_vars = {
            candidate.sentence_id: pulp.LpVariable(
                f"x_{candidate.sentence_id}",
                lowBound=0,
                upBound=1,
                cat=pulp.LpBinary,
            )
            for candidate in candidates
        }

        objective = pulp.lpSum(
            candidate.score * x_vars[candidate.sentence_id] for candidate in candidates
        )
        # 词数预算约束：sum(len_i * x_i) <= B。
        problem += pulp.lpSum(
            candidate.word_count * x_vars[candidate.sentence_id] for candidate in candidates
        ) <= self.word_budget, "word_budget"

        # 角色覆盖约束：保证 objective/methods/results/limitations 至少命中指定数量。
        for role, min_count in self.min_role_coverage.items():
            if min_count <= 0:
                continue
            role_candidates = [c for c in candidates if c.role == role]
            if not role_candidates:
                continue
            bounded_min = min(min_count, len(role_candidates))
            problem += (
                pulp.lpSum(x_vars[c.sentence_id] for c in role_candidates) >= bounded_min
            ), f"role_cover_{role}"

        pair_vars: dict[tuple[int, int], tuple[float, object]] = {}
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                sim = self._sentence_similarity(candidates[left].text, candidates[right].text)
                if sim < self.similarity_threshold:
                    continue
                cid_left = candidates[left].sentence_id
                cid_right = candidates[right].sentence_id
                z_var = pulp.LpVariable(
                    f"z_{cid_left}_{cid_right}",
                    lowBound=0,
                    upBound=1,
                    cat=pulp.LpBinary,
                )
                pair_vars[(cid_left, cid_right)] = (sim, z_var)
                # 线性化约束：z_ij = x_i AND x_j，用于冗余惩罚项。
                problem += z_var <= x_vars[cid_left], f"pair_l_{cid_left}_{cid_right}"
                problem += z_var <= x_vars[cid_right], f"pair_r_{cid_left}_{cid_right}"
                problem += z_var >= x_vars[cid_left] + x_vars[cid_right] - 1, (
                    f"pair_lb_{cid_left}_{cid_right}"
                )

        if pair_vars:
            # 目标函数增加冗余惩罚：-lambda * sim_ij * z_ij。
            objective -= pulp.lpSum(
                self.redundancy_penalty * sim * var for sim, var in pair_vars.values() # type: ignore
            ) 

        problem += objective, "selection_objective"

        try:
            status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
        except Exception:
            # 求解器异常时回退贪心，不影响主流程。
            return None
        status_name = pulp.LpStatus.get(status, "Unknown")
        if status_name not in {"Optimal", "Feasible"}:
            return None

        selected: list[SentenceCandidate] = []
        for candidate in candidates:
            value = x_vars[candidate.sentence_id].value()
            if value is not None and value >= 0.5:
                selected.append(candidate)

        if not selected:
            return None
        return selected

    def _select_with_greedy(self, candidates: list[SentenceCandidate]) -> list[SentenceCandidate]:
        # 回退策略：先满足角色覆盖，再按性价比(分数/词数)补句。
        selected: list[SentenceCandidate] = []
        selected_ids: set[int] = set()
        budget_left = self.word_budget

        for role, min_count in self.min_role_coverage.items():
            if min_count <= 0:
                continue
            role_candidates = [c for c in candidates if c.role == role and c.sentence_id not in selected_ids]
            role_candidates.sort(key=lambda c: c.score / max(1, c.word_count), reverse=True)
            for candidate in role_candidates:
                if min_count <= 0:
                    break
                if candidate.word_count <= budget_left:
                    selected.append(candidate)
                    selected_ids.add(candidate.sentence_id)
                    budget_left -= candidate.word_count
                    min_count -= 1

        remaining = [c for c in candidates if c.sentence_id not in selected_ids]
        remaining.sort(key=lambda c: c.score / max(1, c.word_count), reverse=True)
        for candidate in remaining:
            if candidate.word_count > budget_left:
                continue
            penalty = self._redundancy_penalty(candidate, selected)
            adjusted = candidate.score - penalty
            if adjusted <= 0:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.sentence_id)
            budget_left -= candidate.word_count
            if budget_left <= 0:
                break

        if selected:
            return selected

        smallest = min(candidates, key=lambda c: c.word_count)
        return [smallest] if smallest.word_count <= self.word_budget else []

    def _redundancy_penalty(
        self,
        candidate: SentenceCandidate,
        selected: list[SentenceCandidate],
    ) -> float:
        if not selected:
            return 0.0
        max_sim = max(self._sentence_similarity(candidate.text, item.text) for item in selected)
        return self.redundancy_penalty * max_sim

    def _sentence_similarity(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        union = left_tokens | right_tokens
        return len(overlap) / max(1, len(union))

    def _tokens(self, text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z][A-Za-z\-]{1,}", text.lower())}
