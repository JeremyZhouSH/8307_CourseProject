from __future__ import annotations

from dataclasses import dataclass

from src.agent.planner import AgentAction
from src.agent.state import PipelineState


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class ReviewDecision:
    should_retry: bool = False
    retry_action: AgentAction | None = None
    note: str = ""


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class AgentReviewer:
    """根据执行结果判断是否需要重试。"""

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, min_faithfulness: float = 0.65, retry_limit: int = 1) -> None:
        self.min_faithfulness = min_faithfulness
        self.retry_limit = max(0, retry_limit)

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def review(self, state: PipelineState, action: AgentAction, error: Exception | None = None) -> ReviewDecision:
        if error is not None:
            if state.retry_count < self.retry_limit:
                return ReviewDecision(
                    should_retry=True,
                    retry_action=AgentAction(name=action.name, params=action.params, reason="工具执行异常，重试"),
                    note=f"{action.name} 发生异常，触发重试",
                )
            return ReviewDecision(note=f"{action.name} 发生异常且超过重试上限")

        if action.name != "verify_summary":
            return ReviewDecision()

        score = float(state.verification.get("faithfulness_score", 0.0))
        unsupported_count = len(state.verification.get("unsupported_claims", []))
        if score >= self.min_faithfulness:
            if unsupported_count <= 1:
                return ReviewDecision(note=f"faithfulness={score:.3f} 达标")
            if state.retry_count < self.retry_limit:
                return ReviewDecision(
                    should_retry=True,
                    retry_action=AgentAction(
                        name="retry_with_rule_extractor",
                        reason=f"unsupported_claims={unsupported_count} 偏多，重试",
                    ),
                    note="unsupported_claims 偏多，触发重试",
                )
            return ReviewDecision(note=f"unsupported_claims={unsupported_count} 偏多但超过重试上限")

        if state.retry_count >= self.retry_limit:
            return ReviewDecision(note=f"faithfulness={score:.3f} 未达标，但超过重试上限")

        if state.extractor_strategy != "rule":
            return ReviewDecision(
                should_retry=True,
                retry_action=AgentAction(
                    name="retry_with_rule_extractor",
                    reason=f"faithfulness={score:.3f} 偏低，切换规则抽取重跑",
                ),
                note="触发策略切换重试",
            )
        return ReviewDecision(note="已是 rule 策略，继续输出结果")
