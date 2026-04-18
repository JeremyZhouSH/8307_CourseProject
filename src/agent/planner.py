from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.state import PipelineState


@dataclass
class AgentAction:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class AgentPlanner:
    """根据当前状态选择下一步动作。"""

    def __init__(self) -> None:
        self._forced_actions: list[AgentAction] = []

    def enqueue(self, action: AgentAction) -> None:
        self._forced_actions.append(action)

    def next_action(self, state: PipelineState) -> AgentAction:
        if self._forced_actions:
            return self._forced_actions.pop(0)

        if not state.input_path or not state.output_path:
            return AgentAction(name="resolve_paths", reason="初始化输入输出路径")
        if state.needs_clarification:
            return AgentAction(name="finish", reason="等待用户澄清后再继续")
        if not state.clarification_checked:
            return AgentAction(name="clarify_goal", reason="执行目标澄清检查")
        if not state.strategy_adapted:
            return AgentAction(name="adapt_strategy_from_memory", reason="根据历史记忆调整策略")
        if not state.document_text:
            return AgentAction(name="load_document", reason="读取原始文档")
        if not state.sections:
            return AgentAction(name="split_sections", reason="进行章节切分")
        if not self._has_key_info(state):
            return AgentAction(name="extract_key_info", reason="抽取关键信息")
        if not state.structured_summary:
            return AgentAction(name="build_structured_summary", reason="生成结构化摘要")
        if not state.final_summary:
            return AgentAction(name="build_final_summary", reason="生成最终摘要")
        if state.should_try_llm and not state.llm_summary_attempted:
            return AgentAction(name="maybe_refine_with_llm", reason="可选 LLM 增强摘要")
        if not state.verification:
            return AgentAction(name="verify_summary", reason="进行支持性校验")
        if not state.output_written:
            return AgentAction(name="write_output", reason="写出 JSON 结果")
        return AgentAction(name="finish", reason="流程完成")

    def _has_key_info(self, state: PipelineState) -> bool:
        if not state.key_info:
            return False
        return any(state.key_info.get(role) for role in ("objective", "methods", "results", "limitations"))
