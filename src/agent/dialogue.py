from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agent.state import PipelineState


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class ClarificationField:
    name: str
    prompt: str
    required: bool = True
    field_type: str = "string"


@dataclass
# 类作用：封装相关状态与方法，负责该模块的核心能力。
class ClarificationRequest:
    question_id: str
    question: str
    fields: list[ClarificationField]

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "fields": [asdict(field) for field in self.fields],
        }


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class DialogueManager:
    """对话澄清：在关键输入缺失或目标不明确时暂停并提问。"""

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def clarify(self, state: PipelineState) -> ClarificationRequest | None:
        fields: list[ClarificationField] = []
        request = state.user_request.strip()
        if request and len(request) < 4:
            fields.append(
                ClarificationField(
                    name="user_request",
                    prompt="你的目标描述太短，请补充更具体的任务要求。",
                )
            )

        if not state.input_path.strip():
            fields.append(
                ClarificationField(
                    name="input_path",
                    prompt="请提供待处理论文的输入路径。",
                )
            )

        elif not Path(state.input_path).exists():
            fields.append(
                ClarificationField(
                    name="input_path",
                    prompt=f"输入文件不存在：{state.input_path}。请确认路径后重试。",
                )
            )

        if not fields:
            return None
        return ClarificationRequest(
            question_id="clarify_inputs",
            question="继续执行前需要补充以下信息：",
            fields=fields,
        )
