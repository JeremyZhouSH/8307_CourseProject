from __future__ import annotations

from pathlib import Path

from src.utils.io import load_yaml


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class PromptManager:
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(self, prompts: dict[str, str]) -> None:
        self.prompts = prompts

    @classmethod
    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def from_yaml(cls, path: str | Path) -> "PromptManager":
        # 统一把 YAML 值转成字符串，避免模板渲染时类型不一致。
        data = load_yaml(path)
        return cls({key: str(value) for key, value in data.items()})

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def render(self, prompt_name: str, **kwargs: object) -> str:
        template = self.prompts.get(prompt_name)
        if template is None:
            raise KeyError(f"Prompt not found: {prompt_name}")
        return template.format(**kwargs)

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def list_prompts(self) -> list[str]:
        return sorted(self.prompts.keys())
