from __future__ import annotations

from pathlib import Path

from src.utils.io import load_yaml


class PromptManager:
    def __init__(self, prompts: dict[str, str]) -> None:
        self.prompts = prompts

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromptManager":
        data = load_yaml(path)
        return cls({key: str(value) for key, value in data.items()})

    def render(self, prompt_name: str, **kwargs: object) -> str:
        template = self.prompts.get(prompt_name)
        if template is None:
            raise KeyError(f"Prompt not found: {prompt_name}")
        return template.format(**kwargs)

    def list_prompts(self) -> list[str]:
        return sorted(self.prompts.keys())
