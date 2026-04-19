from __future__ import annotations

from pathlib import Path

from src.utils.io import read_text


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class DocumentLoader:
    """从本地读取纯文本论文。"""

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def load(self, path: str | Path) -> str:
        return read_text(path)
