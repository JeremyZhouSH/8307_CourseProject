from __future__ import annotations

from pathlib import Path

from src.agent.state import PipelineState
from src.pipeline import SummarizationPipeline


class AgentController:
    """对流水线的一层轻量封装，方便上层统一调用。"""

    def __init__(self, pipeline: SummarizationPipeline | None = None) -> None:
        self.pipeline = pipeline or SummarizationPipeline()

    def run(self, input_path: str | Path | None = None, output_path: str | Path | None = None) -> PipelineState:
        # 直接透传到底层流水线，保持控制器行为简单可预测。
        return self.pipeline.run(input_path=input_path, output_path=output_path)
