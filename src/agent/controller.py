from __future__ import annotations

from pathlib import Path

from src.agent.state import PipelineState
from src.pipeline import SummarizationPipeline


class AgentController:
    """Thin wrapper for pipeline orchestration."""

    def __init__(self, pipeline: SummarizationPipeline | None = None) -> None:
        self.pipeline = pipeline or SummarizationPipeline()

    def run(self, input_path: str | Path | None = None, output_path: str | Path | None = None) -> PipelineState:
        return self.pipeline.run(input_path=input_path, output_path=output_path)
