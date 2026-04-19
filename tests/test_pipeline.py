from __future__ import annotations

import json
from pathlib import Path

from src.pipeline import SummarizationPipeline


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_pipeline_runs_end_to_end(tmp_path) -> None:
    input_text = """
ABSTRACT
This paper proposes an explainable summarization workflow.

METHODS
We use a modular architecture with six components.

RESULTS
The workflow generates coherent summaries with stable behavior.
""".strip()

    input_file = tmp_path / "paper.txt"
    output_file = tmp_path / "summary.json"
    input_file.write_text(input_text, encoding="utf-8")

    pipeline = SummarizationPipeline()
    state = pipeline.run(input_path=input_file, output_path=output_file)

    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))

    assert state.final_summary
    assert payload["final_summary"]
    assert payload["verification"]["faithfulness_score"] >= 0
    assert len(payload["sections"]) >= 1


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_pipeline_fallback_when_llm_call_fails(monkeypatch, tmp_path) -> None:
    input_text = """
INTRODUCTION
This paper studies a robust summarization workflow.

METHODS
We use a modular architecture.

RESULTS
The workflow remains stable under failures.
""".strip()

    input_file = tmp_path / "paper.txt"
    output_file = tmp_path / "summary.json"
    input_file.write_text(input_text, encoding="utf-8")

    pipeline = SummarizationPipeline(
        config_path=Path("config/default.yaml"),
        prompts_path=Path("config/prompts.yaml"),
    )

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def fail_complete(prompt: str) -> str:
        raise RuntimeError("simulated llm failure")

    monkeypatch.setattr(pipeline.llm_client, "complete", fail_complete)

    state = pipeline.run(input_path=input_file, output_path=output_file)

    assert state.final_summary
    assert "No summary available." not in state.final_summary
    assert output_file.exists()
