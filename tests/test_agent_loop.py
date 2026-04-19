from __future__ import annotations

import json

from src.agent.controller import AgentController
from src.agent.memory import AgentMemoryStore
from src.agent.reviewer import AgentReviewer
from src.pipeline import SummarizationPipeline


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_agent_controller_runs_planning_loop(tmp_path) -> None:
    input_text = """
ABSTRACT
This paper proposes a modular summarization workflow.

METHODS
We use a staged pipeline with parser extractor summarizer and verifier.

RESULTS
The workflow produces stable summaries in local execution.
""".strip()

    input_file = tmp_path / "paper.txt"
    output_file = tmp_path / "agent_summary.json"
    input_file.write_text(input_text, encoding="utf-8")

    memory_file = tmp_path / "memory.jsonl"
    controller = AgentController(
        pipeline=SummarizationPipeline(),
        memory_store=AgentMemoryStore(memory_file),
    )
    state = controller.run(input_path=input_file, output_path=output_file)

    assert output_file.exists()
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["final_summary"]
    assert payload["verification"]["faithfulness_score"] >= 0.0
    assert state.output_written is True
    assert state.step_history
    assert state.step_history[-1]["action"] == "finish"
    assert memory_file.exists()
    assert memory_file.read_text(encoding="utf-8").strip()


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_agent_controller_can_retry_when_quality_low(tmp_path) -> None:
    input_text = """
INTRODUCTION
The work studies a robust summarization workflow.

METHODS
The method uses modular components.

RESULTS
The system keeps coherent output.
""".strip()

    input_file = tmp_path / "paper.txt"
    output_file = tmp_path / "agent_retry_summary.json"
    input_file.write_text(input_text, encoding="utf-8")

    pipeline = SummarizationPipeline()
    reviewer = AgentReviewer(min_faithfulness=1.1, retry_limit=1)
    memory_file = tmp_path / "memory.jsonl"
    controller = AgentController(
        pipeline=pipeline,
        reviewer=reviewer,
        memory_store=AgentMemoryStore(memory_file),
    )
    state = controller.run(input_path=input_file, output_path=output_file)

    assert state.retry_count >= 1
    assert any(step["action"] == "retry_with_rule_extractor" for step in state.step_history)
    assert output_file.exists()


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_agent_controller_can_request_clarification(tmp_path) -> None:
    output_file = tmp_path / "clarify_summary.json"
    missing_input = tmp_path / "missing_paper.txt"

    controller = AgentController(
        pipeline=SummarizationPipeline(),
        memory_store=AgentMemoryStore(tmp_path / "memory.jsonl"),
    )
    state = controller.run(
        input_path=missing_input,
        output_path=output_file,
        user_request="请",
    )

    assert state.needs_clarification is True
    assert state.clarification_question
    assert state.clarification_request
    assert state.clarification_request["question_id"] == "clarify_inputs"
    assert isinstance(state.clarification_request["fields"], list)
    assert state.output_written is False


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_agent_controller_can_resume_with_clarification_answers(tmp_path) -> None:
    input_text = """
ABSTRACT
This paper proposes a controllable summarization process.

METHODS
We apply planning and tool-based execution.

RESULTS
The output remains stable after clarification-driven resume.
""".strip()
    input_file = tmp_path / "paper.txt"
    input_file.write_text(input_text, encoding="utf-8")
    output_file = tmp_path / "resume_summary.json"

    controller = AgentController(
        pipeline=SummarizationPipeline(),
        memory_store=AgentMemoryStore(tmp_path / "memory.jsonl"),
    )
    # 第一次先触发澄清。
    first_state = controller.run(
        input_path=tmp_path / "not_found.txt",
        output_path=output_file,
        user_request="请",
    )
    assert first_state.needs_clarification is True

    # 第二次通过结构化答案回填，继续执行并产出结果。
    resumed = controller.run(
        output_path=output_file,
        clarification_answers={
            "input_path": str(input_file),
            "user_request": "生成摘要并写出JSON",
        },
    )
    assert resumed.needs_clarification is False
    assert resumed.output_written is True
    assert output_file.exists()
