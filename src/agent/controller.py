from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.dialogue import DialogueManager
from src.agent.memory import AgentMemoryStore, MemoryRecord
from src.agent.planner import AgentAction, AgentPlanner
from src.agent.reviewer import AgentReviewer
from src.agent.state import PipelineState
from src.agent.tools import AgentTools
from src.pipeline import SummarizationPipeline


class AgentController:
    """带决策与规划循环的 Agent 控制器。"""

    def __init__(
        self,
        pipeline: SummarizationPipeline | None = None,
        planner: AgentPlanner | None = None,
        reviewer: AgentReviewer | None = None,
        memory_store: AgentMemoryStore | None = None,
        dialogue: DialogueManager | None = None,
    ) -> None:
        self.pipeline = pipeline or SummarizationPipeline()
        agent_cfg = self.pipeline.config.get("agent", {})
        retry_limit = int(agent_cfg.get("retry_limit", 1))
        min_faithfulness = float(agent_cfg.get("min_faithfulness", 0.65))
        self.max_steps = int(agent_cfg.get("max_steps", 24))
        memory_path = str(agent_cfg.get("memory_path", "data/outputs/agent_memory.jsonl"))
        self.planner = planner or AgentPlanner()
        self.reviewer = reviewer or AgentReviewer(
            min_faithfulness=min_faithfulness,
            retry_limit=retry_limit,
        )
        self.memory_store = memory_store or AgentMemoryStore(path=memory_path)
        self.dialogue = dialogue or DialogueManager()
        self.tools = AgentTools(
            pipeline=self.pipeline,
            memory_store=self.memory_store,
            dialogue=self.dialogue,
        )

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        user_request: str = "",
        clarification_answers: dict[str, Any] | None = None,
    ) -> PipelineState:
        state = PipelineState()
        state.should_try_llm = bool(self.pipeline.use_llm_final_summary and not self.pipeline.llm_client.use_mock)
        state.extractor_strategy = str(getattr(self.pipeline.extractor, "strategy", "unknown"))
        answers = clarification_answers or {}
        effective_input = answers.get("input_path", input_path)
        effective_output = answers.get("output_path", output_path)
        effective_request = str(answers.get("user_request", user_request)).strip()
        state.user_request = effective_request

        self.planner.enqueue(
            AgentAction(
                name="resolve_paths",
                params={"input_path": effective_input, "output_path": effective_output},
                reason="初始化路径",
            )
        )

        for step_index in range(self.max_steps):
            action = self.planner.next_action(state)
            error: Exception | None = None

            try:
                self.tools.execute(action, state)
            except Exception as exc:  # pragma: no cover - errors handled by reviewer branch
                error = exc

            decision = self.reviewer.review(state=state, action=action, error=error)
            state.step_history.append(
                {
                    "step": step_index + 1,
                    "action": action.name,
                    "reason": action.reason,
                    "retry_count": state.retry_count,
                    "error": str(error) if error else "",
                    "review_note": decision.note,
                }
            )

            if error is not None and not decision.should_retry:
                raise error

            if decision.should_retry and decision.retry_action is not None:
                state.retry_count += 1
                self.planner.enqueue(decision.retry_action)
                continue

            if action.name == "finish":
                break

        if not state.output_written and state.verification:
            self.tools.execute(AgentAction(name="write_output", reason="兜底写输出"), state)

        if state.verification:
            self.memory_store.append(
                MemoryRecord(
                    request=state.user_request,
                    summary=state.final_summary,
                    extractor_strategy=state.extractor_strategy,
                    faithfulness_score=float(state.verification.get("faithfulness_score", 0.0)),
                    retry_count=state.retry_count,
                    note="needs_clarification" if state.needs_clarification else "",
                )
            )
        return state
