from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.agent.dialogue import DialogueManager
from src.agent.memory import AgentMemoryStore
from src.agent.planner import AgentAction
from src.agent.state import PipelineState
from src.pipeline import SummarizationPipeline
from src.utils.io import write_json


class AgentTools:
    """把流水线能力封装成可动态选择的工具集合。"""

    def __init__(
        self,
        pipeline: SummarizationPipeline,
        memory_store: AgentMemoryStore,
        dialogue: DialogueManager,
    ) -> None:
        self.pipeline = pipeline
        self.memory_store = memory_store
        self.dialogue = dialogue
        self._tool_map: dict[str, Callable[[PipelineState, AgentAction], None]] = {
            "clarify_goal": self._clarify_goal,
            "adapt_strategy_from_memory": self._adapt_strategy_from_memory,
            "resolve_paths": self._resolve_paths,
            "load_document": self._load_document,
            "split_sections": self._split_sections,
            "extract_key_info": self._extract_key_info,
            "build_structured_summary": self._build_structured_summary,
            "build_final_summary": self._build_final_summary,
            "maybe_refine_with_llm": self._maybe_refine_with_llm,
            "verify_summary": self._verify_summary,
            "write_output": self._write_output,
            "retry_with_rule_extractor": self._retry_with_rule_extractor,
            "finish": self._finish,
        }

    def execute(self, action: AgentAction, state: PipelineState) -> None:
        tool = self._tool_map.get(action.name)
        if tool is None:
            raise KeyError(f"Unknown tool/action: {action.name}")
        tool(state, action)

    def _clarify_goal(self, state: PipelineState, action: AgentAction) -> None:
        state.clarification_checked = True
        clarification = self.dialogue.clarify(state)
        if clarification is None:
            state.needs_clarification = False
            state.clarification_question = ""
            state.clarification_request = {}
            return
        state.needs_clarification = True
        state.clarification_question = clarification.question
        state.clarification_request = clarification.to_dict()

    def _adapt_strategy_from_memory(self, state: PipelineState, action: AgentAction) -> None:
        state.strategy_adapted = True
        query = state.user_request or state.input_path
        hits = self.memory_store.retrieve(query=query, top_k=3)
        state.memory_hits = hits
        suggested = self.memory_store.best_strategy_for(query=query, min_score=0.7)
        if suggested and hasattr(self.pipeline.extractor, "strategy"):
            self.pipeline.extractor.strategy = suggested
            state.extractor_strategy = suggested

    def _resolve_paths(self, state: PipelineState, action: AgentAction) -> None:
        input_path = action.params.get("input_path")
        output_path = action.params.get("output_path")
        resolved_input = self.pipeline._resolve_io_path(  # noqa: SLF001
            input_path,
            str(self.pipeline.config.get("input_path", "data/samples/sample_paper.txt")),
        )
        resolved_output = self.pipeline._resolve_io_path(  # noqa: SLF001
            output_path,
            str(self.pipeline.config.get("output_path", "data/outputs/summary.json")),
        )
        state.input_path = str(resolved_input)
        state.output_path = str(resolved_output)

    def _load_document(self, state: PipelineState, action: AgentAction) -> None:
        state.document_text = self.pipeline.loader.load(Path(state.input_path))

    def _split_sections(self, state: PipelineState, action: AgentAction) -> None:
        state.sections = self.pipeline.splitter.split(state.document_text)

    def _extract_key_info(self, state: PipelineState, action: AgentAction) -> None:
        state.key_info = self.pipeline.extractor.extract(state.sections)

    def _build_structured_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.structured_summary = self.pipeline.structured_summarizer.summarize(
            state.sections,
            state.key_info,
        )

    def _build_final_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.final_summary = self.pipeline.final_summarizer.summarize(state.structured_summary)

    def _maybe_refine_with_llm(self, state: PipelineState, action: AgentAction) -> None:
        state.llm_summary_attempted = True
        if not state.should_try_llm or self.pipeline.llm_client.use_mock:
            return
        llm_summary = self.pipeline._generate_final_summary_with_llm(state.structured_summary)  # noqa: SLF001
        if llm_summary:
            state.final_summary = llm_summary

    def _verify_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.verification = self.pipeline.verifier.check(
            final_summary=state.final_summary,
            key_info=state.key_info,
            document_text=state.document_text,
        )

    def _write_output(self, state: PipelineState, action: AgentAction) -> None:
        payload = {
            "input_path": state.input_path,
            "sections": [asdict(section) for section in state.sections],
            "key_info": state.key_info,
            "structured_summary": state.structured_summary,
            "final_summary": state.final_summary,
            "verification": state.verification,
            "agent_trace": state.step_history,
            "clarification_question": state.clarification_question,
            "clarification_request": state.clarification_request,
            "memory_hits": state.memory_hits,
        }
        write_json(payload, state.output_path)
        state.output_written = True

    def _retry_with_rule_extractor(self, state: PipelineState, action: AgentAction) -> None:
        if hasattr(self.pipeline.extractor, "strategy"):
            self.pipeline.extractor.strategy = "rule"
        state.extractor_strategy = "rule"
        state.key_info = {}
        state.structured_summary = {}
        state.final_summary = ""
        state.verification = {}
        state.output_written = False
        state.llm_summary_attempted = False

    def _finish(self, state: PipelineState, action: AgentAction) -> None:
        state.agent_finished = True
