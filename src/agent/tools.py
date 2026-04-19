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


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class AgentTools:
    """把流水线能力封装成可动态选择的工具集合。"""

    # 注册工具名到函数的映射，供 Planner 按名称调用。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
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

    # 执行单个工具动作；若名称未注册则直接报错。
    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def execute(self, action: AgentAction, state: PipelineState) -> None:
        tool = self._tool_map.get(action.name)
        if tool is None:
            raise KeyError(f"Unknown tool/action: {action.name}")
        tool(state, action)

    # 检查是否需要澄清输入；若需要则写入问题结构并暂停流程。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
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

    # 从长期记忆检索相似任务，必要时调整 extractor 策略。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _adapt_strategy_from_memory(self, state: PipelineState, action: AgentAction) -> None:
        state.strategy_adapted = True
        query = state.user_request or state.input_path
        hits = self.memory_store.retrieve(query=query, top_k=3)
        state.memory_hits = hits
        suggested = self.memory_store.best_strategy_for(query=query, min_score=0.7)
        if suggested and hasattr(self.pipeline.extractor, "strategy"):
            self.pipeline.extractor.strategy = suggested
            state.extractor_strategy = suggested

    # 解析输入输出路径并写回 state，供后续工具统一使用。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
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

    # 加载原始文档文本。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _load_document(self, state: PipelineState, action: AgentAction) -> None:
        state.document_text = self.pipeline.loader.load(Path(state.input_path))

    # 执行章节切分。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _split_sections(self, state: PipelineState, action: AgentAction) -> None:
        state.sections = self.pipeline.splitter.split(state.document_text)

    # 抽取 objective/methods/results/limitations 关键信息。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _extract_key_info(self, state: PipelineState, action: AgentAction) -> None:
        state.key_info = self.pipeline.extractor.extract(state.sections)

    # 构建结构化摘要 JSON 结果。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_structured_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.structured_summary = self.pipeline.structured_summarizer.summarize(
            state.sections,
            state.key_info,
        )

    # 由结构化摘要生成可读最终摘要。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_final_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.final_summary = self.pipeline.final_summarizer.summarize(state.structured_summary)

    # 可选 LLM 重写：仅在允许且非 mock 模式下执行。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _maybe_refine_with_llm(self, state: PipelineState, action: AgentAction) -> None:
        state.llm_summary_attempted = True
        if not state.should_try_llm or self.pipeline.llm_client.use_mock:
            return
        llm_summary = self.pipeline._generate_final_summary_with_llm(state.structured_summary)  # noqa: SLF001
        if llm_summary:
            state.final_summary = llm_summary

    # 计算摘要忠实度并写入 state.verification。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _verify_summary(self, state: PipelineState, action: AgentAction) -> None:
        state.verification = self.pipeline.verifier.check(
            final_summary=state.final_summary,
            key_info=state.key_info,
            document_text=state.document_text,
        )

    # 将完整结果（含 agent 轨迹）落盘到 output_path。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
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

    # 重试前切到 rule 策略，并清空中间结果避免污染下一轮。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
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

    # 标记流程结束，供 Planner/Controller 感知退出状态。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _finish(self, state: PipelineState, action: AgentAction) -> None:
        state.agent_finished = True
