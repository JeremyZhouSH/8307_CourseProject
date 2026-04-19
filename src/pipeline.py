from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.agent.state import PipelineState
from src.extractor.key_info_extractor import KeyInfoExtractor
from src.llm.client import LLMClient
from src.llm.prompts import PromptManager
from src.parser.document_loader import DocumentLoader
from src.parser.section_splitter import SectionSplitter
from src.summarizer.final_summarizer import FinalSummarizer
from src.summarizer.structured_summarizer import StructuredSummarizer
from src.utils.io import load_yaml, write_json
from src.verifier.faithfulness_checker import FaithfulnessChecker

logger = logging.getLogger(__name__)


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class SummarizationPipeline:
    # 初始化整条摘要流水线：加载配置、组装模块、准备 LLM 客户端。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(
        self,
        config_path: str | Path = "config/default.yaml",
        prompts_path: str | Path = "config/prompts.yaml",
    ) -> None:
        # 统一以项目根目录解析相对路径，避免工作目录差异影响运行。
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self._resolve_path(config_path)
        self.prompts_path = self._resolve_path(prompts_path)

        # 配置与提示词在初始化阶段一次性加载。
        self.config: dict[str, Any] = load_yaml(self.config_path)
        self.prompts = PromptManager.from_yaml(self.prompts_path)

        llm_cfg = self._resolve_llm_config(self.config.get("llm", {}))
        pipeline_cfg = self.config.get("pipeline", {})
        self.use_llm_final_summary = bool(pipeline_cfg.get("use_llm_final_summary", True))

        self.loader = DocumentLoader()
        self.splitter = SectionSplitter(max_sections=int(self.config.get("max_sections", 20)))
        extractor_cfg = dict(self.config.get("extractor", {}))
        if "min_sentence_chars" not in extractor_cfg:
            # 与旧配置兼容：若 extractor 未配置该参数，则沿用 pipeline 配置。
            extractor_cfg["min_sentence_chars"] = int(pipeline_cfg.get("min_sentence_chars", 30))
        self.extractor = KeyInfoExtractor(extractor_cfg=extractor_cfg)
        self.structured_summarizer = StructuredSummarizer(
            max_sentences=int(pipeline_cfg.get("max_summary_sentences", 2))
        )
        self.final_summarizer = FinalSummarizer()
        self.verifier = FaithfulnessChecker()
        max_tokens_value = llm_cfg.get("max_tokens")
        max_tokens = int(max_tokens_value) if max_tokens_value is not None else None
        self.llm_client = LLMClient(
            use_mock=bool(llm_cfg.get("use_mock", True)),
            model=str(llm_cfg.get("model", "local-mock")),
            provider=str(llm_cfg.get("provider", "openai_compatible")),
            api_key=str(llm_cfg.get("api_key")) if llm_cfg.get("api_key") else None,
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://api.openai.com/v1")),
            chat_endpoint=str(llm_cfg.get("chat_endpoint", "/chat/completions")),
            timeout_seconds=float(llm_cfg.get("timeout_seconds", 30.0)),
            system_prompt=str(llm_cfg.get("system_prompt", "")) or None,
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=max_tokens,
            local_max_input_length=int(llm_cfg.get("local_max_input_length", 2048)),
            local_max_new_tokens=int(llm_cfg.get("local_max_new_tokens", 256)),
            local_device_map=str(llm_cfg.get("local_device_map", "auto")),
            local_torch_dtype=str(llm_cfg.get("local_torch_dtype", "auto")),
            local_trust_remote_code=bool(llm_cfg.get("local_trust_remote_code", False)),
        )

    # 解析配置文件路径：相对路径统一映射到项目根目录。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    # 解析输入/输出路径：缺失时使用默认值，且统一转绝对路径。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _resolve_io_path(self, path_value: str | Path, fallback: str) -> Path:
        path = Path(path_value) if path_value else Path(fallback)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    # 把环境变量覆盖到 llm 配置，便于不改 yaml 直接切换模型。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _resolve_llm_config(self, llm_cfg: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(llm_cfg)

        # 通过 SMART_LLM__* 环境变量覆盖配置，方便本地切换模型与密钥。
        env_api_key = os.getenv("SMART_LLM__API_KEY", "").strip()
        env_base_url = os.getenv("SMART_LLM__BASE_URL", "").strip()
        env_model = os.getenv("SMART_LLM__MODEL_NAME", "").strip()
        env_provider = os.getenv("SMART_LLM__PROVIDER", "").strip()
        env_use_mock = os.getenv("SMART_LLM__USE_MOCK", "").strip().lower()

        if env_api_key:
            resolved["api_key"] = env_api_key
            if not env_use_mock:
                resolved["use_mock"] = False
        if env_base_url:
            resolved["base_url"] = env_base_url
        if env_model:
            resolved["model"] = env_model
        if env_provider:
            resolved["provider"] = env_provider

        if env_use_mock in {"1", "true", "yes", "on"}:
            resolved["use_mock"] = True
        elif env_use_mock in {"0", "false", "no", "off"}:
            resolved["use_mock"] = False

        return resolved

    # 构建最终摘要重写 prompt：把结构化摘要序列化后填入模板。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_llm_final_summary_prompt(self, structured_summary: dict[str, Any]) -> str:
        # 结构化结果先序列化再注入 prompt，提升可追溯性。
        serialized = json.dumps(structured_summary, ensure_ascii=False, indent=2)
        return self.prompts.render("final_summary", structured_summary=serialized)

    # 调用 LLM 生成最终摘要，空字符串时返回 None 交由上游回退。
    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _generate_final_summary_with_llm(self, structured_summary: dict[str, Any]) -> str | None:
        prompt = self._build_llm_final_summary_prompt(structured_summary)
        result = self.llm_client.complete(prompt).strip()
        return result or None

    # 执行端到端摘要流程，并返回包含中间状态的 PipelineState。
    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> PipelineState:
        # Step 0. 初始化运行状态，并解析输入输出路径。
        state = PipelineState()

        resolved_input = self._resolve_io_path(
            input_path,
            str(self.config.get("input_path", "data/samples/sample_paper.txt")),
        )
        resolved_output = self._resolve_io_path(
            output_path,
            str(self.config.get("output_path", "data/outputs/summary.json")),
        )

        state.input_path = str(resolved_input)
        state.output_path = str(resolved_output)

        # Step 1. 读取原文并按章节切分。
        state.document_text = self.loader.load(resolved_input)
        state.sections = self.splitter.split(state.document_text)

        # Step 2. 抽取 objective/methods/results/limitations 等关键信息。
        state.key_info = self.extractor.extract(state.sections)

        # Step 3. 生成结构化摘要与本地最终摘要。
        state.structured_summary = self.structured_summarizer.summarize(
            state.sections,
            state.key_info,
        )
        state.final_summary = self.final_summarizer.summarize(state.structured_summary)

        # Step 4. 可选：若启用真实 LLM，则用 LLM 重写最终摘要。
        if self.use_llm_final_summary and not self.llm_client.use_mock:
            try:
                llm_summary = self._generate_final_summary_with_llm(state.structured_summary)
                if llm_summary:
                    state.final_summary = llm_summary
            except Exception as exc:
                # 外部 LLM 失败时保留本地确定性摘要，保证流程稳定返回。
                logger.warning(
                    "LLM final-summary refinement failed, fallback to template summary: %s",
                    exc,
                )

        # Step 5. 对最终摘要做支持性（faithfulness）启发式校验。
        state.verification = self.verifier.check(
            final_summary=state.final_summary,
            key_info=state.key_info,
            document_text=state.document_text,
        )

        # Step 6. 组装输出并写入 JSON 文件。
        payload = {
            "input_path": state.input_path,
            "sections": [asdict(section) for section in state.sections],
            "key_info": state.key_info,
            "structured_summary": state.structured_summary,
            "final_summary": state.final_summary,
            "verification": state.verification,
        }
        write_json(payload, resolved_output)

        return state
