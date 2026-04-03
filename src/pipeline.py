from __future__ import annotations

import json
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


class SummarizationPipeline:
    def __init__(
        self,
        config_path: str | Path = "config/default.yaml",
        prompts_path: str | Path = "config/prompts.yaml",
    ) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self._resolve_path(config_path)
        self.prompts_path = self._resolve_path(prompts_path)

        self.config: dict[str, Any] = load_yaml(self.config_path)
        self.prompts = PromptManager.from_yaml(self.prompts_path)

        llm_cfg = self._resolve_llm_config(self.config.get("llm", {}))
        pipeline_cfg = self.config.get("pipeline", {})
        self.use_llm_final_summary = bool(pipeline_cfg.get("use_llm_final_summary", True))

        self.loader = DocumentLoader()
        self.splitter = SectionSplitter(max_sections=int(self.config.get("max_sections", 20)))
        self.extractor = KeyInfoExtractor()
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
            api_key=str(llm_cfg.get("api_key")) if llm_cfg.get("api_key") else None,
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            base_url=str(llm_cfg.get("base_url", "https://api.openai.com/v1")),
            chat_endpoint=str(llm_cfg.get("chat_endpoint", "/chat/completions")),
            timeout_seconds=float(llm_cfg.get("timeout_seconds", 30.0)),
            system_prompt=str(llm_cfg.get("system_prompt", "")) or None,
            temperature=float(llm_cfg.get("temperature", 0.0)),
            max_tokens=max_tokens,
        )

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _resolve_io_path(self, path_value: str | Path, fallback: str) -> Path:
        path = Path(path_value) if path_value else Path(fallback)
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()

    def _resolve_llm_config(self, llm_cfg: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(llm_cfg)

        env_api_key = os.getenv("SMART_LLM__API_KEY", "").strip()
        env_base_url = os.getenv("SMART_LLM__BASE_URL", "").strip()
        env_model = os.getenv("SMART_LLM__MODEL_NAME", "").strip()
        env_use_mock = os.getenv("SMART_LLM__USE_MOCK", "").strip().lower()

        if env_api_key:
            resolved["api_key"] = env_api_key
            if not env_use_mock:
                resolved["use_mock"] = False
        if env_base_url:
            resolved["base_url"] = env_base_url
        if env_model:
            resolved["model"] = env_model

        if env_use_mock in {"1", "true", "yes", "on"}:
            resolved["use_mock"] = True
        elif env_use_mock in {"0", "false", "no", "off"}:
            resolved["use_mock"] = False

        return resolved

    def _build_llm_final_summary_prompt(self, structured_summary: dict[str, Any]) -> str:
        serialized = json.dumps(structured_summary, ensure_ascii=False, indent=2)
        return self.prompts.render("final_summary", structured_summary=serialized)

    def _generate_final_summary_with_llm(self, structured_summary: dict[str, Any]) -> str | None:
        prompt = self._build_llm_final_summary_prompt(structured_summary)
        result = self.llm_client.complete(prompt).strip()
        return result or None

    def run(
        self,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
    ) -> PipelineState:
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

        state.document_text = self.loader.load(resolved_input)
        state.sections = self.splitter.split(state.document_text)

        state.key_info = self.extractor.extract(state.sections)
        state.structured_summary = self.structured_summarizer.summarize(
            state.sections,
            state.key_info,
        )
        state.final_summary = self.final_summarizer.summarize(state.structured_summary)
        if self.use_llm_final_summary and not self.llm_client.use_mock:
            try:
                llm_summary = self._generate_final_summary_with_llm(state.structured_summary)
                if llm_summary:
                    state.final_summary = llm_summary
            except Exception:
                # Keep deterministic fallback summary when external LLM call fails.
                pass
        state.verification = self.verifier.check(
            final_summary=state.final_summary,
            key_info=state.key_info,
            document_text=state.document_text,
        )

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
