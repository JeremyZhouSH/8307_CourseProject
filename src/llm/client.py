from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


# 类作用：封装相关状态与方法，负责该模块的核心能力。
class LLMClient:
    """最小可用 LLM 客户端：支持 mock、OpenAI 兼容 API、HF 本地推理。"""

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def __init__(
        self,
        use_mock: bool = True,
        model: str = "local-mock",
        provider: str = "openai_compatible",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        chat_endpoint: str = "/chat/completions",
        timeout_seconds: float = 30.0,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        local_max_input_length: int = 2048,
        local_max_new_tokens: int = 256,
        local_device_map: str = "auto",
        local_torch_dtype: str = "auto",
        local_trust_remote_code: bool = False,
    ) -> None:
        self.use_mock = use_mock
        self.model = model
        self.provider = provider.strip().lower()
        self.api_key_env = api_key_env
        self.api_key = (api_key or os.getenv(api_key_env, "")).strip()
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint if chat_endpoint.startswith("/") else f"/{chat_endpoint}"
        self.timeout_seconds = timeout_seconds
        self.system_prompt = (system_prompt or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_max_input_length = int(local_max_input_length)
        self.local_max_new_tokens = int(local_max_new_tokens)
        self.local_device_map = local_device_map
        self.local_torch_dtype = local_torch_dtype
        self.local_trust_remote_code = bool(local_trust_remote_code)
        self._local_tokenizer: Any = None
        self._local_model: Any = None

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def complete(self, prompt: str) -> str:
        if self.use_mock:
            # mock 模式返回可读预览，便于离线调试。
            prefix = f"[MOCK:{self.model}]"
            preview = prompt.strip().replace("\n", " ")[:200]
            return f"{prefix} {preview}"

        if self.provider in {"openai_compatible", "openai", "api"}:
            return self._complete_openai_compatible(prompt)
        if self.provider in {"hf_local", "local_hf", "local"}:
            return self._complete_local(prompt)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _complete_openai_compatible(self, prompt: str) -> str:
        if not self.api_key:
            raise ValueError(
                f"Missing API key for LLM call. Set {self.api_key_env} or pass api_key to LLMClient."
            )
        payload = self._build_payload(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = self._chat_url()
        response = self._post_json(url=url, payload=payload, headers=headers)
        content = self._extract_content(response)
        if not content:
            raise RuntimeError("LLM response did not include assistant content.")
        return content

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_payload(self, prompt: str) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _chat_url(self) -> str:
        endpoint = self.chat_endpoint
        # 防止 base_url 与 endpoint 同时带 /v1 导致重复路径。
        if self.base_url.endswith("/v1") and endpoint.startswith("/v1/"):
            endpoint = endpoint[3:]
        return f"{self.base_url}{endpoint}"

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _post_json(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url=url, data=body, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned non-JSON response: {raw[:300]}") from exc

        if not isinstance(data, dict):
            raise RuntimeError("LLM response JSON root must be an object.")
        return data

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _extract_content(self, response: dict[str, Any]) -> str:
        # 兼容 Responses 风格字段。
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        # 兼容 Chat Completions 风格字段。
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        message = first_choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_chunks = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text_value = part.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        text_chunks.append(text_value.strip())
                return "\n".join(text_chunks).strip()

        text = first_choice.get("text")
        if isinstance(text, str):
            return text.strip()

        return ""

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _complete_local(self, prompt: str) -> str:
        tokenizer, model = self._ensure_local_model()
        full_prompt = self._build_local_prompt(prompt)
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.local_max_input_length,
        )
        model_inputs = {k: v.to(model.device) for k, v in inputs.items()}
        do_sample = self.temperature > 0
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=self.local_max_new_tokens or self.max_tokens or 256,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
        )
        generated = outputs[0]
        # 对 decoder-only 模型仅保留新生成片段，对 encoder-decoder 直接解码即可。
        input_len = model_inputs["input_ids"].shape[-1]
        gen_tokens = generated[input_len:] if generated.shape[-1] > input_len else generated
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        if text:
            return text
        fallback = tokenizer.decode(generated, skip_special_tokens=True).strip()
        return fallback

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _build_local_prompt(self, prompt: str) -> str:
        if not self.system_prompt:
            return prompt
        return f"{self.system_prompt}\n\n{prompt}"

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _ensure_local_model(self) -> tuple[Any, Any]:
        if self._local_tokenizer is not None and self._local_model is not None:
            return self._local_tokenizer, self._local_model

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except Exception as exc:
            raise RuntimeError("Local HF provider requires transformers and torch.") from exc

        dtype = self._parse_torch_dtype(torch, self.local_torch_dtype)
        model_kwargs: dict[str, Any] = {"trust_remote_code": self.local_trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if self.local_device_map and self.local_device_map != "none":
            model_kwargs["device_map"] = self.local_device_map

        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            trust_remote_code=self.local_trust_remote_code,
        )
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model, **model_kwargs)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)

        if not hasattr(model, "device"):
            model = model.to("cpu")

        self._local_tokenizer = tokenizer
        self._local_model = model
        return tokenizer, model

    # 函数作用：内部辅助逻辑，服务当前类/模块主流程。
    def _parse_torch_dtype(self, torch_module: Any, value: str) -> Any:
        normalized = (value or "auto").strip().lower()
        if normalized in {"auto", ""}:
            return None
        if normalized in {"float16", "fp16"}:
            return torch_module.float16
        if normalized in {"bfloat16", "bf16"}:
            return torch_module.bfloat16
        if normalized in {"float32", "fp32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported local_torch_dtype: {value}")
