from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


class LLMClient:
    """最小可用 LLM 客户端：支持 mock 与 OpenAI 兼容接口。"""

    def __init__(
        self,
        use_mock: bool = True,
        model: str = "local-mock",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        chat_endpoint: str = "/chat/completions",
        timeout_seconds: float = 30.0,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.use_mock = use_mock
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = (api_key or os.getenv(api_key_env, "")).strip()
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint if chat_endpoint.startswith("/") else f"/{chat_endpoint}"
        self.timeout_seconds = timeout_seconds
        self.system_prompt = (system_prompt or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        if self.use_mock:
            # mock 模式返回可读预览，便于离线调试。
            prefix = f"[MOCK:{self.model}]"
            preview = prompt.strip().replace("\n", " ")[:200]
            return f"{prefix} {preview}"

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

    def _chat_url(self) -> str:
        endpoint = self.chat_endpoint
        # 防止 base_url 与 endpoint 同时带 /v1 导致重复路径。
        if self.base_url.endswith("/v1") and endpoint.startswith("/v1/"):
            endpoint = endpoint[3:]
        return f"{self.base_url}{endpoint}"

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
