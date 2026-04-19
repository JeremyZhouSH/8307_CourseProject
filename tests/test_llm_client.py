from __future__ import annotations

import pytest

from src.llm.client import LLMClient


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_returns_mock_response() -> None:
    client = LLMClient(use_mock=True, model="local-test")
    result = client.complete("Hello\nWorld")

    assert result.startswith("[MOCK:local-test] ")
    assert "Hello World" in result


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    env_name = "OPENAI_API_KEY_FOR_TEST"
    monkeypatch.delenv(env_name, raising=False)

    client = LLMClient(use_mock=False, model="gpt-test", api_key_env=env_name)

    with pytest.raises(ValueError, match=env_name):
        client.complete("ping")


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_real_branch_builds_payload_and_returns_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(
        use_mock=False,
        model="gpt-test",
        api_key="sk-test",
        base_url="https://example.com/",
        chat_endpoint="v1/chat/completions",
        system_prompt="You are concise.",
        temperature=0.2,
        max_tokens=64,
    )

    captured: dict[str, object] = {}

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def fake_post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        captured["url"] = url
        captured["payload"] = payload
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "final answer"}}]}

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    result = client.complete("Summarize this.")

    assert result == "final answer"
    assert captured["url"] == "https://example.com/v1/chat/completions"
    assert captured["headers"] == {
        "Authorization": "Bearer sk-test",
        "Content-Type": "application/json",
    }

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "gpt-test"
    assert payload["temperature"] == 0.2
    assert payload["max_tokens"] == 64
    assert payload["messages"] == [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Summarize this."},
    ]


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_real_branch_supports_list_content(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(use_mock=False, model="gpt-test", api_key="sk-test")

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def fake_post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "output_text", "text": "First line"},
                            {"type": "output_text", "text": "Second line"},
                        ]
                    }
                }
            ]
        }

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    result = client.complete("Summarize this.")

    assert result == "First line\nSecond line"


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_real_branch_avoids_double_v1_in_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(
        use_mock=False,
        model="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com/v1",
        chat_endpoint="/v1/chat/completions",
    )
    captured: dict[str, object] = {}

    # 函数作用：执行当前步骤的核心逻辑，并返回处理结果。
    def fake_post_json(
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        captured["url"] = url
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(client, "_post_json", fake_post_json)

    result = client.complete("ping")

    assert result == "ok"
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_hf_local_branch_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(
        use_mock=False,
        provider="hf_local",
        model="medical-model-local",
        api_key_env="OPENAI_API_KEY_FOR_TEST",
    )

    monkeypatch.setattr(client, "_complete_local", lambda prompt: "local medical summary")

    result = client.complete("Summarize this paper.")
    assert result == "local medical summary"


# 测试函数作用：验证一个具体场景，确保行为与预期一致。
def test_complete_raises_for_unsupported_provider() -> None:
    client = LLMClient(use_mock=False, provider="unknown-provider", model="x")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        client.complete("ping")
