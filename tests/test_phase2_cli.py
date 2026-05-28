from __future__ import annotations

from pathlib import Path

import pytest

from ai_chatbot.cli import command_name
from ai_chatbot.config import load_config
from ai_chatbot.db import ChatStore
from ai_chatbot.session import ChatSession


def test_command_name_handles_model_subcommands() -> None:
    assert command_name("/model list") == "/model list"
    assert command_name("/model set llama-3.1-8b-instant") == "/model set"
    assert command_name("/request req_123") == "/request"
    assert command_name("/exit") == "/exit"


def test_load_config_from_temp_models_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text(
        """
[provider]
name = "test"
base_url_env = "LLM_BASE_URL"
api_key_env = "LLM_API_KEY"
chat_completions_path = "/chat/completions"

[defaults]
model = "test-model"
timeout_seconds = 30
max_concurrency = 1
requests_per_minute = 6

[models."test-model"]
display_name = "Test Model"
family = "test"
use_case = "unit tests"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/api/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_CHAT_COMPLETIONS_PATH", "/chat/completions")
    monkeypatch.setenv("DEFAULT_MODEL", "test-model")
    monkeypatch.setenv("MAX_OUTPUT_TOKENS", "400")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "chatbot.sqlite3"))

    config = load_config(models_path)

    assert config.default_model == "test-model"
    assert config.chat_url() == "https://example.test/api/v1/chat/completions"
    assert config.max_output_tokens == 400
    assert config.sqlite_path == tmp_path / "chatbot.sqlite3"


def test_session_switches_and_clears_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    models_path = tmp_path / "models.toml"
    models_path.write_text(
        """
[provider]
name = "test"

[defaults]
model = "model-a"

[models."model-a"]
display_name = "Model A"
family = "test"
use_case = "default"

[models."model-b"]
display_name = "Model B"
family = "test"
use_case = "backup"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_MODEL", "model-a")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "chatbot.sqlite3"))

    session = ChatSession(load_config(models_path), conversation_id="conv_test")
    session.history.append({"role": "user", "content": "hello"})

    session.set_model("model-b")
    session.clear()

    assert session.active_model == "model-b"
    assert session.history == []


def test_store_initializes_schema_and_records_request(tmp_path: Path) -> None:
    async def run() -> None:
        store = ChatStore(tmp_path / "chatbot.sqlite3")
        await store.initialize()
        conversation_id = await store.start_conversation("model-a")
        await store.add_message(conversation_id, "user", "hello", "model-a")
        await store.record_llm_request(
            request_id="req_test",
            conversation_id=conversation_id,
            model="model-a",
            provider="test",
            messages=[{"role": "user", "content": "hello"}],
            response_content="ok",
            status_code=200,
            success=True,
            latency_ms=12.5,
            input_tokens=2,
            output_tokens=1,
            total_tokens=3,
        )

        messages = await store.list_messages(conversation_id)
        request = await store.get_request("req_test")

        assert messages[0]["content"] == "hello"
        assert request is not None
        assert request.success is True
        assert request.prompt_chars > 0
        assert request.prompt_hash

    import asyncio

    asyncio.run(run())
