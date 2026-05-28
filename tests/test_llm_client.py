from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from ai_chatbot.config import AppConfig, load_config
from ai_chatbot.failure_simulator import FailureSimulator
from ai_chatbot.llm_client import LLMClient, LLMClientError


def load_test_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AppConfig:
    models_path = tmp_path / "models.toml"
    models_path.write_text(
        """
[provider]
name = "test"

[defaults]
model = "model-a"
timeout_seconds = 30
max_concurrency = 2
requests_per_minute = 6

[models."model-a"]
display_name = "Model A"
family = "test"
use_case = "unit tests"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEFAULT_MODEL", "model-a")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "chatbot.sqlite3"))
    return load_config(models_path)


def test_client_retries_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(500, json={"error": "temporary"}, request=request)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
                request=request,
            )

        client = LLMClient(
            load_test_config(monkeypatch, tmp_path),
            FailureSimulator(),
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await client.chat("model-a", [{"role": "user", "content": "hi"}])
        finally:
            await client.close()

        assert response.content == "ok"
        assert response.retry_attempts == 1
        assert calls == 2

    asyncio.run(run())


def test_client_does_not_retry_non_retryable_http_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(401, json={"error": "bad key"}, request=request)

        client = LLMClient(
            load_test_config(monkeypatch, tmp_path),
            FailureSimulator(),
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(LLMClientError) as exc_info:
                await client.chat("model-a", [{"role": "user", "content": "hi"}])
        finally:
            await client.close()

        assert exc_info.value.status_code == 401
        assert exc_info.value.retry_attempts == 0
        assert calls == 1

    asyncio.run(run())


def test_failure_simulator_retries_then_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("simulated failure should happen before HTTP")

        client = LLMClient(
            load_test_config(monkeypatch, tmp_path),
            FailureSimulator(enabled=True, rate=1.0, kind="429"),
            transport=httpx.MockTransport(handler),
        )
        try:
            with pytest.raises(LLMClientError) as exc_info:
                await client.chat("model-a", [{"role": "user", "content": "hi"}])
        finally:
            await client.close()

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_attempts == 3

    asyncio.run(run())


def test_request_payload_does_not_exceed_configured_tokens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def run() -> None:
        seen_payload: dict[str, object] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal seen_payload
            seen_payload = json.loads(request.content.decode("utf-8"))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                request=request,
            )

        config = load_test_config(monkeypatch, tmp_path)
        client = LLMClient(
            config,
            FailureSimulator(),
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.chat("model-a", [{"role": "user", "content": "hi"}])
        finally:
            await client.close()

        assert seen_payload["max_tokens"] == config.max_output_tokens

    asyncio.run(run())
