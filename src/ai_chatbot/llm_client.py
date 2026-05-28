from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from ai_chatbot.config import AppConfig

Message = dict[str, str]


@dataclass(frozen=True)
class ChatResponse:
    request_id: str
    model: str
    content: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(config.timeout_seconds))

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, model: str, messages: list[Message]) -> ChatResponse:
        self._config.validate_model(model)
        request_id = f"req_{uuid.uuid4().hex}"
        started = time.perf_counter()

        try:
            response = await self._client.post(
                self._config.chat_url(),
                headers={
                    "Authorization": f"Bearer {self._config.provider.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": self._config.max_output_tokens,
                    "temperature": 0,
                },
            )
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LLMClientError(
                f"Provider returned HTTP {exc.response.status_code} for request {request_id}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError(f"Request {request_id} timed out") from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Request {request_id} failed: {exc.__class__.__name__}") from exc

        payload = response.json()
        content = extract_content(payload)
        if not content:
            raise LLMClientError(f"Request {request_id} returned an empty assistant response")

        usage = payload.get("usage") or {}
        return ChatResponse(
            request_id=request_id,
            model=model,
            content=content,
            latency_ms=latency_ms,
            input_tokens=as_optional_int(usage.get("prompt_tokens")),
            output_tokens=as_optional_int(usage.get("completion_tokens")),
            total_tokens=as_optional_int(usage.get("total_tokens")),
        )


def extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None
