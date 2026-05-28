from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

from ai_chatbot.config import AppConfig
from ai_chatbot.failure_simulator import FailureSimulator, SimulatedProviderError

Message = dict[str, str]
logger = logging.getLogger(__name__)
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_ATTEMPTS = 3


@dataclass(frozen=True)
class ChatResponse:
    request_id: str
    model: str
    content: str
    latency_ms: float
    status_code: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    retry_attempts: int


class LLMClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        request_id: str,
        model: str,
        latency_ms: float,
        status_code: int | None = None,
        error_type: str = "provider_error",
        retry_attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.model = model
        self.latency_ms = latency_ms
        self.status_code = status_code
        self.error_type = error_type
        self.retry_attempts = retry_attempts


class AsyncRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self._requests_per_minute = requests_per_minute
        self._starts: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            while self._starts and now - self._starts[0] >= 60:
                self._starts.popleft()

            if len(self._starts) >= self._requests_per_minute:
                sleep_seconds = 60 - (now - self._starts[0])
                logger.info(
                    "llm_request_rate_limited",
                    extra={"sleep_seconds": round(max(sleep_seconds, 0), 3)},
                )
                await asyncio.sleep(max(sleep_seconds, 0))
                now = time.monotonic()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()

            self._starts.append(time.monotonic())


class LLMClient:
    def __init__(
        self,
        config: AppConfig,
        failure_simulator: FailureSimulator,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._failure_simulator = failure_simulator
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._rate_limiter = AsyncRateLimiter(config.requests_per_minute)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10,
                read=config.timeout_seconds,
                write=10,
                pool=10,
            ),
            limits=httpx.Limits(
                max_connections=config.max_concurrency + 2,
                max_keepalive_connections=config.max_concurrency,
            ),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(self, model: str, messages: list[Message]) -> ChatResponse:
        self._config.validate_model(model)
        request_id = f"req_{uuid.uuid4().hex}"
        started = time.perf_counter()
        prompt_chars = sum(len(message.get("content", "")) for message in messages)

        logger.info(
            "llm_request_started",
            extra={
                "request_id": request_id,
                "model": model,
                "prompt_chars": prompt_chars,
            },
        )

        async with self._semaphore:
            await self._rate_limiter.acquire()
            return await self._chat_with_retries(request_id, model, messages, started)

    async def _chat_with_retries(
        self, request_id: str, model: str, messages: list[Message], started: float
    ) -> ChatResponse:
        retry_attempts = 0
        while True:
            try:
                return await self._chat_once(
                    request_id=request_id,
                    model=model,
                    messages=messages,
                    started=started,
                    retry_attempts=retry_attempts,
                )
            except LLMClientError as exc:
                if (
                    not is_retryable(exc.status_code, exc.error_type)
                    or retry_attempts >= MAX_RETRY_ATTEMPTS
                ):
                    raise
                retry_attempts += 1
                delay_seconds = min(2**retry_attempts * 0.25, 2.0)
                logger.warning(
                    "llm_request_retrying",
                    extra={
                        "request_id": request_id,
                        "model": model,
                        "retry_attempt": retry_attempts,
                        "delay_seconds": delay_seconds,
                        "status_code": exc.status_code,
                        "error_type": exc.error_type,
                    },
                )
                await asyncio.sleep(delay_seconds)

    async def _chat_once(
        self,
        *,
        request_id: str,
        model: str,
        messages: list[Message],
        started: float,
        retry_attempts: int,
    ) -> ChatResponse:
        try:
            await self._failure_simulator.maybe_fail(request_id, model)
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
        except SimulatedProviderError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            if exc.kind == "empty_response":
                logger.warning(
                    "llm_request_failed",
                    extra={
                        "request_id": request_id,
                        "model": model,
                        "latency_ms": latency_ms,
                        "status_code": exc.status_code,
                        "error_type": "schema",
                        "retry_attempts": retry_attempts,
                    },
                )
                raise LLMClientError(
                    f"Request {request_id} returned a simulated empty response",
                    request_id=request_id,
                    model=model,
                    latency_ms=latency_ms,
                    status_code=exc.status_code,
                    error_type="schema",
                    retry_attempts=retry_attempts,
                ) from exc

            error_type = "http"
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": exc.status_code,
                    "error_type": error_type,
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Provider returned simulated HTTP {exc.status_code} for request {request_id}",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                status_code=exc.status_code,
                error_type=error_type,
                retry_attempts=retry_attempts,
            ) from exc
        except httpx.HTTPStatusError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": exc.response.status_code,
                    "error_type": "http",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Provider returned HTTP {exc.response.status_code} for request {request_id}",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                status_code=exc.response.status_code,
                error_type="http",
                retry_attempts=retry_attempts,
            ) from exc
        except httpx.TimeoutException as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": None,
                    "error_type": "timeout",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Request {request_id} timed out",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                error_type="timeout",
                retry_attempts=retry_attempts,
            ) from exc
        except httpx.HTTPError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": None,
                    "error_type": "network",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Request {request_id} failed: {exc.__class__.__name__}",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                error_type="network",
                retry_attempts=retry_attempts,
            ) from exc
        except ValueError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": None,
                    "error_type": "schema",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Request {request_id} returned malformed provider data",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                error_type="schema",
                retry_attempts=retry_attempts,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": response.status_code,
                    "error_type": "schema",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Request {request_id} returned malformed provider data",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                status_code=response.status_code,
                error_type="schema",
                retry_attempts=retry_attempts,
            ) from exc
        content = extract_content(payload)
        if not content:
            logger.warning(
                "llm_request_failed",
                extra={
                    "request_id": request_id,
                    "model": model,
                    "latency_ms": latency_ms,
                    "status_code": response.status_code,
                    "error_type": "schema",
                    "retry_attempts": retry_attempts,
                },
            )
            raise LLMClientError(
                f"Request {request_id} returned an empty assistant response",
                request_id=request_id,
                model=model,
                latency_ms=latency_ms,
                status_code=response.status_code,
                error_type="schema",
                retry_attempts=retry_attempts,
            )

        usage = payload.get("usage") or {}
        logger.info(
            "llm_request_completed",
            extra={
                "request_id": request_id,
                "model": model,
                "latency_ms": latency_ms,
                "status_code": response.status_code,
                "response_chars": len(content),
                "input_tokens": as_optional_int(usage.get("prompt_tokens")),
                "output_tokens": as_optional_int(usage.get("completion_tokens")),
                "total_tokens": as_optional_int(usage.get("total_tokens")),
                "retry_attempts": retry_attempts,
            },
        )
        return ChatResponse(
            request_id=request_id,
            model=model,
            content=content,
            latency_ms=latency_ms,
            status_code=response.status_code,
            input_tokens=as_optional_int(usage.get("prompt_tokens")),
            output_tokens=as_optional_int(usage.get("completion_tokens")),
            total_tokens=as_optional_int(usage.get("total_tokens")),
            retry_attempts=retry_attempts,
        )


def extract_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


def as_optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def is_retryable(status_code: int | None, error_type: str) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or error_type in {"timeout", "network"}
