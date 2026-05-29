"""In-memory conversation session that orchestrates chat turns and persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ai_chatbot.config import AppConfig
from ai_chatbot.db import ChatStore
from ai_chatbot.llm_client import ChatResponse, LLMClient, LLMClientError, Message

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    """Holds conversation state: active model, message history, and conversation ID."""

    config: AppConfig
    conversation_id: str
    active_model: str = field(init=False)
    history: list[Message] = field(default_factory=list)
    thinking_enabled: bool = field(init=False)
    thinking_budget_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.active_model = self.config.default_model
        self.thinking_enabled = self.config.thinking_enabled
        self.thinking_budget_tokens = self.config.thinking_budget_tokens

    def set_model(self, model_id: str) -> None:
        """Switch the active model, validating it exists in config first."""
        previous_model = self.active_model
        self.config.validate_model(model_id)
        self.active_model = model_id
        logger.info(
            "model_switched",
            extra={
                "conversation_id": self.conversation_id,
                "from_model": previous_model,
                "to_model": model_id,
            },
        )

    def set_thinking_enabled(self, enabled: bool) -> None:
        """Toggle thinking on or off for this session."""
        previous = self.thinking_enabled
        self.thinking_enabled = enabled
        logger.info(
            "thinking_toggled",
            extra={
                "conversation_id": self.conversation_id,
                "thinking_enabled": enabled,
                "previous": previous,
            },
        )

    def set_thinking_budget(self, budget: int) -> None:
        """Set the session-level thinking token budget (minimum 1024)."""
        if budget < 1024:
            raise ValueError("Thinking budget must be at least 1024 tokens")
        self.thinking_budget_tokens = budget
        logger.info(
            "thinking_budget_changed",
            extra={
                "conversation_id": self.conversation_id,
                "thinking_budget_tokens": budget,
            },
        )

    def effective_thinking_budget(self) -> int | None:
        """Return the thinking budget if enabled and supported by the active model."""
        if not self.thinking_enabled:
            return None
        model_config = self.config.models.get(self.active_model)
        if model_config is None or model_config.thinking_budget_tokens <= 0:
            return None
        return min(self.thinking_budget_tokens, model_config.thinking_budget_tokens)

    def clear(self) -> None:
        """Drop all in-memory message history."""
        self.history.clear()

    async def send(self, client: LLMClient, store: ChatStore, content: str) -> ChatResponse:
        """Send a user message to the LLM, persist both sides, return the response."""
        logger.info(
            "chat_user_message_received",
            extra={
                "conversation_id": self.conversation_id,
                "model": self.active_model,
                "message_chars": len(content),
            },
        )
        self.history.append({"role": "user", "content": content})
        await store.add_message(self.conversation_id, "user", content, self.active_model)

        request_messages = list(self.history)
        try:
            response = await client.chat(
                self.active_model,
                request_messages,
                thinking_budget=self.effective_thinking_budget(),
            )
        except LLMClientError as exc:
            await store.record_llm_request(
                request_id=exc.request_id,
                conversation_id=self.conversation_id,
                model=exc.model,
                provider=self.config.provider.name,
                messages=request_messages,
                response_content=None,
                status_code=exc.status_code,
                success=False,
                latency_ms=exc.latency_ms,
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
                error_type=exc.error_type,
                error_message=str(exc),
                retry_attempt=exc.retry_attempts,
            )
            logger.warning(
                "llm_request_failed_persisted",
                extra={
                    "request_id": exc.request_id,
                    "conversation_id": self.conversation_id,
                    "model": exc.model,
                    "latency_ms": exc.latency_ms,
                    "status_code": exc.status_code,
                    "error_type": exc.error_type,
                    "retry_attempts": exc.retry_attempts,
                },
            )
            raise

        await store.record_llm_request(
            request_id=response.request_id,
            conversation_id=self.conversation_id,
            model=response.model,
            provider=self.config.provider.name,
            messages=request_messages,
            response_content=response.content,
            status_code=response.status_code,
            success=True,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )
        logger.info(
            "llm_request_completed_persisted",
            extra={
                "request_id": response.request_id,
                "conversation_id": self.conversation_id,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "status_code": response.status_code,
                "retry_attempts": response.retry_attempts,
            },
        )
        self.history.append({"role": "assistant", "content": response.content})
        await store.add_message(self.conversation_id, "assistant", response.content, response.model)
        return response
