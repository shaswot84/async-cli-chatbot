from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ai_chatbot.config import AppConfig
from ai_chatbot.db import ChatStore
from ai_chatbot.llm_client import ChatResponse, LLMClient, LLMClientError, Message

logger = logging.getLogger(__name__)


@dataclass
class ChatSession:
    config: AppConfig
    conversation_id: str
    active_model: str = field(init=False)
    history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_model = self.config.default_model

    def set_model(self, model_id: str) -> None:
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

    def clear(self) -> None:
        self.history.clear()

    async def send(self, client: LLMClient, store: ChatStore, content: str) -> ChatResponse:
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
            response = await client.chat(self.active_model, request_messages)
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
