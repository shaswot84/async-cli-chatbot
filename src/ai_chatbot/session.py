from __future__ import annotations

from dataclasses import dataclass, field

from ai_chatbot.config import AppConfig
from ai_chatbot.llm_client import ChatResponse, LLMClient, Message


@dataclass
class ChatSession:
    config: AppConfig
    active_model: str = field(init=False)
    history: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_model = self.config.default_model

    def set_model(self, model_id: str) -> None:
        self.config.validate_model(model_id)
        self.active_model = model_id

    def clear(self) -> None:
        self.history.clear()

    async def send(self, client: LLMClient, content: str) -> ChatResponse:
        self.history.append({"role": "user", "content": content})
        response = await client.chat(self.active_model, self.history)
        self.history.append({"role": "assistant", "content": response.content})
        return response
