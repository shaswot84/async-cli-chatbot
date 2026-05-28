from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from ai_chatbot.llm_client import Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    default_model TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS llm_requests (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    prompt_chars INTEGER NOT NULL,
    response_chars INTEGER,
    prompt_hash TEXT NOT NULL,
    status_code INTEGER,
    success INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    error_type TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS api_failures (
    id TEXT PRIMARY KEY,
    request_id TEXT,
    failure_kind TEXT NOT NULL,
    status_code INTEGER,
    message TEXT NOT NULL,
    retry_attempt INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (request_id) REFERENCES llm_requests(id)
);
"""


@dataclass(frozen=True)
class StoredRequest:
    id: str
    conversation_id: str
    model: str
    provider: str
    prompt_chars: int
    response_chars: int | None
    prompt_hash: str
    status_code: int | None
    success: bool
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error_type: str | None
    error_message: str | None
    created_at: str


class ChatStore:
    def __init__(self, sqlite_path: Path) -> None:
        self.sqlite_path = sqlite_path

    async def initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()

    async def start_conversation(self, default_model: str) -> str:
        conversation_id = f"conv_{uuid.uuid4().hex}"
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.execute(
                """
                INSERT INTO conversations (id, started_at, default_model)
                VALUES (?, ?, ?)
                """,
                (conversation_id, utc_now(), default_model),
            )
            await db.commit()
        return conversation_id

    async def end_conversation(self, conversation_id: str) -> None:
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.execute(
                "UPDATE conversations SET ended_at = ? WHERE id = ?",
                (utc_now(), conversation_id),
            )
            await db.commit()

    async def add_message(
        self, conversation_id: str, role: str, content: str, model: str | None = None
    ) -> str:
        message_id = f"msg_{uuid.uuid4().hex}"
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.execute(
                """
                INSERT INTO messages (id, conversation_id, role, content, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, conversation_id, role, content, model, utc_now()),
            )
            await db.commit()
        return message_id

    async def record_llm_request(
        self,
        *,
        request_id: str,
        conversation_id: str,
        model: str,
        provider: str,
        messages: list[Message],
        response_content: str | None,
        status_code: int | None,
        success: bool,
        latency_ms: float,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        prompt_text = serialize_messages(messages)
        response_chars = len(response_content) if response_content is not None else None
        async with aiosqlite.connect(self.sqlite_path) as db:
            await db.execute(
                """
                INSERT INTO llm_requests (
                    id, conversation_id, model, provider, prompt_chars, response_chars,
                    prompt_hash, status_code, success, latency_ms, input_tokens, output_tokens,
                    total_tokens, error_type, error_message, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    conversation_id,
                    model,
                    provider,
                    len(prompt_text),
                    response_chars,
                    prompt_hash(prompt_text),
                    status_code,
                    int(success),
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    error_type,
                    error_message,
                    utc_now(),
                ),
            )
            if not success and error_type and error_message:
                await db.execute(
                    """
                    INSERT INTO api_failures (
                        id, request_id, failure_kind, status_code, message,
                        retry_attempt, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        f"fail_{uuid.uuid4().hex}",
                        request_id,
                        error_type,
                        status_code,
                        error_message,
                        utc_now(),
                    ),
                )
            await db.commit()

    async def list_messages(self, conversation_id: str) -> list[dict[str, str | None]]:
        async with aiosqlite.connect(self.sqlite_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT role, content, model, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            )
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_request(self, request_id: str) -> StoredRequest | None:
        async with aiosqlite.connect(self.sqlite_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT *
                FROM llm_requests
                WHERE id = ?
                """,
                (request_id,),
            )
            row = await cursor.fetchone()

        if row is None:
            return None
        return StoredRequest(
            id=row["id"],
            conversation_id=row["conversation_id"],
            model=row["model"],
            provider=row["provider"],
            prompt_chars=row["prompt_chars"],
            response_chars=row["response_chars"],
            prompt_hash=row["prompt_hash"],
            status_code=row["status_code"],
            success=bool(row["success"]),
            latency_ms=row["latency_ms"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            created_at=row["created_at"],
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def serialize_messages(messages: list[Message]) -> str:
    return json.dumps(messages, sort_keys=True, separators=(",", ":"))


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def request_to_rows(request: StoredRequest) -> list[tuple[str, Any]]:
    return [
        ("request_id", request.id),
        ("conversation_id", request.conversation_id),
        ("model", request.model),
        ("provider", request.provider),
        ("success", request.success),
        ("status_code", request.status_code),
        ("latency_ms", request.latency_ms),
        ("prompt_chars", request.prompt_chars),
        ("response_chars", request.response_chars),
        ("input_tokens", request.input_tokens),
        ("output_tokens", request.output_tokens),
        ("total_tokens", request.total_tokens),
        ("prompt_hash", request.prompt_hash),
        ("error_type", request.error_type),
        ("error_message", request.error_message),
        ("created_at", request.created_at),
    ]
