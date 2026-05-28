from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

RESERVED_LOG_RECORD_KEYS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "message",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

SENSITIVE_KEY_FRAGMENTS = ("api_key", "authorization", "secret", "headers")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        for key, value in record.__dict__.items():
            if key in RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            payload[key] = sanitize_log_value(key, value)
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = []
        for key, value in record.__dict__.items():
            if key in RESERVED_LOG_RECORD_KEYS or key.startswith("_"):
                continue
            fields.append(f"{key}={sanitize_log_value(key, value)}")
        suffix = f" {' '.join(fields)}" if fields else ""
        return f"{record.levelname.lower()} {record.getMessage()}{suffix}"


def setup_logging(log_level: str, json_logs: bool) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(coerce_log_level(log_level))


def coerce_log_level(log_level: str) -> int:
    return getattr(logging, log_level.upper(), logging.INFO)


def sanitize_log_value(key: str, value: Any) -> Any:
    lowered = key.lower()
    if any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS):
        return "[redacted]"
    if "token" in lowered and not isinstance(value, (int, float)) and value is not None:
        return "[redacted]"
    if isinstance(value, dict):
        return {
            nested_key: sanitize_log_value(nested_key, nested_value)
            for nested_key, nested_value in value.items()
        }
    if isinstance(value, list):
        return [sanitize_log_value(key, item) for item in value]
    return value
