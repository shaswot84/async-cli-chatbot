from __future__ import annotations

import json
import logging

from ai_chatbot.logging_setup import JsonFormatter, sanitize_log_value


def test_json_formatter_emits_structured_event_without_secret() -> None:
    record = logging.LogRecord(
        name="ai_chatbot.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="llm_request_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req_test"
    record.model = "model-a"
    record.latency_ms = 12.5
    record.total_tokens = 42
    record.api_key = "secret"

    payload = json.loads(JsonFormatter().format(record))

    assert payload["event"] == "llm_request_completed"
    assert payload["request_id"] == "req_test"
    assert payload["model"] == "model-a"
    assert payload["latency_ms"] == 12.5
    assert payload["total_tokens"] == 42
    assert payload["api_key"] == "[redacted]"
    assert "secret" not in json.dumps(payload)


def test_sanitize_nested_sensitive_values() -> None:
    sanitized = sanitize_log_value(
        "payload",
        {
            "Authorization": "Bearer secret",
            "access_token": "secret",
            "safe": "ok",
            "nested": [{"api_key": "secret"}],
        },
    )

    assert sanitized == {
        "Authorization": "[redacted]",
        "access_token": "[redacted]",
        "safe": "ok",
        "nested": [{"api_key": "[redacted]"}],
    }
