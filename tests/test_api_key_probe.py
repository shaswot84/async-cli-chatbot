from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "api_key_probe.py"
SPEC = importlib.util.spec_from_file_location("api_key_probe", MODULE_PATH)
assert SPEC is not None
api_key_probe = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["api_key_probe"] = api_key_probe
SPEC.loader.exec_module(api_key_probe)


def make_config() -> Any:
    return api_key_probe.RuntimeConfig(
        provider=api_key_probe.ProviderConfig(
            name="test-provider",
            base_url="https://example.test/v1",
            api_key="secret-key",
            chat_completions_path="/chat/completions",
        ),
        settings=api_key_probe.PreflightSettings(
            default_model="llama-3.1-8b-instant",
            timeout_seconds=60,
            max_concurrency=2,
            requests_per_minute=20,
            max_tokens=64,
            temperature=0.0,
            safe_requests_per_minute=6,
            burst_concurrency=2,
            rate_test_requests=8,
            report_path=Path("reports/test.md"),
        ),
        models=["llama-3.1-8b-instant", "backup-model"],
    )


def test_validate_config_accepts_safe_config() -> None:
    assert api_key_probe.validate_config(make_config()) == []


def test_validate_config_rejects_unknown_default_model() -> None:
    config = make_config()
    bad_config = api_key_probe.RuntimeConfig(
        provider=config.provider,
        settings=api_key_probe.PreflightSettings(
            default_model="missing-model",
            timeout_seconds=config.settings.timeout_seconds,
            max_concurrency=config.settings.max_concurrency,
            requests_per_minute=config.settings.requests_per_minute,
            max_tokens=config.settings.max_tokens,
            temperature=config.settings.temperature,
            safe_requests_per_minute=config.settings.safe_requests_per_minute,
            burst_concurrency=config.settings.burst_concurrency,
            rate_test_requests=config.settings.rate_test_requests,
            report_path=config.settings.report_path,
        ),
        models=config.models,
    )

    assert "DEFAULT_MODEL must exist in config/models.toml" in api_key_probe.validate_config(
        bad_config
    )


def test_parse_openai_response_extracts_content_and_usage() -> None:
    parsed = api_key_probe.parse_openai_response(
        {
            "id": "chatcmpl_123",
            "model": "llama-3.1-8b-instant",
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        }
    )

    assert parsed["content"] == "ok"
    assert parsed["openai_compatible"] is True
    assert parsed["usage_complete"] is True


def test_classify_failure_retryability() -> None:
    assert api_key_probe.classify_failure(429) == "retryable"
    assert api_key_probe.classify_failure(503) == "retryable"
    assert api_key_probe.classify_failure(401) == "non_retryable"
    assert api_key_probe.classify_failure(None, "timeout") == "retryable"


def test_sanitize_error_message_redacts_secrets() -> None:
    config = make_config()
    message = "Authorization: Bearer secret-key failed for https://example.test/v1"

    sanitized = api_key_probe.sanitize_error_message(message, config)

    assert "secret-key" not in sanitized
    assert "Bearer secret-key" not in sanitized
    assert "https://example.test/v1" not in sanitized


def test_markdown_report_contains_decision_without_secret() -> None:
    config = make_config()
    report = api_key_probe.build_report(
        config,
        {
            "basic": {
                "config_errors": [],
                "smoke": {
                    "model_id": "llama-3.1-8b-instant",
                    "success": True,
                    "openai_compatible": True,
                },
            },
            "models": {
                "results": [
                    {
                        "model_id": "llama-3.1-8b-instant",
                        "success": True,
                        "status_code": 200,
                        "latency_ms": 100,
                        "response_chars": 2,
                        "usage_complete": True,
                        "openai_compatible": True,
                    },
                    {
                        "model_id": "backup-model",
                        "success": True,
                        "status_code": 200,
                        "latency_ms": 90,
                        "response_chars": 2,
                        "usage_complete": True,
                        "openai_compatible": True,
                    },
                ]
            },
            "rate-limit": {
                "concurrency": [
                    {
                        "concurrency": 1,
                        "requests_failed": 0,
                        "429_count": 0,
                    }
                ],
                "rpm": {
                    "429_count": 0,
                    "requests_failed": 0,
                },
            },
        },
    )

    markdown = api_key_probe.render_markdown(report)

    assert "GO:" in markdown
    assert "secret-key" not in markdown
    assert "https://example.test/v1" not in markdown
