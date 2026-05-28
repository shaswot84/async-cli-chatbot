from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import tomllib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urljoin, urlparse

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "models.toml"
DEFAULT_RAW_REPORT_PATH = PROJECT_ROOT / "reports" / "api_key_preflight_raw.json"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422}
STOP_STATUS_CODES = {401, 403}


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    chat_completions_path: str


@dataclass(frozen=True)
class PreflightSettings:
    default_model: str
    timeout_seconds: float
    max_concurrency: int
    requests_per_minute: int
    max_tokens: int
    temperature: float
    safe_requests_per_minute: int
    burst_concurrency: int
    rate_test_requests: int
    report_path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    provider: ProviderConfig
    settings: PreflightSettings
    models: list[str]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def env_str(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value


def load_runtime_config(config_path: Path = DEFAULT_CONFIG_PATH) -> RuntimeConfig:
    load_dotenv(PROJECT_ROOT / ".env")

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    provider_section = config.get("provider", {})
    defaults_section = config.get("defaults", {})
    models_section = config.get("models", {})
    model_ids = [model_id for model_id in models_section if isinstance(model_id, str) and model_id]

    provider_base_env = provider_section.get("base_url_env", "LLM_BASE_URL")
    provider_key_env = provider_section.get("api_key_env", "LLM_API_KEY")
    path_default = provider_section.get("chat_completions_path", "/chat/completions")

    default_model = env_str("DEFAULT_MODEL", str(defaults_section.get("model", "")))
    timeout_seconds = env_float(
        "REQUEST_TIMEOUT_SECONDS", float(defaults_section.get("timeout_seconds", 60))
    )
    max_concurrency = env_int("MAX_CONCURRENCY", int(defaults_section.get("max_concurrency", 2)))
    rpm = env_int("REQUESTS_PER_MINUTE", int(defaults_section.get("requests_per_minute", 20)))

    report_path = Path(
        os.environ.get("PREFLIGHT_REPORT_PATH", "reports/api_key_preflight_report.md")
    )
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path

    return RuntimeConfig(
        provider=ProviderConfig(
            name=str(provider_section.get("name", "multi-llm-router")),
            base_url=env_str(provider_base_env),
            api_key=env_str(provider_key_env),
            chat_completions_path=env_str("LLM_CHAT_COMPLETIONS_PATH", path_default),
        ),
        settings=PreflightSettings(
            default_model=default_model,
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            requests_per_minute=rpm,
            max_tokens=env_int("PREFLIGHT_MAX_TOKENS", 64),
            temperature=env_float("PREFLIGHT_TEMPERATURE", 0.0),
            safe_requests_per_minute=env_int("PREFLIGHT_SAFE_REQUESTS_PER_MINUTE", 6),
            burst_concurrency=env_int("PREFLIGHT_BURST_CONCURRENCY", 2),
            rate_test_requests=env_int("PREFLIGHT_RATE_TEST_REQUESTS", 8),
            report_path=report_path,
        ),
        models=model_ids,
    )


def sanitized_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "invalid-url"
    return f"{parsed.scheme}://{parsed.netloc}"


def validate_config(config: RuntimeConfig) -> list[str]:
    errors: list[str] = []
    parsed = urlparse(config.provider.base_url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        errors.append("LLM_BASE_URL must be an absolute HTTP(S) URL")
    if config.provider.chat_completions_path.startswith("http"):
        errors.append("LLM_CHAT_COMPLETIONS_PATH must be a path, not a full URL")
    if not config.provider.chat_completions_path.startswith("/"):
        errors.append("LLM_CHAT_COMPLETIONS_PATH must start with /")
    if config.settings.default_model not in config.models:
        errors.append("DEFAULT_MODEL must exist in config/models.toml")
    if not config.models:
        errors.append("config/models.toml must define at least one model")

    return errors


def chat_url(config: RuntimeConfig) -> str:
    base = config.provider.base_url.rstrip("/") + "/"
    path = config.provider.chat_completions_path.lstrip("/")
    return urljoin(base, path)


def classify_failure(status_code: int | None, error_type: str | None = None) -> str:
    if error_type in {"timeout", "network"}:
        return "retryable"
    if status_code in RETRYABLE_STATUS_CODES:
        return "retryable"
    if status_code in NON_RETRYABLE_STATUS_CODES:
        return "non_retryable"
    if status_code is None:
        return "unknown"
    if 500 <= status_code:
        return "retryable"
    if 400 <= status_code:
        return "non_retryable"
    return "unknown"


def sanitize_error_message(message: str, config: RuntimeConfig | None = None) -> str:
    sanitized = message
    if config is not None:
        secret_values = [
            config.provider.api_key,
            f"Bearer {config.provider.api_key}",
            config.provider.base_url,
        ]
        for secret in secret_values:
            if secret:
                sanitized = sanitized.replace(secret, "[redacted]")
    return sanitized[:300]


def parse_openai_response(payload: dict[str, Any]) -> dict[str, Any]:
    content = ""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""

    usage = payload.get("usage") or {}
    return {
        "provider_id": payload.get("id"),
        "provider_model": payload.get("model"),
        "content": content if isinstance(content, str) else "",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "openai_compatible": bool(content),
        "usage_complete": all(
            isinstance(usage.get(key), int)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        ),
    }


def build_payload(model: str, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


async def tiny_chat_request(
    client: httpx.AsyncClient,
    config: RuntimeConfig,
    model: str,
    prompt: str = "Reply with exactly: ok",
    max_tokens: int | None = None,
) -> dict[str, Any]:
    request_id = f"req_{uuid.uuid4().hex}"
    started = time.perf_counter()
    status_code: int | None = None

    try:
        response = await client.post(
            chat_url(config),
            json=build_payload(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens or config.settings.max_tokens,
                temperature=config.settings.temperature,
            ),
            headers={
                "Authorization": f"Bearer {config.provider.api_key}",
                "Content-Type": "application/json",
            },
        )
        latency_ms = (time.perf_counter() - started) * 1000
        status_code = response.status_code

        if response.is_success:
            parsed = parse_openai_response(response.json())
            return {
                "request_id": request_id,
                "model_id": model,
                "success": bool(parsed["content"]),
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "response_chars": len(parsed["content"]),
                "error_type": None if parsed["content"] else "schema",
                "sanitized_error_message": None
                if parsed["content"]
                else "Response did not include choices[0].message.content",
                **parsed,
            }

        return {
            "request_id": request_id,
            "model_id": model,
            "success": False,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "response_chars": 0,
            "error_type": "http",
            "retryability": classify_failure(status_code),
            "sanitized_error_message": sanitize_error_message(response.text, config),
        }
    except httpx.TimeoutException as exc:
        return failed_result(request_id, model, started, status_code, "timeout", str(exc), config)
    except httpx.HTTPError as exc:
        return failed_result(request_id, model, started, status_code, "network", str(exc), config)
    except (json.JSONDecodeError, ValueError) as exc:
        return failed_result(request_id, model, started, status_code, "schema", str(exc), config)


def failed_result(
    request_id: str,
    model: str,
    started: float,
    status_code: int | None,
    error_type: str,
    message: str,
    config: RuntimeConfig,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model_id": model,
        "success": False,
        "status_code": status_code,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "response_chars": 0,
        "error_type": error_type,
        "retryability": classify_failure(status_code, error_type),
        "sanitized_error_message": sanitize_error_message(message, config),
    }


async def run_basic(config: RuntimeConfig) -> dict[str, Any]:
    errors = validate_config(config)
    result: dict[str, Any] = {
        "name": "basic",
        "started_at": utc_now(),
        "config": sanitized_config(config),
        "config_errors": errors,
        "smoke": None,
    }

    if errors:
        result["success"] = False
        return result

    timeout = httpx.Timeout(config.settings.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        smoke = await tiny_chat_request(
            client,
            config,
            config.settings.default_model,
            max_tokens=8,
        )
        result["smoke"] = smoke
        result["success"] = bool(smoke["success"])
    return result


async def run_models(config: RuntimeConfig) -> dict[str, Any]:
    timeout = httpx.Timeout(config.settings.timeout_seconds)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in config.models:
            result = await tiny_chat_request(client, config, model)
            results.append(result)
            if result.get("status_code") in STOP_STATUS_CODES:
                break
            if len([item for item in results if item.get("status_code") == 429]) >= 2:
                break

    return {
        "name": "models",
        "started_at": utc_now(),
        "results": results,
        "success": any(result["success"] for result in results),
    }


async def run_latency(config: RuntimeConfig, working_models: list[str]) -> dict[str, Any]:
    timeout = httpx.Timeout(config.settings.timeout_seconds)
    model_stats: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for model in working_models:
            samples = [await tiny_chat_request(client, config, model) for _ in range(3)]
            latencies = [sample["latency_ms"] for sample in samples if sample["success"]]
            model_stats.append(
                {
                    "model_id": model,
                    "samples": samples,
                    "min_latency_ms": round(min(latencies), 2) if latencies else None,
                    "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
                    "p95_latency_ms": percentile_95(latencies),
                    "failure_count": len([sample for sample in samples if not sample["success"]]),
                }
            )
    return {"name": "latency", "started_at": utc_now(), "results": model_stats}


async def run_rate_limit(config: RuntimeConfig) -> dict[str, Any]:
    timeout = httpx.Timeout(config.settings.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        concurrency_results = []
        for concurrency in range(1, config.settings.burst_concurrency + 1):
            semaphore = asyncio.Semaphore(concurrency)

            async def guarded_request(
                index: int, request_semaphore: asyncio.Semaphore = semaphore
            ) -> dict[str, Any]:
                async with request_semaphore:
                    return await tiny_chat_request(
                        client,
                        config,
                        config.settings.default_model,
                        prompt=f"Reply with exactly: ok {index}",
                        max_tokens=8,
                    )

            started = time.perf_counter()
            requests = await asyncio.gather(
                *(guarded_request(index) for index in range(concurrency))
            )
            latencies = [request["latency_ms"] for request in requests]
            concurrency_results.append(
                {
                    "concurrency": concurrency,
                    "requests_started": concurrency,
                    "requests_succeeded": len([item for item in requests if item["success"]]),
                    "requests_failed": len([item for item in requests if not item["success"]]),
                    "429_count": len([item for item in requests if item.get("status_code") == 429]),
                    "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
                    "max_latency_ms": round(max(latencies), 2) if latencies else None,
                    "wall_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
            )
            if any(request.get("status_code") in STOP_STATUS_CODES for request in requests):
                break

        rpm_results = []
        delay_seconds = 60 / config.settings.safe_requests_per_minute
        for index in range(config.settings.rate_test_requests):
            result = await tiny_chat_request(
                client,
                config,
                config.settings.default_model,
                prompt=f"Reply with exactly: ok rate {index}",
                max_tokens=8,
            )
            rpm_results.append(result)
            if result.get("status_code") in STOP_STATUS_CODES:
                break
            if (
                result.get("status_code") == 429
                and len([item for item in rpm_results if item.get("status_code") == 429]) >= 2
            ):
                break
            if index < config.settings.rate_test_requests - 1:
                await asyncio.sleep(delay_seconds)

    return {
        "name": "rate-limit",
        "started_at": utc_now(),
        "concurrency": concurrency_results,
        "rpm": {
            "safe_requests_per_minute": config.settings.safe_requests_per_minute,
            "requests_started": len(rpm_results),
            "requests_succeeded": len([item for item in rpm_results if item["success"]]),
            "requests_failed": len([item for item in rpm_results if not item["success"]]),
            "429_count": len([item for item in rpm_results if item.get("status_code") == 429]),
            "results": rpm_results,
        },
    }


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) < 2:
        return round(values[0], 2)
    return round(statistics.quantiles(values, n=20, method="inclusive")[18], 2)


def sanitized_config(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "provider_name": config.provider.name,
        "provider_base_url": sanitized_url(config.provider.base_url),
        "chat_completions_path": config.provider.chat_completions_path,
        "api_key_present": bool(config.provider.api_key),
        "default_model": config.settings.default_model,
        "timeout_seconds": config.settings.timeout_seconds,
        "max_concurrency": config.settings.max_concurrency,
        "requests_per_minute": config.settings.requests_per_minute,
        "preflight_max_tokens": config.settings.max_tokens,
        "preflight_safe_requests_per_minute": config.settings.safe_requests_per_minute,
        "preflight_burst_concurrency": config.settings.burst_concurrency,
        "preflight_rate_test_requests": config.settings.rate_test_requests,
        "models": config.models,
    }


def recommended_default_model(
    config: RuntimeConfig, model_results: list[dict[str, Any]]
) -> str | None:
    successful = [result for result in model_results if result["success"]]
    if any(result["model_id"] == config.settings.default_model for result in successful):
        return config.settings.default_model
    if not successful:
        return None
    return cast(str, min(successful, key=lambda result: result["latency_ms"])["model_id"])


def recommended_concurrency(rate_result: dict[str, Any] | None) -> int | None:
    if not rate_result:
        return None
    passing = [
        item["concurrency"]
        for item in rate_result.get("concurrency", [])
        if item["requests_failed"] == 0 and item["429_count"] == 0
    ]
    return max(passing) if passing else 1


def recommended_rpm(config: RuntimeConfig, rate_result: dict[str, Any] | None) -> int | None:
    if not rate_result:
        return None
    rpm = rate_result.get("rpm", {})
    if rpm.get("429_count", 0) == 0 and rpm.get("requests_failed", 0) == 0:
        return min(config.settings.safe_requests_per_minute, config.settings.requests_per_minute)
    return max(1, config.settings.safe_requests_per_minute // 2)


def decision(config: RuntimeConfig, report: dict[str, Any]) -> tuple[str, list[str], str]:
    model_results = report.get("models", {}).get("results", [])
    default_result = next(
        (item for item in model_results if item.get("model_id") == config.settings.default_model),
        report.get("basic", {}).get("smoke"),
    )
    passed_models = [item for item in model_results if item.get("success")]
    backup_models = [
        item for item in passed_models if item.get("model_id") != config.settings.default_model
    ]
    rate_result = report.get("rate-limit")

    blockers: list[str] = []
    warnings: list[str] = []

    if report.get("basic", {}).get("config_errors"):
        blockers.append("configuration is incomplete or unsafe")
    if not default_result or not default_result.get("success"):
        blockers.append("DEFAULT_MODEL did not pass a smoke test")
    if not backup_models and model_results:
        blockers.append("no backup model passed")
    if passed_models and not any(item.get("openai_compatible") for item in passed_models):
        blockers.append("response content could not be parsed reliably")
    if rate_result and recommended_concurrency(rate_result) is None:
        blockers.append("safe concurrency could not be determined")
    if rate_result and recommended_rpm(config, rate_result) is None:
        blockers.append("safe request rate could not be determined")

    if passed_models and any(not item.get("usage_complete") for item in passed_models):
        warnings.append("some successful responses did not include complete token usage")
    if model_results and len(passed_models) < len(config.models):
        warnings.append("some configured models failed")

    if blockers:
        return "NO-GO", blockers, "Fix provider/model access before Phase 2."
    if warnings:
        return "GO WITH WARNINGS", warnings, "Proceed to Phase 2 with passing models only."
    return "GO", ["all Phase 1 gates passed"], "Proceed to Phase 2."


def build_report(config: RuntimeConfig, results: dict[str, Any]) -> dict[str, Any]:
    model_results = results.get("models", {}).get("results", [])
    rate_result = results.get("rate-limit")
    status, reasons, next_action = decision(config, results)
    return {
        "generated_at": utc_now(),
        "sanitized_config": sanitized_config(config),
        **results,
        "recommendations": {
            "default_model": recommended_default_model(config, model_results),
            "max_concurrency": recommended_concurrency(rate_result),
            "requests_per_minute": recommended_rpm(config, rate_result),
            "request_timeout_seconds": config.settings.timeout_seconds,
        },
        "decision": {
            "status": status,
            "reasons": reasons,
            "next_action": next_action,
        },
    }


def write_reports(report: dict[str, Any], markdown_path: Path) -> None:
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RAW_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RAW_REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    config = report["sanitized_config"]
    model_results = report.get("models", {}).get("results", [])
    passed = [item["model_id"] for item in model_results if item.get("success")]
    failed = [item["model_id"] for item in model_results if not item.get("success")]
    recs = report["recommendations"]
    decision_section = report["decision"]

    lines = [
        "# API Key Preflight Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Provider",
        "",
        f"- Provider: {config['provider_name']}",
        f"- Base URL: {config['provider_base_url']}",
        f"- Chat path: {config['chat_completions_path']}",
        f"- API key present: {config['api_key_present']}",
        "",
        "## Models",
        "",
        f"- Models tested: {', '.join(config['models'])}",
        f"- Models passed: {', '.join(passed) if passed else 'none'}",
        f"- Models failed: {', '.join(failed) if failed else 'none'}",
        "",
        "## Model Results",
        "",
        "| Model | Result | HTTP | Latency ms | Response chars | Token usage | Error |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]

    for item in model_results:
        lines.append(
            "| {model} | {result} | {status} | {latency} | {chars} | {usage} | {error} |".format(
                model=item["model_id"],
                result="PASS" if item.get("success") else "FAIL",
                status=item.get("status_code") or "",
                latency=item.get("latency_ms") or "",
                chars=item.get("response_chars") or 0,
                usage="complete" if item.get("usage_complete") else "missing/partial",
                error=item.get("error_type") or "",
            )
        )

    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            f"- DEFAULT_MODEL: {recs['default_model'] or 'unknown'}",
            f"- MAX_CONCURRENCY: {recs['max_concurrency'] or 'unknown'}",
            f"- REQUESTS_PER_MINUTE: {recs['requests_per_minute'] or 'unknown'}",
            f"- REQUEST_TIMEOUT_SECONDS: {recs['request_timeout_seconds']}",
            "",
            "## Decision",
            "",
            f"{decision_section['status']}:",
        ]
    )

    lines.extend(f"- {reason}" for reason in decision_section["reasons"])
    lines.extend(["", decision_section["next_action"], ""])
    return "\n".join(lines)


async def run_suite(suite: str, config: RuntimeConfig) -> dict[str, Any]:
    results: dict[str, Any] = {}

    if suite in {"all", "basic"}:
        results["basic"] = await run_basic(config)
        smoke = results["basic"].get("smoke")
        if smoke and smoke.get("status_code") in STOP_STATUS_CODES:
            return results

    if suite in {"all", "models"}:
        results["models"] = await run_models(config)

    if suite == "all":
        working_models = [
            item["model_id"]
            for item in results.get("models", {}).get("results", [])
            if item.get("success")
        ]
        if working_models:
            results["latency"] = await run_latency(config, working_models)

    if suite in {"all", "rate-limit"}:
        results["rate-limit"] = await run_rate_limit(config)

    return results


def print_existing_report() -> int:
    report_path = PROJECT_ROOT / "reports" / "api_key_preflight_report.md"
    if not report_path.exists():
        print("No preflight report found. Run `make preflight` first.", file=sys.stderr)
        return 1
    print(report_path.read_text(encoding="utf-8"))
    return 0


async def async_main(args: argparse.Namespace) -> int:
    if args.suite == "report":
        return print_existing_report()

    try:
        config = load_runtime_config(Path(args.config))
    except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    results = await run_suite(args.suite, config)
    report = build_report(config, results)
    write_reports(report, config.settings.report_path)

    print(render_markdown(report))
    return 0 if report["decision"]["status"] in {"GO", "GO WITH WARNINGS"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 API-key feasibility probe")
    parser.add_argument(
        "--suite",
        choices=["all", "basic", "models", "rate-limit", "report"],
        default="all",
        help="Probe suite to run",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config/models.toml",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
