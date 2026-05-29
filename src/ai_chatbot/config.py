"""Application configuration loaded from TOML files and environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_PATH = PROJECT_ROOT / "config" / "models.toml"


@dataclass(frozen=True)
class ProviderConfig:
    """Connection details for an OpenAI-compatible LLM provider."""

    name: str
    base_url: str
    api_key: str
    chat_completions_path: str


@dataclass(frozen=True)
class ModelConfig:
    """Metadata for a single model listed in models.toml."""

    model_id: str
    display_name: str
    family: str
    use_case: str
    thinking_budget_tokens: int = 0
    streaming_capable: bool = True


@dataclass(frozen=True)
class AppConfig:
    """All runtime configuration for the chatbot application."""

    provider: ProviderConfig
    default_model: str
    timeout_seconds: float
    max_concurrency: int
    requests_per_minute: int
    max_output_tokens: int
    sqlite_path: Path
    log_level: str
    log_json: bool
    simulate_failures: bool
    simulate_failure_rate: float
    simulate_failure_kind: str
    thinking_enabled: bool
    thinking_budget_tokens: int
    streaming_enabled: bool
    theme: str
    models: dict[str, ModelConfig]

    def chat_url(self) -> str:
        """Build the full chat completions endpoint URL from base URL and path."""
        base = self.provider.base_url.rstrip("/") + "/"
        path = self.provider.chat_completions_path.lstrip("/")
        return urljoin(base, path)

    def validate_model(self, model_id: str) -> None:
        """Raise ValueError if the given model_id is not in the configured models."""
        if model_id not in self.models:
            available = ", ".join(self.models)
            raise ValueError(f"Unknown model `{model_id}`. Available models: {available}")

    def resolve_model(self, partial: str) -> str:
        """Resolve a partial model ID to its canonical form.

        Exact matches are returned immediately. Otherwise, models whose ID
        contains the input as a substring are collected. If exactly one
        candidate is found it is returned. Zero or multiple candidates raise
        ValueError with an appropriate message.
        """
        if partial in self.models:
            return partial

        candidates = [mid for mid in self.models if partial in mid]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                f"`{partial}` matches multiple models: {', '.join(candidates)}. "
                f"Please be more specific."
            )
        available = ", ".join(self.models)
        raise ValueError(f"Unknown model `{partial}`. Available models: {available}")


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load key=value pairs from a .env file, skipping already-set variables."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_str(name: str, default: str | None = None) -> str:
    """Read a required string environment variable, raising on missing or empty."""
    value = os.environ.get(name, default)
    if value is None or not value.strip():
        raise ValueError(f"Missing required environment variable: {name}")
    return value.strip()


def env_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def env_float(name: str, default: float) -> float:
    """Read a positive float environment variable."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_config(models_path: Path = DEFAULT_MODELS_PATH) -> AppConfig:
    """Load and validate the full application configuration from env and TOML."""
    load_dotenv()

    with models_path.open("rb") as models_file:
        data = tomllib.load(models_file)

    provider_data = data.get("provider", {})
    defaults_data = data.get("defaults", {})
    models_data = data.get("models", {})

    provider = ProviderConfig(
        name=str(provider_data.get("name", "multi-llm-router")),
        base_url=env_str(str(provider_data.get("base_url_env", "LLM_BASE_URL"))),
        api_key=env_str(str(provider_data.get("api_key_env", "LLM_API_KEY"))),
        chat_completions_path=env_str(
            "LLM_CHAT_COMPLETIONS_PATH",
            str(provider_data.get("chat_completions_path", "/chat/completions")),
        ),
    )

    models = {
        model_id: ModelConfig(
            model_id=model_id,
            display_name=str(model_data.get("display_name", model_id)),
            family=str(model_data.get("family", "unknown")),
            use_case=str(model_data.get("use_case", "")),
            thinking_budget_tokens=int(model_data.get("thinking_budget_tokens", 0)),
            streaming_capable=bool(model_data.get("streaming_capable", True)),
        )
        for model_id, model_data in models_data.items()
    }

    config = AppConfig(
        provider=provider,
        default_model=env_str("DEFAULT_MODEL", str(defaults_data.get("model", ""))),
        timeout_seconds=env_float(
            "REQUEST_TIMEOUT_SECONDS", float(defaults_data.get("timeout_seconds", 60))
        ),
        max_concurrency=env_int("MAX_CONCURRENCY", int(defaults_data.get("max_concurrency", 2))),
        requests_per_minute=env_int(
            "REQUESTS_PER_MINUTE", int(defaults_data.get("requests_per_minute", 6))
        ),
        max_output_tokens=env_int("MAX_OUTPUT_TOKENS", 400),
        sqlite_path=env_path("SQLITE_PATH", PROJECT_ROOT / "data" / "chatbot.sqlite3"),
        log_level=env_str("LOG_LEVEL", "INFO").upper(),
        log_json=env_bool("LOG_JSON", True),
        simulate_failures=env_bool("SIMULATE_FAILURES", False),
        simulate_failure_rate=env_ratio("SIMULATE_FAILURE_RATE", 0.0),
        simulate_failure_kind=env_str("SIMULATE_FAILURE_KIND", "429"),
        thinking_enabled=env_bool("THINKING_ENABLED", False),
        thinking_budget_tokens=env_int("THINKING_BUDGET_TOKENS", 16000),
        streaming_enabled=env_bool("STREAMING_ENABLED", True),
        theme=env_str("THEME", "dark"),
        models=models,
    )

    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    """Validate essential config fields; raises ValueError on the first problem."""
    parsed = urlparse(config.provider.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM_BASE_URL must be an absolute HTTP(S) URL")
    if not config.provider.chat_completions_path.startswith("/"):
        raise ValueError("LLM_CHAT_COMPLETIONS_PATH must start with /")
    if not config.models:
        raise ValueError("config/models.toml must define at least one model")
    config.validate_model(config.default_model)
    for model_id, model_config in config.models.items():
        tbt = model_config.thinking_budget_tokens
        if tbt != 0 and tbt < 1024:
            raise ValueError(
                f"Model `{model_id}` thinking_budget_tokens must be 0 (disabled) or >= 1024, got {tbt}"
            )


def sanitized_config(config: AppConfig) -> dict[str, str | int | float | bool]:
    """Return a copy of the config safe for display (no API key, no full URL)."""
    parsed = urlparse(config.provider.base_url)
    return {
        "provider": config.provider.name,
        "base_url": f"{parsed.scheme}://{parsed.netloc}",
        "chat_path": config.provider.chat_completions_path,
        "api_key_present": bool(config.provider.api_key),
        "default_model": config.default_model,
        "timeout_seconds": config.timeout_seconds,
        "max_concurrency": config.max_concurrency,
        "requests_per_minute": config.requests_per_minute,
        "max_output_tokens": config.max_output_tokens,
        "sqlite_path": str(config.sqlite_path),
        "log_level": config.log_level,
        "log_json": config.log_json,
        "simulate_failures": config.simulate_failures,
        "simulate_failure_rate": config.simulate_failure_rate,
        "simulate_failure_kind": config.simulate_failure_kind,
        "thinking_enabled": config.thinking_enabled,
        "thinking_budget_tokens": config.thinking_budget_tokens,
        "streaming_enabled": config.streaming_enabled,
        "theme": config.theme,
    }


def env_path(name: str, default: Path) -> Path:
    """Read an env var as a filesystem path, resolving relative paths against project root."""
    raw = os.environ.get(name)
    path = Path(raw.strip()) if raw and raw.strip() else default
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable (true/false, 1/0, yes/no, on/off)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def env_ratio(name: str, default: float) -> float:
    """Read a float env var clamped to [0.0, 1.0]."""
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    if value > 1:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value
