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
    name: str
    base_url: str
    api_key: str
    chat_completions_path: str


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    display_name: str
    family: str
    use_case: str


@dataclass(frozen=True)
class AppConfig:
    provider: ProviderConfig
    default_model: str
    timeout_seconds: float
    max_concurrency: int
    requests_per_minute: int
    max_output_tokens: int
    sqlite_path: Path
    models: dict[str, ModelConfig]

    def chat_url(self) -> str:
        base = self.provider.base_url.rstrip("/") + "/"
        path = self.provider.chat_completions_path.lstrip("/")
        return urljoin(base, path)

    def validate_model(self, model_id: str) -> None:
        if model_id not in self.models:
            available = ", ".join(self.models)
            raise ValueError(f"Unknown model `{model_id}`. Available models: {available}")


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


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
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def load_config(models_path: Path = DEFAULT_MODELS_PATH) -> AppConfig:
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
        models=models,
    )

    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    parsed = urlparse(config.provider.base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("LLM_BASE_URL must be an absolute HTTP(S) URL")
    if not config.provider.chat_completions_path.startswith("/"):
        raise ValueError("LLM_CHAT_COMPLETIONS_PATH must start with /")
    if not config.models:
        raise ValueError("config/models.toml must define at least one model")
    config.validate_model(config.default_model)


def sanitized_config(config: AppConfig) -> dict[str, str | int | float | bool]:
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
    }


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    path = Path(raw.strip()) if raw and raw.strip() else default
    return path if path.is_absolute() else PROJECT_ROOT / path
