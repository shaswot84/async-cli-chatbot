# async-cli-ai-chatbot

> Asynchronous multi-model CLI chatbot with streaming, themes, image generation,
> extended reasoning, and local SQLite observability.

## Quick start

```bash
cp .env.example .env                    # then edit LLM_API_KEY and LLM_BASE_URL
uv sync                                 # install dependencies
make preflight                          # verify API key and model access
make run                                # start chatting
```

---

## Features

| Category | Capability |
|---|---|
| **Chat** | Multi-model REPL, partial-ID model matching, `/model set qwen` resolves to `qwen/qwen3-32b` |
| **Streaming** | SSE-based token-by-token output via Rich `Live` panels, toggleable per session |
| **Thinking** | Extended reasoning with configurable token budgets, per-model in `models.toml` |
| **Image generation** | `/image <prompt>` via `/images/generations` endpoint, auto-saves to `data/images/` |
| **Themes** | 6 colour themes (dark, light, monokai, dracula, nord, gruvbox), switchable at runtime |
| **Persistence** | SQLite: conversations, messages, request telemetry, API failures |
| **Resilience** | Concurrency semaphore, sliding-window RPM limiter, exponential backoff (3 retries) |
| **Failure simulation** | Inject 429, 500, timeout, malformed JSON, empty/slow responses at configurable rates |
| **Preflight probe** | Smoke-test provider access, model availability, latency, and rate limits before Phase 2 |
| **Observability** | Structured JSON logs with secret redaction, per-request token/latency tracking |
| **Debug mode** | `--debug` flag shows sanitized config, token counts, thinking/streaming status per turn |

---

## Slash commands

### Session
| Command | Action |
|---|---|
| `/help` | Show all commands |
| `/config` | Show sanitized runtime configuration |
| `/clear` | Clear in-memory conversation history |
| `/exit`, `/quit` | Exit the REPL |

### Models
| Command | Action |
|---|---|
| `/model list` | Show all models with thinking/streaming/image columns |
| `/model current` | Show active model and its capabilities |
| `/model set <id>` | Switch models — supports partial IDs (`qwen3` → `qwen/qwen3-32b`) |

### Thinking
| Command | Action |
|---|---|
| `/think` | Show status, session budget, model max, effective budget |
| `/think on` / `off` | Toggle extended reasoning |
| `/think budget <N>` | Set token budget (min 1024) |

### Streaming
| Command | Action |
|---|---|
| `/stream` | Show streaming status and model support |
| `/stream on` / `off` | Toggle SSE response streaming |

### Image generation
| Command | Action |
|---|---|
| `/image <prompt>` | Generate images via the configured image model |

### Failure simulation
| Command | Action |
|---|---|
| `/fail on` / `off` | Toggle failure injection |
| `/fail rate <0.0-1.0>` | Set failure probability |
| `/fail kind <kind>` | Set failure mode: `429`, `500`, `timeout`, `malformed_json`, `empty_response`, `slow_response` |

### Themes
| Command | Action |
|---|---|
| `/theme` | Show current theme |
| `/theme list` | List all 6 available themes |
| `/theme set <name>` | Switch theme (monokai, dracula, nord, gruvbox, dark, light) |

### Persistence
| Command | Action |
|---|---|
| `/history` | Show persisted messages for the current conversation |
| `/request <id>` | Show stored LLM request telemetry |

---

## Environment variables

### Provider
| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | *(required)* | Provider API key |
| `LLM_BASE_URL` | *(required)* | Provider base URL |
| `LLM_CHAT_COMPLETIONS_PATH` | `/chat/completions` | Chat endpoint path |

### Request defaults
| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | *(required)* | Model used on startup |
| `REQUEST_TIMEOUT_SECONDS` | `60` | HTTP read timeout |
| `MAX_CONCURRENCY` | `2` | Max concurrent requests |
| `REQUESTS_PER_MINUTE` | `6` | Rate limiter ceiling |
| `MAX_OUTPUT_TOKENS` | `400` | Max tokens per response |

### Features
| Variable | Default | Description |
|---|---|---|
| `THINKING_ENABLED` | `false` | Enable extended reasoning at startup |
| `THINKING_BUDGET_TOKENS` | `16000` | Default thinking token budget |
| `STREAMING_ENABLED` | `true` | Enable SSE streaming at startup |
| `THEME` | `dark` | UI colour theme |
| `IMAGE_N` | `1` | Number of images per `/image` request |

### Failure simulation
| Variable | Default | Description |
|---|---|---|
| `SIMULATE_FAILURES` | `false` | Enable at startup |
| `SIMULATE_FAILURE_RATE` | `0.0` | Probability 0.0–1.0 |
| `SIMULATE_FAILURE_KIND` | `429` | Default failure kind |

### Persistence & logging
| Variable | Default | Description |
|---|---|---|
| `SQLITE_PATH` | `data/chatbot.sqlite3` | Database file path |
| `LOG_LEVEL` | `INFO` | DEBUG, INFO, WARNING, ERROR |
| `LOG_JSON` | `true` | Emit structured JSON logs |

### Preflight
| Variable | Default | Description |
|---|---|---|
| `PREFLIGHT_MAX_TOKENS` | `400` | Tokens per probe request |
| `PREFLIGHT_TEMPERATURE` | `0.0` | Probe request temperature |
| `PREFLIGHT_SAFE_REQUESTS_PER_MINUTE` | `6` | Conservative RPM for rate test |
| `PREFLIGHT_BURST_CONCURRENCY` | `2` | Max burst for concurrency test |
| `PREFLIGHT_RATE_TEST_REQUESTS` | `8` | Number of sustained-rate probes |
| `PREFLIGHT_REPORT_PATH` | `reports/api_key_preflight_report.md` | Report output path |

---

## models.toml

Define your provider and models in `src/ai_chatbot/models.toml`. Each model supports optional per-model fields:

| Field | Type | Default | Description |
|---|---|---|---|
| `display_name` | string | model ID | Human-readable name |
| `family` | string | `"unknown"` | Model family (openai, llama, gemini, qwen, etc.) |
| `use_case` | string | `""` | Short description |
| `thinking_budget_tokens` | int | `0` | Max thinking tokens; `0` = disabled |
| `streaming_capable` | bool | `true` | Whether the model supports SSE streaming |
| `image_capable` | bool | `false` | Whether the model generates images |

```toml
[provider]
name = "multi-llm-router"
base_url_env = "LLM_BASE_URL"
api_key_env = "LLM_API_KEY"
chat_completions_path = "/chat/completions"
image_generations_path = "/images/generations"

[defaults]
model = "llama-3.1-8b-instant"
timeout_seconds = 60
max_concurrency = 2
requests_per_minute = 6

[models."llama-3.1-8b-instant"]
display_name = "Llama 3.1 8B Instant"
family = "llama"
use_case = "fast free-tier default"

[models."qwen/qwen3-32b"]
display_name = "Qwen3 32B"
family = "qwen"
use_case = "strong open model reasoning"
thinking_budget_tokens = 16384

[models."gemini-3-pro-image-preview"]
display_name = "Gemini 3 Pro Image Preview"
family = "gemini"
use_case = "image generation"
image_capable = true
```

---

## Project structure

```
.
├── data/
│   ├── .gitkeep
│   ├── chatbot.sqlite3            # SQLite database (created at runtime)
│   └── images/                    # Generated images saved here
├── reports/                       # Preflight probe output
├── scripts/
│   └── api_key_probe.py           # Phase 1: API-key feasibility probe
├── src/ai_chatbot/
│   ├── models.toml                # Provider + per-model definitions
│   ├── cli.py                     # Typer app, REPL, slash commands
│   ├── config.py                  # AppConfig, ModelConfig, env/TOML loading
│   ├── db.py                      # SQLite schema + async ChatStore
│   ├── failure_simulator.py       # Configurable failure injection
│   ├── llm_client.py              # Async HTTP client, streaming, retries, image gen
│   ├── logging_setup.py           # JSON/plain formatters, secret redaction
│   ├── session.py                 # In-memory conversation state
│   └── themes.py                  # 6 predefined colour themes
├── tests/
├── .env.example
├── Makefile
└── pyproject.toml
```

---

## Entity-relationship diagram

```
 ────────────────────────────────────────────────────────
  conversations                messages
 ───────────────────────       ─────────────────────
 │ id            PK │───<      │ id            PK │
 │ started_at       │          │ conversation_id FK
 │ ended_at         │          │ role            │  user | assistant
 │ default_model    │          │ content         │
 ────────────────────          │ model           │  nullable
                               │ created_at      │
                               ───────────────────

 ────────────────────────────────────────────────────────
  llm_requests                 api_failures
 ────────────────────────      ─────────────────────
 │ id            PK │───<      │ id            PK │
 │ conversation_id FK │        │ request_id    FK │  nullable
 │ model             │         │ failure_kind     │  timeout|network|http|schema
 │ provider          │         │ status_code      │
 │ prompt_chars      │         │ message          │
 │ response_chars    │         │ retry_attempt    │
 │ prompt_hash       │         │ created_at       │
 │ status_code       │         ────────────────────
 │ success           │
 │ latency_ms        │
 │ input_tokens      │
 │ output_tokens     │
 │ total_tokens      │
 │ error_type        │
 │ error_message     │
 │ created_at        │
 ─────────────────────
```

All PKs are `TEXT` UUIDs (`conv_abc123`, `msg_def456`, `req_ghi789`). Timestamps are ISO-8601 UTC.

---

## Architecture

### Request pipeline

```
User input
  │
  ├─ /command ───> command_handlers() dispatch
  │
  └─ message ───> ChatSession.send()
                    │
                    ├─ append user message to history
                    ├─ persist to SQLite
                    │
                    ├─ streaming? ──> LLMClient.chat_stream()
                    │                   │
                    │                   └─ SSE → StreamChunk → Rich Live panel
                    │
                    └─ non-streaming ──> LLMClient.chat()
                                          │
                                          ├─ Semaphore (max_concurrency)
                                          ├─ AsyncRateLimiter (requests_per_minute)
                                          ├─ FailureSimulator.maybe_fail()
                                          │
                                          └─ _chat_once()
                                               │
                                               ├─ POST {thinking?, stream?} → provider
                                               ├─ Parse ChatResponse
                                               └─ Retry ×3 (exp backoff, 0.25s–2s)

  /image <prompt> ──> LLMClient.generate_image()
                        │
                        └─ POST {model, prompt, n} → /images/generations
                           └─ ImageResult → save to data/images/
```

### Resilience

| Mechanism | Implementation |
|---|---|
| Concurrency limit | `asyncio.Semaphore` (`MAX_CONCURRENCY`) |
| Rate limiting | Sliding-window RPM limiter (`REQUESTS_PER_MINUTE`) |
| Retries | 3 attempts, exponential backoff 0.25s → 2s cap |
| Retryable statuses | 429, 500, 502, 503, 504 + timeout/network errors |
| Failure simulation | 6 kinds, toggleable at runtime with configurable probability |

---

## Development

```bash
make test       # pytest (16 tests)
make lint       # ruff check
make fmt        # ruff format
make typecheck  # mypy --strict
make db-reset   # rm data/chatbot.sqlite3
make clean      # remove cache + generated reports
```

### Tech stack

| Component | Library |
|---|---|
| CLI | [Typer](https://typer.tiangolo.com/) |
| Terminal | [Rich](https://rich.readthedocs.io/) |
| HTTP | [httpx](https://www.python-httpx.org/) (async) |
| Database | [aiosqlite](https://github.com/omnilib/aiosqlite) |
| Linting | [Ruff](https://docs.astral.sh/ruff/) |
| Types | [Mypy](https://mypy-lang.org/) (`--strict`) |
| Tests | [pytest](https://docs.pytest.org/) |
| Packaging | [uv](https://docs.astral.sh/uv/) + [hatchling](https://hatch.pypa.io/) |
