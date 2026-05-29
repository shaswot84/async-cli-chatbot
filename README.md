# async-cli-ai-chatbot

Asynchronous command-line AI chatbot with SQLite persistence, structured logging,
resilience patterns, and extended reasoning support.

## Features

- **Multi-model chat** — switch between any OpenAI-compatible model at runtime
- **SQLite persistence** — conversations, messages, request telemetry, and failure records
- **Structured logging** — JSON or plain-text logs with secret redaction
- **Resilience** — concurrency semaphore, RPM rate limiter, exponential-backoff retries
- **Failure simulation** — inject artificial failures (429, 500, timeout, etc.) to test error handling
- **Extended reasoning** — enable thinking/reasoning for models that support it, controllable from the CLI
- **Preflight probe** — smoke-test API key, model access, and rate limits before building on the provider

## Project structure

```
.
├── config/
│   └── models.toml            # Provider, defaults, and per-model definitions
├── data/
│   └── .gitkeep
├── reports/                   # Preflight report output
├── scripts/
│   └── api_key_probe.py       # Phase 1 API-key feasibility probe
├── src/ai_chatbot/
│   ├── __init__.py
│   ├── cli.py                 # Typer app and slash-command handlers
│   ├── config.py              # AppConfig, ModelConfig, env/TOML loading
│   ├── db.py                  # SQLite schema and async ChatStore
│   ├── failure_simulator.py   # Configurable failure injection
│   ├── llm_client.py          # Async HTTP client with retries and rate limiting
│   ├── logging_setup.py       # JSON/plain formatters with redaction
│   └── session.py             # In-memory conversation state
├── tests/
│   ├── test_api_key_probe.py
│   ├── test_llm_client.py
│   ├── test_logging_setup.py
│   └── test_phase2_cli.py
├── .env.example
├── .python-version
├── Makefile
├── pyproject.toml
└── README.md
```

## Entity-Relationship Diagram

```
┌──────────────┐       ┌──────────────┐
│ conversations │       │   messages   │
├──────────────┤       ├──────────────┤
│ id        PK │───<   │ id        PK │
│ started_at   │       │ conversation_id FK
│ ended_at     │       │ role          │   user | assistant
│ default_model│       │ content       │
└──────────────┘       │ model         │   nullable
                       │ created_at    │
                       └──────────────┘

┌──────────────┐       ┌──────────────┐
│ llm_requests │       │ api_failures │
├──────────────┤       ├──────────────┤
│ id        PK │───<   │ id        PK │
│ conversation_id FK    │ request_id FK   nullable
│ model         │       │ failure_kind │   timeout | network | http | schema
│ provider      │       │ status_code  │
│ prompt_chars  │       │ message      │
│ response_chars│       │ retry_attempt│
│ prompt_hash   │       │ created_at   │
│ status_code   │       └──────────────┘
│ success       │
│ latency_ms    │
│ input_tokens  │
│ output_tokens │
│ total_tokens  │
│ error_type    │
│ error_message │
│ created_at    │
└──────────────┘
```

All PKs are `TEXT` UUIDs (e.g. `conv_abc123`, `msg_def456`, `req_ghi789`).
Timestamps are ISO-8601 UTC strings.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Install

```bash
git clone <repo-url>
cd internship

# Copy and edit the environment file
cp .env.example .env
# Fill in LLM_API_KEY, LLM_BASE_URL, and adjust any defaults

# Install dependencies
uv sync
```

### Verify

```bash
# Run the API-key feasibility probe
make preflight

# If the probe reports GO, start the chatbot
make run
```

## Configuration

### Environment variables

All runtime configuration is read from the environment (typically via `.env`).

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | (required) | Provider API key |
| `LLM_BASE_URL` | (required) | Provider base URL (e.g. `https://api.example.com/v1`) |
| `LLM_CHAT_COMPLETIONS_PATH` | `/chat/completions` | Path appended to base URL |
| `DEFAULT_MODEL` | (required) | Model used on startup; must exist in `models.toml` |
| `REQUEST_TIMEOUT_SECONDS` | `60` | HTTP read timeout in seconds |
| `MAX_CONCURRENCY` | `2` | Max concurrent requests |
| `REQUESTS_PER_MINUTE` | `6` | Max requests per minute (rate limiter) |
| `MAX_OUTPUT_TOKENS` | `400` | Max tokens in each response |
| `SQLITE_PATH` | `data/chatbot.sqlite3` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_JSON` | `true` | Emit JSON-structured logs |
| `SIMULATE_FAILURES` | `false` | Enable failure simulation at startup |
| `SIMULATE_FAILURE_RATE` | `0.0` | Probability a request fails (0.0–1.0) |
| `SIMULATE_FAILURE_KIND` | `429` | Failure kind to simulate |
| `THINKING_ENABLED` | `false` | Enable extended reasoning at startup |
| `THINKING_BUDGET_TOKENS` | `16000` | Session-level thinking token budget |

### models.toml

Defines the provider connection and available models. Each model supports an optional
`thinking_budget_tokens` field (0 or absent = thinking disabled for that model).

```toml
[provider]
name = "multi-llm-router"
base_url_env = "LLM_BASE_URL"
api_key_env = "LLM_API_KEY"
chat_completions_path = "/chat/completions"

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
```

## Usage

### Preflight probe

Before using the chatbot, run the API-key feasibility probe to verify provider
access, test each model, and get safe concurrency/rate-limit recommendations.

```bash
make preflight          # Run all checks
make preflight-basic    # Smoke test the default model only
make preflight-models   # Test each configured model
make preflight-rate     # Concurrency and RPM probe
make preflight-report   # Print the latest Markdown report
```

Reports are written to `reports/api_key_preflight_report.md` and
`reports/api_key_preflight_raw.json`.

The probe emits a GO / GO WITH WARNINGS / NO-GO decision with actionable
recommendations for `DEFAULT_MODEL`, `MAX_CONCURRENCY`, and
`REQUESTS_PER_MINUTE`.

### Chat REPL

```bash
make run         # Normal mode
make dev         # Debug mode (shows config, token counts, thinking status)
```

Inside the REPL, type any message to chat. Prefix with `/` for commands.

### Slash commands

| Command | Action |
|---|---|
| `/help` | Show all commands |
| `/model list` | Show configured models (active one marked with `*`) |
| `/model current` | Show the active model |
| `/model set <id>` | Switch to a different model |
| `/config` | Show sanitized runtime configuration |
| `/history` | Show persisted conversation messages from SQLite |
| `/request <id>` | Show stored LLM request telemetry |
| `/fail on` | Enable failure simulation |
| `/fail off` | Disable failure simulation |
| `/fail rate <0.0-1.0>` | Set simulated failure probability |
| `/fail kind <kind>` | Set failure kind: `429`, `500`, `timeout`, `malformed_json`, `empty_response`, `slow_response` |
| `/think` | Show thinking status (enabled, budgets, model support) |
| `/think on` | Enable extended reasoning |
| `/think off` | Disable extended reasoning |
| `/think budget <tokens>` | Set thinking token budget (min 1024) |
| `/clear` | Clear in-memory conversation history |
| `/exit` | Quit |

### Inspecting the database

```bash
sqlite3 data/chatbot.sqlite3 ".tables"
# api_failures  conversations  llm_requests  messages

sqlite3 data/chatbot.sqlite3 ".schema"
```

## Architecture

### Request pipeline

```
User input
  │
  ├─ /command ──> command_handlers() dispatch
  │
  └─ message ───> ChatSession.send()
                    │
                    ├─ append user message to history
                    ├─ persist to SQLite
                    │
                    └─ LLMClient.chat(model, messages, thinking_budget)
                         │
                         ├─ Semaphore (max_concurrency)
                         ├─ AsyncRateLimiter (requests_per_minute)
                         ├─ FailureSimulator.maybe_fail()
                         │
                         └─ _chat_once()
                              │
                              ├─ POST {thinking?} to provider
                              ├─ Parse ChatResponse
                              └─ Retry on retryable failures (3 attempts, exp backoff)
```

### Resilience

| Mechanism | Implementation |
|---|---|
| Concurrency limit | `asyncio.Semaphore` (env: `MAX_CONCURRENCY`) |
| Rate limiting | Sliding-window RPM limiter (env: `REQUESTS_PER_MINUTE`) |
| Retries | 3 attempts with exponential backoff (0.25s → 2s cap) |
| Retryable codes | 429, 500, 502, 503, 504, plus timeout/network errors |
| Failure simulation | Probabilistic injection of 6 failure kinds, toggleable at runtime |

### Observability

- **Structured logs**: Every request emits `llm_request_started`, `llm_request_completed`
  (or `llm_request_failed`) with request ID, model, latency, tokens, status, and thinking
  budget. All secrets (API key, tokens, headers) are redacted.
- **SQLite telemetry**: Full record of every request (tokens, latency, errors) and every
  API failure (with retry count).
- **Debug mode** (`--debug`): Prints sanitized config on startup and per-response token
  counts, retry attempts, and thinking status.

## Development

```bash
make test       # Run test suite (16 tests)
make lint       # Ruff linting
make fmt        # Ruff formatting
make typecheck  # Mypy strict type checking
make db-reset   # Delete the SQLite database
make clean      # Remove cache dirs and generated reports
```

### Tech stack

| Component | Library |
|---|---|
| CLI framework | [Typer](https://typer.tiangolo.com/) |
| Terminal output | [Rich](https://rich.readthedocs.io/) |
| HTTP client | [httpx](https://www.python-httpx.org/) (async) |
| Database | [aiosqlite](https://github.com/omnilib/aiosqlite) |
| Linter | [Ruff](https://docs.astral.sh/ruff/) |
| Type checker | [Mypy](https://mypy-lang.org/) (strict mode) |
| Test runner | [pytest](https://docs.pytest.org/) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
