# Async CLI AI Chatbot

Phase 1 implements an API-key feasibility probe before the chatbot is built.

## Setup

```bash
cp .env.example .env
# edit .env with your provider base URL and API key
uv sync
```

## Phase 1 commands

```bash
make preflight-basic      # config + default-model smoke test
make preflight-models     # test each configured model once
make preflight-rate       # conservative concurrency/RPM probe
make preflight            # run all Phase 1 checks
make preflight-report     # print the latest Markdown report
```

Reports are written to:

```text
reports/api_key_preflight_report.md
reports/api_key_preflight_raw.json
```

## Phase 2 CLI

```bash
make run
```

Inside the REPL:

```text
/help
/model list
/model current
/model set <model_id>
/history
/request <request_id>
/clear
/exit
```

Phase 3 persists conversations, messages, request metadata, and failures in SQLite:

```bash
sqlite3 data/chatbot.sqlite3 ".tables"
```

Expected tables:

```text
api_failures   conversations  llm_requests  messages
```

Phase 4 emits structured logs during `make dev`. Each chat turn includes JSON log
events with request ID, conversation ID, model, latency, status, and token metadata
without API keys or raw prompts.
