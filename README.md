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

