# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-file Gradio chatbot (`app.py`) that acts as a D&D 5e Dungeon Master, backed by Google's Gemma 4 model via the OpenRouter API (OpenAI-compatible `chat.completions`). Deployed to a Hugging Face Space (`sytse06/dend`) via GitHub Actions on every push to `main`.

## Commands

```bash
# Setup (Python 3.12, matches the deployed Space)
uv venv --python /opt/homebrew/opt/python@3.12/bin/python3.12
uv pip install -r requirements.txt --python .venv/bin/python

# Run locally
.venv/bin/python app.py
```

No test suite or linter is configured.

## Architecture

Everything lives in `app.py`:

- `roll_dice(dice_type)` — validates against `{4,6,8,10,12,20,100}`, returns a random result as a string.
- `get_dm_response(message, history)` — builds the message list (`system` + `history` + `user`), calls `client.chat.completions.create(model="google/gemma-4-26b-a4b-it:free", ...)` against OpenRouter, wrapped in a `logfire.span`.
- `respond(message, history)` — Gradio submit handler; appends user/assistant turns to `history` in messages format.
- `gr.Blocks` UI — chat log + textbox on the left, dice-roller buttons on the right.

Chat history is `list[dict]` with `{"role", "content"}` keys throughout (Gradio 6's `gr.Chatbot` only supports the messages format — the old tuples format was removed entirely, not just deprecated).

## Key version constraints

Pinned deliberately after debugging real breakage — don't bump without re-verifying against a real API call and a real `demo.launch()`:

- **`openai>=1.0.0`**: used purely as a generic OpenAI-compatible HTTP client (`base_url="https://openrouter.ai/api/v1"`), not talking to OpenAI itself. `client.chat.completions.create(...)`, not the Mistral SDK's `chat.complete(...)`.
- **`google/gemma-4-26b-a4b-it:free`**: OpenRouter's free-tier Gemma 4 MoE model ID (25.2B total/3.8B active params). Free tier is capped at 20 req/min and 50 req/day unless the account has ever bought $10+ in OpenRouter credits (permanent bump to 1000 req/day). The dense sibling `google/gemma-4-31b-it:free` is a one-line swap if quality matters more than latency.
- **`gradio==6.20.0`**: `gr.Textbox` has no `show_copy_button` kwarg anymore — use `buttons=["copy"]`. `theme=` goes on `.launch()`, not `gr.Blocks()`.
- **`python_version: "3.12"`** in `README.md` frontmatter must match the local venv's Python (currently Homebrew 3.12).

## Hugging Face Space quirks

- `README.md` frontmatter (`title`/`emoji`/`colorFrom`/`colorTo`/`sdk`/`sdk_version`/`python_version`/`app_file`/`pinned`) is HF Spaces config, not decoration — keep it in sync with `requirements.txt`.
- `@spaces.GPU`-decorated `_zerogpu_startup_check()` in `app.py` is a deliberate no-op, called once at import time. HF's default ZeroGPU hardware refuses to start a Space with zero `@spaces.GPU` functions, and downgrading to `cpu-basic` requires a PRO subscription (enforced server-side). This app does no local GPU work — it's a pure API-proxy app — so the decorator exists purely to satisfy the startup check. Never call it per-request; that would burn ZeroGPU quota on real traffic.
- Logfire (`logfire.configure(..., send_to_logfire="if-token-present")`) is a safe no-op without a `LOGFIRE_TOKEN` — don't add guards around it.

## Deploy pipeline (`.github/workflows/deploy.yml`)

Push to `main` → GitHub Actions force-pushes to the HF Space's git remote over SSH.

- `actions/checkout` uses `fetch-depth: 0` — HF's git server rejects shallow-clone pushes ("shallow update not allowed").
- The HF Space remote is `git@hf.co:spaces/sytse06/dend` — Spaces require the `spaces/` prefix; the bare `git@hf.co:sytse06/dend` form 404s.
- `known_hosts` is pre-seeded with a pinned host key rather than `ssh-keyscan hf.co`, which is unreliable in CI.
- Auth is `secrets.HF_SSH_KEY` (a deploy key registered on the HF account, not `HF_TOKEN`).

## Environment variables (`.env`, see `.env.example`)

- `OPENROUTER_API_KEY` — required; `client` is `None` without it and `get_dm_response` returns an error string instead of raising.
- `HF_TOKEN` — not required for the GitHub Actions deploy (that uses SSH) or for running `app.py` locally.
- `LOGFIRE_TOKEN` — optional, no-op if unset.
