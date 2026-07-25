# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-file Gradio chatbot (`app.py`) that acts as a D&D 5e Dungeon Master, backed by the Mistral API. Deployed to a Hugging Face Space (`sytse06/dend`) via GitHub Actions on every push to `main`.

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
- `get_dm_response(message, history)` — builds the Mistral message list (`system` + `history` + `user`), calls `client.chat.complete(model="mistral-medium-latest", ...)`, wrapped in a `logfire.span`.
- `respond(message, history)` — Gradio submit handler; appends user/assistant turns to `history` in messages format.
- `gr.Blocks` UI — chat log + textbox on the left, dice-roller buttons on the right.

Chat history is `list[dict]` with `{"role", "content"}` keys throughout (Gradio 6's `gr.Chatbot` only supports the messages format — the old tuples format was removed entirely, not just deprecated).

## Key version constraints

Pinned deliberately after debugging real breakage — don't bump without re-verifying against a real Mistral call and a real `demo.launch()`:

- **`mistralai==2.7.2`** (via `>=1.0.0`): API is `from mistralai.client import Mistral`, `client.chat.complete(...)`. The old `MistralClient`/`ChatMessage` classes don't exist in this version. Model ID must be `mistral-medium-latest`, not `mistral-medium`.
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

- `MISTRAL_API_KEY` — required; `client` is `None` without it and `get_dm_response` returns an error string instead of raising.
- `HF_TOKEN` — only needed for manual HF operations, not for the GitHub Actions deploy (that uses SSH).
- `LOGFIRE_TOKEN` — optional, no-op if unset.
