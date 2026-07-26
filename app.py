import os
import random
import time
import gradio as gr
import logfire
import spaces
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# Load environment variables from .env file
load_dotenv()

logfire.configure(service_name="dnd-dm-assistant", send_to_logfire="if-token-present")
logfire.instrument_httpx(capture_all=True)


@spaces.GPU
def _zerogpu_startup_check() -> None:
    """No-op: this app does no local GPU work, but HF's ZeroGPU hardware
    refuses to start a Space with zero @spaces.GPU functions. Called once
    at import time only, never on the chat/dice request path."""
    return None


_zerogpu_startup_check()

# =============================================================================
# LLM Provider Configuration
# =============================================================================

# Provider configurations
# Ordered by priority - first available provider will be used
PROVIDERS = {
    "nvidia": {
        "api_key_env": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "supports_streaming": True,
        "max_tokens": 1536,
        "extra_params": {
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": 1024,
            }
        },
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-4-26b-a4b-it:free",
        "supports_streaming": True,
        "max_tokens": 150,
        "extra_params": {},
    },
}

# Determine which provider to use (priority order: NVIDIA, then OpenRouter)
def get_active_provider():
    """Get the first available provider based on API key availability."""
    for provider_name, config in PROVIDERS.items():
        api_key = os.environ.get(config["api_key_env"])
        if api_key:
            return provider_name, config, api_key
    return None, None, None


PROVIDER_NAME, PROVIDER_CONFIG, API_KEY = get_active_provider()

# Initialize the active client
client = (
    OpenAI(
        base_url=PROVIDER_CONFIG["base_url"],
        api_key=API_KEY,
    )
    if PROVIDER_NAME and PROVIDER_CONFIG and API_KEY
    else None
)

MODEL = PROVIDER_CONFIG["default_model"] if PROVIDER_CONFIG else "google/gemma-4-26b-a4b-it:free"

# System prompt for the D&D DM
SYSTEM_PROMPT = """
You are a Dungeon Master for D&D 5e running a game for a casual, possibly rules-unfamiliar player.

Follow these rules on every reply:
1. If the player asks a rules/mechanics question or seems confused, answer directly and simply. Do not add new plot events in that reply.
2. When you require a dice roll, name the exact die (e.g. "roll a d20") and what it's for, in the same message as the request.
3. When the player reports a roll result, resolve that specific check: state clearly whether it succeeded or failed and what happens as a direct result. Then STOP and ask "what do you do?" -- do not request another roll in that same reply, even if the outcome suggests a new danger. Wait for the player's next action before calling for another check.
4. Keep replies to 4-6 sentences. Include at least one concrete sensory detail (sight, sound, or smell) per description -- vivid, not overwrought.
5. Tone: serious adventure with light humor.
"""

# =============================================================================
# Rolling context summarization
# =============================================================================
# The Chatbot's displayed history grows forever, but resending the entire raw
# transcript to the LLM on every turn eventually hits context-length limits
# and gets slower/costlier well before that. Instead, only a small verbatim
# tail of recent turns plus a compact rolling summary of everything earlier
# gets sent to the LLM -- the full display log is untouched by this.

SUMMARY_TRIGGER_MESSAGES = 20  # fold older turns in once the tail exceeds this
SUMMARY_KEEP_TAIL_MESSAGES = 6  # always keep this many most-recent messages verbatim


def summarize_adventure(existing_summary: str, turns_to_fold: list[dict]) -> str:
    """Fold older turns into a compact rolling summary via one plain LLM call."""
    prompt = (
        "Summarize this D&D adventure so far in 4-6 sentences, preserving key plot "
        "points, locations, NPCs, items, and unresolved threads. This summary will "
        "replace the detailed history below to keep future context manageable.\n\n"
    )
    if existing_summary:
        prompt += f"Previous summary: {existing_summary}\n\n"
    prompt += "Events to fold in:\n" + "\n".join(f"{t['role']}: {t['content']}" for t in turns_to_fold)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=450,
        )
        return response.choices[0].message.content or existing_summary
    except Exception:
        logfire.exception("Adventure summarization failed, keeping previous summary")
        return existing_summary


def build_context_messages(summary: str, tail: list[dict]) -> list[dict]:
    system_content = SYSTEM_PROMPT
    if summary:
        system_content += f"\n\nAdventure so far (summary of earlier events): {summary}"
    return [{"role": "system", "content": system_content}, *tail]


# NVIDIA's free-tier Nemotron 3 Ultra endpoint shares a small worker pool across
# all API Catalog users and occasionally rejects requests with "Worker local
# total request limit reached" -- confirmed transient/congestion-driven (NVIDIA
# forum reports, not our own request rate) via real testing, not something a
# fixed local rate limit could prevent. No documented Retry-After header exists
# for this error, so retry with a fixed backoff rather than trusting one.
MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = (1, 3)
RETRYABLE_MESSAGE_SUBSTRINGS = ("resource exhausted", "worker local total request limit")


def _is_retryable(e: Exception) -> bool:
    if isinstance(e, RateLimitError):
        return True
    message = str(e).lower()
    return any(s in message for s in RETRYABLE_MESSAGE_SUBSTRINGS)


def roll_dice(dice_type: str) -> str:
    """Roll a dice and return the result as a string."""
    try:
        sides = int(dice_type.replace("d", ""))
        if sides not in {4, 6, 8, 10, 12, 20, 100}:
            return f"Invalid dice: {dice_type}"
        result = random.randint(1, sides)
        return f"{result}"
    except ValueError:
        return f"Invalid dice: {dice_type}"


def get_dm_response(message: str, summary: str, tail: list[dict]) -> str:
    """Get a response from the DM via the configured LLM provider.

    `tail` is the small verbatim window of recent turns; `summary` covers
    everything older (see build_context_messages). For streaming-capable
    providers with thinking enabled, returns the response string which may
    contain reasoning content interspersed with final content. Gradio 6
    Chatbot handles the display automatically.
    """
    with logfire.span(
        "get_dm_response",
        message_length=len(message),
        tail_turns=len(tail),
        has_summary=bool(summary),
        provider=PROVIDER_NAME,
    ):
        if client is None:
            logfire.error("No LLM provider API key configured")
            return "Error: No LLM provider configured. Set NVIDIA_API_KEY or OPENROUTER_API_KEY environment variable."

        messages = build_context_messages(summary, tail) + [{"role": "user", "content": message}]

        create_params = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": PROVIDER_CONFIG.get("max_tokens", 150),
        }

        if PROVIDER_CONFIG and "extra_params" in PROVIDER_CONFIG:
            create_params.update(PROVIDER_CONFIG["extra_params"])

        # Use streaming for providers that support it
        supports_streaming = PROVIDER_CONFIG.get("supports_streaming", False)
        if supports_streaming:
            create_params["stream"] = True

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.chat.completions.create(**create_params)

                # Handle streaming response
                if create_params.get("stream"):
                    full_content = ""
                    full_reasoning = ""
                    finish_reason = None
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        finish_reason = chunk.choices[0].finish_reason or finish_reason
                        delta = chunk.choices[0].delta
                        full_reasoning += getattr(delta, "reasoning_content", None) or ""
                        full_content += getattr(delta, "content", None) or ""

                    if full_reasoning:
                        logfire.info("dm reasoning", reasoning=full_reasoning)

                    # If the model got cut off before finishing, some providers (NVIDIA's
                    # Nemotron included) flush the entire in-progress reasoning into the
                    # final content chunk as a compatibility fallback -- so content can't
                    # be trusted at all when truncated, not just when it looks empty.
                    if not full_content.strip() or finish_reason == "length":
                        logfire.error(
                            "DM response truncated by token budget",
                            finish_reason=finish_reason,
                            reasoning_length=len(full_reasoning),
                            content_length=len(full_content),
                        )
                        return "The DM pauses, lost in thought for a moment too long. Try rephrasing your action."

                    return full_content
                else:
                    return response.choices[0].message.content

            except Exception as e:
                if attempt < MAX_ATTEMPTS - 1 and _is_retryable(e):
                    delay = RETRY_BACKOFF_SECONDS[min(attempt, len(RETRY_BACKOFF_SECONDS) - 1)]
                    logfire.warn(
                        "Retrying after transient provider error",
                        attempt=attempt + 1,
                        delay=delay,
                        error=str(e),
                    )
                    time.sleep(delay)
                    continue
                logfire.exception(f"{PROVIDER_NAME or 'LLM'} request failed")
                return f"Error: Failed to get response. {str(e)}"


def respond(
    message: str, history: list[dict], context_state: dict
) -> tuple[str, list[dict], dict]:
    """Handle user message and return DM response.

    `history` is the full display log (grows forever, shown to the player).
    `context_state` is the compact {"summary", "tail"} used to build the
    LLM's context -- see the rolling summarization section above.
    """
    summary = context_state.get("summary", "")
    tail = context_state.get("tail", [])

    response = get_dm_response(message, summary, tail)

    new_tail = tail + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    if len(new_tail) > SUMMARY_TRIGGER_MESSAGES:
        to_fold, new_tail = new_tail[:-SUMMARY_KEEP_TAIL_MESSAGES], new_tail[-SUMMARY_KEEP_TAIL_MESSAGES:]
        summary = summarize_adventure(summary, to_fold)

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", new_history, {"summary": summary, "tail": new_tail}


# Create Gradio interface
with gr.Blocks(title="D&D Dungeon Master") as demo:
    gr.Markdown("# 🎲 D&D Dungeon Master Assistant")
    gr.Markdown("Play a text-based D&D adventure. The AI DM will guide your party through an epic story.")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(height=500, label="Adventure Log")
            context_state = gr.State({"summary": "", "tail": []})
            msg = gr.Textbox(
                label="Your Action",
                placeholder="e.g., I investigate the door...",
                lines=1,
            )
            msg.submit(respond, [msg, chatbot, context_state], [msg, chatbot, context_state])
        
        with gr.Column(scale=1):
            gr.Markdown("### 🎲 Dice Roller")
            with gr.Row():
                d4_btn = gr.Button("d4", variant="secondary")
                d6_btn = gr.Button("d6", variant="secondary")
                d8_btn = gr.Button("d8", variant="secondary")
            with gr.Row():
                d10_btn = gr.Button("d10", variant="secondary")
                d12_btn = gr.Button("d12", variant="secondary")
                d20_btn = gr.Button("d20", variant="secondary")
            with gr.Row():
                d100_btn = gr.Button("d100", variant="secondary")
            
            dice_result = gr.Textbox(
                label="Result",
                interactive=False,
                buttons=["copy"],
            )
    
    # Connect dice buttons
    d4_btn.click(lambda: roll_dice("d4"), outputs=dice_result)
    d6_btn.click(lambda: roll_dice("d6"), outputs=dice_result)
    d8_btn.click(lambda: roll_dice("d8"), outputs=dice_result)
    d10_btn.click(lambda: roll_dice("d10"), outputs=dice_result)
    d12_btn.click(lambda: roll_dice("d12"), outputs=dice_result)
    d20_btn.click(lambda: roll_dice("d20"), outputs=dice_result)
    d100_btn.click(lambda: roll_dice("d100"), outputs=dice_result)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
