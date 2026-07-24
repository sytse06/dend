import os
import random
import gradio as gr
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Load Mistral API key from environment variable
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Initialize Mistral client only if API key is available
client = MistralClient(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

# System prompt for the D&D DM
SYSTEM_PROMPT = """
You are an experienced Dungeon Master for Dungeons & Dragons 5th Edition.
Provide an epic, challenging, and creative adventure for the players.
Use the rules of D&D 5e and ask for dice rolls when necessary.
Describe locations, NPCs, and events in detail.
Keep the tone serious but with a touch of humor.
Respond in 2-4 sentences maximum to keep the game flowing.
"""


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


def get_dm_response(message: str, history: list[tuple[str, str]]) -> str:
    """Get a response from the Mistral DM."""
    if client is None:
        return "Error: Mistral API key not configured. Set MISTRAL_API_KEY environment variable."
    
    try:
        # Build messages including system prompt and conversation history
        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        for user_msg, assistant_msg in history:
            messages.append(ChatMessage(role="user", content=user_msg))
            messages.append(ChatMessage(role="assistant", content=assistant_msg))
        messages.append(ChatMessage(role="user", content=message))
        
        # Get response from Mistral
        response = client.chat(
            model="mistral-medium",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: Failed to get response from Mistral. {str(e)}"


def respond(message: str, history: list[tuple[str, str]]) -> tuple[str, list[tuple[str, str]]]:
    """Handle user message and return DM response."""
    response = get_dm_response(message, history)
    return "", history + [(message, response)]


# Create Gradio interface
with gr.Blocks(title="D&D Dungeon Master", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎲 D&D Dungeon Master Assistant")
    gr.Markdown("Play a text-based D&D adventure. The AI DM will guide your party through an epic story.")
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(height=500, label="Adventure Log")
            msg = gr.Textbox(
                label="Your Action",
                placeholder="e.g., I investigate the door...",
                lines=1,
            )
            msg.submit(respond, [msg, chatbot], [msg, chatbot])
        
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
                show_copy_button=True,
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
    demo.launch()
