import os
import base64
import mimetypes
import gradio as gr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the same folder

# ----------------------------------------------------
# Config
# ----------------------------------------------------
# Note: qwen/qwen3.6-27b is currently Groq's vision-capable model, served
# as a preview (fine for testing, not guaranteed stable for production).
MODEL = "qwen/qwen3.6-27b"

SYSTEM_PROMPT = (
    "You are a helpful visual assistant. When an image is provided, "
    "describe and reason about it accurately before answering the question."
)
MAX_IMAGES_PER_TURN = 5

API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY not found. Set it in your shell (export/setx) or in a .env file "
        "next to this script, e.g.: GROQ_API_KEY=your-key-here"
    )

client = Groq(api_key=API_KEY)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def encode_image_to_data_url(filepath: str) -> str:
    """Reads a local image file and returns a base64 data: URL."""
    mime_type, _ = mimetypes.guess_type(filepath)
    mime_type = mime_type or "image/jpeg"
    with open(filepath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def build_user_content(text: str, files: list[str]) -> list[dict]:
    """Builds an OpenAI/Groq-style multimodal content block list."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for filepath in files[:MAX_IMAGES_PER_TURN]:
        content.append({
            "type": "image_url",
            "image_url": {"url": encode_image_to_data_url(filepath)},
        })
    return content


def history_to_messages(history: list[dict]) -> list[dict]:
    """
    Converts Gradio's stored history (messages format, text-only for past
    turns) back into API messages. Past images aren't resent - only the
    text is retained - to keep payload size and cost down on long chats.
    """
    messages = []
    for turn in history:
        role = turn.get("role")
        content = turn.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


# ----------------------------------------------------------
# Chat handler
# ----------------------------------------------------------
def respond(message, history):
    """
    message: dict like {"text": "...", "files": ["/tmp/xyz.png", ...]}
             (Gradio's multimodal ChatInterface format)
    history: prior turns in messages format
    """
    text = message.get("text", "") or ""
    files = message.get("files", []) or []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_to_messages(history))
    messages.append({"role": "user", "content": build_user_content(text, files)})

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
            stream=True,
        )
    except Exception as e:
        yield f"Error contacting Groq API: {e}"
        return

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial


# --------------------------------------------------------
# UI
# --------------------------------------------------------
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CUSTOM_CSS = """
#chatbot-container {max-width: 860px; margin: 0 auto;}
.gradio-container {background: #f7f9fb;}
footer {visibility: hidden}
"""

with gr.Blocks(theme=theme, css=CUSTOM_CSS) as demo:
    gr.ChatInterface(
        fn=respond,
        multimodal=True,
        title="Groq Vision Chat",
        description=(
            "Attach an image and ask a question - powered by a Groq-hosted "
            "vision-language model. Text-only messages work too."
        ),
        chatbot=gr.Chatbot(
            height=560,
            elem_id="chatbot-container",
            avatar_images=(None, "https://groq.com/favicon.ico"),
        ),
        textbox=gr.MultimodalTextbox(
            placeholder="Ask a question, optionally attach an image...",
            file_types=["image"],
            file_count="multiple",
            container=False,
            scale=7,
        ),
        submit_btn="Send",
    )

if __name__ == "__main__":
    demo.queue().launch(  
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )