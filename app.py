from __future__ import annotations

import gradio as gr

from chatbot.assistant import ChatbotAssistant

# Project / bot name (good for Hugging Face Space title)
# Pick a unique name so your Hugging Face Space doesn't look like a common template.
PROJECT_NAME = "LumenPath"
BOT_NAME = "Lumi"

# Optional: add your links (used in the sidebar)
HF_SPACE_URL = "https://huggingface.co/spaces/<your-username>/<your-space-name>"
GITHUB_URL = "https://github.com/<your-username>/<your-repo>"


def load_assistant() -> ChatbotAssistant:
    assistant = ChatbotAssistant(
        intents_path='intents.json',
        confidence_threshold=0.65,
        unknown_log_path='unknown_questions.jsonl',
        faq_path='faq.json',
    )
    assistant.parse_intents()
    # `model_meta.json` is produced by `python train.py`. Include it in your HF Space repo.
    assistant.load_model('chatbot_model.pth', meta_path='model_meta.json')
    return assistant


ASSISTANT = None
LOAD_ERROR = None
try:
    ASSISTANT = load_assistant()
except Exception as e:
    LOAD_ERROR = str(e)


def chat(message: str, history: list[dict], debug: bool):
    history = history or []

    if not message or not message.strip():
        return history, ""

    if LOAD_ERROR or ASSISTANT is None:
        history = history + [
            {"role": "user", "content": message.strip()},
            {
                "role": "assistant",
                "content": (
                    "I couldn't start due to a missing dependency/data file (common on first deploy).\n\n"
                    "Fix (model): make sure these files exist in your Space repo:\n"
                    "- chatbot_model.pth\n- model_meta.json\n\n"
                    "Fix (NLTK): if the error mentions 'punkt' or 'punkt_tab', redeploy after updating chatbot/nlp.py so it auto-downloads NLTK data.\n\n"
                    f"Error: {LOAD_ERROR}"
                ),
            },
        ]
        return history, ""

    res = ASSISTANT.process_message(message.strip())

    text = res.text
    if debug:
        text = f"{text}\n\n—\nintent: {res.intent} | confidence: {res.confidence:.2f}"

    history = history + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": text},
    ]

    return history, ""


def clear_history():
    return []


css = """
#title {text-align:center; margin-bottom: 0.25rem;}
.gradio-container {max-width: 1100px !important; margin: 0 auto !important;}

/* Right-side feature bar */
#sidebar {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 14px;
  padding: 14px;
}
#sidebar h3 { margin-top: 0.25rem; }
#sidebar .note { color: #475569; font-size: 0.95rem; }
"""


theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
)

sidebar_md = f"""### {PROJECT_NAME} — a hybrid study + career copilot

**Attractive • Authentic • Unique • Reliable • Built for real student workflows**

#### What makes it *stand out* (resume-ready)
- **Hybrid brain (not generic):** intent classifier (PyTorch) + FAQ retrieval fallback (TF‑IDF)
- **Confidence gate:** low-confidence messages don’t get random answers; they route to retrieval / clarification
- **Active-learning loop:** unknown questions are stored in `unknown_questions.jsonl` to expand the dataset and retrain
- **Model/intent safety:** `model_meta.json` locks vocabulary + intent order to prevent “wrong-intent” drift after updates
- **Debug visibility:** optional intent + confidence trace for explainability during demos
- **Extensible actions:** intent → function hooks (easy to add features without rewriting the model)

#### Reliability, illustrated
> You
> → NLP normalize + tokenize
> → Intent Model (predict + confidence)
> → If high: respond from curated intent responses
> → If low: FAQ retriever answer OR ask to rephrase
> → Log unknown for future training

#### Quick prompts (for your demo video)
- How to prepare for exams?
- Python roadmap
- Resume tips

#### Links
- Hugging Face Space: {HF_SPACE_URL}
- GitHub repo: {GITHUB_URL}
"""


with gr.Blocks(title=PROJECT_NAME) as demo:
    gr.Markdown(f"# {PROJECT_NAME}", elem_id="title")

    with gr.Row(equal_height=True):
        # Main chat area
        with gr.Column(scale=7):
            debug = gr.Checkbox(value=False, label="Debug (show intent + confidence)")

            chatbot = gr.Chatbot(height=460, layout="bubble")

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask: exam plan for 7 days, Python roadmap, resume tips, etc.",
                    scale=8,
                )
                send = gr.Button("Send", variant="primary", scale=1)

            with gr.Row():
                clear = gr.Button("Clear chat")

        # Right-side feature bar
        with gr.Column(scale=3, min_width=280):
            with gr.Group(elem_id="sidebar"):
                gr.Markdown(sidebar_md)

    send.click(chat, inputs=[msg, chatbot, debug], outputs=[chatbot, msg])
    msg.submit(chat, inputs=[msg, chatbot, debug], outputs=[chatbot, msg])
    clear.click(clear_history, inputs=None, outputs=chatbot)


demo.queue()

demo.launch(theme=theme, css=css)
