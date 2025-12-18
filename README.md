# LumenPath (Lumi)
LumenPath is a hybrid assistant for study, coding, and early-career guidance.
It is intentionally designed to be deployable, explainable, and continuously improvable, not a generic “chatbot wrapper”.

## Attractive, authentic, unique, reliable
### Attractive
The UI is a clean Gradio chat with a right-side “how it works” panel so a viewer can understand the system in seconds.

### Authentic
LumenPath answers from a controlled set of intent responses and a curated FAQ file.
When the model is unsure, it prefers clarification or retrieval over guessing.

### Unique
This project uses a hybrid approach:
- Intent classification (PyTorch) for fast, structured routing
- FAQ retrieval fallback (TF-IDF via scikit-learn) for low-confidence queries

### Reliable
Reliability is built into the flow:
- Confidence gating to reduce incorrect answers
- A safe fallback path (retrieval match or “please rephrase”)
- Model metadata locking (`model_meta.json`) to avoid vocabulary/intent drift after updates
- Debug mode that shows intent + confidence for evaluation and demos

## How it works
1) Message preprocessing and tokenization (NLTK)
2) Intent prediction (PyTorch model)
3) If confidence is high: respond using the mapped intent responses
4) If confidence is low:
   - Try FAQ retrieval (TF-IDF)
   - Otherwise ask the user to rephrase
5) Log unknown questions to `unknown_questions.jsonl` so you can expand data and retrain

## Tech stack
- Python
- Gradio (UI)
- PyTorch (intent classifier)
- NLTK (tokenization + lemmatization)
- scikit-learn (optional: TF-IDF FAQ retrieval fallback)

## Project structure
- `app.py` — Gradio UI (Hugging Face Spaces entry point)
- `main.py` — CLI runner (local terminal testing)
- `train.py` — training script for the intent classifier
- `intents.json` — training patterns + intent responses
- `faq.json` — retrieval knowledge base used when confidence is low
- `unknown_questions.jsonl` — logs low-confidence user queries for future improvement
- `chatbot/assistant.py` — orchestration logic (confidence gate + retrieval fallback)
- `chatbot/retrieval.py` — TF-IDF retriever (auto-disables if scikit-learn is missing)
- `chatbot/nlp.py` — NLTK setup (auto-downloads required data; Spaces-friendly)
- `chatbot_model.pth` — trained PyTorch weights
- `model_meta.json` — vocabulary + intent order + dimensions (prevents drift)
- `dimensions.json` — backward-compatible dimensions file

## Run locally
Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Start the Gradio app:

```bash
python app.py
```

Or run the CLI version:

```bash
python main.py
```

## Train / retrain the model
Training produces both the model weights and metadata.

```bash
python train.py --epochs 200
```

Outputs:
- `chatbot_model.pth`
- `model_meta.json`
- `dimensions.json`

## Improve it (active-learning style)
When the classifier confidence is low, the message is appended to:
- `unknown_questions.jsonl`

A simple improvement loop:
1) Review `unknown_questions.jsonl`
2) Add better patterns/responses to `intents.json` and/or add items to `faq.json`
3) Retrain with `train.py`
4) Redeploy

## Add FAQ items (retrieval)
Edit `faq.json` and add more `q` / `a` items. The retriever will use TF-IDF similarity.

## Add new actions (intent → function hook)
`ChatbotAssistant` supports mapping an intent tag to a Python function.
Example exists in `main.py` (`function_mappings`).

## Deploy to Hugging Face Spaces
Recommended repo contents for a Space:
- `app.py`
- `requirements.txt`
- `chatbot/` package
- `intents.json`, `faq.json`
- `chatbot_model.pth`, `model_meta.json` (and optionally `dimensions.json`)

In `app.py`, set your links:
- [https://huggingface.co/spaces/<your-username>/<your-space-name>](https://huggingface.co/spaces/adityachute01/Lumi)
- [`GITHUB_URL`](https://github.com/InfinityAditya/Lumi)

## Links
- Hugging Face Space: [https://huggingface.co/spaces/<your-username>/<your-space-name>](https://huggingface.co/spaces/adityachute01/Lumi)
- GitHub: [https://github.com/<your-username>/<your-repo>](https://github.com/InfinityAditya/Lumi)

## License
Add a license file if you plan to make this public/open-source.
