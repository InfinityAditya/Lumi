from __future__ import annotations

import random

from chatbot.assistant import ChatbotAssistant


def get_stocks():
    stocks = ['AAPL', 'META', 'NVDA', 'GS', 'MSFT']
    print(random.sample(stocks, 3))


def main() -> int:
    assistant = ChatbotAssistant(
        'intents.json',
        function_mappings={'stocks': get_stocks},
        confidence_threshold=0.65,
        unknown_log_path='unknown_questions.jsonl',
        faq_path='faq.json',
    )

    # Parse intents so we can return responses for predicted tags.
    assistant.parse_intents()

    # Load model. If you retrain using train.py, it will create model_meta.json.
    assistant.load_model('chatbot_model.pth', meta_path='model_meta.json')

    print('LumenPath ready! Commands: /quit, /help, /debug')
    debug = False

    while True:
        message = input('You: ').strip()
        if not message:
            continue

        if message.lower() in {'/quit', '/exit'}:
            break

        if message.lower() == '/help':
            print('Commands:')
            print('  /quit  - exit the chatbot')
            print('  /debug - toggle debug output (intent + confidence)')
            continue

        if message.lower() == '/debug':
            debug = not debug
            print(f'Debug mode: {"ON" if debug else "OFF"}')
            continue

        res = assistant.process_message(message)
        if debug:
            print(f"[intent={res.intent} confidence={res.confidence:.2f}]")
        print(f"Bot: {res.text}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
