from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from chatbot.assistant import ChatbotAssistant
from chatbot.model import ChatbotModel


def main() -> int:
    parser = argparse.ArgumentParser(description='Train the intent classifier chatbot model')
    parser.add_argument('--intents', default='intents.json', help='Path to intents.json')
    parser.add_argument('--model-out', default='chatbot_model.pth', help='Output .pth path')
    parser.add_argument('--meta-out', default='model_meta.json', help='Output meta json path')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    assistant = ChatbotAssistant(args.intents)
    assistant.parse_intents()

    X = []
    y = []
    for words, intent in assistant.documents:
        X.append([1 if w in set(words) else 0 for w in assistant.vocabulary])
        y.append(assistant.intents.index(intent))

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ChatbotModel(input_size=X.shape[1], output_size=len(assistant.intents))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        running = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - loss={running/len(loader):.4f}")

    torch.save(model.state_dict(), args.model_out)

    meta = {
        'input_size': int(X.shape[1]),
        'output_size': int(len(assistant.intents)),
        'vocabulary': assistant.vocabulary,
        'intents': assistant.intents,
    }

    Path(args.meta_out).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    # Keep the old file too for backward compatibility.
    Path('dimensions.json').write_text(
        json.dumps({'input_size': meta['input_size'], 'output_size': meta['output_size']}),
        encoding='utf-8',
    )

    print(f"Saved model to {args.model_out}, meta to {args.meta_out}, and dimensions.json")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
