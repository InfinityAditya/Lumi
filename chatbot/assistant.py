from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import ChatbotModel
from .nlp import bag_of_words, ensure_nltk_data, tokenize_and_lemmatize
from .retrieval import FAQRetriever


@dataclass
class ChatbotResponse:
    text: str
    intent: str
    confidence: float


class ChatbotAssistant:
    def __init__(
        self,
        intents_path: str,
        function_mappings: Optional[Dict[str, Callable[[], Any]]] = None,
        confidence_threshold: float = 0.65,
        unknown_log_path: str = 'unknown_questions.jsonl',
        faq_path: str = 'faq.json',
    ):
        ensure_nltk_data()

        self.intents_path = intents_path
        self.function_mappings = function_mappings or {}
        self.confidence_threshold = confidence_threshold
        self.unknown_log_path = unknown_log_path

        self.model: Optional[ChatbotModel] = None
        self.documents: List[Tuple[List[str], str]] = []
        self.vocabulary: List[str] = []
        self.intents: List[str] = []
        self.intents_responses: Dict[str, List[str]] = {}

        self._faq_retriever: Optional[FAQRetriever] = None
        self._load_faq_if_present(faq_path)

    def _load_faq_if_present(self, faq_path: str) -> None:
        if not os.path.exists(faq_path):
            return

        try:
            with open(faq_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            questions = [item['q'] for item in data.get('items', [])]
            answers = [item['a'] for item in data.get('items', [])]
            if not questions:
                return

            retriever = FAQRetriever(questions, answers)
            if retriever.enabled:
                self._faq_retriever = retriever
        except Exception:
            # Don't crash if FAQ is malformed.
            self._faq_retriever = None

    def parse_intents(self) -> None:
        if not os.path.exists(self.intents_path):
            raise FileNotFoundError(f"Could not find intents file: {self.intents_path}")

        with open(self.intents_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        for intent in intents_data.get('intents', []):
            tag = intent['tag']

            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent.get('responses', [])

            for pattern in intent.get('patterns', []):
                pattern_words = tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append((pattern_words, tag))

        self.vocabulary = sorted(set(self.vocabulary))

    def encode(self, message: str) -> List[int]:
        tokens = tokenize_and_lemmatize(message)
        return bag_of_words(tokens, self.vocabulary)

    def load_model(self, model_path: str, meta_path: str = 'model_meta.json') -> None:
        if not os.path.exists(meta_path):
            # Backward compatible with your old format (dimensions.json)
            # NOTE: this format does NOT store the intent order or vocabulary.
            # If intents.json has changed since the model was trained, you MUST retrain.
            with open('dimensions.json', 'r', encoding='utf-8') as f:
                dims = json.load(f)

            if self.intents and len(self.intents) != int(dims['output_size']):
                raise RuntimeError(
                    "Model metadata not found (model_meta.json). Your intents.json likely changed. "
                    "Please retrain the model: `python train.py --epochs 200`, then rerun main.py."
                )

            self.model = ChatbotModel(int(dims['input_size']), int(dims['output_size']))
            self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
            return

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self.vocabulary = meta['vocabulary']
        self.intents = meta['intents']

        self.model = ChatbotModel(meta['input_size'], meta['output_size'])
        self.model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    def _log_unknown(self, message: str) -> None:
        try:
            with open(self.unknown_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'message': message}) + '\n')
        except Exception:
            pass

    def process_message(self, message: str) -> ChatbotResponse:
        if not self.model:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        bag = self.encode(message)
        x = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs).item())
        intent = self.intents[idx]
        confidence = float(probs[idx].item())

        # If confidence is low, try retrieval fallback.
        if confidence < self.confidence_threshold:
            if self._faq_retriever:
                match = self._faq_retriever.query(message)
                if match:
                    return ChatbotResponse(text=match.answer, intent='faq_retrieval', confidence=match.score)

            self._log_unknown(message)
            return ChatbotResponse(
                text="I'm not fully sure about that yet. Can you rephrase or ask something more specific?",
                intent='unknown',
                confidence=confidence,
            )

        # Optional function hook.
        if intent in self.function_mappings:
            try:
                self.function_mappings[intent]()
            except Exception:
                pass

        responses = self.intents_responses.get(intent, [])
        if not responses:
            return ChatbotResponse(text="Okay.", intent=intent, confidence=confidence)

        return ChatbotResponse(text=random.choice(responses), intent=intent, confidence=confidence)
