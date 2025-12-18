from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RetrievalMatch:
    answer: str
    score: float


class FAQRetriever:
    """Lightweight retrieval fallback.

    Uses scikit-learn if installed. If not installed, it gracefully disables retrieval.
    """

    def __init__(self, questions: List[str], answers: List[str]):
        self._enabled = False
        self._questions = questions
        self._answers = answers

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

            self._TfidfVectorizer = TfidfVectorizer
            self._cosine_similarity = cosine_similarity
        except Exception:
            self._TfidfVectorizer = None
            self._cosine_similarity = None
            return

        self._vectorizer = self._TfidfVectorizer(stop_words='english')
        self._matrix = self._vectorizer.fit_transform(self._questions)
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def query(self, text: str, min_score: float = 0.35) -> Optional[RetrievalMatch]:
        if not self._enabled:
            return None

        v = self._vectorizer.transform([text])
        sims = self._cosine_similarity(v, self._matrix)[0]
        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score < min_score:
            return None

        return RetrievalMatch(answer=self._answers[best_idx], score=best_score)
