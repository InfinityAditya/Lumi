from __future__ import annotations

import os
import re
from typing import Iterable, List

import nltk
from nltk.stem import WordNetLemmatizer


def _ensure_download_dir() -> str:
    """Return a writable dir for NLTK data and ensure NLTK can find it.

    Hugging Face Spaces often start with an empty filesystem cache, so we download
    the minimal assets at runtime into a writable directory.
    """

    # Prefer an explicit env var, otherwise default to a local cache folder.
    download_dir = os.environ.get('NLTK_DATA') or os.path.join(
        os.path.expanduser('~'), '.cache', 'nltk_data'
    )
    os.makedirs(download_dir, exist_ok=True)

    # Ensure NLTK searches this folder first.
    if download_dir not in nltk.data.path:
        nltk.data.path.insert(0, download_dir)

    return download_dir


def ensure_nltk_data() -> None:
    """Download required NLTK assets if missing."""

    download_dir = _ensure_download_dir()

    # Newer NLTK versions may require both 'punkt' and 'punkt_tab'.
    for pkg, resource in [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet', 'corpora/wordnet'),
        # WordNet lemmatizer commonly needs this too.
        ('omw-1.4', 'corpora/omw-1.4'),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True, download_dir=download_dir)


_lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(text: str) -> List[str]:
    # Basic cleanup: keep words/numbers, drop most punctuation.
    text = re.sub(r"[^a-zA-Z0-9\s'_-]", " ", text)
    tokens = nltk.word_tokenize(text)
    return [_lemmatizer.lemmatize(t.lower()) for t in tokens]


def bag_of_words(tokens: Iterable[str], vocabulary: List[str]) -> List[int]:
    token_set = set(tokens)
    return [1 if word in token_set else 0 for word in vocabulary]
