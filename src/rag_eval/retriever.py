"""Retriever implementations for the RAG evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


@dataclass
class KeywordRetriever:
    """Simple lexical retriever based on token overlap."""

    corpus: Sequence[str]

    def retrieve(self, question: str, top_k: int = 3) -> list[str]:
        q_tokens = _tokenize(question)
        scored_docs: list[tuple[float, str]] = []

        for doc in self.corpus:
            doc_tokens = _tokenize(doc)
            if not doc_tokens:
                continue
            overlap = len(q_tokens & doc_tokens)
            score = overlap / len(doc_tokens)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]


class StaticRetriever:
    """Retriever helper for tests and scripted experiments."""

    def __init__(self, retrieved_docs: Iterable[str]):
        self._retrieved_docs = list(retrieved_docs)

    def retrieve(self, _: str, top_k: int = 3) -> list[str]:
        return self._retrieved_docs[:top_k]
