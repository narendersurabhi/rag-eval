"""Generator implementations for the RAG evaluation harness."""

from __future__ import annotations

import re
from typing import Sequence


class SimpleGenerator:
    """Heuristic generator that selects context snippets as answers."""

    def generate(self, question: str, retrieved_docs: Sequence[str]) -> str:
        if not retrieved_docs:
            return "I do not have enough context to answer."

        q_tokens = set(re.findall(r"\w+", question.lower()))
        best_doc = max(
            retrieved_docs,
            key=lambda doc: len(q_tokens & set(re.findall(r"\w+", doc.lower()))),
        )

        match = re.search(r"capital(?: city)? is ([A-Za-z\s]+?)(?:[\.,]|$)", best_doc, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

        first_sentence = re.split(r"(?<=[.!?])\s+", best_doc.strip())[0]
        return first_sentence.strip()
