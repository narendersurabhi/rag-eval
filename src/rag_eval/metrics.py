"""Metric implementations for retrieval-augmented generation evaluation."""

from __future__ import annotations

import re
from typing import Sequence


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def answer_accuracy(answer: str, ground_truth: str) -> float:
    return float(_normalize(answer) == _normalize(ground_truth))


def retrieval_recall(retrieved_docs: Sequence[str], relevant_docs: Sequence[str]) -> float:
    if not relevant_docs:
        return 1.0
    hits = sum(1 for doc in relevant_docs if doc in retrieved_docs)
    return hits / len(relevant_docs)


def context_precision(retrieved_docs: Sequence[str], relevant_docs: Sequence[str]) -> float:
    if not retrieved_docs:
        return 0.0
    relevant_tokens = 0
    total_tokens = 0

    relevant_token_set = set()
    for doc in relevant_docs:
        relevant_token_set.update(_normalize(doc).split())

    for doc in retrieved_docs:
        doc_tokens = _normalize(doc).split()
        total_tokens += len(doc_tokens)
        relevant_tokens += sum(1 for tok in doc_tokens if tok in relevant_token_set)

    if total_tokens == 0:
        return 0.0
    return relevant_tokens / total_tokens


def faithfulness(answer: str, retrieved_docs: Sequence[str]) -> float:
    if not answer.strip():
        return 0.0
    answer_tokens = set(_normalize(answer).split())
    context_tokens = set()
    for doc in retrieved_docs:
        context_tokens.update(_normalize(doc).split())

    if not answer_tokens:
        return 0.0
    supported = len(answer_tokens & context_tokens)
    return supported / len(answer_tokens)


def hallucination_rate(answer: str, retrieved_docs: Sequence[str]) -> float:
    return 1.0 - faithfulness(answer, retrieved_docs)
