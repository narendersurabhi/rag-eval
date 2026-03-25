"""Metric implementations for retrieval-augmented generation evaluation."""

from __future__ import annotations

import re
from typing import Sequence

from .types import RetrievedDocument, docs_to_ids, docs_to_text


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def answer_accuracy(answer: str, ground_truth: str) -> float:
    return float(_normalize(answer) == _normalize(ground_truth))


def retrieval_recall(
    retrieved_docs: Sequence[str | RetrievedDocument],
    relevant_docs: Sequence[str],
) -> float:
    if not relevant_docs:
        return 1.0

    retrieved_normalized = {_normalize(doc) for doc in docs_to_text(retrieved_docs)}
    relevant_normalized = [_normalize(doc) for doc in relevant_docs]
    hits = sum(1 for doc in relevant_normalized if doc in retrieved_normalized)
    return hits / len(relevant_docs)


def retrieval_recall_by_id(
    retrieved_docs: Sequence[str | RetrievedDocument],
    relevant_doc_ids: Sequence[str],
) -> float:
    """Compute recall using stable document IDs when available."""

    if not relevant_doc_ids:
        return 1.0

    retrieved_ids = docs_to_ids(retrieved_docs)
    hits = sum(1 for doc_id in relevant_doc_ids if doc_id in retrieved_ids)
    return hits / len(relevant_doc_ids)


def context_precision(
    retrieved_docs: Sequence[str | RetrievedDocument],
    relevant_docs: Sequence[str],
) -> float:
    retrieved_text = docs_to_text(retrieved_docs)
    if not retrieved_text:
        return 0.0
    relevant_tokens = 0
    total_tokens = 0

    relevant_token_set = set()
    for doc in relevant_docs:
        relevant_token_set.update(_normalize(doc).split())

    for doc in retrieved_text:
        doc_tokens = _normalize(doc).split()
        total_tokens += len(doc_tokens)
        relevant_tokens += sum(1 for tok in doc_tokens if tok in relevant_token_set)

    if total_tokens == 0:
        return 0.0
    return relevant_tokens / total_tokens


def faithfulness(answer: str, retrieved_docs: Sequence[str | RetrievedDocument]) -> float:
    if not answer.strip():
        return 0.0
    answer_tokens = set(_normalize(answer).split())
    context_tokens = set()
    for doc in docs_to_text(retrieved_docs):
        context_tokens.update(_normalize(doc).split())

    if not answer_tokens:
        return 0.0
    supported = len(answer_tokens & context_tokens)
    return supported / len(answer_tokens)


def hallucination_rate(answer: str, retrieved_docs: Sequence[str | RetrievedDocument]) -> float:
    return 1.0 - faithfulness(answer, retrieved_docs)
