"""Shared types for retrieval outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class RetrievedDocument:
    """Represents one retrieved item from a retriever backend."""

    text: str
    doc_id: str | None = None
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def docs_to_text(docs: Sequence[str | RetrievedDocument]) -> list[str]:
    """Convert mixed retrieval results into text-only strings."""

    output: list[str] = []
    for doc in docs:
        if isinstance(doc, RetrievedDocument):
            output.append(doc.text)
        else:
            output.append(doc)
    return output


def docs_to_ids(docs: Sequence[str | RetrievedDocument]) -> set[str]:
    """Extract non-null identifiers from retrieval results."""

    return {
        doc.doc_id
        for doc in docs
        if isinstance(doc, RetrievedDocument) and doc.doc_id is not None
    }
