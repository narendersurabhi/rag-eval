"""Prompt rubrics for LLM-judge evaluation."""

from __future__ import annotations

DEFAULT_RUBRIC = [
    "Correctness",
    "Faithfulness to the retrieved context",
    "Completeness",
]


def build_judge_prompt(question: str, context: str, answer: str, ground_truth: str) -> str:
    """Build a JSON-first rubric prompt for judge scoring."""
    rubric_lines = "\n".join(f"{idx + 1}. {criterion}" for idx, criterion in enumerate(DEFAULT_RUBRIC))
    return f"""You are an expert evaluator of AI system outputs.

Evaluate the following answer to a question.

Question:
{question}

Retrieved Context:
{context}

Model Answer:
{answer}

Ground Truth:
{ground_truth}

Evaluate the answer using the following criteria:
{rubric_lines}

Return a JSON response with keys correctness_score, faithfulness_score, completeness_score,
and explanation. Each score must be an integer from 1 to 10.
"""
