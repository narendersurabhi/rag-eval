"""LLM-as-a-judge implementation with a deterministic fallback client."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Protocol, Sequence

from rag_eval import metrics

from .rubrics import build_judge_prompt


class LLMClient(Protocol):
    """Simple protocol for judge model providers."""

    def generate(self, prompt: str) -> str:
        """Return text generated for prompt."""


@dataclass
class JudgeEvaluationResult:
    correctness_score: int
    faithfulness_score: int
    completeness_score: int
    explanation: str

    @property
    def overall_score(self) -> float:
        return round(
            (self.correctness_score + self.faithfulness_score + self.completeness_score) / 3,
            3,
        )


class HeuristicJudgeClient:
    """Deterministic fallback that emulates an LLM judge JSON response."""

    def score(self, question: str, context: Sequence[str], answer: str, ground_truth: str) -> JudgeEvaluationResult:
        del question
        correctness = int(round(1 + 9 * metrics.answer_accuracy(answer, ground_truth)))
        faithfulness = int(round(1 + 9 * metrics.faithfulness(answer, context)))

        answer_tokens = set(_normalize(answer).split())
        truth_tokens = set(_normalize(ground_truth).split())
        overlap = len(answer_tokens & truth_tokens)
        completeness_ratio = overlap / len(truth_tokens) if truth_tokens else 1.0
        completeness = int(round(1 + 9 * completeness_ratio))

        explanation = (
            "Heuristic judge fallback: scores are derived from exact-match accuracy, "
            "context faithfulness, and ground-truth token coverage."
        )
        return JudgeEvaluationResult(
            correctness_score=max(1, min(10, correctness)),
            faithfulness_score=max(1, min(10, faithfulness)),
            completeness_score=max(1, min(10, completeness)),
            explanation=explanation,
        )

    def generate(self, prompt: str) -> str:
        del prompt
        raise NotImplementedError("Use score() for the heuristic client.")


class LLMJudge:
    """Judge that requests rubric-aligned scoring from an LLM client."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client
        self._heuristic = HeuristicJudgeClient()

    def evaluate(self, question: str, context: Sequence[str], answer: str, ground_truth: str) -> JudgeEvaluationResult:
        if self.llm_client is None:
            return self._heuristic.score(question=question, context=context, answer=answer, ground_truth=ground_truth)

        prompt = build_judge_prompt(
            question=question,
            context="\n".join(context),
            answer=answer,
            ground_truth=ground_truth,
        )
        response = self.llm_client.generate(prompt)
        payload = json.loads(response)

        return JudgeEvaluationResult(
            correctness_score=int(payload["correctness_score"]),
            faithfulness_score=int(payload["faithfulness_score"]),
            completeness_score=int(payload["completeness_score"]),
            explanation=str(payload["explanation"]),
        )


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))
