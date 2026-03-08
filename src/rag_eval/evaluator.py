"""Evaluation orchestration for RAG systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from . import metrics


@dataclass
class LLMJudgeResult:
    score: float
    verdict: str
    rationale: str


class HeuristicLLMJudge:
    """A deterministic stand-in for an LLM-as-a-judge evaluation step."""

    def evaluate(self, question: str, answer: str, context: Sequence[str], ground_truth: str) -> LLMJudgeResult:
        acc = metrics.answer_accuracy(answer, ground_truth)
        faith = metrics.faithfulness(answer, context)
        score = round((0.6 * acc) + (0.4 * faith), 3)

        if score >= 0.9:
            verdict = "excellent"
        elif score >= 0.7:
            verdict = "good"
        elif score >= 0.4:
            verdict = "partial"
        else:
            verdict = "poor"

        rationale = (
            f"Question: {question} | Accuracy={acc:.2f}, Faithfulness={faith:.2f}. "
            f"The answer is judged as {verdict}."
        )
        return LLMJudgeResult(score=score, verdict=verdict, rationale=rationale)


class RAGEvaluator:
    """Computes core RAG metrics for a single QA sample."""

    def __init__(self, judge: HeuristicLLMJudge | None = None):
        self.judge = judge or HeuristicLLMJudge()

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: Sequence[str],
        ground_truth: str,
        relevant_docs: Sequence[str],
    ) -> dict[str, float | str]:
        judge_result = self.judge.evaluate(
            question=question,
            answer=answer,
            context=retrieved_docs,
            ground_truth=ground_truth,
        )

        return {
            "retrieval_recall": metrics.retrieval_recall(retrieved_docs, relevant_docs),
            "context_precision": metrics.context_precision(retrieved_docs, relevant_docs),
            "answer_accuracy": metrics.answer_accuracy(answer, ground_truth),
            "faithfulness": metrics.faithfulness(answer, retrieved_docs),
            "hallucination_rate": metrics.hallucination_rate(answer, retrieved_docs),
            "llm_judge_score": judge_result.score,
            "llm_judge_verdict": judge_result.verdict,
            "llm_judge_rationale": judge_result.rationale,
        }
