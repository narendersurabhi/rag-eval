"""Evaluation orchestration for RAG systems."""

from __future__ import annotations

from typing import Sequence

from evaluation.judges import JudgeRunner, LLMJudge

from . import metrics


class RAGEvaluator:
    """Computes core RAG metrics plus rubric-based LLM judge scores."""

    def __init__(self, judge: LLMJudge | None = None):
        self.judge_runner = JudgeRunner(judge=judge)

    def evaluate(
        self,
        question: str,
        answer: str,
        retrieved_docs: Sequence[str],
        ground_truth: str,
        relevant_docs: Sequence[str],
    ) -> dict[str, float | str]:
        judge_result = self.judge_runner.evaluate_sample(
            question=question,
            context=retrieved_docs,
            answer=answer,
            ground_truth=ground_truth,
        )

        return {
            "retrieval_recall": metrics.retrieval_recall(retrieved_docs, relevant_docs),
            "context_precision": metrics.context_precision(retrieved_docs, relevant_docs),
            "answer_accuracy": metrics.answer_accuracy(answer, ground_truth),
            "faithfulness": metrics.faithfulness(answer, retrieved_docs),
            "hallucination_rate": metrics.hallucination_rate(answer, retrieved_docs),
            "judge_correctness": judge_result.correctness_score,
            "judge_faithfulness": judge_result.faithfulness_score,
            "judge_completeness": judge_result.completeness_score,
            "judge_overall": judge_result.overall_score,
            "judge_explanation": judge_result.explanation,
        }
