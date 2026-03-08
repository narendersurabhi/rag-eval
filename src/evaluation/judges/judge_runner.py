"""Utilities for running and aggregating LLM-judge evaluations."""

from __future__ import annotations

from typing import Sequence

from .llm_judge import JudgeEvaluationResult, LLMJudge


class JudgeRunner:
    """Applies an LLMJudge across samples and aggregates scores."""

    def __init__(self, judge: LLMJudge | None = None):
        self.judge = judge or LLMJudge()

    def evaluate_sample(self, question: str, context: Sequence[str], answer: str, ground_truth: str) -> JudgeEvaluationResult:
        return self.judge.evaluate(question=question, context=context, answer=answer, ground_truth=ground_truth)

    @staticmethod
    def aggregate(results: Sequence[JudgeEvaluationResult]) -> dict[str, float]:
        if not results:
            return {
                "judge_correctness": 0.0,
                "judge_faithfulness": 0.0,
                "judge_completeness": 0.0,
                "judge_overall": 0.0,
            }

        n = len(results)
        return {
            "judge_correctness": round(sum(r.correctness_score for r in results) / n, 3),
            "judge_faithfulness": round(sum(r.faithfulness_score for r in results) / n, 3),
            "judge_completeness": round(sum(r.completeness_score for r in results) / n, 3),
            "judge_overall": round(sum(r.overall_score for r in results) / n, 3),
        }
