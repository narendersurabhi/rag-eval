"""Run an end-to-end RAG evaluation experiment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_eval import KeywordRetriever, RAGEvaluator, SimpleGenerator


def _bar(metric: float, width: int = 20) -> str:
    count = int(round(metric * width))
    return "█" * count + "-" * (width - count)


def main() -> None:
    dataset_path = ROOT / "datasets" / "qa_dataset.json"
    results_path = ROOT / "results" / "rag_eval_results.json"

    data = json.loads(dataset_path.read_text())
    corpus = [doc for row in data for doc in row["relevant_documents"]]

    retriever = KeywordRetriever(corpus=corpus)
    generator = SimpleGenerator()
    evaluator = RAGEvaluator()

    per_sample = []
    for sample in data:
        question = sample["question"]
        relevant_docs = sample["relevant_documents"]
        retrieved_docs = retriever.retrieve(question, top_k=3)
        answer = generator.generate(question, retrieved_docs)
        metric_values = evaluator.evaluate(
            question=question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            ground_truth=sample["ground_truth"],
            relevant_docs=relevant_docs,
        )
        per_sample.append(
            {
                "question": question,
                "ground_truth": sample["ground_truth"],
                "answer": answer,
                "retrieved_docs": retrieved_docs,
                "metrics": metric_values,
            }
        )

    aggregate = {
        key: sum(row["metrics"][key] for row in per_sample if isinstance(row["metrics"][key], (int, float)))
        / len(per_sample)
        for key in [
            "retrieval_recall",
            "context_precision",
            "answer_accuracy",
            "faithfulness",
            "hallucination_rate",
            "judge_correctness",
            "judge_faithfulness",
            "judge_completeness",
            "judge_overall",
        ]
    }

    output = {"aggregate": aggregate, "samples": per_sample}
    results_path.write_text(json.dumps(output, indent=2))

    print("RAG Evaluation Summary")
    print(f"Retrieval Recall:   {aggregate['retrieval_recall']:.2f}")
    print(f"Context Precision:  {aggregate['context_precision']:.2f}")
    print(f"Answer Accuracy:    {aggregate['answer_accuracy']:.2f}")
    print(f"Faithfulness:       {aggregate['faithfulness']:.2f}")
    print(f"Hallucination Rate: {aggregate['hallucination_rate']:.2f}")
    print("\nLLM Judge Scores")
    print(f"Correctness:       {aggregate['judge_correctness']:.2f}")
    print(f"Faithfulness:      {aggregate['judge_faithfulness']:.2f}")
    print(f"Completeness:      {aggregate['judge_completeness']:.2f}")
    print(f"Overall:           {aggregate['judge_overall']:.2f}")

    print("\nHallucination Rate (lower is better)")
    hall = aggregate["hallucination_rate"]
    print(f"Baseline RAG {_bar(1 - hall)} {hall:.2f}")


if __name__ == "__main__":
    main()
