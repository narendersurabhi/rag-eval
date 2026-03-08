# rag-eval

A lightweight evaluation harness for retrieval-augmented generation (RAG) pipelines.

## RAG Evaluation Harness

Evaluate retrieval-augmented generation pipelines using reproducible benchmarks.

### Metrics

- Retrieval Recall
- Context Precision
- Answer Accuracy
- Faithfulness
- Hallucination Rate
- LLM Judge Correctness
- LLM Judge Faithfulness
- LLM Judge Completeness
- LLM Judge Overall Score

### Repository structure

```text
rag-eval/
├── src/rag_eval/
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   └── metrics.py
├── src/evaluation/judges/
│   ├── llm_judge.py
│   ├── rubrics.py
│   └── judge_runner.py
├── datasets/
│   └── qa_dataset.json
├── experiments/
│   └── run_rag_eval.py
└── results/
```

### Run evaluation

```bash
python experiments/run_rag_eval.py
```

The script writes results to `results/rag_eval_results.json`.

## LLM Judge Evaluation

The harness includes an LLM-as-a-judge module that scores model answers with a rubric when deterministic metrics alone are not enough.

### Rubric criteria

1. Correctness
2. Faithfulness to retrieved context
3. Completeness

### Example judge output

```json
{
  "correctness_score": 9,
  "faithfulness_score": 10,
  "completeness_score": 8,
  "explanation": "The answer is correct, grounded in context, and mostly complete."
}
```

### Example aggregate judge scores

| Metric       | Score |
| ------------ | ----- |
| Correctness  | 8.7   |
| Faithfulness | 9.1   |
| Completeness | 8.3   |
| Overall      | 8.7   |
