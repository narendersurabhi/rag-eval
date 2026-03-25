# rag-eval

A lightweight evaluation harness for retrieval-augmented generation (RAG) pipelines.

## RAG Evaluation Harness

Evaluate retrieval-augmented generation pipelines using reproducible benchmarks.

### Metrics

- Retrieval Recall
- Retrieval Recall by ID (optional; recommended for production retrievers)
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
│   ├── metrics.py
│   └── types.py
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

### Run baseline evaluation

```bash
python experiments/run_rag_eval.py
```

The script writes results to `results/rag_eval_results.json`.

## Using this repo with Qdrant hybrid retrieval (BM25 + vector)

The baseline `KeywordRetriever` is only a local demo. For your production stack, you can pass your own retrieval outputs into `RAGEvaluator`.

### 1) Return `RetrievedDocument` objects from your retriever

```python
from rag_eval import RetrievedDocument

# convert Qdrant + BM25 fused results into this shape
RetrievedDocument(
    text=payload["text"],
    doc_id=payload["doc_id"],     # strongly recommended
    score=fused_score,
    source="qdrant_hybrid",
    metadata={"bm25": bm25_score, "vector": vector_score},
)
```

### 2) Provide a dataset with gold relevant document IDs

For each QA sample, store both:
- `relevant_documents` (for text-based metrics)
- `relevant_doc_ids` (for robust retrieval scoring)

### 3) Evaluate with ID-aware recall

```python
metrics = evaluator.evaluate(
    question=question,
    answer=answer,
    retrieved_docs=retrieved_docs,          # list[RetrievedDocument]
    ground_truth=ground_truth,
    relevant_docs=relevant_documents,
    relevant_doc_ids=relevant_doc_ids,      # enables retrieval_recall_by_id
)
```

### Why this matters for hybrid retrieval

When you fuse BM25 and vector search, returned text can differ slightly (chunking, normalization, metadata formatting). Exact string matching underestimates retrieval quality. `retrieval_recall_by_id` lets you evaluate retrieval on stable IDs instead of raw text.


### Metric definitions

Detailed metric formulas and per-metric logic are documented in `docs/evaluation_metric_logic.md`.

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
