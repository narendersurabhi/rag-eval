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
- LLM Judge Score (heuristic LLM-as-a-judge baseline)

### Repository structure

```text
rag-eval/
├── src/rag_eval/
│   ├── retriever.py
│   ├── generator.py
│   ├── evaluator.py
│   └── metrics.py
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

### Example result table

| Model        | Retrieval Recall | Answer Accuracy | Hallucination Rate |
| ------------ | ---------------- | --------------- | ------------------ |
| Baseline RAG | 0.84             | 0.79            | 0.08               |
| Improved RAG | **0.91**         | **0.86**        | **0.04**           |
