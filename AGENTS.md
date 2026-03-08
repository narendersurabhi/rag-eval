# AGENTS Log and Working Notes

## Repository status
This repository now contains a complete baseline RAG evaluation harness, including retrieval, generation, evaluation metrics, experiment runner, dataset, and output artifacts.

## Agent instructions
- Always read this file before making updates.
- Keep this file updated with each change set.
- Record new modules, scripts, and notable behavior changes.

## Change log

### 2026-03-08
- Added `src/rag_eval` package with:
  - `retriever.py` for keyword overlap retrieval.
  - `generator.py` for heuristic answer generation.
  - `metrics.py` for retrieval recall, context precision, answer accuracy, faithfulness, and hallucination rate.
  - `evaluator.py` for metric orchestration and heuristic LLM Judge scoring.
- Added benchmark dataset at `datasets/qa_dataset.json`.
- Added experiment script `experiments/run_rag_eval.py` to run end-to-end evaluation and write JSON results.
- Added project documentation in `README.md` with a dedicated RAG harness section.
- Generated baseline output file at `results/rag_eval_results.json`.

- Added `.gitignore` rules to exclude Python bytecode and cache directories.
