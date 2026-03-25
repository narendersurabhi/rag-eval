# AGENTS Log and Working Notes

## Repository status
This repository now contains a baseline RAG evaluation harness plus a modular LLM-judge subsystem with rubric-driven scoring and aggregation.

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

### 2026-03-09
- Added a new evaluation package at `src/evaluation/judges/` with:
  - `rubrics.py` for reusable rubric prompt templates.
  - `llm_judge.py` containing an `LLMJudge` interface, JSON parsing flow, and deterministic fallback `HeuristicJudgeClient`.
  - `judge_runner.py` for per-sample execution and aggregate judge metrics.
- Added package init files at `src/evaluation/__init__.py` and `src/evaluation/judges/__init__.py`.
- Updated `src/rag_eval/evaluator.py` to use the new judge module and return rubric-aligned scores (`judge_correctness`, `judge_faithfulness`, `judge_completeness`, `judge_overall`) plus explanation text.
- Updated `experiments/run_rag_eval.py` to aggregate and print judge rubric scores in the evaluation summary.
- Refreshed documentation in `README.md` with an `LLM Judge Evaluation` section and updated repository structure.
- Re-ran the evaluation script and regenerated `results/rag_eval_results.json` with the new judge metrics.

### 2026-03-25
- Added `src/rag_eval/types.py` with a `RetrievedDocument` dataclass and helpers to normalize mixed retrieval outputs (strings + structured docs).
- Updated `src/rag_eval/metrics.py` to support structured retrieval docs and introduced `retrieval_recall_by_id` for stable ID-based retrieval evaluation.
- Updated `src/rag_eval/evaluator.py` to accept structured retrieved docs, convert context for LLM judge calls, and optionally emit `retrieval_recall_by_id` when `relevant_doc_ids` are provided.
- Updated `src/rag_eval/__init__.py` to export `RetrievedDocument`.
- Expanded `README.md` with a Qdrant hybrid retrieval integration guide, including how to map BM25+vector fused outputs and use ID-aware recall.

### 2026-03-25 (follow-up)
- Added `docs/evaluation_metric_logic.md` as a dedicated metric-logic reference that explains formulas, inputs, edge-case behavior, and judge aggregation logic.
- Updated `README.md` to link to the new metric logic document under a dedicated "Metric definitions" subsection.
