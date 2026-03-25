# Evaluation Metric Logic

This document explains exactly how each metric is computed in this repository, including deterministic metrics and LLM-judge outputs.

## Normalization used by multiple metrics

The helper `_normalize(text)` lowercases text and keeps only `\w+` tokens, then rejoins with single spaces.

- Example: `"Paris, France!" -> "paris france"`

This normalization is used by:
- `answer_accuracy`
- `retrieval_recall` (text variant)
- `context_precision`
- `faithfulness`
- heuristic judge completeness logic

---

## 1) Retrieval Recall (`retrieval_recall`)

**Purpose:** fraction of gold relevant documents that were retrieved.

**Inputs:**
- `retrieved_docs`: `Sequence[str | RetrievedDocument]`
- `relevant_docs`: `Sequence[str]`

**Logic:**
1. Convert retrieved docs to text with `docs_to_text`.
2. Normalize all retrieved texts and gold relevant texts.
3. Count a hit when a normalized gold relevant document exactly matches one normalized retrieved document.
4. Return `hits / len(relevant_docs)`.
5. If `relevant_docs` is empty, return `1.0`.

**Formula:**
`retrieval_recall = |relevant_docs ∩ retrieved_docs| / |relevant_docs|` (after normalization).

---

## 2) Retrieval Recall by ID (`retrieval_recall_by_id`)

**Purpose:** robust retrieval recall for production systems where text can differ due to chunking/formatting.

**Inputs:**
- `retrieved_docs`: `Sequence[str | RetrievedDocument]`
- `relevant_doc_ids`: `Sequence[str]`

**Logic:**
1. Extract retrieved IDs with `docs_to_ids` (only non-null `doc_id`).
2. Count a hit when a gold `relevant_doc_id` is in retrieved IDs.
3. Return `hits / len(relevant_doc_ids)`.
4. If `relevant_doc_ids` is empty, return `1.0`.

**Formula:**
`retrieval_recall_by_id = |relevant_doc_ids ∩ retrieved_doc_ids| / |relevant_doc_ids|`.

---

## 3) Context Precision (`context_precision`)

**Purpose:** token-level precision of retrieved context against the token set derived from gold relevant docs.

**Inputs:**
- `retrieved_docs`: `Sequence[str | RetrievedDocument]`
- `relevant_docs`: `Sequence[str]`

**Logic:**
1. Build `relevant_token_set` from normalized tokens across all relevant docs.
2. For each retrieved doc token:
   - increment `total_tokens`
   - increment `relevant_tokens` if token exists in `relevant_token_set`
3. Return `relevant_tokens / total_tokens`.
4. If no retrieved docs or `total_tokens == 0`, return `0.0`.

**Formula:**
`context_precision = (# retrieved tokens appearing in gold token set) / (# all retrieved tokens)`.

---

## 4) Answer Accuracy (`answer_accuracy`)

**Purpose:** strict exact-match style answer correctness.

**Inputs:**
- `answer: str`
- `ground_truth: str`

**Logic:**
1. Normalize both strings.
2. Return `1.0` if equal, else `0.0`.

**Formula:**
`answer_accuracy = 1 if normalize(answer) == normalize(ground_truth) else 0`.

---

## 5) Faithfulness (`faithfulness`)

**Purpose:** proportion of answer tokens supported by retrieved context tokens.

**Inputs:**
- `answer: str`
- `retrieved_docs`: `Sequence[str | RetrievedDocument]`

**Logic:**
1. Normalize and tokenize answer into a set.
2. Normalize retrieved docs and build union token set for context.
3. Compute `supported = len(answer_tokens ∩ context_tokens)`.
4. Return `supported / len(answer_tokens)`.
5. Return `0.0` for empty/blank answer.

**Formula:**
`faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|`.

---

## 6) Hallucination Rate (`hallucination_rate`)

**Purpose:** complement of faithfulness.

**Logic:**
- `hallucination_rate = 1.0 - faithfulness(answer, retrieved_docs)`.

---

## 7) LLM Judge Criteria

Per-sample judge output contains:
- `correctness_score` (1-10)
- `faithfulness_score` (1-10)
- `completeness_score` (1-10)
- `explanation` (string)

### Judge overall score (`overall_score`)

Computed from the per-sample result as:
- `overall_score = round((correctness + faithfulness + completeness) / 3, 3)`.

### Aggregate judge scores (`JudgeRunner.aggregate`)

For N samples, repository returns rounded averages:
- `judge_correctness = round(sum(correctness_score) / N, 3)`
- `judge_faithfulness = round(sum(faithfulness_score) / N, 3)`
- `judge_completeness = round(sum(completeness_score) / N, 3)`
- `judge_overall = round(sum(overall_score) / N, 3)`

If there are no results, all aggregate judge metrics are `0.0`.

---

## 8) Heuristic fallback judge behavior

When no external LLM client is configured, `HeuristicJudgeClient` is used:

- `correctness_score = clamp_1_10(round(1 + 9 * answer_accuracy))`
- `faithfulness_score = clamp_1_10(round(1 + 9 * faithfulness))`
- `completeness_score = clamp_1_10(round(1 + 9 * completeness_ratio))`

Where:
- `completeness_ratio = |answer_tokens ∩ ground_truth_tokens| / |ground_truth_tokens|` (or `1.0` if ground truth has no tokens)
- `clamp_1_10` bounds values into `[1, 10]`.

This produces deterministic, reproducible rubric-like scores without calling an external model.
