# Ranking Metrics beyond Binary AUC — DCG / NDCG / MAP / MRR

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN).
> Oracle values come from **scikit-learn 1.9.1** and **trec_eval** (via
> `ir_measures` 0.4.3), both read from source on 2026-10-13, never from memory.

**Goal:** Add graded-relevance and multi-query ranking metrics to
`include/nn/utils/metrics.{h,cpp}`: DCG, NDCG, MAP and MRR, with per-query and
aggregate (mean) forms and a `k` cut-off.

**Architecture:** Extend the existing `metrics` module rather than adding a new
file — it already owns the ranking/calibration section and the
`MulticlassAveraging` enum, and these metrics are its direct siblings.
Row-major `(N_queries, N_docs)` tensors, matching the existing convention.
Every expected value in the test file is taken from the oracle run, never
hand-derived.

**Tech Stack:** C++17, repo `Tensor`, `std::vector`, plain `assert`-style test
harness matching `tests/test_calibration_diagnostics.cpp`.

---

## Semantics read from primary sources (the load-bearing part)

These are the rules a reimplementation gets wrong. Each is pinned by a named
test.

### S1 — DCG discount is `log_base`, default 2, index from 0

`_dcg_sample_scores` (`_ranking.py:1698`):
```python
discount = 1 / (np.log(np.arange(y_true.shape[1]) + 2) / np.log(log_base))
```
So `discount[j] = log(2) / log(j+2)`, i.e. rank 1 (j=0) discounts by 1.
**The classic `1/log2(j+1)` starting at 0.6309 is the OFF-BY-ONE trap** — sklearn
indexes from `arange(n)+2`. Pinned by `test_dcg_matches_closed_form`.

### S2 — `k` zeroes the discount BEYOND k, it does not truncate the ideal

`discount[k:] = 0` (`_ranking.py:1700`). This applies to **both** the gain and
the normalizing ideal gain, so NDCG@k is normalised by the ideal **truncated at
k** — not by the full-length ideal. Pinned by `test_ndcg_at_k_uses_truncated_ideal`
(f8 k=2: 3.6309297535714573 / ideal_trunc, and the full ideal would give a
different number).

### S3 — IDCG sorts by `y_true` descending with `ignore_ties=True`

`_ndcg_sample_scores` (`_ranking.py:1942`):
```python
normalizing_gain = _dcg_sample_scores(y_true, y_true, k, ignore_ties=True)
```
**Ties in relevance are resolved arbitrarily** because permuting equal-relevance
items cannot change the gain. Pinned by `test_ndcg_is_one_when_y_score_equals_y_true`
(f4 = 1.0 exactly, where relevance order is not the input order).

### S4 — A query with NO relevant document scores 0.0, and is INCLUDED in the mean

```python
all_irrelevant = normalizing_gain == 0
gain[all_irrelevant] = 0
```
f6 is the whole-corpus degenerate case: `ndcg_score` = **0.0**, and the
0.0 is averaged in (it is not dropped). This is *not* a divide-by-zero guard —
the division simply never happens. Pinned by `test_query_with_no_relevant_docs_scores_zero`.

### S5 — MAP here means the TREC/``mean precision at relevant ranks``, and it is
      **NOT** the `average_precision_score` already shipped in this file

`ir_measures` `AP` on binary relevance equals the hand formula
`mean_i (i / rank_i)` over relevant ranks, on **all 5 fixtures exactly**
(f1 1.0, f3 0.6388888888888888, f5 0.7555555555555555, f7 0.3333333333333333,
f8 0.8666666666666667). sklearn's step-integral AP happens to agree with it on
these fixtures too, but the repo already documents that the shipped
`average_precision_score` is the STEP integral and is **not** the trapezoid —
so the new `mean_average_precision` must be named and pinned separately rather
than reusing the existing symbol. Pinned by `test_map_is_mean_precision_at_relevant_ranks`
plus an explicit `test_map_and_shipped_average_precision_are_separate_symbols`.

### S6 — Relevance grades are THRESHOLDED, not used as weights, for MAP/MRR

`ir_measures` converts qrels to a binary indicator (`{AP: ...}` with
`int(yt[d])` grades) and both `AP` and `RR` are defined on that binary view.
On graded f1, binary AP = 0.75 while the relevance-*weighted* variant gives
0.4166666666666667 — a 1.8× disagreement. The repo's API is a dense `(N, C)`
`Tensor` with no qrels concept, so **`rel > 0` is the binary view** and MAP/MRR
are documented as ignoring grade magnitude. Pinned by `test_map_uses_binary_view_of_graded_relevance`
(f1: 0.75).

### S7 — MRR under ties: trec_eval is PESSIMISTIC (docid-descending tie-break)

Measured, and the tie-break is **insertion-order independent** (verified: all
four dict orderings of the tied case give RR = 1/3), so it is real semantics and
not a Python dict artifact:

| case | trec RR | docid-asc | docid-**desc** | tie-averaged |
|---|---|---|---|---|
| all tied, d0 relevant | 0.3333333333333333 | 1.0 | **0.3333333333333333** | 0.6111111111 |
| d2=0.9, then d0=d1=0.5 tie, d0 relevant | 0.3333333333333333 | 0.5 | **0.3333333333333333** | 0.4166666667 |

The second row is the discriminating one: it separates pessimistic from
docid-ascending (0.5) and from tie-averaged (0.4167). **The repo implementation
uses the pessimistic rule** (relevant doc placed last within its tied score
group), documented at the declaration and pinned by
`test_mrr_is_pessimistic_under_ties`.

### S8 — MRR of a query with no relevant doc is 0.0 and is INCLUDED in the mean

f6, measured: `RR: 0.0`. Pinned by `test_mrr_zero_when_no_relevant_doc`.

### S9 — Aggregate is the unweighted mean over queries

`ndcg_score` returns `float(np.average(gain, weights=sample_weight))` and with
no weights that is the plain mean; `label_ranking_average_precision_score`
divides by `n_samples`. Pinned by `test_aggregate_is_unweighted_mean_of_queries`
(f2, 3 queries: 0.9741648372255328).

---

## Tasks

### Task 1: Write the RED test suite
Create `tests/test_ranking_metrics.cpp` with all fixtures from the oracle run,
asserting the S1–S9 semantics. Build it, confirm it fails to link (functions
absent) — that is the RED state. Commit `test(metrics): RED suite for ranking metrics`.

### Task 2: Declare the API in `metrics.h`
`dcg_score`, `ndcg_score`, `mean_average_precision`, `mean_reciprocal_rank`, plus
per-query `dcg_per_query` / `ndcg_per_query` / `average_precision_per_query` /
`reciprocal_rank_per_query` and a `RankingRelevance` doc note. No new `enum class`
needed (grades are doubles, `k` is a `size_t`). Commit `feat(metrics): ranking
metric declarations`.

### Task 3: Implement DCG/NDCG
`discount[j] = log(log_base)/log(j+2)`, zero beyond k; ideal DCG sorts
`y_true` descending. Query with zero ideal gain → 0.0 (no division). Commit.

### Task 4: Implement MAP/MRR
Binary view `rel > 0`; MAP = mean of `i/rank_i`; MRR = `1/rank` of the first
relevant doc with the pessimistic tied-group rule. Commit.

### Task 5: Mutation harness + verification
`tests/mutate_ranking_metrics.sh` following the hardened three-defence pattern
from the calibration session: baseline taken once and **asserted pristine**,
each `sed` verified with `diff -q`, mtime check, `MUT-BROKEN` distinguished from
`MUT-CAUGHT`. Commit.

### Task 6: Register in Makefile + `nn.h`, aggregate run, commit
Link rule, `tests:` prerequisite, `run_tests` line. Run the full suite to
confirm no new failures. Commit `chore(build): register ranking metrics`.