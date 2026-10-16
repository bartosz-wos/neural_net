# Test-Time Augmentation — Implementation Plan

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN → commit).

**Goal:** Ship `include/nn/utils/tta.{h,cpp}` — a model-free test-time-augmentation
utility that aggregates per-transform logits/probabilities into one prediction,
plus the flip/identity transform generator and the flip-report the paper's
headline finding requires.

**Architecture:** every numerical core is a **free function over `Tensor`s**, so the
whole file is testable with no `Model`, no optimizer and no network — the same
model-free shape `utils/vat.h`, `utils/randomized_smoothing.h` and
`utils/vqcodebook.h` were built with. The only interface to a network is a
`LogitsFn = std::function<Tensor(const Tensor&)>`, reused verbatim from `vat.h`.

**Tech stack:** C++17, `include/nn/core/tensor.h`, `Model`-free.

---

## Citation — verified, and the verification is part of the record

Divya Shanmugam, Davis Blalock, Guha Balakrishnan, John Guttag —
*"Better Aggregation in Test-Time Augmentation"*, **arXiv:2011.11156** (ICCV 2021).

Verified on 2026-10-16 against `https://arxiv.org/abs/2011.11156` directly
(full-text scrape of the abs page: title, the four author links, and the
abstract). The arXiv **export API was rate-limited from this host** ("Rate
exceeded." on three attempts, 25 s apart, http and https, with and without a UA),
so the ID was confirmed from three independent sources instead: the arXiv abs
page itself, NASA ADS (`2020arXiv201111156S`, DOI `10.48550/arXiv.2011.11156`,
"E-Print: ICCV 2021") and emergentmind.

The queue entry proposed this feature with **no citation at all**, on the
reasoning that it "needs no paper-shaped feature". That was right about the
*feature* and it also turned out to be right about a real aggregation subtlety:
the paper's central point is that the **simple average is suboptimal and the
failure is not visible in the aggregate score**. The repo had
`adversarial.{h,cpp}` (attack), `randomized_smoothing.{h,cpp}` (bound) and
`vat.{h,cpp}` (smooth) — three inference-time utilities — and **nothing that
combines several forward passes of one input into one prediction**. Verified
absent by content grep before implementing:

```
grep -ril 'test.time.augment\|test_time_augment\|\btta\b\|ensemble.*predict' include/ tests/
-> 0 hits
```

(Rejected alternatives, also grep-verified: Semantic RENAMINING — 0 hits in
`include/nn/layers/recurrent/`; orthogonality penalties — the only `orthogonal`
hits are Muon/Newton-Schulz orthonormalisation inside the optimizers, not a
regulariser over layer weights.)

---

## THE GAP THIS FILLS, MEASURED NOT ASSUMED

`Model::forward(const Tensor&)` returns a single `Tensor`. Every classifier in
this repo runs **exactly one forward pass per input**. The repo has ~250 layers,
~10 inference-time utilities, and zero code that runs a model on *several
transformations of the same input and combines the results* — the single most
commonly deployed inference-time technique in vision, and the thing the paper
shows can silently trade correct predictions for a better average.

---

## Semantics pinned (each is a natural mistake, each gets a named test)

| # | Semantics | Why it is a trap |
|---|---|---|
| **D1** | Layout is `(T, N, K)` — transform-major, so `L(t, n, k)`. | Row-major `(N, T, K)` silently mixes transforms. |
| **D2** | `tta_mean_logits` averages **logits** and returns logits; it is **not** a probability vector. | A caller that softmaxes it gets a different answer from `tta_mean_probs`. |
| **D3** | `tta_geo_mean_probs` **normalises** the geometric mean. The raw geometric mean sums to `0.945`, not 1. | Omitting it leaves a **sub-probability vector**; value tests catch it, argmax tests do **not** (a positive per-row scale never changes the argmax). |
| **D4** | `tta_argmax` ties go to the **lowest class index** (`>` not `>=`), matching `metrics.cpp:38`. | `>=` silently makes the answer depend on iteration order. |
| **D5** | `tta_majority_vote` ties go to the **lowest class index**, not the earliest transform. | "First transform wins" and "lowest class wins" differ under a 1–1 split. |
| **D6** | `tta_majority_vote` returns a **one-hot** `(N, K)`, not a count vector and not class indices. | Return type is part of the contract; counts are not probabilities. |
| **D7** | `tta_mean_probs` / `tta_geo_mean_probs` are **shift-invariant** (softmax is, per transform). `tta_mean_logits` is **not**. | Measured: adding 7.0 to transform 1 moves `mean_logits` by 3.5 and moves `mean_probs` by **0**. |
| **D8** | `weights` are **normalised internally**; `T == 1` throws. | Ignoring weights changes the answer; `T==1` is the no-augmentation case. |
| **D9** | `tta_flip_report` returns **`changed`, `correct_to_incorrect`, `incorrect_to_correct`** separately, and `changed == c2i + i2c` always. | Reporting only the net change is the paper's exact complaint. |
| **D10** | `tta_hflip` reverses **within each row of a flattened `(H, W)` image**, so the width comes first. | A whole-buffer reversal flips row order too. |

**Structural fact the suite pins (oracle §5): with `T = 2`, aggregation can never
repair a row that *both* members get wrong** — the mean of two probability
vectors is a convex combination, so if both members prefer the same class the
mean prefers it too. Measured: **0 counterexamples in 20 000 random fixtures**.
This is why an end-to-end test needs `T ≥ 3` to demonstrate any gain at all.

---

## Tasks

### Task 1: transforms + `tta_argmax` + `tta_flip_report`

Files: create `include/nn/utils/tta.h`, `include/nn/utils/tta.cpp`,
`tests/test_tta.cpp`; modify `Makefile`, `include/nn/nn.h`.

1. **RED** — write `test_transforms` and `test_flip_report` in `tests/test_tta.cpp`.
   `tta_hflip` involution (`tta_hflip(tta_hflip(X)) == X` bitwise), the width-first
   reversal rule, `tta_identity` being an exact copy, `tta_flip_report` on a fixture
   where `changed == 3`, `c2i == 1`, `i2c == 2` (oracle §7).
2. Run `make build/test_tta` → must fail to link (functions undefined).
3. **GREEN** — implement the four functions.
4. Run `./build/test_tta` → PASS.
5. Commit `feat(utils): test-time augmentation transforms + flip report`.

### Task 2: the four aggregations

1. **RED** — `test_aggregations`: every value in oracle §1 and §6 transcribed
   digit-for-digit, plus `test_geo_identity` (D3: normalised geo == softmax of
   mean logits to 4.4e-16 over 5000 random fixtures; the suite pins it on two
   fixed fixtures), `test_shift_invariance` (D7, both directions), and
   `test_t2_cannot_fix_both_wrong` (the structural fact, on a `T=2` fixture).
2. Verify RED.
3. **GREEN** — implement `tta_mean_logits`, `tta_mean_probs`, `tta_geo_mean_probs`,
   `tta_majority_vote`.
4. Verify GREEN, commit.

### Task 3: weights, validation, end-to-end

1. **RED** — `test_weights` (D8: equal weights reproduce unweighted to `0.0`
   exactly — measured, not tolerance; unequal weights change the answer by 0.109),
   `test_validation` (`T == 1` throws, empty transform set throws, mismatched
   shapes throw), `test_aggregation_beats_members` (the headline: aggregate 0.80
   > every member 0.70/0.70/0.60 **and** > majority vote 0.70), `test_flip_report_integration`
   (the +1 net with 1 regression).
2. Verify RED.
3. **GREEN** — implement weights + `tta_aggregate` dispatcher.
4. Verify GREEN.

### Task 4: verification sweep

- `./build/test_tta` run **3×**, output identical (determinism).
- Mutation harness `tests/mutate_tta.sh`: ≥6 mutations, verdicts counted from
  `[FAIL]` lines; `MUT-BROKEN` never `MUT-CAUGHT`; `print-lib-objs` for the link set
  with a `≥ 10` object-count assertion before any verdict is reported.
- ASan+UBSan build clean.
- `-O2 -march=native -Wall -Wextra` zero warnings.
- `g++ -fsyntax-only` on `nn/nn.h` standalone.
- `make run_tests` — no new failures.

---

## Plan-integrity checks

- Every expected number traces to `~/.hermes/cache/scratch/tta_oracle.py`
  (numpy 2.3.5 / scipy 1.17.1 float64, output saved to `tta_oracle_out.txt`).
  **Not hand-derived.**
- The oracle caught two of my own claims *before* any C++ existed: the D3
  identity is false as literally written (logsumexp is convex, so the raw
  geometric mean is a sub-probability vector), and my D7 shift-invariance claim
  was **backwards** (mean-*probs* is invariant, mean-*logits* is not).
- The end-to-end fixture was rebuilt twice because it failed its own design
  goals: the first version had the aggregate merely *tie* the best member, and
  the second discovered the `(row, transform, class)` / `(T, N, K)` transpose
  error. Both were found by running the oracle, not by reading it.