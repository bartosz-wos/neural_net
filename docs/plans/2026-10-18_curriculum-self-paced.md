# Curriculum / Self-Paced Sample Weighing — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/curriculum.{h,cpp}` — the self-paced sample-weighting
family: explicit regularisers, the SPL-IR latent-robust forms, the λ schedule, and a
model-free weighted-loss core.

**Architecture:** Every numerical core is a free function over plain `double` / `Tensor`,
so the whole scheduler is testable with no `Model`, no optimizer and no network — the
same shape `utils/mup.h` and `utils/ptq.h` were built with. `Tensor` enters only where a
caller already holds a batch of per-sample losses.

**Tech Stack:** C++17, `Tensor`, repo Makefile (`-O2 -march=native -Wall -Wextra`).

---

## Source of truth

Yanbo Fan, Ran He, Jian Liang, Bao-Gang Hu — "Self-Paced Learning: an Implicit
Regularization Perspective", **arXiv:1606.00128** (NeurIPS 2016).

**Citation verified** 2026-10-18 by a `ti:` title query against the arXiv API (returns
exactly this title, exactly these four authors) AND by scraping the abs page's
`citation_*` meta tags. The classic SPL paper (Kumar et al., KDD 2017) is **not on
arXiv**, which is why a title query is the only reliable route to this one. This queue
has produced nine wrong arXiv IDs; the check is not optional.

Table 1 was read from the **LaTeX source** (`https://arxiv.org/e-print/1606.00128v3`,
`paper/table/implicit_spl_regularizer.tex`), not from flattened PDF text — see Task 2.

## The finding that shapes the whole file (measured, in `tests/curriculum_oracle.py`)

Definition 1 condition 1 states `phi = min_{v>=0}[v*t + psi(lam,v)]`, whose argmin is
`v = phi'(t)`, and Algorithm 1 uses `v* = sigma(lam, ell)`.

**The compiled Table 1 satisfies `phi'(t) == t * sigma(t)`, not `phi'(t) == sigma(t)`.**
Measured spread over four λ and `t = 1.7`: `1.18e-10`, `1.33e-10`, `6.84e-11`, `6.75e-11`
for Huber/Cauchy/Welsch/L1-L2. Table 1's σ column is therefore Eq. (7)'s HQ minimiser
`delta(t) = argmin_p [p t²/2 + psi(p)]` — the paper uses `delta` for HQ ("The optimal p*
… is uniquely determined by the corresponding minimizer function `delta(t)`") and
`sigma` for the SPL **weight**; the table heading conflates the two.

Section 3.2 / Figure 1 state a **second** pair of the same law — `phi = lam²·log(1 +
l/lam²)`, `sigma = 1/(1 + l/lam²)` — and *that* one satisfies `phi' == sigma` exactly
(measured `2.56e-09`) and is bounded by 1.

**Consequence for the API.** `curr_weight_*` returns `phi'(ell)`, the genuine `[0,1]`
weight. Table 1's column is exposed separately as `curr_hq_minimiser_*` and named for
what it is. Using it as the weight scales every hard sample by `t` — a clean `1/ell`
factor that an aggregate metric would hide.

## Task 1: RED — write the test file

**Files:** Create `tests/test_curriculum.cpp`

Tests (each pins a named semantics; see D1–D12 in the header):

1. `test_hard_weight_is_a_strict_cut` — D1: `ell < lam` → 1, else 0 (values).
2. `test_hard_boundary_is_a_documented_tie_break` — D2: `ell == lam` → 0, and the
   objective is genuinely **flat** there (argmin not unique). The body text's Eq. (4)
   writes `ell <= lam`; the table writes `ell < lam`. Ship the table's, document it.
3. `test_linear_log_mixture_gamma` — D3: the four remaining explicit regularisers,
   argmin-verified against the oracle; ζ domain guards throw.
4. `test_implicit_weight_is_phi_prime` — D4: Cauchy/Welsch/L1-L2/sec-3.2 weights.
5. `test_implicit_weight_bounded_by_one` — D5: the shipped weight ≤ 1; **Table 1's
   column is not** (it reaches 2), which is the check that distinguishes them.
6. `test_hq_minimiser_is_phi_prime_over_t` — D6: `curr_hq_minimiser_* == w * t` for
   the same fixtures, and differs from `w` for every `t > 0`.
7. `test_lambda_schedule_direction` — D7: `lam *= mu` for all shipped modes,
   `lam /= mu` for L1-L2, matching the paper's sentence.
8. `test_lambda_init_by_quantile` — D8: `lambda_init` at median selects the documented
   half for `hard`, and is monotone in the quantile.
9. `test_weighted_loss_and_gradient` — D9: `E = sum v*ell`, `dE/dv_i == ell_i` **exactly**.
10. `test_effective_mass_and_hard_fraction` — D10: reporting helpers, plus the
    degenerate all-zero loss case.
11. `test_monotone_properties` — D11: `sigma` non-increasing in `t`, non-decreasing in
    `lam` (Definition 1 conditions 4 and 5), on a grid.
12. `test_errors_and_empty_inputs` — D12: negative λ throws, `ell < 0` throws, empty
    tensors throw, all-zero losses do not divide by zero.
13. `test_end_to_end_curriculum_training` — the fixture: a two-cluster problem where
    uniform weighting cannot separate the clusters but the curriculum's growing λ
    does, with **liveness assertions** (the accuracy comparison must SPLIT its
    predictions; a constant predictor scores 0.5 on a balanced task).

Run: `make tests build/test_curriculum && ./build/test_curriculum`
Expected: **compile/link failure** — undefined references. Every failure must be an
undefined reference, not a typo.

## Task 2: GREEN — write the header

**Files:** Create `include/nn/utils/curriculum.h`

- `#include "../core/tensor.h"`, `<vector>`, `<cstddef>`.
- `enum class CurrWeightMode { Hard, Linear, Log, Mixture, Gamma, Cauchy, Welsch,
  L1L2, Sec32 };`
- `double curr_weight(CurrWeightMode, double lambda, double ell, double gamma = 1.0);`
- `double curr_hq_minimiser(CurrWeightMode, double lambda, double ell);` (Table 1's column)
- `double curr_schedule_step(double lambda, double step_factor, bool divide);`
- `double curr_lambda_init(const Tensor& losses, double quantile, CurrWeightMode,
                          double gamma = 1.0);`
- `Tensor curr_weights(const Tensor& losses, double lambda, CurrWeightMode, double gamma = 1.0);`
- `double curr_weighted_loss(const Tensor& losses, const Tensor& weights);`
- `Tensor curr_weighted_loss_gradient(const Tensor& losses);`
- `double curr_effective_mass(const Tensor& weights);`
- `size_t curr_hard_fraction(const Tensor& weights, double tol = 1e-12);`
- Doc-comment each with its measured evidence (the header is the deliverable's spec).

## Task 3: GREEN — write the implementation

**Files:** Create `include/nn/utils/curriculum.cpp`

Mirror `utils/mup.cpp` / `utils/ptq.cpp` structure. Guard every division (this is the
repo's recurring `1/0` trap: muP `inf`, VQ-VAE `decay=1`, PTQ all-zero tensor, PTQ
`relative_l2` — **fifth instance** here). Validate λ > 0 and `ell >= 0`.

## Task 4: Verify GREEN + mutation harness

**Files:** Create `tools/curriculum_mutation.sh`

- `make print-lib-objs` for the link set, **assert count ≥ 10, abort before any verdict**
  (two link-set defects in this repo scored every mutation as a false verdict).
- Snapshot `.cpp` **and** `.h`, restore **atomically** (`cp base tmp && mv -f`).
- `rm -f` intermediates before every build (a stale `.o` reported a phantom verdict).
- Python `str.replace` for the edit — never `sed` with `/` in the replacement.
- **Checksum the tree after every edit**; a `MUT-NOOP` is a harness defect, not a verdict.
- Separate columns for `MUT-BROKEN` (build failed) and `MUT-CRASH` (test aborted).
- Rebuild and re-verify the real binary on exit.
- ≥ 10 mutations, all of which must be caught.

## Task 5: Integrate + full suite

- `#include "utils/curriculum.h"` in `include/nn/nn.h`.
- `make print-lib-objs` count must still be ≥ 10 (catches cross-header redefinition).
- Umbrella compiles standalone:
  `g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`
- `make tests` then `make run_tests` — expect no new failures beyond the ODERNN entry
  already deferred in `NOT_FIXED.md`.

## Task 6: Docs

- `EXPANSION_QUEUE.md`: move the item to `## Done` with a one-line summary.
- `NOT_FIXED.md`: untouched unless a real new bug is found and left.

---

## Self-contradiction scan

- Task 1 test 7 says L1-L2 uses `/mu`; Task 3 says "`lam *= mu` for all shipped modes,
  `lam /= mu` for L1-L2". **These must agree.** The resolution is that `curr_schedule_step`
  takes an explicit `divide` flag and the *default* for L1-L2 is `divide=true`; the header
  documents the paper's sentence verbatim and the test pins both directions. Strip one of
  the two statements if they are still inconsistent when implementing.
- Task 1 test 5 asserts the shipped weight ≤ 1; test 6 asserts the HQ minimiser differs
  from it. Both are true simultaneously only because they are different functions — which
  is the entire point of the finding. Not a contradiction.