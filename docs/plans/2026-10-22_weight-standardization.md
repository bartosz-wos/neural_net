# Weight Standardization Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Ship `include/nn/utils/weight_std.{h,cpp}` — weight standardization (WS) with its exact adjoint, per-channel Lipschitz diagnostics, and a `WeightStandardization` layer that wraps an existing layer — filling the last unfilled normalization direction in the repo.

**Status: SHIPPED 2026-10-22 as `a935ebb`.** 94/94 checks, 14/14 mutations caught with 0 survived, ASan+UBSan clean, umbrella header compiles standalone, zero new full-suite failures. The record of what the oracle and the mutation harness caught is in `## Done` of `EXPANSION_QUEUE.md`; this file is kept as the design record.

**Architecture:** Model-free free functions over `Tensor` as the core (same shape as `utils/hutchinson.h`, `utils/sinkhorn.h`, `utils/ortho_penalty.h`), plus one thin `Layer*` wrapper. The wrapper snapshots raw params, standardizes them in place before `forward()`, and — the load-bearing part — converts the wrapped layer's gradient w.r.t. the *standardized* weights back to a gradient w.r.t. the *raw* weights via the exact adjoint, accumulating into `v_`.

**Tech Stack:** C++17, `Tensor` (`include/nn/core/tensor.h`), `Layer` base class, the repo Makefile, numpy oracle in `tests/`.

---

## 0. Verified facts this plan rests on (measured, not assumed)

**Absence.** `grep -rinE 'weight.?standard|ws_norm|lipschitz|learnable_lip|gain.?schedul' include/ tests/`
returns **only** `tests/test_spectral_norm.cpp`'s test *name* `test_end_to_end_lipschitz`. No
implementation of the concept exists anywhere in the tree.

**Not redundant.** The two nearby mechanisms do different things and are documented as such:

| existing | what it does | what WS adds |
|---|---|---|
| `layers/normalization/weight_norm.h` | `w = g_i * v_i / ||v_i||` — learnable length `g`, **no centering**, *not a `Layer`*, no adjoint | centering + unit variance, fixed projection, exact adjoint |
| `layers/normalization/spectral_norm.h` | constrains the **whole matrix**'s spectral norm to 1, via power iteration, forward-only | per-**channel** bound `√K`, no learnable gain |

**Citations — verified, with three wrong IDs recorded so they are never reintroduced.**

- **PRIMARY (defines WS).** Qiao, Wang, Liu, Shen, Yuille, "Micro-Batch Training with
  Batch-Channel Normalization and Weight Standardization", **arXiv:1903.10520**.
  Verified 2026-10-22 by scraping the arXiv **abs page** `citation_title` /
  `citation_author` meta tags (title and all five authors match verbatim).
- **WS eq. (10)** was read from the paper's own HTML (`ar5iv`), not from memory:
  `Ŵ = (W − W̄) / sqrt(W̄² − W̄̄²)`, where the overbars are **per output channel**, and
  "we do not have any affine transformation on `Ŵ`".
- **SECONDARY (Lipschitz / spectral-norm machinery).** Liu, Williams, Jacobson, Fidler,
  Litany, "Learning Smooth Neural Functions via Lipschitz Regularization",
  **arXiv:2202.08345**. Verified by abs-page meta tags and by a `ti:` title query.
- **NOT ON arXiv — NO ID MAY BE ASSERTED.** "Weight Standardization for Adversarial
  Robustness" (the CVPR 2020 paper) returns **0** results from an arXiv `ti:` title query
  and **0** from OpenAlex `title.search`; `openaccess.thecvf.com` is blocked from this
  host (HTTP 200 with a 2.6 kB block page). The design still stands — it is the same WS
  formula — but no identifier is claimed for it.
- **THREE WRONG IDS, measured today, recorded so they are never reintroduced:**
  `1903.10520` is *not* the adversarial-robustness paper (it is the micro-batch/BCN one —
  this is the trap a memory-typed citation walks straight into);
  `1805.09555` = "Phase Retrieval via Polytope Optimization";
  `1804.08868` = "Rational proofs for quantum computing".

**The env note (re-confirmed).** `python3` on PATH has no `pip` and no numpy. The working
interpreter is
`/home/stefan/.hermes/tools/python-3.14.7+202****0901-linux-x64/bin/python3` (numpy 2.5.4).

## 1. Design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | `eps` goes **inside** the sqrt, `s = sqrt(v + eps)` | matches every normalizer already in the repo; the adjoint is the exact derivative of *that* function |
| D2 | **population** variance, divide by `K` | WS is a fixed projection of exactly `K` weights; a `1/(K-1)` sample std makes `‖ŵ_c‖₂ ≠ √K` and breaks D3 below |
| D3 | per-channel = per **ROW** of the `(out,in)` weight | the paper standardizes "each output channel"; on this repo's 2-D `Tensor` an output channel is a row |
| D4 | core is free functions; the layer is a thin wrapper | testable with no `Model`, no optimizer, no network — the shape every recent util shipped with |
| D5 | the wrapper **accumulates** the adjoint into the raw snapshot | `Layer::backward` accumulates into `gradients()`; without the adjoint the wrapper silently trains the *projection*, not the weights |

## 2. The adjoint (derived before coding; the oracle re-derives it by FD)

With `u_i = W_i − m`, `m = mean(W)`, `v = mean(u²)`, `s = sqrt(v + eps)`,
`Ŵ_i = u_i / s`, and `g = dL/dŴ`:

```
mean(g)  = (1/K) Σ g_i
mean(g·x̂) = (1/K) Σ g_i x̂_i          where x̂ = Ŵ

dL/dW[j] = (1/s) · ( g[j] − mean(g) − x̂[j] · mean(g·x̂) )
```

This is the textbook normalization backward. Two traps it hides: `mean(g·x̂)` is **not**
`mean(g·u)` (they differ by `1/s` — a clean factor), and the `1/s` prefactor is outside
the parenthesis, not inside it.

## 3. Files

- Create: `include/nn/utils/weight_std.h`, `include/nn/utils/weight_std.cpp`
- Create: `tests/ws_oracle.py` (numpy oracle, run **before** any C++ exists)
- Create: `tests/test_weight_std.cpp`
- Create: `tests/mutate_weight_std.sh`
- Modify: `include/nn/nn.h` (add the `#include` next to the other `utils/` lines)
- Modify: `Makefile` (link rule, `tests:` prerequisite, `run_tests` line)

## 4. Tasks (all complete)

1. **numpy oracle** — `tests/ws_oracle.py`, run before any C++ exists. **42/42**, and it
   caught three defects before the first line of C++ (see the Done entry).
2. **failing test** — `tests/test_weight_std.cpp`, verified RED (fails to compile because
   `ws_standardize` is undeclared, and nothing else).
3. **implementation** — minimal code to green.
4. **registration** — `nn.h` + `Makefile`; umbrella header verified standalone.
5. **mutation harness** — `tests/mutate_weight_std.sh`, **14 mutations, 14 caught,
   0 survived, 0 broken, 0 harness defects**, with a post-harness rebuild check.
6. **full suite + commit + queue update**.

## 5. Semantics pinned (each a natural mistake)

- **S1** standardized rows have mean ≈ 0 and population std ≈ 1 (with an explicit
  ddof discriminator).
- **S2** `‖ŵ_c‖₂ = √K` (to 1e-12 — see the ULP finding in the Done entry).
- **S3** standardizing an already-standardized matrix is a **no-op** (idempotence), and
  the non-mutating free function leaves its input untouched.
- **S4** `eps` inside vs outside the sqrt, discriminated by FD and by *direction*.
- **S5** the adjoint matches central differences at two shapes and two `eps`.
- **S7** translation invariance: `dL/dW` sums to zero over a channel; a constant offset
  within a channel is annihilated.
- **S8** per-COLUMN standardization is a different function (the axis discriminator).
- **S10** every guard message is **distinct** and non-trivial; the Lipschitz diagnostics
  agree with the materialized tensor.
- **S13–S16** the wrapper: forward uses standardized weights; `raw_grad` matches FD
  through the whole composition; two backwards against one snapshot **sum** two adjoints;
  `update_weights` moves the RAW weights by exactly `−lr·grad` (descent sign); the inner
  layer is left holding the RAW weights so the transform does not compound.