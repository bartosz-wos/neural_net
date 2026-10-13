# Temperature Scaling — Guo, Pleiss, Sun, Weinberger (arXiv:1706.04599)

> **For Hermes:** implement task-by-task with strict TDD (RED → GREEN).
> Oracle values come from **scipy 1.18.1** (`scipy.special.logsumexp`,
> `scipy.optimize.minimize_scalar`) and the paper's own reference
> implementation (`gpleiss/temperature_scaling`), both read on 2026-10-13,
> never from memory.

**Goal:** Ship the standard post-hoc calibration fix as a first-class,
measurable utility — fit a single temperature `T` on held-out logits against
validation NLL, then re-emit calibrated probabilities.

**Architecture:** New files `include/nn/utils/temperature_scaling.{h,cpp}`.
Temperature scaling is a *post-processing* step over logits, not a layer and
not a metric, so it does not belong in `metrics.{h,cpp}` (which scores a fixed
prediction set) or in the loss files (which are training objectives). It is
model-free: it consumes `(N, K)` logits and `(N, 1)` integer labels and returns
`(N, K)` probabilities. This makes it directly measurable with the
already-shipped `expected_calibration_error` / `classwise_ece` — which is
precisely why the queue deferred it until today.

**Tech Stack:** C++17, repo `Tensor` (row-major, `operator()(i,j)`), no new
dependencies. Golden-section line search over `log T` (see "Why a line search"
below — the objective is **not** convex in `T`).

---

## Citation correction (read the sources before implementing)

**The arXiv ID recorded in `EXPANSION_QUEUE.md` is WRONG.** The queue says
`arXiv:1707.06552`; the arXiv API returns *"Towards a scientific blockchain
framework for reproducible data analysis"* for that ID (verified 2026-10-13).
The correct ID is **arXiv:1706.04599**, *"On Calibration of Modern Neural
Networks"*, Guo, Pleiss, Sun & Weinberger, ICML 2017. Do not re-introduce
`1707.06552`.

---

## Primary-source semantics (each one a natural mistake)

Read from the paper (ar5iv HTML, fetched 2026-10-13) and the authors'
reference implementation (`gpleiss/temperature_scaling`, fetched 2026-10-13).

1. **D1 — the forward is `softmax(z / T)`, i.e. DIVIDE.** Paper Eq. 9:
   `q̂ᵢ = max_k σ_SM(zᵢ/T)^(k)`. The Torch7 implementation makes it explicit:
   "we implement temperature scaling by inserting a `nn.MulConstant` between
   the logits and the softmax, whose parameter is `1/T`" (`:457`). A `Layer`
   elsewhere in this repo (`hopfield.h`) stores an **inverse** temperature
   `beta` — do not conflate the two conventions; `temperature_scaling` exposes
   `T` itself, the paper's parameterisation.
2. **D2 — `T > 0` is a hard precondition, and `T=1` is the identity.** Paper:
   "`T` softens the softmax (i.e. raises the output entropy) with `T > 1`… With
   `T = 1`, we recover the original probability `p̂ᵢ`" (`:392-393`).
3. **D3 — the class prediction CANNOT change.** Paper: "Because the parameter
   `T` does not change the maximum of the softmax function, the class
   prediction `ŷ'ᵢ` remains unchanged… temperature scaling does not affect the
   model's accuracy" (`:396`), confirmed in Table S2. Dividing by a positive
   scalar preserves argmax **exactly** — pin this on a fixture where the
   argmax margins are tiny, since a wrong sign (`z*T`) breaks it only for
   `T<1`.
4. **D4 — the limits are one-sided and both are degenerate.** `T → ∞`:
   `q̂ᵢ → 1/K` ("maximum uncertainty"); `T → 0`: `q̂ᵢ → 1` (point mass)
   (`:393-394`). This matters for the *fit*: on a perfectly separable
   validation set the NLL minimiser runs to `T → 0` (oracle measured
   `T* = 6.9e-4`, `NLL = 0.0`), so the search must be bracketed, not unbounded.
5. **D5 — the objective is NLL on a HELD-OUT set** (`:395`: "`T` is optimized
   with respect to NLL on the validation set"), using the **mean** over the
   batch. `log_loss` in `metrics.h` is a different reduction (it is
   sklearn's clip-based form) and must not be reused as the fit objective.
6. **D6 — the fit is a 1-D problem, and the paper says so.** ":453": "it
   amounts to a one-dimensional convex optimization problem. Using a
   conjugate gradient solver, the optimal temperature can be found in 10
   iterations… even a naive line-search for the optimal temperature is faster
   than any of the other methods." **The paper's word "convex" is not
   verifiable** — see the measurement below. The *recommendation* (line search)
   is sound regardless.

### The measured fact that overrides the paper's "convex" claim

scipy, five fixtures, 3000-point grid:

| fixture | unimodal in `T` | convex in `T` | convex in `log T` | convex in `1/T` |
|---|---|---|---|---|
| 4-row overconfident | yes | **no** | **no** | **no** |
| 4-row underconfident | yes | **no** | **no** | **no** |
| random `K=2`, `n=30` | yes | yes | yes | yes |
| random `K=7`, `n=50` | yes | yes | yes | yes |
| single row `K=3` | yes | **no** | **no** | **no** |

**`L(T)` is unimodal but not convex in any of the three natural
reparameterisations.** A gradient descent or Newton step on `T` is therefore
not justified; the implementation must be a **bracketed line search
(golden-section) over `log T`**, which needs only unimodality. Searching in
`log T` (not `T`) keeps the bracket two-sided over `T ∈ (0, ∞)` — a symmetric
bracket in `T` cannot represent both a `T*` of 0.03 and one of 11.7.

*(Two grid "sign changes" were diagnostics artefacts, not extra modes: one was
a single sample straddling the minimum with `|dL| ~ 1e-7`, the other the
minimum sitting in the first grid cell. The implementation's own termination
test must not rely on a sign-change count for the same reason.)*

---

## Why `T > 1` needs confident ERRORS — a fixture trap already found once

At `T = 1`, `p` is peaked on the argmax, so `Σ_j p_j z_j ≈ z_argmax` and

```
dL/dT ≈ (1/(n T²)) Σ_i [ z_{i,y_i} − z_{i,argmax_i} ]
```

A **correct** row contributes ≈ 0 (its label *is* the argmax); an **error**
contributes `−gap`. So `T` rises above 1 only when confident errors outweigh
correct rows. **A fixture whose label equals the argmax on every row is
perfectly separable, and the fit will always run `T → 0` with `NLL → 0`** —
the oracle measured `T* = 0.0675` on exactly that mistake. Every fixture that
asserts `T > 1` must contain errors with a large argmax-vs-true gap.

---

## Derivation of the gradient (verified against scipy centered FD)

`L(T) = (1/n) Σ_i [ lse(z_i/T) − z_{i,y_i}/T ]`, `p = softmax(z/T)`:

```
d/dT lse(z/T) = −(1/T²) Σ_j p_j z_j        d/dT (z_y/T) = −z_y/T²

=>  dL/dT = (1/(n T²)) Σ_i [ z_{i,y_i} − Σ_j p_j(i) z_j(i) ]
```

scipy centered FD (`eps=1e-6`) agrees to **rel ≤ 1.6e-9** on five fixtures.

**The sign is the thing that gets missed.** The first version of the oracle had
the bracket reversed and disagreed with FD by exactly `rel = 2.000` on every
fixture — a clean fingerprint, and the reason the FD check is not optional.

---

## Public API

```cpp
// Fit T on held-out logits against mean NLL, by golden-section line search
// over log T. Returns T; nll_before/nll_after are optional out-params.
double fit_temperature(const Tensor& logits, const Tensor& labels,
                       double t_min = 1e-2, double t_max = 1e2,
                       int max_iter = 200, size_t tol_iters = 0);

// Calibration NLL of softmax(logits/T) against `labels` (mean over rows).
double temperature_nll(const Tensor& logits, const Tensor& labels, double T);

// softmax(logits / T), (N,K) -> (N,K). T must be > 0.
Tensor temperature_scale(const Tensor& logits, double T);

// Fit T then apply it. Throws std::invalid_argument on shape/label mismatch.
Tensor calibrate_logits(const Tensor& logits, const Tensor& labels, double T);
```

`fit_temperature` returns the **best `T` seen**, not the bracket midpoint, and
returns the midpoint of the final bracket when the bracket has not converged
(so a separable fixture returns `t_min`, a documented clamp — never a value
outside `[t_min, t_max]`).

---

## Tasks

### Task 1: RED — the test file, before any implementation

**Files:** Create `tests/test_temperature_scaling.cpp`

Oracle values (scipy, 2026-10-13) to encode as expectations:

| fixture | value |
|---|---|
| `FIX_A` 3×2 logits `[[2,0],[0,2],[1,1]]`, labels `[0,1,0]`, `T=1` | NLL `0.315667734215` |
| `FIX_A` at `T=2` | NLL `0.439890185199` |
| `FIX_A` at `T=1` dL/dT | `-0.158937229363` |
| `FIX_B` 2×3, `T=0.5 / 1 / 3` | NLL `0.058127734181 / 0.260898941680 / 0.703956193093` |
| `FIX_B` dL/dT | `-0.339406654658 / -0.398332762128 / -0.110241561665` |
| `FIX_C2` 4×3 (see below) fitted `T*` | `11.695103496360` |
| `FIX_C2` NLL `1 → *` | `2.518628011767 → 1.053981950411` |
| `FIX_E` 4×3 (underconfident) fitted `T*` | `0.036395141796` |
| separable 3×3, labels = argmax | `T*` runs to the low clamp, NLL → 0 |

`FIX_C2 = logits [[3,0.5,0],[0.5,3,0],[9,-8,0.5],[4,0,3]], labels [0,1,2,2]`
(argmax `[0,1,0,0]` → **2 errors**, gaps 8.5 and 1.0).

Write 11 test functions following `tests/test_calibration_diagnostics.cpp`'s
exact conventions (`check(name, pass)`, `close(a,b,tol)`, `std::fixed <<
std::setprecision(10)`, `=== Result: N passed, M failed ===`, return
`failed == 0 ? 0 : 1`):

1. `test_validation` — `T <= 0` throws (both `0.0` and `-1.0`);
   `logits.rows == 0` throws; `labels.cols != 1` throws;
   `logits.cols != labels.rows` throws; out-of-range label throws;
   non-integer label throws; `t_min <= 0 || t_max <= t_min` throws.
2. `test_identity_at_one` — `temperature_scale(z, 1.0)` is bit-equal to the
   shipped `softmax` (compute softmax inline from `log_softmax` in
   `losses.h`) to 1e-15, on a 4×3 fixture.
3. `test_divides_not_multiplies` — **the headline direction test.** On
   `FIX_A` at `T=2`, assert `temperature_scale(z,2)` equals the oracle
   probabilities `[[0.73105858,0.26894142],[0.26894142,0.73105858],[0.5,0.5]]`
   and that multiplying by T gives a **different** number (so the test cannot
   pass if the op is inverted).
4. `test_argmax_invariance` — for `T ∈ {0.01, 0.5, 1, 7, 1000}`, argmax of
   `temperature_scale(z,T)` equals argmax of `softmax(z)` on a fixture whose
   margins are small (`0.03`, `0.02`) — the case a wrong sign breaks.
5. `test_temperature_limits` — `T=1e-8` → max prob `1.0` (≤1e-6);
   `T=1e8` → every row within 1e-6 of `1/K`.
6. `test_binary_matches_sigmoid` — for `K=2`, `p[1] == 1/(1+exp((z0-z1)/T))`
   to 1e-12. *(This exact identity is the one the oracle got wrong twice —
   pin it.)*
7. `test_nll_matches_oracle` — `FIX_A` at `T=1` and `T=2`, `FIX_B` at
   `T=0.5/1/3`, to 1e-12.
8. `test_fit_finds_oracle_temperature` — `fit_temperature(FIX_C2)` returns
   `11.695103496360` to `1e-5` relative; NLL strictly decreases.
9. `test_fit_handles_separable_and_underconfident` — separable fixture
   returns the low clamp with NLL ≈ 0; `FIX_E` returns `0.036395141796` to
   `1e-4` relative.
10. `test_gradient_matches_fd` — centered FD `(L(T+e)−L(T−e))/(2e)` at
    `e=1e-6` vs the closed form on 3 fixtures, to 1e-6 relative. The closed
    form is asserted **in the test** (independent of the impl) so the test
    cannot pass by comparing the impl against itself.
11. `test_fit_improves_shipped_ece` — **the integration test.** Build a
    deliberately overconfident `(N,K)` logit set, compute
    `expected_calibration_error(y_true, p1, 15, 1.0)` before and after
    `calibrate_logits`, and assert ECE **decreases** and NLL decreases. This
    is what the feature is *for*.

**Step 1:** write the file. **Step 2:** `make build/test_temperature_scaling`
→ must FAIL to compile (`temperature_scaling.h` not found) — that is the RED.

### Task 2: GREEN — `temperature_scale` + `temperature_nll`

**Files:** Create `include/nn/utils/temperature_scaling.h`, `.cpp`

Only what tests 1-7 need. `temperature_nll` reuses `log_softmax` from
`utils/losses.h` (already public, already max-subtracted) — do **not** re-derive
it; a second copy is a drift risk.

**Step 3:** implement. **Step 4:** `make build/test_temperature_scaling &&
./build/test_temperature_scaling` → tests 1-7 pass.

### Task 3: GREEN — `fit_temperature`

**Files:** modify `.cpp` (add the line search + header decl)

Golden-section on `log T` over `[log t_min, log t_max]`, evaluating `L` and
`dL/dT` at each probe. Terminate when the bracket is below `tol` or
`max_iter` is hit; **return the best `L` seen**, and clamp into
`[t_min, t_max]`. Do not assume convexity (measured false above); do not stop
on a sign-change count (see the grid artefact).

**Step 5:** implement. **Step 6:** run → tests 8-9 pass.

### Task 4: GREEN — `calibrate_logits` + integration test

**Files:** modify `.h`/`.cpp`

`calibrate_logits` = `fit_temperature` then `temperature_scale`.

**Step 7:** implement. **Step 8:** run → all 11 tests pass.

### Task 5: wire up

**Files:** `include/nn/nn.h` (add `#include "utils/temperature_scaling.h"`),
`Makefile` (link rule + `tests:` prerequisite + a `run_tests` line)

**Step 9:** verify the umbrella header compiles **standalone** (the
umbrella-only check that focused tests cannot do):
```
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```
**Step 10:** `make run_tests` → confirm no new failures vs the baseline.

### Task 6: mutation harness + docs

**Files:** Create `tests/mutate_temperature_scaling.sh`

Defences that are now **mandatory** (each was a real bug in a prior session):
1. Take the baseline **once, before any mutation**, and **assert it is
   pristine** by grepping lines that exist only in the correct impl
   (`z / T`, `z_{i,y_i} − Σ_j p_j z_j`, the golden-section constant).
2. **Verify each `sed` applied** with `diff -q` before building; report
   `MUT-NOOP` separately.
3. **Reject any run whose binary is older than the mutated source**
   (an absolute `make` target silently reports "up to date" and never
   rebuilds — this produced 7 false `MUT-SURVIVED` in the metrics session).
4. Compile failure is `MUT-BROKEN`, never `MUT-CAUGHT`.

Mutations: M1 `/T` → `*T`; M2 drop the `1/(n T²)` factor; M3 flip the
gradient bracket's sign; M4 drop `Σ_j p_j z_j`; M5 golden-section → return
`T=1`; M6 use `sum` instead of `mean` in the NLL; M7 ignore `t_min/t_max`
clamping.

Then: `EXPANSION_QUEUE.md` → move the item to `## Done`; add a
`docs/plans/` pointer; commit.