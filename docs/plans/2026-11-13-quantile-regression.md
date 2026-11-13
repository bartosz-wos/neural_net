# Quantile Regression (Linear, Multi-Level) — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship a linear quantile-regression layer (`include/nn/layers/dense/quantile_regression.{h,cpp}`) that maps a feature vector to the **K conditional quantiles** of a scalar response at K prescribed levels, trained by the pinball (quantile) loss, with the crossing property measured rather than assumed.

**Architecture:** a `Layer` subclass (like `Dense` and the shipped `OrdinalRegression`) with two parameter tensors — a slope matrix `weights` of shape `(K, D)` (one row per quantile level) and an intercept row `bias` of shape `(1, K)`. The forward pass returns `(N, K)` quantile predictions. The layer owns the **sum over levels of the per-level mean pinball loss** (the classical Koenker–Bassett objective) and therefore its own `pinball_loss` / `pinball_gradient` pair, because the loss depends on the *level vector* — which is a property of the model, not of the caller.

**Tech Stack:** C++17, the repo's `Tensor`, the project's `-O2 -march=native -Wall -Wextra` build, a float64 numpy+scipy oracle in `tests/`, an oracle-mutation harness in `tests/`, and a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

```
grep -rinE 'quantile.regress|pinball|quantile.?head|non.?crossing' include/ tests/
-> include/nn/utils/conformal.h:11,25,74   (the CQR paper and the NO-CONSUMER note)
-> include/nn/utils/isotonic_regression.h:134 (quantile regression named OUT OF SCOPE)
-> include/nn/utils/distribution_losses.h:106 (the `QuantileLoss` class itself)
-> tests/test_distribution_losses.cpp       (its own unit tests)
```

**ZERO IMPLEMENTATION** of a quantile *regression* — nothing anywhere produces a quantile prediction.

Two shipped files state the gap in their own words, and they are the strongest possible evidence because they are *about this repo*:

| shipped | what it says |
|---|---|
| `utils/conformal.h:74` | "`utils/distribution_losses.h` ships `QuantileLoss` (the pinball loss). That is the TRAINING LOSS CQR needs — and it is the strongest evidence that CQR belongs here: **a shipped primitive with NO CONSUMER**." |
| `utils/isotonic_regression.h:134` | "QUANTILE REGRESSION / CQR-style monotone quantile fitting. A different objective (pinball loss, not squared loss)." |

And the consumer is already shipped and already unusable:

```cpp
// utils/conformal.h:259
Tensor cqr_score(const Tensor& lower, const Tensor& upper, const Tensor& y);
```

`cqr_score` needs a **lower** and an **upper** quantile prediction. The repo ships the conformal machinery that consumes them and **no producer at all**.

### It is not redundant with the shipped neighbours

- `utils/distribution_losses.h` `QuantileLoss` is a **loss**, and it takes **one** `q` for the whole `(N, K)` tensor — it cannot express "column 0 is the 0.05-quantile, column 1 is the 0.95-quantile". It is the *objective*, not the *model*.
- `utils/isotonic_regression.h` fits a monotone map on a **single** response; it is a *post-hoc calibration* of a score, not a conditional quantile model, and its header explicitly excludes the pinball objective.
- `utils/conformal.h` is **post-hoc**: it calibrates a prediction that was already made. A quantile regression produces the distribution *from the data*.

### The composition is the point

The layer's output composes with the shipped conformal machinery with **zero glue code**:

```
layer.forward(X)          -> (N, K) quantile predictions, column k at level tau_k
layer.monotone(preds)     -> (N, K) with each row sorted ascending (no crossing)
cqr_score(lower, upper, y) -> (N, 1) conformalised quantile-regression score
cqr_interval(l, u, qhat)  -> (1, 2) the prediction interval
```

---

## The mathematics, and the places it is easy to get wrong

**1. THE PINBALL LOSS AND THE TIE CONVENTION.** For level `tau` and residual `e = y - pred`,

```
rho_tau(e) = e * tau          if e >= 0     (under-prediction)
           = e * (tau - 1)    if e <  0     (over-prediction)
dL/dpred   = -tau/N           if e >= 0
           = (1 - tau)/N      if e <  0
```

`rho_tau` is **not differentiable at `e = 0`**, so the subgradient there is a *set*. The repo already has a convention — `utils/distribution_losses.h` `QuantileLoss::backward` uses the **under-prediction branch at `e == 0`** — and the layer **must match it exactly**, or the layer and the shipped loss disagree on the same data. That equality is pinned as an exact cross-class check (K=1 layer vs `QuantileLoss`), which is why the convention is not a free choice here.

**2. THE REDUCTION IS "MEAN OVER THE BATCH, SUM OVER THE LEVELS", NOT "MEAN OVER EVERYTHING".** The shipped `QuantileLoss` divides by `N` (rows) and sums the columns. The layer must use the same reduction, or the K=1 equality does not hold. The consequence is documented: **the total loss scales with K**, so the effective learning rate does too.

**3. THE `K = 1` CASE MUST BE BIT-FOR-BIT THE SHIPPED `QuantileLoss`.** This is the strongest available composition check and it pins the loss, the gradient, the tie convention and the reduction *simultaneously*. If any of the four drifts, the cross-class equality fails.

**4. THE FITTED QUANTILE FUNCTIONS CAN CROSS, AND THE LAYER MEASURES IT.** The K levels are fitted **independently** (the classical Koenker–Bassett model: each level's pinball loss depends only on its own column), so nothing in the model forces `q_{tau_1}(x) <= q_{tau_2}(x)`. That is a property of the *estimator*, not a bug — and it is exactly what breaks a conformal interval (`lower > upper` yields a negative-width interval). The layer therefore ships a **crossing diagnostic** (`crossing_count`) and an explicit **pointwise monotone rearrangement** (`monotone`), and the header states plainly which one is the model's output and which is a projection.

**5. THE INTERCEPT-ONLY FIT RECOVERS THE SAMPLE QUANTILE, IN CLOSED FORM.** With `X = 0`, the prediction is the intercept alone and the minimiser of the empirical pinball loss is the inverse-ECDF quantile `y_(ceil(N*tau))` (1-indexed). This is a **definition-level** target: the trained intercept is checked against a quantity computed from the sample, not against a re-run of the implementation. It is also why the training fixture uses `N = 10` and levels whose `N*tau` is **not** an integer: when `N*tau` is an integer the minimiser is an *interval* of values and the subgradient convention picks an endpoint, so the target would be ambiguous.

**6. THE SUBGRADIENT CONVENTION SELECTS THE LEFT ENDPOINT OF THE MINIMISER INTERVAL.** At a kink the layer's convention treats `y_i == pred` as the `y_i >= pred` side, so the fixed-point condition is `#{y_i < pred} = N*tau`; when `N*tau` is an integer that pins `pred` to the *left* end of the interval `[y_(m), y_(m+1)]`. The oracle measures the oscillation amplitude so the test's tolerance is derived rather than guessed.

---

## The shipped API

```cpp
// include/nn/layers/dense/quantile_regression.h
class QuantileRegression : public Layer {
public:
    // in_features D >= 1; `quantiles` are K levels in (0, 1), STRICTLY INCREASING.
    QuantileRegression(size_t in_features, std::vector<double> quantiles);

    // (N, D) features -> (N, K) quantile predictions, column k at level tau_k.
    Tensor forward(const Tensor& input) override;

    // grad_output is dL/d(pred), shape (N, K) -- exactly what
    // `pinball_gradient` produces. Accumulates into the parameter gradients
    // and returns dL/d(input), shape (N, D).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;

    Tensor get_weights() const override;     // (K, D)
    Tensor get_gradients() const override;   // (K, D)
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "QuantileRegression"; }

    const std::vector<double>& quantiles() const;   // the K levels, ascending
    size_t num_levels() const;
    size_t in_features() const;

    // The pinball objective, at the layer's OWN levels.
    //   L = (1/N) * sum_i sum_k rho_{tau_k}(y_i - pred_ik)
    Tensor pinball_loss(const Tensor& pred, const Tensor& y) const;       // (1, 1)
    Tensor pinball_gradient(const Tensor& pred, const Tensor& y) const;   // (N, K)

    // The pointwise monotone rearrangement: each row of `pred` sorted
    // ascending. A projection, NOT the model's output.
    Tensor monotone(const Tensor& pred) const;

    // The number of ADJACENT level pairs (k, k+1) whose predictions are
    // inverted, summed over the rows. Zero iff every row is non-decreasing.
    size_t crossing_count(const Tensor& pred) const;

    // Column k of `pred` as (N, 1) -- the shape CQR consumes.
    Tensor level_column(const Tensor& pred, size_t k) const;
};
```

`y` is `(N, 1)` — **one response per row, compared against all K levels**. This differs from the shipped `QuantileLoss`, which takes an elementwise `(N, K)` target; the header says so loudly, because mixing the two up silently produces garbage.

---

## Tasks

### Task 1: The float64 oracle — the specification

**Objective:** write `tests/quantile_regression_oracle.py` and run it green.

**Status: DONE — 3215 passed, 0 failed.** It validates `rho_tau` against
`scipy.integrate.quad` on the subgradient integral (definition-level), the
gradient against central finite differences, the intercept-only minimiser
against the closed-form inverse-ECDF quantile *and* against a brute-force grid
search, and every layer fixture against finite differences of the loss. It
prints the C++ fixture constants.

**It caught FOUR defects in its own first drafts, all four in the
specification:** the quadrature's orientation was reversed for negative
residuals (every negative-`u` check failed by a factor of exactly `-1`, which
identified the orientation rather than a bad definition); the closed form was
compared TWO-SIDEDLY against a dense grid, but the loss is piecewise linear
with kinks at the sample values, so a finite grid cannot land on one and the
comparison had to become ONE-SIDED; convergence was measured over the WHOLE
trajectory, which is dominated by the travel from an initial intercept of `0`
to targets as large as `9.5`, so it is now measured over the trajectory TAIL;
and the gradient section used `np.outer`, which produces an `(N, N)` matrix
instead of the required `(D,)` vector.

**A non-converging configuration is recorded rather than hidden:** `lr=0.05`
over 400 steps has a measured tail deviation of `1.91` and genuinely does not
converge, so the oracle ASSERTS that it does not, and the shipped fixture
uses `lr=0.5` over 200 steps.

**The crossing hunt uses SciPy, because the obvious approach was infeasible:**
the first draft's vectorised NumPy subgradient loop needed roughly 160
million NumPy calls and did not finish; it was killed. `Nelder-Mead` produced
the same fixtures in `0.086` seconds.

### Task 2: Measure the oracle's own sensitivity

**Objective:** write `tests/quantile_regression_mutate_oracle.py`, which breaks the oracle's model code in N specific ways and requires that it FAIL each time.

**Status: DONE.** A green oracle proves nothing about its own sensitivity
(§5f-bis of the debugging skill). Fifteen mutations, each targeting a
specific decision — including the quadrature orientation, the one-sided grid
comparison, the inverse-ECDF index, the level decoupling, and the sign of the
training simulation's update. The oracle's OWN first-draft defects are among
the mutations, so the harness proves it would have caught them.

**The FIRST run reported 1 survivor and 1 broken, and BOTH were defects in
the harness rather than in the oracle:** the "level decoupling" mutation
DELETED A CHECK instead of breaking a value, so the oracle could only get
quieter — a mutation that removes the thing that would catch it measures
nothing; and the two-sided-grid mutation had a FORMAT-STRING ARITY BUG, so
the mutated oracle CRASHED (rc=1 with zero reported failures) instead of
failing. Final: **15 caught / 0 survived / 0 broken / 0 noop / 0 anchor.**

### Task 3: The RED C++ test

**Objective:** write `tests/test_quantile_regression.cpp` against the oracle's fixtures, and watch it fail.

**Status: DONE — 173 passed, 0 failed.** Written against the oracle's
generated fixture constants, so the expectations and the points are pinned
together (an RNG is a stream; two draws are two fixtures).

### Task 4: The implementation

**Objective:** write `include/nn/layers/dense/quantile_regression.{h,cpp}`.

**Status: DONE.** `weights` is `(K, D)`, `bias` is `(1, K)`, the forward
output is `(N, K)`, and the objective is the sum over levels of each level's
mean pinball loss. The tie convention matches the shipped `QuantileLoss`
exactly (`e >= 0` uses `tau`), and the suite asserts that a `K = 1` layer is
BIT-FOR-BIT equal to it.

### Task 5: Wire it into the public surface and the test target

**Objective:** add the include to `include/nn/nn.h`, the build rule and the `tests:` / `run_tests:` lines to the `Makefile`.

**Status: DONE.** The umbrella header compiles standalone
(`g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'`).

### Task 6: The C++ mutation harness

**Objective:** write `tests/mutate_quantile_regression.sh`, which breaks the implementation in N specific ways and requires that the suite catch each one.

**Status: DONE — 16 caught / 0 survived / 0 broken / 0 crashed / 0 noop / 0 anchor.**

**Its FIRST run reported 4 survivors, and every one was a real coverage gap:**
the WEIGHT gradient's accumulation contract was untested (only the BIAS
gradient was checked); the input-width guard was tested by asserting "it
threw", so removing the guard left the suite GREEN because the downstream
`Tensor::operator*` dimension check throws INSTEAD (the discriminator is the
MESSAGE, not the throw); and the initialisation check used an ALL-ZERO INPUT,
so the predictions were zero for ANY weights and the assertion was VACUOUS.
All three were fixed.

### Task 7: The aggregate gates

**Objective:** the full `make run_tests` sweep, an ASan+UBSan build of the new suite, and a clean-tree re-measurement of the one pre-existing failure.

**Status: DONE.** The full sweep ran every registered suite: the only failure
is `test_neural_ode`, which was re-measured IDENTICALLY on a STASHED
pre-work tree (47 passed / 1 failed, the same `ODERNN dense1.W grad vs FD
(line 561)`) — the long-standing entry in `NOT_FIXED.md`, not a regression.
The ASan+UBSan build of the new suite is CLEAN.

### Task 8: Record the work

**Objective:** move the item to `## Done` in `EXPANSION_QUEUE.md` with the measured results.

**Status: DONE.**

---

## References

- Roger Koenker and Gilbert Bassett, Jr., "Regression Quantiles", *Econometrica* **46**(1):33–50, 1978. **Verified 2026-11-13 by title search** against the Econometric Society's own index page (`econometricsociety.org/publications/econometrica/1978/01/01/regression-quantiles`), which gives the volume, issue and page range verbatim. **No arXiv ID is asserted**: the paper predates arXiv by three decades and is not on it.
- The pinball loss's identity as an integral of its own subgradient, `rho_tau(u) = integral_0^u (tau - 1{v < 0}) dv`, is the definition-level reference the oracle checks against `scipy.integrate.quad`.
