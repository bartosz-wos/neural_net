# Ordinal Regression (Cumulative Link) — Implementation Plan

> **For Hermes:** use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** ship a cumulative-link ordinal-regression layer (`include/nn/layers/dense/ordinal_regression.{h,cpp}`) that maps a feature vector to a distribution over K *ordered* classes, with mathematically correct gradients and numerically stable class probabilities.

**Architecture:** a `Layer` subclass (like `Dense`) with two parameter tensors — a slope vector `beta` of shape `(1, D)` and a raw cut-point vector `raw_cuts` of shape `(1, K-1)`. The forward pass returns **log class probabilities** of shape `(N, K)`, because the probability-space gradient is provably zero where the true gradient is O(1) (FINDING 2 in the oracle). The backward pass consumes `dL/d(log P)` — exactly what the repo's existing `utils/losses.h` `NLLLoss::backward` already produces — and returns `dL/dx`.

**Tech Stack:** C++17, the repo's `Tensor`, the project's `-O2 -march=native -Wall -Wextra` build, a float64 numpy/scipy oracle in `tests/`, an oracle-mutation harness in `tests/`, and a bash mutation harness in `tests/`.

---

## Why this feature, measured not assumed

```
grep -rinE 'ordinal.regress|cumulative.link|probit|ordered.logit' include/
-> only `gaussian_process.h`'s prose mention of "Bernoulli-probit GPs"
   and `utils/rank_correlation.cpp:8`'s note about not hand-rolling the
   probit function. ZERO IMPLEMENTATION.
```

The repo has ~250 layers and 60+ utility files and **no ordinal model**. What it has instead is a *nominal* classifier plus a calibration layer:

| shipped | what it is | what it cannot do |
|---|---|---|
| `layers/dense/embedding.h` | a lookup table | no ordered-output model |
| `utils/losses.h` `NLLLoss` | NLL against **log-softmax** | assumes K *unordered* classes; the class index carries no order |
| `utils/metrics.h` | accuracy / ECE / Brier | a class-index accuracy treats "2 vs 3" as as wrong as "2 vs 9" |
| `utils/temperature_scaling.h` | one scalar `T` | recalibrates, does not *order* |

**THE COMPOSITION IS THE POINT.** An ordinal model emits a distribution over an *ordered* label set, so its natural readout is the **expected class** `E[Y|x] = sum_k k P(Y=k|x)`, a real number, not an argmax. That is exactly what a Likert-scale response needs, and the repo ships no way to produce it.

## The mathematics, and the four places it is easy to get wrong

**1. THE PROBABILITY OF AN INTERIOR CLASS IS A DIFFERENCE OF TWO CUMULATIVES, AND THE NAIVE DIFFERENCE LOSES ALL OF IT.**
`P(Y=k) = sigma(a_k) - sigma(a_{k-1})` with `a_k = theta_k - eta`. When `eta` is large and negative both sigmoids round to exactly `1.0` and the difference is **exactly zero** — the class is silently assigned probability 0 while its true probability is `2.685e-18`. The stable identity is `sigma(u) - sigma(v) = sigma(u)·sigma(-v)·(1 - e^{-(u-v)})`, whose every factor is O(1). **Measured** on the oracle's F3 fixture: naive `0.0` against a 60-digit truth of `2.6854720659566001e-18`.

**2. THE SAME CANCELLATION SITS INSIDE THE GRADIENT, ONE LEVEL DEEPER — WHICH IS WHY THE LAYER OUTPUTS LOG PROBABILITIES.**
The probability-space gradient carries `sigma(a)·(1 - sigma(a))`, which is **exactly zero** where `sigma(a)` has rounded to 1.0. **Measured**: the probability-space derivative is `0` while the finite difference is `-1` — a 100 % relative error that no care in the probability computation can repair. In log space the factors are `1 - c = sigmoid(-a)` and `-c`, both stable. **The layer's forward returns log P, and its backward consumes `dL/d(log P)`** — which is exactly what the repo's `NLLLoss::backward` already emits, so the two compose with no glue.

**3. THE ORDERING IS STRUCTURAL, AND THE GAPS MUST NOT BE RECOVERED BY DIFFERENCING.**
The cut points are stored as raw parameters `r` with `theta_1 = r_1` and `theta_k = theta_{k-1} + softplus(r_k)`, so no optimiser step can produce `theta_k <= theta_{k-1}` and no class probability can be negative. **But** the cumulative sum can *absorb* a tiny gap: `theta_1 = 100` with a gap of `3.7e-44` gives `theta_2 == theta_1` exactly in float64, so differencing reports a gap of 0. The gap enters the probability **and** the gradient, so the implementation keeps it as its own quantity computed from `r` directly.

**4. THE INTERIOR CLASSES' PROBABILITY DEPENDS ON THE *SPREAD* OF TWO CUT POINTS, NOT ONLY THEIR POSITIONS.**
`log P_k = log sigma(a_k) + log sigma(-a_{k-1}) + log(1 - e^{-gap_{k-1}})` carries an extra term `q_{k-1} = 1/(e^{gap_{k-1}} - 1)`. An implementation that differentiates only the positions gets the cut-point gradient **wrong**. Worse: `q` appears with *opposite signs* in the two theta-partials and cancels, and with a tiny gap `q ≈ 2.7e43`, so the cancellation destroys the answer — the oracle's own first draft returned **exactly 0** where the finite difference returned `-1`. The fix is to differentiate in `(u, gaps)` coordinates, where `q` never has to cancel.

### The three-case `d/dgap` rule (the part that is easy to get wrong)

A gap shift moves **every** cut point above it, so it touches **every** class at or above it — in particular the last class. For an interior class `k`:

| case | contribution |
|---|---|
| `j <= k-2` (both `theta_k` and `theta_{k-1}` shift) | `e_k - c_{k-1}`, **no `q`** |
| `j == k-1` (only `theta_k` shifts, and the gap itself changes) | `e_k + q_{k-1}` |
| `j >= k` | nothing |

An implementation that credits the gap to a single class gets `dL/dr` wrong for every fixture whose observed class is not adjacent to the perturbed gap. **The oracle's second draft had exactly this bug and the finite differences caught it.**

### The semantics that no one expects

**S1. THERE IS NO INTERCEPT.** `P(Y<=k) = sigmoid(theta_k - eta)` is invariant under shifting every `theta` by `c` and `eta` by `c`, so an intercept would be *exactly* redundant with a uniform shift of the cut points. Pinning it to zero removes the degeneracy instead of hiding it.

**S2. `e = 1 - c` IS COMPUTED AS `sigmoid(-a)`, NEVER AS `1.0 - c`.** When `c` has rounded to `1.0`, `1.0 - c` is exactly `0.0` while `sigmoid(-a)` is `4.2e-18`. **Measured**: a fixture whose observed class is the *first* one and whose `a` is large positive has its entire gradient carried by this factor, and the unstable form returns `0` against a true `-4.248e-18`.

**S3. A HIGHER LATENT SCORE MUST MOVE MASS TOWARDS HIGHER CLASSES**, so `P(Y<=k)` must **decrease** in `eta` and the expected class must **rise**. The direction is a measured property of the model, not a convention.

**S4. THE LAYER RETURNS LOG PROBABILITIES, NOT PROBABILITIES.** This is not a stylistic choice — it is FINDING 2. `exp()` of the returned log-probs gives the probabilities back when the caller wants them.

## The shipped API

```cpp
// include/nn/layers/dense/ordinal_regression.h
class OrdinalRegression : public Layer {
public:
    OrdinalRegression(size_t in_features, size_t num_classes);

    // (N, D) features -> (N, K) LOG class probabilities.
    Tensor forward(const Tensor& input) override;

    // grad_output is dL/d(log P), shape (N, K) -- the shape the repo's
    // NLLLoss::backward already produces. Returns dL/d(input), shape (N, D).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return beta_; }
    Tensor get_gradients() const override { return grad_beta_; }
    std::vector<Tensor*> parameters() override { return {&beta_, &raw_cuts_}; }
    std::vector<Tensor*> gradients() override { return {&grad_beta_, &grad_raw_cuts_}; }
    void zero_grad() override;

    size_t num_classes() const { return num_classes_; }
    size_t in_features() const { return in_features_; }

    // The ORDERED cut points actually in force, shape (1, K-1).
    // Strictly increasing by construction.
    Tensor cut_points() const;

    // The expected class E[Y|x] = sum_k k*P(Y=k|x), shape (N, 1).
    // The natural readout of an ordinal model -- a real number, not an argmax.
    Tensor expected_class(const Tensor& input) const;

private:
    size_t in_features_, num_classes_;
    Tensor beta_;            // (1, D)   the slope
    Tensor raw_cuts_;        // (1, K-1) the raw cut-point parameters
    Tensor grad_beta_, grad_raw_cuts_;
    Tensor last_input_;      // cached for backward
};
```

## Tasks

### Task 1: The float64 oracle — the specification

**Objective:** write `tests/ordinal_regression_oracle.py`, the float64 specification, and run it green.

**Status: DONE.** `322 checks passed, 0 failed`. It validates the closed form against `scipy.integrate.quad` on the logistic density (definition-level), against a 60-digit `decimal` reference, and the analytic gradient against central finite differences. It found **four defects in its own first drafts**, all of which changed the specification:
1. the stable closed form was missing the `e^{-v}` factor (caught by the quadrature cross-check);
2. the 60-digit `softplus` had `1 - exp(-t)` where it needed `1 + exp(-t)`, producing **negative probabilities** (caught by the same cross-check);
3. the analytic gradient was written in probability space, where the `c·(1-c)` factor cancels catastrophically (caught by the FD check: analytic `0` against FD `-1`);
4. the `(u, gaps)` reparameterisation credited a gap shift to one class only (caught by the FD check: `dL/dr[1]` analytic `0` against FD `0.3407`).

### Task 2: Measure the oracle's own sensitivity

**Objective:** write `tests/ordinal_regression_mutate_oracle.py` and require that every mutation of the oracle's MODEL code is caught.

**Status: DONE.** `10 caught, 0 survived, 0 broken, 0 noop, 0 anchor`.

The first run reported **2 survivors** (the unstable `1.0 - c` factor) and **1 broken** (a mutation that crashed instead of reporting). Both were holes in the *specification*, not in the implementation, and the fix was **two new fixtures** — F5 (observed class is the first one, so the whole gradient is carried by `1 - c`) and F6 (four classes, so an interior class's bulk coefficient is) — plus a guard so a degenerate gap reports a failure instead of crashing.

### Task 3: The RED C++ test

**Objective:** write `tests/test_ordinal_regression.cpp` against the oracle's emitted fixture block, and watch it fail.

### Task 4: The implementation

**Objective:** write `include/nn/layers/dense/ordinal_regression.{h,cpp}` to make Task 3 pass.

### Task 5: Wire it into the public surface and the test target

**Objective:** add the include to `include/nn/nn.h`, add the test target to the `Makefile`, and verify the umbrella header compiles standalone.

### Task 6: The C++ mutation harness

**Objective:** write `tests/mutate_ordinal_regression.sh` proving the C++ suite catches the defects it was written for.

### Task 7: The aggregate gates

**Objective:** full suite, ASan+UBSan, zero warnings at the project's real `-O2 -march=native -Wall -Wextra`, umbrella header compiles standalone.

### Task 8: Record the work

**Objective:** move the queue item to `## Done`, commit, push.

## References

- McCullagh, P. — "Regression analysis of ordinal data", *JRSS-B* 42(2):109-142, 1980. The cumulative link model.
- Ordinal logistic regression: the `q` term's role and the `(u, gaps)` reparameterisation are discussed in Agresti, *An Introduction to Categorical Data Analysis*, ch. 6.
- **NO arXiv ID IS ASSERTED** — the 1980 paper predates arXiv's coverage of statistics journals, and this repo's expansion queue has produced many wrong arXiv identifiers.
