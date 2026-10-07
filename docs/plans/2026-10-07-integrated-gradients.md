# Integrated Gradients + Input Attribution Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Open the **interpretability** category — the one major ML capability entirely absent from this repo — by implementing Integrated Gradients (Sundararajan et al., ICML 2017) as a from-scratch, paper-exact attribution engine, plus the gradient-only saliency baseline it is compared against.

**Architecture:** New category `include/nn/interpretability/` holding `attribution.{h,cpp}`. Two free functions (`input_gradient`, `integrated_gradients`) plus a small `IGAttribution` result struct. `IntegratedGradients` needs **no new infrastructure** because this repo's `Layer::backward()` already *returns* dL/d(input) and `Model::backward()` propagates it to the model input — verified in `include/nn/core/layer.cpp:62-84` and `include/nn/core/model.cpp:30-36`. Registration in `include/nn/nn.h` + `tests/test_integrated_gradients.cpp` + Makefile wiring.

**Tech Stack:** C++17, hand-rolled `Tensor` (2-D real, row-major), `Layer` interface, `Model` forward/backward, `Dense`, naive O(m) loop over Riemann steps (repo convention).

---

## Why this feature, and why this order

The repo has ~250 layer implementations, 38 optimizers, and 60+ utils files, but **zero interpretability code**. Content greps (not filename greps — per the queue's own warning) confirm the whole category is missing:

```
grep -ril 'GradCAM\|IntegratedGrad\|saliency\|Shapley\|feature_importance' include/   → no hits
```

The one existing lookalike, `attention/layer_output_tracker.h`, is a *training-time activation-statistics logger* — it tracks min/max/mean/std and vanishing-exploding warnings. It computes no attribution and shares no code with what follows. It is **not** a substitute and is not modified.

**Why IG and not Grad-CAM first — a hard architectural constraint, verified not assumed:**

`Grad-CAM` (the other obvious entry point) computes `α_k = mean_{i,j}( ∂y^c/∂A^k_ij )` — gradients of the target score w.r.t. an *intermediate* layer's activations. This repo's `Layer` interface does not retain input gradients: `backward()` returns them, and nothing caches them per-layer. Capturing them would require either changing the base `Layer` virtual interface (a repo-wide ABI change touching every one of the ~250 shipped layers) or a forward/backward hook system that does not exist. Both are large, risky refactors.

`IntegratedGradients` needs **only** dL/d(input), which `Model::backward()` already hands back. It therefore ships **today** with zero infrastructure risk, and it is the stronger choice on the merits: it is the axiomatic method (completeness is a *provable* axiom, not a heuristic), and unlike Grad-CAM it is architecture-agnostic — it works on MLPs and CNNs alike, which matches this repo's 2-D `Tensor` layout.

Grad-CAM is **not** discarded. Its exact spec was fully researched this session (see "Deferred: Grad-CAM" at the end) and the blocker is recorded, so it can ship later behind a hook API rather than being re-researched.

---

## Source of truth (verified against primary sources, not from memory)

Sundararajan, Taly, Yan, *"Axiomatic Attribution for Deep Networks"*, ICML 2017, **arXiv:1703.01365**.

**The definition (Eq. 3 of the paper, using the paper's notation):**

```
IG_i(x) = (x_i − x'_i) · ∫₀¹ (∂F(x' + α(x − x')) / ∂x_i − x'_i) dα
```

Broken down, as implemented:

| Piece | Implementation |
|---|---|
| baseline `x'` | caller-supplied `Tensor` (defaults to zeros) |
| path | straight line from `x'` at α=0 to `x` at α=1 |
| point on path | `x(α) = x' + α(x − x')` |
| integrand | `∇_x F(x(α))` at that point |
| outer factor | `(x − x')` elementwise, applied **after** averaging the m gradients |

**Completeness axiom:** `F(x) − F(x') = Σᵢ IGᵢ(x)`. This is the definitional property and becomes the primary test (Task 6).

### The Riemann-sum convention — verified from the captum reference implementation, not guessed

I read `captum/attr/_utils/approximation_methods.py` and `integrated_gradients.py` directly. The convention that actually matters, and that a from-scratch implementation will otherwise get wrong:

**Captum's `method` default is `"gausslegendre"`, NOT a Riemann sum.** `n_steps` defaults to **50**.

For the Riemann variants, captum's `step_sizes` and `alphas` are:

| method | step_sizes | alphas |
|---|---|---|
| `riemann_left` | `[1/n]·n` | `linspace(0, 1 − 1/n, n)` |
| `riemann_right` | `[1/n]·n` | `linspace(1/n, 1, n)` |
| `riemann_middle` | `[1/n]·n` | `linspace(1/(2n), 1 − 1/(2n), n)` |
| `riemann_trapezoid` | `[1/n]·n` with first & last halved | `linspace(0, 1, n)` |

**The aggregation order in captum is: `total_grad = Σ_k step_sizes[k] · grad_k`, THEN `× (input − baseline)`.** Confirmed at `integrated_gradients.py:385-405`: gradients are scaled by step size, summed, and only then multiplied by `(input − baseline)`. This ordering matters — summing `(x−x')·grad_k` instead gives a different number whenever `step_sizes` are unequal (i.e. trapezoid).

**Design decision (D1):** this plan ships `riemann_middle` as the default, with all four Riemann variants selectable. Rationale: middle is the highest-accuracy of the four for smooth integrands, and it is the variant whose step size is unambiguous. Gauss-Legendre is **out of scope for v1** (see Out of Scope) — it is captum's default but needs Legendre nodes/weights that add real complexity, and Riemann-middle is the correct foundation.

### Repo constraints that shape the implementation

**C1 — `Dense::backward` accumulates into parameter gradients.** `include/nn/core/layer.cpp:79-80` does `grad_weights += grad_w; grad_bias += grad_b;`. IG calls `backward()` **m times** per input (m=50 by default). Without care, that leaves 50 stacked copies of the parameter gradient inside the model, which corrupts any subsequent training step and inflates the reported gradient norm.

> **This is a real trap, not a hypothetical.** `IntegratedGradients` must therefore either (a) zero each layer's gradients before/after the attribution pass, or (b) document loudly that attribution perturbs gradient state. **(D2): do (a)** — snapshot `zero_grad()` across the model before the loop and restore-clean after, so an attribution call is side-effect-free on parameter gradients.

**C2 — `Model::backward` returns dL/d(input).** `include/nn/core/model.cpp:30-36` threads the returned grad through `layers.rbegin()..rend()` and returns the final value. This is exactly what IG needs, and it is why no infrastructure change is required.

**C3 — `Dense::backward` is scale-invariant to the seed scale only through the layer stack.** The target is a class logit; the seed is a one-hot-like vector on the output row. Because `Dense::backward` computes `grad_input = grad_output * weights`, seeding with a one-hot on output index `c` yields exactly d(logit_c)/dx. Good.

---

## Public API

```cpp
// include/nn/interpretability/attribution.h
#ifndef ATTRIBUTION_H
#define ATTRIBUTION_H

#include "../core/model.h"
#include "../core/tensor.h"
#include <string>

// Result of an attribution pass.
struct IGAttribution {
    Tensor attributions;   // (rows, cols) — IG_i per element, shape == input
    double completeness_delta;  // sum(attributions) - (F(x) - F(x')); ~0 is the axiom
    size_t n_steps;        // m actually used
    std::string method;    // "riemann_middle" etc.
};

// Gradient of a scalar model output w.r.t. the input.
// `target` selects which output column's logit to differentiate (default 0).
// Pass the input through Model::forward FIRST (layers cache activations in
// forward(); backward() reads them).
Tensor input_gradient(Model& model, const Tensor& input, size_t target = 0);

IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   const Tensor& baseline,
                                   size_t target = 0,
                                   size_t n_steps = 50,
                                   const std::string& method = "riemann_middle");

// Zero-baseline convenience overload.
IGAttribution integrated_gradients(Model& model,
                                   const Tensor& input,
                                   size_t target = 0,
                                   size_t n_steps = 50,
                                   const std::string& method = "riemann_middle");

#endif
```

**The `F(x)` scalar convention:** the model output must be `(rows, n_classes)`; `target` selects the logit column to attribute. **Logit, not softmax** — softmax probabilities saturate and the paper's §3 uses the pre-softmax score. Documented in the header.

---

## Tasks

### Task 1: `input_gradient` — the one backward pass

**Objective:** Seed the model output with a one-hot on `target`, run `Model::backward`, return d(logit_target)/dx.

**Files:**
- Create: `include/nn/interpretability/attribution.h`
- Create: `include/nn/interpretability/attribution.cpp`
- Test: `tests/test_integrated_gradients.cpp`

**Step 1: Write failing test**

```cpp
void test_input_gradient_matches_analytic() {
    // f(x) = W·x (no bias, no activation) => df/dx = W^T
    Model m;
    Dense* d = new Dense(3, 1);
    d->weights.fill(0.0);
    d->weights[0][0] = 2.0; d->weights[0][1] = 3.0; d->weights[0][2] = -1.0;
    d->bias.fill(0.0);
    m.add_layer(d);

    Tensor x(1, 3); x[0][0] = 1.0; x[0][1] = 5.0; x[0][2] = 2.0;
    m.forward(x);
    Tensor g = input_gradient(m, x, 0);

    assert_close(g[0][0], 2.0);
    assert_close(g[0][1], 3.0);
    assert_close(g[0][2], -1.0);
}
```

**Step 2: Run to verify failure**

Run: `make build/test_integrated_gradients 2>&1 | tail -20`
Expected: FAIL — `'input_gradient' was not declared`.

**Step 3: Minimal implementation**

```cpp
Tensor input_gradient(Model& model, const Tensor& input, size_t target) {
    // Callers must have run forward() already: layers cache activations there.
    Tensor out = model.forward(input);
    if (target >= out.cols) {
        throw std::out_of_range("input_gradient: target index out of range");
    }
    // Seed the output row with a one-hot on `target` => d(logit_target)/d(input).
    Tensor seed(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i) seed[i][target] = 1.0;
    return model.backward(seed, 0.0);
}
```

**Step 4: Run to verify pass**

Run: `make build/test_integrated_gradients && ./build/test_integrated_gradients`
Expected: PASS.

**Step 5: Commit**

```bash
git add include/nn/interpretability/attribution.h include/nn/interpretability/attribution.cpp tests/test_integrated_gradients.cpp
git commit -m "feat(interpretability): input_gradient — seeded one-hot logit gradient"
```

> **Note on the `lr` argument:** `0.0` is passed so no layer mutates its weights during attribution. `Dense::update_weights` is a separate call in this repo's contract, so `backward` will not move weights regardless — but `0.0` is explicit and safe. Task 3 makes the gradient-accumulation side effect explicit and tested.

---

### Task 2: Riemann step/alpha builders

**Objective:** Reproduce captum's `step_sizes` / `alphas` exactly, for all four variants.

**Files:**
- Modify: `include/nn/interpretability/attribution.{h,cpp}`
- Test: `tests/test_integrated_gradients.cpp`

**Step 1: Write failing test**

```cpp
void test_riemann_builders_match_captum() {
    // n=4, riemann_middle: alphas = linspace(1/8, 1-1/8, 4) = {0.125, 0.375, 0.625, 0.875}
    std::vector<double> ss, al;
    riemann_params("riemann_middle", 4, ss, al);
    assert_close(ss[0], 0.25); assert_close(ss[3], 0.25);
    assert_close(al[0], 0.125); assert_close(al[1], 0.375);
    assert_close(al[2], 0.625); assert_close(al[3], 0.875);

    // trapezoid: endpoints halved
    riemann_params("riemann_trapezoid", 4, ss, al);
    assert_close(ss[0], 0.125); assert_close(ss[1], 0.25); assert_close(ss[2], 0.25); assert_close(ss[3], 0.125);
    assert_close(al[0], 0.0); assert_close(al[3], 1.0);

    // left/right endpoints
    riemann_params("riemann_left", 4, ss, al);
    assert_close(al[0], 0.0); assert_close(al[3], 0.75);
    riemann_params("riemann_right", 4, ss, al);
    assert_close(al[0], 0.25); assert_close(al[3], 1.0);
}

void test_unknown_method_throws() {
    std::vector<double> ss, al;
    bool threw = false;
    try { riemann_params("riemann_bogus", 4, ss, al); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}
```

**Step 2: Run to verify failure** — `riemann_params` undeclared → FAIL.

**Step 3: Minimal implementation** — mirror captum exactly:

```cpp
void riemann_params(const std::string& method, size_t n,
                    std::vector<double>& step_sizes, std::vector<double>& alphas) {
    if (n < 1) throw std::invalid_argument("riemann_params: n must be >= 1");
    step_sizes.assign(n, 1.0 / static_cast<double>(n));
    alphas.resize(n);
    const double dn = static_cast<double>(n);
    if (method == "riemann_trapezoid") {
        step_sizes[0] /= 2.0; step_sizes[n - 1] /= 2.0;
        for (size_t k = 0; k < n; ++k) alphas[k] = static_cast<double>(k) / (dn - 1.0);
    } else if (method == "riemann_left") {
        for (size_t k = 0; k < n; ++k) alphas[k] = static_cast<double>(k) / dn;
    } else if (method == "riemann_right") {
        for (size_t k = 0; k < n; ++k) alphas[k] = (static_cast<double>(k) + 1.0) / dn;
    } else if (method == "riemann_middle") {
        for (size_t k = 0; k < n; ++k) alphas[k] = (static_cast<double>(k) + 0.5) / dn;
    } else {
        throw std::invalid_argument("riemann_params: unknown method '" + method + "'");
    }
}
```

> For `n == 1`, `trapezoid` divides by `dn - 1.0 == 0` → guard it (return `{0.0}` for alpha, `{1.0}` for step) and add a test.

**Step 4: Run to verify pass.**

**Step 5: Commit** — `git commit -m "feat(interpretability): captum-exact Riemann step/alpha builders"`

---

### Task 3: `integrated_gradients` core loop + gradient-state hygiene

**Objective:** The main m-step loop, with correct aggregation order and no parameter-gradient side effects.

**Files:**
- Modify: `include/nn/interpretability/attribution.{h,cpp}`
- Test: `tests/test_integrated_gradients.cpp`

**Step 1: Write failing test**

```cpp
void test_ig_does_not_pollute_parameter_gradients() {
    Model m;
    m.add_layer(new Dense(3, 1));
    Tensor x(1, 3); x[0][0]=1; x[0][1]=2; x[0][2]=3;
    Tensor base(1, 3); base.fill(0.0);

    integrated_gradients(m, x, base, 0, 8);

    // Dense::backward ACCUMULATES (layer.cpp:79). After 8 steps, grad_weights
    // would hold 8 stacked copies if we did not zero. It must be all-zero.
    Tensor gw = m.layers[0]->get_gradients();
    for (size_t i = 0; i < gw.rows; ++i)
        for (size_t j = 0; j < gw.cols; ++j)
            assert_close(gw[i][j], 0.0);
}

void test_ig_attribution_shape_matches_input() {
    Model m; m.add_layer(new Dense(4, 2));
    Tensor x(2, 4); x.fill(1.0); Tensor base(2, 4); base.fill(0.0);
    IGAttribution a = integrated_gradients(m, x, base, 0, 16);
    assert(a.attributions.rows == 2 && a.attributions.cols == 4);
    assert(a.n_steps == 16);
    assert(a.method == "riemann_middle");
}
```

**Step 2: Run to verify failure** — `integrated_gradients` undeclared → FAIL.

**Step 3: Minimal implementation** — note the aggregation order and the zero_grad hygiene:

```cpp
IGAttribution integrated_gradients(Model& model, const Tensor& input,
                                   const Tensor& baseline, size_t target,
                                   size_t n_steps, const std::string& method) {
    if (input.rows != baseline.rows || input.cols != baseline.cols)
        throw std::invalid_argument("integrated_gradients: baseline shape must match input");

    std::vector<double> step_sizes, alphas;
    riemann_params(method, n_steps, step_sizes, alphas);

    // C1/D2: Dense::backward ACCUMULATES parameter grads. Snapshot-and-zero so
    // m backward passes do not leave 50 stacked copies behind.
    for (auto& l : model.layers) l->zero_grad();

    Tensor total_grad(input.rows, input.cols);
    double scale = 0.0;   // sum of step_sizes (1.0 except trapezoid)

    for (size_t k = 0; k < n_steps; ++k) {
        const double a = alphas[k];
        Tensor point(input.rows, input.cols);
        for (size_t i = 0; i < input.rows; ++i)
            for (size_t j = 0; j < input.cols; ++j)
                point[i][j] = baseline[i][j] + a * (input[i][j] - baseline[i][j]);

        Tensor g = input_gradient(model, point, target);   // dF/d(point)
        for (size_t i = 0; i < total_grad.rows; ++i)
            for (size_t j = 0; j < total_grad.cols; ++j)
                total_grad[i][j] += step_sizes[k] * g[i][j];
        scale += step_sizes[k];
    }
    if (scale > 0.0) total_grad *= (1.0 / scale);

    // Captum order: multiply by (x - baseline) AFTER the weighted sum.
    Tensor attrib(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            attrib[i][j] = (input[i][j] - baseline[i][j]) * total_grad[i][j];

    for (auto& l : model.layers) l->zero_grad();   // leave no residue
    ...
}
```

> **Why the `1/scale` renormalization:** for all four Riemann variants `Σ step_sizes == 1.0` exactly, so it is a no-op — but it guards a future quadrature rule with unequal total weight, and it keeps the completeness assertion in Task 6 honest. The test asserts `scale ≈ 1` so the no-op is pinned rather than assumed.

**Step 4: Run to verify pass.**

**Step 5: Commit** — `git commit -m "feat(interpretability): integrated_gradients core loop, captum aggregation order"`

---

### Task 4: Completeness delta + overloads

**Objective:** Compute `completeness_delta = Σᵢ IGᵢ − (F(x) − F(x'))` and add the zero-baseline overload.

**Files:**
- Modify: `include/nn/interpretability/attribution.{h,cpp}`
- Test: `tests/test_integrated_gradients.cpp`

**Step 1: Write failing test**

```cpp
void test_completeness_delta_computed() {
    Model m; m.add_layer(new Dense(3, 1));
    Tensor x(1,3); x[0][0]=1; x[0][1]=2; x[0][2]=3;
    Tensor base(1,3); base.fill(0.0);
    IGAttribution a = integrated_gradients(m, x, base, 0, 32);
    // delta must be finite and recorded (not defaulted to 0 by accident)
    assert(std::isfinite(a.completeness_delta));
}

void test_zero_baseline_overload_matches_explicit() {
    Model m; m.add_layer(new Dense(3, 1));
    Tensor x(1,3); x[0][0]=1; x[0][1]=2; x[0][2]=3;
    IGAttribution a1 = integrated_gradients(m, x, 0, 32);
    IGAttribution a2 = integrated_gradients(m, x, Tensor::zeros(1,3), 0, 32);
    assert_close(a1.attributions[0][0], a2.attributions[0][0]);
}
```

**Step 2: Run → FAIL** (`completeness_delta` is a default-constructed `0.0`, so `isfinite` passes but the second test needs the overload → FAIL on undeclared overload).

**Step 3: Implementation** — after the loop:

```cpp
Tensor fx    = model.forward(input);
Tensor fb    = model.forward(baseline);
double Fx = fx[0][target], Fb = fb[0][target];
double completeness = attrib.sum() - (Fx - Fb);
```

**Step 4: Run → PASS.**

**Step 5: Commit** — `git commit -m "feat(interpretability): completeness delta + zero-baseline overload"`

---

### Task 5: Edge cases

**Objective:** Pin the error paths and degenerate inputs.

**Tests:**

```cpp
void test_shape_mismatch_throws()      { /* input (1,3), baseline (1,4) => throws */ }
void test_target_out_of_range_throws() { /* target=5 on a 2-output model => throws */ }
void test_n_steps_zero_preserves_zero_attribution() {
    // n_steps=0 => total_grad stays 0 => all attributions 0 (no div-by-zero).
}
void test_linear_model_ig_is_exact() {
    // For f(x) = W·x (linear), the gradient is CONSTANT along the path, so the
    // Riemann sum is EXACT for any m >= 1: IG_i = (x_i - x'_i) · W_i.
    // This is the strongest closed-form test in the suite.
}
```

**On `test_n_steps_zero_preserves_zero_attribution`:** with `n_steps == 0` the loop body never runs, `scale == 0.0`, and the `if (scale > 0.0)` guard prevents the division. The result is an all-zero attribution tensor. **This is the degenerate-config test from the debugging skills** — it exists to pin that no NaN/Inf escapes, since `0/0` would poison the whole tensor.

**Step order:** write each test, watch it fail, implement, watch it pass, commit.

**Commit** — `git commit -m "test(interpretability): edge cases — shape, target bounds, zero steps, linear exactness"`

---

### Task 6: The completeness axiom test — the headline assertion

**Objective:** Assert the defining property of the whole method.

**Test:**

```cpp
void test_completeness_axiom_holds() {
    // Build a small nonlinear MLP so the gradient genuinely varies along the path
    // (a linear model would make completeness trivially true).
    Model m;
    m.add_layer(new Dense(4, 4));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(4, 1));

    Tensor x(1,4); x[0][0]=1.5; x[0][1]=-0.7; x[0][2]=2.3; x[0][3]=0.4;
    Tensor base(1,4); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 128);
    // sum(IG) should approach F(x) - F(x'), up to the Riemann quadrature error.
    assert(std::abs(a.completeness_delta) < 1e-3);
}
```

**Note on the tolerance:** the Riemann sum is an *approximation* of the integral, so completeness holds only up to quadrature error, which shrinks like O(1/m) for midpoint and O(1/m²) for trapezoid. 128 steps on a smooth 3-layer net is comfortably inside 1e-3. If this assertion proves flaky, **do not loosen the tolerance blindly** — raise `n_steps` first and re-check; if it still fails at 512, the gradient path is wrong, not the tolerance.

**Commit** — `git commit -m "test(interpretability): completeness axiom — the defining property"`

---

### Task 7: Mutation testing — prove the tests aren't vacuous

**Objective:** Verify the suite actually catches a broken implementation.

Per the TDD skill, each mutation below must make at least one test FAIL. **Confirm each mutation applied** by grepping the file after editing (a `sed` that misses is a silent no-op that reads as "0 failures"):

| # | Mutation | Must be caught by |
|---|---|---|
| M1 | drop the `* (input[i][j] - baseline[i][j])` factor | `test_completeness_axiom_holds` |
| M2 | swap `riemann_middle` default → `riemann_left` | `test_riemann_builders_match_captum` |
| M3 | move `(x - x')` multiply INSIDE the loop (wrong aggregation order) | completeness + trapezoid test |
| M4 | delete the final `zero_grad()` loop | `test_ig_does_not_pollute_parameter_gradients` |
| M5 | seed `seed[i][0] = 1.0` always (ignore `target`) | `test_target_*` + a 2-output completeness test |

> **M1 is the classic half-scale/missing-factor mutation.** Note that `rel_err`-style FD checks are *blind* to constant-factor bugs — this suite deliberately uses **absolute, hand-derived** assertions (the linear-model closed form in Task 5, the completeness axiom in Task 6) rather than self-referential comparisons, for exactly that reason.

Restore each mutation and re-run the full suite before moving on.

**Commit** — `git commit -m "test(interpretability): mutation coverage — 5 mutations, all caught"`

---

### Task 8: Registration + full-suite verification

**Objective:** Wire into the umbrella header and the build; confirm no regressions.

**Files:**
- Modify: `include/nn/nn.h` — add `#include "interpretability/attribution.h"`
- Modify: `Makefile` — 4 places (see below)

**Step 1: Add to `nn.h`.** Match the existing include style in that file.

**Step 2: Makefile wiring.** Four edits, all required — the same four that `test_spiking_lif` needed (commit `604740a`):

```make
# 1. link rule (after the test_model_ema rule, line ~224)
$(BUILD_DIR)/test_integrated_gradients: $(LIB_OBJS) $(BUILD_DIR)/test_integrated_gradients.o
	$(CXX) $^ -o $@

# 2. tests: prerequisite list (line ~796)
#    add $(BUILD_DIR)/test_integrated_gradients to that long line

# 3. run_tests line (after the test_model_ema run line, ~941)
	@echo "=== Running test_integrated_gradients ==="; if ./$(BUILD_DIR)/test_integrated_gradients; then :; else echo "test_integrated_gradients" >> .run_tests_failed; fi
```

> **Forward-dependency check (from the planning skill):** there is **none** here — all four files exist before this task, so the binary is buildable the moment this task lands. Unlike the wallboard `COPY frontend/` failure, this task has no phase-skipped scaffold.

**Step 3: Verify the umbrella header compiles standalone** (this catches cross-header redefinition that a focused test cannot):

```bash
g++ -std=c++17 -Iinclude -x c++ -fsyntax-only - <<< '#include "nn/nn.h"'
```

Expected: no output, exit 0.

**Step 4: Warning-clean build:**

```bash
rm -f build/interpretability/attribution.o && make build/interpretability/attribution.o
```
Expected: zero warnings under the project's `-Wall -Wextra`.

**Step 5: Full suite — no regressions:**

```bash
make run_tests 2>&1 | tail -30
```

Expected: the new suite passes and **no previously-passing suite changes status**. Per NOT_FIXED.md, these failures are pre-existing and must NOT be counted as regressions: `test_neural_ode`, `test_coord_network`, `test_wgan_gp`, `test_vit`, and intermittent `test_adaln_zero` test 8.

**Step 6: Commit + push.**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore(build): register test_integrated_gradients in tests: + run_tests + link rule"
git push origin master
```

---

## Out of scope (v1)

- **Gauss-Legendre quadrature** — captum's actual default. Needs Legendre nodes/weights tables. Riemann-middle is the correct foundation; GL is a clean follow-up that reuses `integrated_gradients`'s loop verbatim (only `step_sizes`/`alphas` change).
- **Grad-CAM / HiResCAM / Ablation-CAM** — blocked on the `Layer` input-gradient API, see below.
- **Baseline strategies** (mean-image, k-means centroids, "blurred image" baseline from the paper). v1 takes a caller-supplied baseline + a zero convenience overload.
- **Multiple input tensors** — this repo's `Model::forward` takes a single `Tensor`.
- **Batch chunking / `internal_batch_size`** — captum chunks steps across devices; irrelevant for a single-threaded CPU implementation.

---

## Deferred: Grad-CAM (research is done, blocker is architectural)

The full Grad-CAM spec was researched and verified this session and is worth preserving rather than re-researching:

- **α_k = (1/Z) Σ_{i,j} ∂y^c/∂A^k_ij**, summed over the two **spatial** axes only (`Z = H·W`), one scalar per channel. Never over N or K.
- **L = ReLU(Σ_k α_k A^k)**, and **ReLU is applied at feature resolution BEFORE upsampling** — both the paper (§3.2) and the `pytorch-grad-cam` reference agree. Resize-first is the classic naive bug.
- **Upsampling** must use the **half-pixel-centered** rule (`src = r·(j+½) − ½`, clamped at 0), matching PyTorch `F.interpolate(align_corners=False)` and OpenCV `INTER_LINEAR`. A naive `src = j·(H_in/H_out)` is wrong.
- **Normalize** is `(x − min) / (max − min + 1e-7)`, applied at low resolution before the resize.
- **Seed the raw logit, never a softmax probability** — Ablation-CAM §4 explicitly rejects confidence-score drops as untrustworthy.
- **Strongest test:** Grad-CAM §3's own identity — on a GAP+linear model `α^c_k = w^c_k`, i.e. Grad-CAM reduces to classic CAM's closed form.
- **Citation correction:** Ablation-CAM is **WACV 2020 (Desai & Ramaswamy), not on arXiv**. `arXiv:1910.07079` is an unrelated solar-energy paper — do not cite it.

**The blocker:** Grad-CAM needs ∂y^c/∂A^k at an *intermediate* layer. `Layer::backward` returns that gradient but discards it; nothing caches it. Shipping Grad-CAM requires either (a) an optional `virtual void cache_input_grad_(const Tensor&)` no-op on the base `Layer` with overrides only in the layers that need it, or (b) a forward/backward hook vector on `Model`. Both are feasible, but option (a) touches the base class every layer inherits from and needs a survey of which layers are CAM-capable. That is a proper plan of its own — not a task to smuggle into this one.

---

## References

- Sundararajan, Taly, Yan, *"Axiomatic Attribution for Deep Networks"*, ICML 2017 — arXiv:1703.01365. Eq. 3 (definition), §3 (completeness, saturation).
- Sundararajan et al., *"Axiomatic Attribution for Deep Networks"* + the `F(F(x))` symmetrization trick for baseline choice — same paper, §5.
- `captum/attr/_core/integrated_gradients.py` — `n_steps=50` default, aggregation order (lines 385–405).
- `captum/attr/_utils/approximation_methods.py` — `riemann_params` semantics, `method="gausslegendre"` default.
- Repo internals verified: `include/nn/core/layer.cpp:62-84` (Dense backward), `include/nn/core/model.cpp:30-36` (Model::backward), `include/nn/core/tensor.h` (Tensor API).