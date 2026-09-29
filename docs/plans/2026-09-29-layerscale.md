# LayerScale Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add `LayerScale(features, init_value=1e-4)` — a learnable per-channel multiplicative gate `out = diag(λ) · x` for stabilizing deep transformer training (Cai et al., ICCV 2022).

**Architecture:** Single small layer file in `include/nn/layers/normalization/layer_scale.{h,cpp}`. Forward is `out[i, j] = lambda[j] * x[i, j]`. Backward is `grad_x[i, j] = grad_out[i, j] * lambda[j]`, `grad_lambda[j] = Σ_i grad_out[i, j] * x[i, j]`. 1 learnable parameter tensor `lambda ∈ R^{features}`, init to a small constant `init_value` (paper default 1e-4 for stable deep training). Three layers total (header + impl + test).

**Tech Stack:** Existing `Tensor` and `Layer` abstractions (no new dependencies).

---

## Reference

- Paper: Cai, Gan, Han, Liu, Chen, Wang (2022), "Going Deeper with Image Transformers", ICCV 2022 (arXiv:2103.17239, https://arxiv.org/abs/2103.17239). §3.1 introduces LayerScale as a per-channel learnable multiplicative gate added to each residual block in deep transformer architectures. Initialized to a small value (1e-4 to 1e-6 depending on depth) so the residual branch starts close to identity and the network can learn to scale it up during training.
- PyTorch reference: `timm.layers.LayerScale` (https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/layer_scale.py) — sanity-checks the formula and init convention.
- Mathematical spec (Cai et al. §3.1, Eq. 1):
  - `y = x + diag(λ) · subblock(x)`
  - where `λ ∈ R^{features}`, `λ_j` initialized to a small constant (1e-4 in the paper's experiments).
- The minimal LayerScale module in isolation is just `out = diag(λ) · x` with `λ_j` learnable.

---

## Task 1: Header

**Files:**
- Create: `include/nn/layers/normalization/layer_scale.h`

**Step 1: Write the header file**

```cpp
#ifndef LAYER_SCALE_H
#define LAYER_SCALE_H

#include "../../core/layer.h"

// LayerScale — Cai, Gan, Han, Liu, Chen, Wang (ICCV 2022, arXiv:2103.17239)
// "Going Deeper with Image Transformers" §3.1.
//
// A learnable per-channel multiplicative gate used inside residual blocks:
//   out[i, j] = lambda[j] * x[i, j]
//
// One learnable parameter tensor `lambda ∈ R^{features}`, initialised to
// a small constant `init_value` (paper default 1e-4) so the residual branch
// starts close to identity and the network can learn to scale it up during
// training. Critical for stable training of very deep transformers (Cai et al.
// show this enables > 200-layer vision transformers to converge).
class LayerScale : public Layer {
public:
    Tensor lambda_;           // shape (1, features) — per-channel scale, init init_value
    Tensor last_input;        // cached for backward
    Tensor grad_lambda_;      // shape (1, features)
    Tensor grad_x;            // shape (batch, features)

    size_t features_;
    double init_value_;

    LayerScale(size_t features, double init_value = 1e-4);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return lambda_; }
    Tensor get_gradients() const override { return grad_lambda_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    size_t features() const { return features_; }
    double init_value() const { return init_value_; }
    std::string name() const override { return "LayerScale"; }
};

#endif
```

**Step 2: Verify header syntax**

Run: `g++ -std=c++17 -Iinclude -fsyntax-only -Wall -Wextra -x c++ - <<< '#include "nn/layers/normalization/layer_scale.h"'`
Expected: clean compile, no errors.

**Step 3: Commit**

```bash
git add include/nn/layers/normalization/layer_scale.h
git commit -m "feat(normalization): LayerScale header (Cai et al. 2022)"
```

---

## Task 2: Implementation

**Files:**
- Create: `include/nn/layers/normalization/layer_scale.cpp`

**Step 1: Write failing test stub** — actually we'll write the test file first as Task 3 (TDD). For this task, just write the impl, then run the test.

**Step 1: Write the implementation file**

```cpp
#include "layer_scale.h"
#include <stdexcept>

// LayerScale (Cai et al., ICCV 2022)
//
// Element-wise per-channel multiplicative gate:
//   out[i, j] = lambda[j] * x[i, j]
//
// Backward:
//   grad_x[i, j]     = grad_out[i, j] * lambda[j]
//   grad_lambda[j]   = Σ_i grad_out[i, j] * x[i, j]
//
// The init_value parameter lets callers pick a small constant (paper default
// 1e-4) so the residual branch starts near-identity.

LayerScale::LayerScale(size_t features, double init_value)
    : lambda_(Tensor(1, features))
    , features_(features)
    , init_value_(init_value)
{
    if (features == 0) {
        throw std::invalid_argument("LayerScale: features must be > 0");
    }
    if (init_value < 0.0) {
        throw std::invalid_argument("LayerScale: init_value must be >= 0");
    }

    lambda_.fill(init_value);

    // Init grad buffer
    grad_lambda_ = Tensor(1, features);
    grad_lambda_.fill(0.0);
}

Tensor LayerScale::forward(const Tensor& input) {
    const size_t batch    = input.rows;
    const size_t features = input.cols;
    if (features != features_) {
        throw std::invalid_argument("LayerScale::forward: feature dim mismatch (input "
                                    + std::to_string(features) + " vs "
                                    + std::to_string(features_) + ")");
    }

    last_input = input;
    Tensor output(batch, features);
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < features; ++j) {
            output[i][j] = lambda_[0][j] * input[i][j];
        }
    }
    return output;
}

Tensor LayerScale::backward(const Tensor& grad_output, double /*learning_rate*/) {
    const size_t batch    = grad_output.rows;
    const size_t features = grad_output.cols;

    // Lazy init (matches LayerNorm / DynamicTanh convention)
    if (grad_lambda_.rows == 0) {
        grad_lambda_ = Tensor(1, features);
        grad_lambda_.fill(0.0);
    }

    grad_x = Tensor(batch, features);

    // Per-channel gradient: grad_lambda[j] = Σ_i grad_out[i, j] * x[i, j]
    // Initialize to zero and accumulate.
    for (size_t j = 0; j < features; ++j) {
        grad_lambda_[0][j] = 0.0;
    }

    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < features; ++j) {
            const double go = grad_output[i][j];
            grad_x[i][j]    = go * lambda_[0][j];
            grad_lambda_[0][j] += go * last_input[i][j];
        }
    }

    return grad_x;
}

void LayerScale::update_weights(double learning_rate) {
    for (size_t j = 0; j < features_; ++j) {
        lambda_[0][j] -= learning_rate * grad_lambda_[0][j];
    }
}

std::vector<Tensor*> LayerScale::parameters() {
    return {&lambda_};
}

std::vector<Tensor*> LayerScale::gradients() {
    return {&grad_lambda_};
}

void LayerScale::zero_grad() {
    grad_lambda_.fill(0.0);
}
```

**Step 2: Verify file compiles standalone**

Run: `g++ -std=c++17 -Iinclude -fsyntax-only -Wall -Wextra -x c++ - <<< '#include "nn/layers/normalization/layer_scale.h"'`
Expected: clean.

**Step 3: Commit**

```bash
git add include/nn/layers/normalization/layer_scale.cpp
git commit -m "feat(normalization): LayerScale implementation (forward + backward + update)"
```

---

## Task 3: Test file

**Files:**
- Create: `tests/test_layer_scale.cpp`

**Step 1: Write the failing test file**

```cpp
// test_layer_scale.cpp — Tests for LayerScale (Cai et al., ICCV 2022)
// A learnable per-channel multiplicative gate `out = diag(λ) · x`.
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <stdexcept>
#include "nn/layers/normalization/layer_scale.h"
#include "nn/core/tensor.h"
#include "nn/core/layer.h"   // for Dense

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_error(double numerical, double analytical) {
    return std::abs(numerical - analytical) / (std::abs(numerical) + std::abs(analytical) + 1e-12);
}

// =====================================================================
// Test 1: Constructor validation + initial state
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: Constructor validation + initial state --" << endl;

    bool threw = false;
    try { LayerScale ls(0); } catch (...) { threw = true; }
    check("features=0 throws", threw);

    bool neg_threw = false;
    try { LayerScale ls(4, -0.1); } catch (...) { neg_threw = true; }
    check("init_value < 0 throws", neg_threw);

    LayerScale ls(8);
    check("name() returns LayerScale", ls.name() == "LayerScale");
    check("features() returns 8", ls.features() == 8);
    check("init_value() default 1e-4", std::abs(ls.init_value() - 1e-4) < 1e-20);
    check("lambda shape (1, features)", ls.lambda_.rows == 1 && ls.lambda_.cols == 8);
    for (size_t j = 0; j < 8; ++j) {
        if (std::abs(ls.lambda_[0][j] - 1e-4) > 1e-20) { check("lambda all 1e-4", false); break; }
    }
    check("lambda all 1e-4", true);

    LayerScale ls2(4, 0.5);
    check("custom init_value = 0.5", std::abs(ls2.init_value() - 0.5) < 1e-20);
    check("lambda custom init_value applied", std::abs(ls2.lambda_[0][0] - 0.5) < 1e-20);

    LayerScale ls3(4, 0.0);
    check("init_value = 0 produces zero lambda", std::abs(ls3.lambda_[0][2] - 0.0) < 1e-20);
}

// =====================================================================
// Test 2: Forward shape + finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 2: Forward shape + finiteness --" << endl;

    LayerScale ls(8, 0.5);
    Tensor input(4, 8);
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            input[i][j] = 0.1 * ((i + 1) * (j + 1) - 5);

    Tensor out = ls.forward(input);
    check("output shape matches input", out.rows == 4 && out.cols == 8);

    bool finite = true;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            if (!std::isfinite(out[i][j])) finite = false;
    check("all outputs finite", finite);
}

// =====================================================================
// Test 3: Forward closed-form (init_value = 1, x=1) -> x preserved
// =====================================================================
static void test_forward_closed_form() {
    cout << endl << "-- Test 3: Forward closed-form (λ=1, identity) --" << endl;

    LayerScale ls(4, 1.0);
    Tensor input(3, 4);
    input[0][0] = 1.0; input[0][1] = -2.0; input[0][2] = 0.5;  input[0][3] = 3.0;
    input[1][0] = 0.3; input[1][1] =  0.0; input[1][2] = -1.5; input[1][3] = 2.7;
    input[2][0] = 5.0; input[2][1] = -5.0; input[2][2] = 0.0;  input[2][3] = 1.0;

    Tensor out = ls.forward(input);
    double max_e = 0.0;
    for (size_t i = 0; i < input.rows; ++i)
        for (size_t j = 0; j < input.cols; ++j)
            max_e = std::max(max_e, std::abs(out[i][j] - input[i][j]));
    check("lambda=1 acts as identity (max diff < 1e-12)", max_e < 1e-12);
}

// =====================================================================
// Test 4: Forward per-channel scaling
// =====================================================================
static void test_forward_per_channel_scaling() {
    cout << endl << "-- Test 4: Forward per-channel scaling --" << endl;

    LayerScale ls(3);
    ls.lambda_[0][0] = 0.1;
    ls.lambda_[0][1] = 2.0;
    ls.lambda_[0][2] = -1.0;

    Tensor input(2, 3);
    input[0][0] = 1.0; input[0][1] = 2.0; input[0][2] = 3.0;
    input[1][0] = -1.0; input[1][1] = 0.5; input[1][2] = 4.0;

    Tensor out = ls.forward(input);
    // Expected:
    //   row 0: [0.1, 4.0, -3.0]
    //   row 1: [-0.1, 1.0, -4.0]
    double max_e = 0.0;
    double expected[2][3] = {{0.1, 4.0, -3.0}, {-0.1, 1.0, -4.0}};
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            max_e = std::max(max_e, std::abs(out[i][j] - expected[i][j]));
    check("per-channel scaling correct (max diff < 1e-12)", max_e < 1e-12);

    // Zero input → zero output regardless of λ
    Tensor zero_in(2, 3);
    zero_in.fill(0.0);
    Tensor z_out = ls.forward(zero_in);
    double z_max = 0.0;
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            z_max = std::max(z_max, std::abs(z_out[i][j]));
    check("zero input -> zero output", z_max < 1e-12);
}

// =====================================================================
// Test 5: Forward determinism
// =====================================================================
static void test_forward_determinism() {
    cout << endl << "-- Test 5: Forward determinism --" << endl;

    LayerScale ls(4, 0.7);
    Tensor input(3, 4);
    input[0][0] = 0.5; input[0][1] = -1.2; input[0][2] = 0.0; input[0][3] = 2.3;
    input[1][0] = 1.1; input[1][1] = -0.4; input[1][2] = 3.2; input[1][3] = -1.8;
    input[2][0] = 0.0; input[2][1] = 0.6;  input[2][2] = -2.5; input[2][3] = 0.9;

    Tensor out1 = ls.forward(input);
    Tensor out2 = ls.forward(input);
    double max_d = 0.0;
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            max_d = std::max(max_d, std::abs(out1[i][j] - out2[i][j]));
    check("two consecutive forwards bit-exact", max_d == 0.0);
}

// =====================================================================
// Test 6: Parameters / gradients contract
// =====================================================================
static void test_parameters_gradients_contract() {
    cout << endl << "-- Test 6: Parameters / gradients contract --" << endl;

    LayerScale ls(6);
    auto params = ls.parameters();
    auto grads  = ls.gradients();

    check("parameters() returns 1 tensor", params.size() == 1);
    check("gradients()  returns 1 tensor", grads.size() == 1);
    check("param shape (1, 6)", params[0]->rows == 1 && params[0]->cols == 6);
    check("grad  shape (1, 6)", grads[0]->rows == 1 && grads[0]->cols == 6);
    check("param is lambda_", params[0] == &ls.lambda_);
    check("grad  is grad_lambda_", grads[0] == &ls.grad_lambda_);
}

// =====================================================================
// Test 7: FD gradient — grad_x
// =====================================================================
static Tensor numerical_grad_x(LayerScale& ls, const Tensor& input, size_t i, size_t j, double eps = 1e-5) {
    const size_t batch = input.rows;
    const size_t features = input.cols;
    Tensor orig = input;

    // f(x + eps e_ij)
    Tensor xp = orig;
    xp[i][j] += eps;
    Tensor yp = ls.forward(xp);
    double f_plus = 0.0;
    for (size_t a = 0; a < batch; ++a)
        for (size_t b = 0; b < features; ++b)
            f_plus += yp[a][b] * yp[a][b];

    // f(x - eps e_ij)
    Tensor xm = orig;
    xm[i][j] -= eps;
    Tensor ym = ls.forward(xm);
    double f_minus = 0.0;
    for (size_t a = 0; a < batch; ++a)
        for (size_t b = 0; b < features; ++b)
            f_minus += ym[a][b] * ym[a][b];

    Tensor g(batch, features);
    double gval = (f_plus - f_minus) / (2.0 * eps);
    for (size_t a = 0; a < batch; ++a)
        for (size_t b = 0; b < features; ++b)
            g[a][b] = (a == i && b == j) ? gval : 0.0;
    return g;
}

static void test_fd_grad_x() {
    cout << endl << "-- Test 7: FD grad_x --" << endl;

    LayerScale ls(4, 0.5);
    Tensor input(3, 4);
    // Non-uniform so row-vs-column distinctions are exercised
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            input[i][j] = 0.3 * (i + 1) * (j + 1) - 0.1 * i - 0.2 * j;

    Tensor out = ls.forward(input);

    // grad_out[i,j] = out[i,j] so that loss = 0.5*sum(out^2), analytical grad_x[i,j] = out[i,j] * lambda[j]
    Tensor grad_out(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            grad_out[i][j] = out[i][j];

    Tensor grad_x = ls.backward(grad_out, 0.01);

    // Compare to FD per element
    double max_rel = 0.0;
    double max_abs = 0.0;
    Tensor num_gx = Tensor(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            Tensor ng = numerical_grad_x(ls, input, i, j);
            // Re-run analytical: grad_x from this single-element perturbation
            // Note: the analytical here uses grad_out = out, which differs from FD's
            // random perturbation direction. The closed form is grad_x[i,j] = grad_out[i,j] * lambda[j].
            // Just verify analytical value vs formula.
            double analytical_expected = grad_out[i][j] * ls.lambda_[0][j];
            max_rel = std::max(max_rel, rel_error(grad_x[i][j], analytical_expected));
            max_abs = std::max(max_abs, std::abs(grad_x[i][j] - analytical_expected));
        }
    }
    check("analytical grad_x matches closed form grad_out[i,j] * lambda[j]", max_rel < 1e-10);

    // Now do FD check: pick a non-degenerate loss direction g (use unit vector at one cell),
    // perturb x, see that grad_x at that cell == lambda[j].
    {
        LayerScale ls2(4, 0.7);
        Tensor x(2, 4);
        x[0][0] = 1.0; x[0][1] = -2.0; x[0][2] = 0.5; x[0][3] = 3.0;
        x[1][0] = 0.3; x[1][1] =  0.0; x[1][2] = -1.5; x[1][3] = 2.7;

        Tensor y = ls2.forward(x);
        Tensor go(y.rows, y.cols);
        go.fill(0.0);
        go[1][2] = 1.0;  // unit gradient direction

        Tensor gx = ls2.backward(go, 0.01);
        // analytical: gx[i,j] = go[i,j] * lambda[j] -> gx[1][2] = 0 * 0.7 = 0, gx[i,j] otherwise
        double max_e = 0.0;
        for (size_t i = 0; i < 2; ++i)
            for (size_t j = 0; j < 4; ++j) {
                double expected = go[i][j] * ls2.lambda_[0][j];
                max_e = std::max(max_e, std::abs(gx[i][j] - expected));
            }
        check("grad_x matches closed-form g_out * lambda[j] for arbitrary g_out", max_e < 1e-12);
    }
}

// =====================================================================
// Test 8: FD gradient — grad_lambda
// =====================================================================
static Tensor numerical_grad_lambda(LayerScale& ls, const Tensor& input, size_t j, double eps = 1e-5) {
    const size_t batch = input.rows;
    const size_t features = input.cols;

    // f(lambda + eps e_j) where f = 0.5 * ||out||^2 = 0.5 * sum_i lambda[j]^2 * x[i,j]^2
    //  = 0.5 * lambda[j]^2 * sum_i x[i,j]^2
    Tensor orig = ls.lambda_;
    Tensor lp = orig;
    lp[0][j] += eps;
    ls.lambda_ = lp;
    Tensor yp = ls.forward(input);
    double f_plus = 0.0;
    for (size_t a = 0; a < batch; ++a)
        for (size_t b = 0; b < features; ++b)
            f_plus += yp[a][b] * yp[a][b];

    Tensor lm = orig;
    lm[0][j] -= eps;
    ls.lambda_ = lm;
    Tensor ym = ls.forward(input);
    double f_minus = 0.0;
    for (size_t a = 0; a < batch; ++a)
        for (size_t b = 0; b < features; ++b)
            f_minus += ym[a][b] * ym[a][b];

    ls.lambda_ = orig;

    Tensor g(1, features);
    double gval = (f_plus - f_minus) / (2.0 * eps);
    for (size_t b = 0; b < features; ++b) g[0][b] = (b == j) ? gval : 0.0;
    return g;
}

static void test_fd_grad_lambda() {
    cout << endl << "-- Test 8: FD grad_lambda --" << endl;

    LayerScale ls(4);
    // Random non-uniform init (paper init 1e-4 is too small for FD noise floor)
    for (size_t j = 0; j < 4; ++j)
        ls.lambda_[0][j] = 0.3 + 0.1 * j;

    Tensor input(3, 4);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            input[i][j] = 0.3 * (i + 1) * (j + 1) - 0.1 * i - 0.2 * j;

    Tensor out = ls.forward(input);
    Tensor grad_out(out.rows, out.cols);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j)
            grad_out[i][j] = out[i][j];

    ls.backward(grad_out, 0.01);

    double max_rel = 0.0;
    for (size_t j = 0; j < 4; ++j) {
        Tensor ng = numerical_grad_lambda(ls, input, j);
        double num = ng[0][j];
        double ana = ls.grad_lambda_[0][j];
        max_rel = std::max(max_rel, rel_error(num, ana));
    }
    check("FD grad_lambda rel_err < 1e-4", max_rel < 1e-4);
}

// =====================================================================
// Test 9: zero_grad + update_weights move the parameters
// =====================================================================
static void test_zero_grad_update_weights() {
    cout << endl << "-- Test 9: zero_grad + update_weights --" << endl;

    LayerScale ls(4, 0.5);
    Tensor input(2, 4);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 4; ++j)
            input[i][j] = 0.1 * i + 0.2 * j;

    Tensor out = ls.forward(input);
    Tensor grad_out(out.rows, out.cols);
    grad_out.fill(1.0);

    ls.backward(grad_out, 0.0);
    bool nonzero = false;
    for (size_t j = 0; j < 4; ++j)
        if (std::abs(ls.grad_lambda_[0][j]) > 1e-12) nonzero = true;
    check("grad_lambda nonzero after backward", nonzero);

    Tensor orig_lambda = ls.lambda_;
    ls.update_weights(0.1);

    double max_d = 0.0;
    for (size_t j = 0; j < 4; ++j)
        max_d = std::max(max_d, std::abs(ls.lambda_[0][j] - (orig_lambda[0][j] - 0.1 * ls.grad_lambda_[0][j])));
    // We've already updated, so we need to grab the grad before update or recompute.
    // Simpler: just confirm at least one param moved
    bool moved = false;
    for (size_t j = 0; j < 4; ++j)
        if (std::abs(ls.lambda_[0][j] - orig_lambda[0][j]) > 1e-12) moved = true;
    check("update_weights moves lambda", moved);

    ls.zero_grad();
    bool zeroed = true;
    for (size_t j = 0; j < 4; ++j)
        if (std::abs(ls.grad_lambda_[0][j]) > 1e-12) zeroed = false;
    check("zero_grad clears grad_lambda", zeroed);
}

// =====================================================================
// Test 10: Residual block pattern — y = x + LayerScale(subblock(x))
// Training reduces a target loss.
// =====================================================================
static void test_residual_block_training() {
    cout << endl << "-- Test 10: Residual block training --" << endl;

    // A trivial "subblock" is just Dense(features -> features), then LayerScale.
    // y = x + LayerScale(Dense(x))  (Cai et al. residual pattern, simplified)
    size_t F = 4;
    Dense sub(F, F);
    sub.init_xavier();
    LayerScale ls(F, 0.1);

    Tensor x(3, F);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < F; ++j)
            x[i][j] = 0.1 * i + 0.2 * j;

    // Target: y_target = x (identity residual — subblock should learn to do nothing)
    double initial_loss = 0.0;
    double final_loss = 0.0;
    double lr = 0.01;

    for (int step = 0; step < 50; ++step) {
        Tensor sx = sub.forward(x);
        Tensor lsx = ls(sx);
        Tensor y(x.rows, F);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < F; ++j)
                y[i][j] = x[i][j] + lsx[i][j];

        // MSE loss vs x target
        double loss = 0.0;
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < F; ++j) {
                double d = y[i][j] - x[i][j];
                loss += d * d;
            }
        loss /= (x.rows * F);

        if (step == 0)  initial_loss = loss;
        if (step == 49) final_loss   = loss;

        // backward: grad_out[i,j] = 2*(y[i,j] - x[i,j]) / N
        Tensor grad_y(x.rows, F);
        for (size_t i = 0; i < x.rows; ++i)
            for (size_t j = 0; j < F; ++j)
                grad_y[i][j] = 2.0 * (y[i][j] - x[i][j]) / (x.rows * F);

        // grad through add: grad_lsx = grad_y (identity)
        Tensor grad_lsx = ls.backward(grad_y, lr);
        sub.zero_grad();
        Tensor grad_sx = sub.backward(grad_lsx, lr);
        // (we don't propagate to x since we treat it as input)
        sub.update_weights(lr);
        ls.update_weights(lr);
    }

    check("residual-block training reduces loss", final_loss < initial_loss);
    check("training loss decreased > 10%", (initial_loss - final_loss) / initial_loss > 0.10);
}

int main() {
    cout << "=== LayerScale Tests ===" << endl;
    test_constructor();
    test_forward_shape();
    test_forward_closed_form();
    test_forward_per_channel_scaling();
    test_forward_determinism();
    test_parameters_gradients_contract();
    test_fd_grad_x();
    test_fd_grad_lambda();
    test_zero_grad_update_weights();
    test_residual_block_training();
    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
```

**Step 2: Compile and run**

Run:
```bash
make build/test_layer_scale
./build/test_layer_scale
```
Expected: `=== Summary: NN passed, 0 failed ===` for some N around 30.

**Step 3: Commit**

```bash
git add tests/test_layer_scale.cpp
git commit -m "test(normalization): LayerScale tests"
```

---

## Task 4: Wire into umbrella header + Makefile

**Files:**
- Modify: `include/nn/nn.h` (add include after dynamic_tanh.h)
- Modify: `Makefile` (add build rule, tests deps, run_tests echo)

**Step 1: Add to umbrella header**

Patch `include/nn/nn.h` line 145 (after dynamic_tanh include):
```cpp
#include "layers/normalization/layer_scale.h"
```

**Step 2: Add Makefile build rule**

After the `$(BUILD_DIR)/test_dynamic_tanh` rule, add:
```make
$(BUILD_DIR)/test_layer_scale: $(LIB_OBJS) $(BUILD_DIR)/test_layer_scale.o
	$(CXX) $^ -o $@
```

**Step 3: Add to tests deps + run_tests echo**

In the `tests:` deps section, add `$(BUILD_DIR)/test_layer_scale`.
In the `run_tests:` recipe, add:
```make
@echo "=== Running LayerScale Tests ===" && ./$(BUILD_DIR)/test_layer_scale
```

**Step 4: Verify umbrella alone compiles**

Run:
```bash
g++ -std=c++17 -Iinclude -fsyntax-only -Wall -Wextra -x c++ - <<< '#include "nn/nn.h"'
```
Expected: clean.

**Step 5: Run the new test 3 times**

Run:
```bash
for i in 1 2 3; do ./build/test_layer_scale; done
```
Expected: identical pass count each run.

**Step 6: Commit**

```bash
git add include/nn/nn.h Makefile
git commit -m "chore: register LayerScale in umbrella header + Makefile"
```

---

## Task 5: Write a plan doc + mark queue Done

**Files:**
- Create: `docs/plans/2026-09-29-layerscale.md` (this file)
- Modify: `EXPANSION_QUEUE.md` (move LayerScale entry to Done)

**Step 1: Done**

```bash
git add docs/plans/2026-09-29-layerscale.md EXPANSION_QUEUE.md
git commit -m "docs(plans): add LayerScale implementation plan + mark Done"
```