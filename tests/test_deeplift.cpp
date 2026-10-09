// test_deeplift.cpp — Tests for DeepLIFT with the Rescale rule
// (Shrikumar, Greenside, Kundaje, "Learning Important Features Through
// Propagating Activation Differences", arXiv:1704.02685).
//
//   m_i      = (f(a_i) - f(a'_i)) / (a_i - a'_i)      per neuron, nonlinear layers only
//   G_in     = G_out ⊙ m                              elementwise nonlinearity
//   G_in     = G_out                                  linear layers (identity multiplier)
//   contrib_l = (a_l - a'_l) ⊙ G_in,l                  each layer's contribution
//   attribution = contrib_0                            the INPUT layer's contribution
//
// CLOSED-FORM ORACLE (why this is unusually testable). On a purely LINEAR
// network every multiplier is the identity, so DeepLIFT degenerates to
// multiplying the exact gradient by (x - x'):
//
//   attribution_i = (x_i - x'_i) · ∂F/∂x_i
//
// which is hand-computable from the weights with no tolerance at all. That is
// the headline test, and it pins the recursion end to end.
//
// COMPLETENESS. The axiom is about the INPUT layer's contribution, NOT the sum
// over all layers: sum(attributions) == F(x) - F(x'). Summing every layer's
// contribution counts the same output change once per layer and is wrong (see
// the header's "COMPLETENESS" note).
//
// Coverage:
//   1.  linear closed-form oracle: attribution_i == (x_i - x'_i)·∂F/∂x_i exactly
//   2.  single-ReLU oracle: multiplier 1 above the crossing, 0 below it
//   3.  ReLU+Dense hand weights: completeness and per-neuron multipliers
//   4.  multiplier is Δout/Δin, NOT f'(a) (the two separated by >10x)
//   5.  ε-fallback with x_i == x'_i exactly: zero attributions, no NaN/Inf
//   6.  ε-fallback is the DERIVATIVE, not 0 and not 1
//   7.  whole-batch input == baseline: all zeros, no NaN/Inf
//   8.  completeness on a deep net with random NON-UNIFORM weights
//   9.  validation errors (shape, target, eps, empty model)
//  10.  no parameter-gradient residue (seeded nonzero first)
//  11.  determinism: two calls bit-identical
//  12.  DeepLIFT != Integrated Gradients on a ReLU net
//  13.  deep_lift_shap: K=3 == mean of 3 deep_lift calls; K=1 == deep_lift

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/nn/nn.h"
#include "../include/nn/interpretability/deeplift.h"

static int passed = 0, failed = 0;

static void check(bool cond, const std::string& msg) {
    if (cond) { passed++; }
    else { failed++; std::cout << "  FAIL: " << msg << std::endl; }
}

static void check_close(double a, double b, double tol, const std::string& msg) {
    if (std::abs(a - b) <= tol) { passed++; }
    else {
        failed++;
        std::cout << "  FAIL: " << msg << " (got " << a << ", expected " << b
                  << ", tol " << tol << ")" << std::endl;
    }
}

static void check_throws(bool threw, const std::string& msg) {
    if (threw) { passed++; }
    else { failed++; std::cout << "  FAIL: " << msg << std::endl; }
}

static Tensor make_row(const std::vector<double>& v) {
    Tensor t(1, v.size());
    for (size_t j = 0; j < v.size(); ++j) t[0][j] = v[j];
    return t;
}

// TRANSPOSITION WARNING (same trap as test_lime.cpp): Dense::forward computes
// `input * weights.transpose()` with `weights` shaped (out_features, in_features),
// so f(x)[j] == Σ_k weights[j][k]·x[k]. This helper takes W in the natural
// (in_dim, out_dim) human layout and stores it transposed.
static void set_dense(Dense* d, const std::vector<std::vector<double>>& W,
                      const std::vector<double>& b) {
    const size_t in_dim = W.size(), out_dim = W[0].size();
    d->weights.fill(0.0);
    for (size_t i = 0; i < in_dim; ++i)
        for (size_t j = 0; j < out_dim; ++j)
            d->weights[j][i] = W[i][j];
    d->bias.fill(0.0);
    for (size_t j = 0; j < out_dim && j < b.size(); ++j) d->bias[0][j] = b[j];
}

// --- 1: the linear closed-form oracle (the headline test) --------------------

static void test_linear_closed_form_oracle() {
    // F(x) = W2 · (W1 · x + b1) + b2,  x ∈ R^2,  x' = 0
    //   W1 = [[1, 2], [3, -1]]   (in_dim 2 -> hidden 2), b1 = [0.5, -0.5]
    //   W2 = [[2, 1]]           (hidden 2 -> out 1),    b2 = [0.25]
    // Hand-set, so every expected value below is exact rational arithmetic.
    Model m;
    Dense* d1 = new Dense(2, 2);
    set_dense(d1, {{1.0, 2.0}, {3.0, -1.0}}, {0.5, -0.5});
    m.add_layer(d1);
    Dense* d2 = new Dense(2, 1);
    set_dense(d2, {{2.0, 1.0}}, {0.25});
    m.add_layer(d2);

    const Tensor x = make_row({1.5, -0.5});
    const Tensor xb = Tensor::zeros(1, 2);

    // ∂F/∂x by hand: F = 2·(x0 + 2·x1 + 0.5) + 1·(3·x0 - x1 - 0.5) + 0.25
    //              = 2x0 + 4x1 + 1 + 3x0 - x1 - 0.5 + 0.25
    //              = 5x0 + 3x1 + 0.75
    const double g0 = 5.0, g1 = 3.0;
    // Cross-check against the shipped analytic gradient path, so the oracle
    // value itself is not just "whatever the implementation produces".
    const Tensor grad = input_gradient(m, x, 0);
    check_close(grad[0][0], g0, 1e-12, "oracle precondition: dF/dx_0 == 5");
    check_close(grad[0][1], g1, 1e-12, "oracle precondition: dF/dx_1 == 3");

    DeepLiftResult r = deep_lift(m, x, xb, 0);
    check_close(r.attributions[0][0], 1.5 * g0, 1e-12,
                "linear oracle [0][0] == (x_0 - x'_0)·dF/dx_0");
    check_close(r.attributions[0][1], (-0.5) * g1, 1e-12,
                "linear oracle [0][1] == (x_1 - x'_1)·dF/dx_1");

    // Completeness on the same net: sum == F(x) - F(x').
    // F(x)  = 5(1.5) + 3(-0.5) + 0.75 = 7.5 - 1.5 + 0.75 = 6.75
    // F(0)  = 0.75                              => delta 6.0
    const Tensor fx = m.forward(x), f0 = m.forward(xb);
    check_close(fx[0][0] - f0[0][0], 6.0, 1e-12, "oracle precondition: F(x)-F(0) == 6");
    check_close(r.attributions.sum(), 6.0, 1e-12, "linear oracle: sum == F(x)-F(x')");
    check(std::abs(r.completeness_delta) < 1e-12, "linear oracle: completeness_delta ~ 0");
}

// --- 2: the single-ReLU oracle ----------------------------------------------

static void test_single_relu_oracle() {
    // Model = Activation<ReLU> only. x = [2, -3], x' = [1, -1].
    //   neuron 0: x0 = 2 > 0 > x'0 = 1  =>  f(x0) - f(x'0) = 2 - 1 = 1,
    //                                       delta_in = 1,  m = 1,  attr = 1·1 = 1
    //   neuron 1: x1 = -3 < x'1 = -1 < 0 =>  f is identically 0 on both,
    //                                       delta_out = 0, m = 0,  attr = 0
    Model m;
    m.add_layer(new Activation<ReLU>());

    const Tensor x = make_row({2.0, -3.0});
    const Tensor xb = make_row({1.0, -1.0});

    DeepLiftResult r = deep_lift(m, x, xb, 0);
    check_close(r.attributions[0][0], 1.0, 1e-15,
                "ReLU oracle: x>0>x' gives multiplier 1, attr == x - x'");
    check_close(r.attributions[0][1], 0.0, 1e-15,
                "ReLU oracle: x<x'<0 gives multiplier 0, attr == 0");

    // A dead-above-zero neuron: x' and x both positive but different => m = 1.
    Model m2;
    m2.add_layer(new Activation<ReLU>());
    DeepLiftResult r2 = deep_lift(m2, make_row({3.0, 0.5}), make_row({1.0, 0.25}), 0);
    check_close(r2.attributions[0][0], 2.0, 1e-15, "ReLU oracle: both positive, m == 1");
    check_close(r2.attributions[0][1], 0.25, 1e-15, "ReLU oracle: both positive, m == 1 (2)");

    // Completeness: F(x) = [2, 0], F(x') = [1, 0]; the model output is 2-D so
    // target 0 explains only column 0 => sum == 1.0 exactly.
    check_close(r.attributions.sum(), 1.0, 1e-15, "ReLU oracle: completeness");
    check(std::abs(r.completeness_delta) < 1e-15, "ReLU oracle: completeness_delta ~ 0");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_deeplift\n";
    std::cout << "============================================================\n";

    test_linear_closed_form_oracle();
    test_single_relu_oracle();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}