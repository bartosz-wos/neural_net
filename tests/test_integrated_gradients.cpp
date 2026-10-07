// test_integrated_gradients.cpp — Tests for Integrated Gradients
// (Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks",
// ICML 2017, arXiv:1703.01365) and the gradient-only baseline it builds on.
//
// IG:  IG_i(x) = (x_i - x'_i) * integral_0^1 d(alpha) [ dF(x' + a(x - x'))/dx_i ]
//
// Coverage:
//   1.  input_gradient matches the analytic gradient of a linear map
//   2.  input_gradient targets the requested output column
//   3.  riemann_params reproduce captum's step_sizes/alphas for all 4 variants
//   4.  unknown riemann method throws
//   5.  IG leaves no residue in parameter gradients (Dense::backward accumulates)
//   6.  IG attribution shape/metadata match the request
//   7.  completeness delta is finite and recorded
//   8.  zero-baseline overload == explicit zeros baseline
//   9.  shape mismatch between input and baseline throws
// 10.  out-of-range target throws
// 11.  n_steps == 0 yields all-zero attributions (no NaN from 0/0)
// 12.  linear model: IG is EXACT for any m >= 1 (closed form (x-x')*W)
// 13.  completeness axiom holds on a nonlinear MLP
// 14.  more steps => smaller completeness delta (convergence)
// 15.  attribution is antisymmetric under swapping input and baseline

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/nn/nn.h"

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

// Build f(x) = W·x exactly (zero bias, no activation), so the gradient is
// the transpose of W and IG is analytically exact.
static Model make_linear_model(double w00, double w01, double w02) {
    Model m;
    Dense* d = new Dense(3, 1);
    d->weights.fill(0.0);
    d->weights[0][0] = w00;
    d->weights[0][1] = w01;
    d->weights[0][2] = w02;
    d->bias.fill(0.0);
    d->grad_weights.fill(0.0);
    d->grad_bias.fill(0.0);
    m.add_layer(d);
    return m;
}

// ---------------------------------------------------------------- 1
void test_input_gradient_matches_analytic() {
    Model m = make_linear_model(2.0, 3.0, -1.0);
    Tensor x(1, 3);
    x[0][0] = 1.0; x[0][1] = 5.0; x[0][2] = 2.0;

    Tensor g = input_gradient(m, x, 0);

    check(g.rows == 1 && g.cols == 3, "test_input_gradient: shape preserved");
    check_close(g[0][0], 2.0, 1e-12, "d/dx0");
    check_close(g[0][1], 3.0, 1e-12, "d/dx1");
    check_close(g[0][2], -1.0, 1e-12, "d/dx2");
}

// ---------------------------------------------------------------- 2
void test_input_gradient_targets_requested_column() {
    // Two independent outputs: f0 = 2*x0, f1 = 5*x1. The gradient w.r.t. output
    // `target` must pick that column's weights only.
    Model m;
    Dense* d = new Dense(2, 2);
    d->weights.fill(0.0);
    d->weights[0][0] = 2.0;  // f0 = 2*x0
    d->weights[1][1] = 5.0;  // f1 = 5*x1
    d->bias.fill(0.0);
    d->grad_weights.fill(0.0);
    d->grad_bias.fill(0.0);
    m.add_layer(d);

    Tensor x(1, 2);
    x[0][0] = 1.0; x[0][1] = 1.0;

    Tensor g0 = input_gradient(m, x, 0);
    check_close(g0[0][0], 2.0, 1e-12, "target=0 -> d(f0)/dx0");
    check_close(g0[0][1], 0.0, 1e-12, "target=0 -> d(f0)/dx1 == 0");

    Tensor g1 = input_gradient(m, x, 1);
    check_close(g1[0][0], 0.0, 1e-12, "target=1 -> d(f1)/dx0 == 0");
    check_close(g1[0][1], 5.0, 1e-12, "target=1 -> d(f1)/dx1");
}

// ---------------------------------------------------------------- 3
void test_riemann_params_match_captum() {
    std::vector<double> ss, al;

    // middle: alphas = linspace(1/(2n), 1-1/(2n), n)
    riemann_params("riemann_middle", 4, ss, al);
    check_close(ss[0], 0.25, 1e-12, "middle step size");
    check_close(ss[3], 0.25, 1e-12, "middle step size last");
    check_close(al[0], 0.125, 1e-12, "middle alpha 0");
    check_close(al[1], 0.375, 1e-12, "middle alpha 1");
    check_close(al[2], 0.625, 1e-12, "middle alpha 2");
    check_close(al[3], 0.875, 1e-12, "middle alpha 3");

    // trapezoid: endpoints halved, alphas span [0,1] inclusive.
    // NOTE: the base step is 1/(n-1), NOT captum's 1/n. captum pairs
    // 1/(n-1)-spaced nodes with a 1/n step, so its weights sum to (n-1)/n and
    // its trapezoidal rule under-integrates by that factor. A quadrature rule
    // whose weights sum to 0.75 cannot integrate the constant 1 exactly, so
    // we use the matching base step: for n=4, [1/12, 1/3, 1/3, 1/12], sum 1.
    riemann_params("riemann_trapezoid", 4, ss, al);
    check_close(ss[0], 1.0 / 6.0, 1e-12, "trap step 0 halved");
    check_close(ss[1], 1.0 / 3.0, 1e-12, "trap step 1");
    check_close(ss[2], 1.0 / 3.0, 1e-12, "trap step 2");
    check_close(ss[3], 1.0 / 6.0, 1e-12, "trap step 3 halved");
    check_close(al[0], 0.0, 1e-12, "trap alpha 0");
    check_close(al[3], 1.0, 1e-12, "trap alpha last");

    // left: alphas = k/n
    riemann_params("riemann_left", 4, ss, al);
    check_close(al[0], 0.0, 1e-12, "left alpha 0");
    check_close(al[3], 0.75, 1e-12, "left alpha last");

    // right: alphas = (k+1)/n
    riemann_params("riemann_right", 4, ss, al);
    check_close(al[0], 0.25, 1e-12, "right alpha 0");
    check_close(al[3], 1.0, 1e-12, "right alpha last");

    // All four variants must have step sizes summing to exactly 1.
    for (const char* mtd : {"riemann_left", "riemann_right",
                            "riemann_middle", "riemann_trapezoid"}) {
        riemann_params(mtd, 7, ss, al);
        double sum = 0.0;
        for (double v : ss) sum += v;
        check_close(sum, 1.0, 1e-12, std::string("step sizes sum to 1 for ") + mtd);
    }
}

// ---------------------------------------------------------------- 4
void test_unknown_method_throws() {
    std::vector<double> ss, al;
    bool threw = false;
    try { riemann_params("riemann_bogus", 4, ss, al); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "unknown riemann method throws");
}

// ---------------------------------------------------------------- 5
void test_ig_does_not_pollute_parameter_gradients() {
    // Dense::backward ACCUMULATES into grad_weights (layer.cpp:79). IG calls
    // backward m times; without hygiene, m stacked copies are left behind.
    Model m;
    Dense* d = new Dense(3, 1);
    m.add_layer(d);

    Tensor x(1, 3); x[0][0] = 1; x[0][1] = 2; x[0][2] = 3;
    Tensor base(1, 3); base.fill(0.0);

    integrated_gradients(m, x, base, 0, 8);

    Tensor gw = m.layers[0]->get_gradients();
    for (size_t i = 0; i < gw.rows; ++i)
        for (size_t j = 0; j < gw.cols; ++j)
            check_close(gw[i][j], 0.0, 0.0, "parameter gradients zeroed after IG");
}

// ---------------------------------------------------------------- 6
void test_ig_attribution_shape_matches_input() {
    Model m;
    m.add_layer(new Dense(4, 2));
    Tensor x(2, 4); x.fill(1.0);
    Tensor base(2, 4); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 16);
    check(a.attributions.rows == 2, "attribution rows match input");
    check(a.attributions.cols == 4, "attribution cols match input");
    check(a.n_steps == 16, "n_steps recorded");
    check(a.method == "riemann_middle", "method recorded, default is middle");
}

// ---------------------------------------------------------------- 7
void test_completeness_delta_computed() {
    Model m;
    m.add_layer(new Dense(3, 1));
    Tensor x(1, 3); x[0][0] = 1; x[0][1] = 2; x[0][2] = 3;
    Tensor base(1, 3); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 32);
    check(std::isfinite(a.completeness_delta), "completeness delta finite");

    // Delta must actually equal sum(IG) - (F(x) - F(x')), not default to 0.
    Tensor fx = m.forward(x);
    Tensor fb = m.forward(base);
    double expected = a.attributions.sum() - (fx[0][0] - fb[0][0]);
    check_close(a.completeness_delta, expected, 1e-12, "delta formula");
}

// ---------------------------------------------------------------- 8
void test_zero_baseline_overload_matches_explicit() {
    Model m;
    m.add_layer(new Dense(3, 1));
    Tensor x(1, 3); x[0][0] = 1; x[0][1] = 2; x[0][2] = 3;

    IGAttribution a1 = integrated_gradients(m, x, 0, 32);
    IGAttribution a2 = integrated_gradients(m, x, Tensor::zeros(1, 3), 0, 32);
    for (int j = 0; j < 3; ++j)
        check_close(a1.attributions[0][j], a2.attributions[0][j], 1e-12,
                    "zero-baseline overload matches explicit");
}

// ---------------------------------------------------------------- 9
void test_shape_mismatch_throws() {
    Model m;
    m.add_layer(new Dense(3, 1));
    Tensor x(1, 3); x.fill(1.0);
    Tensor base(1, 4); base.fill(0.0);

    bool threw = false;
    try { integrated_gradients(m, x, base, 0, 8); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "baseline/input shape mismatch throws");
}

// --------------------------------------------------------------- 10
void test_target_out_of_range_throws() {
    Model m;
    m.add_layer(new Dense(3, 2));
    Tensor x(1, 3); x.fill(1.0);
    Tensor base(1, 3); base.fill(0.0);

    bool threw = false;
    try { integrated_gradients(m, x, base, 5, 8); }
    catch (const std::out_of_range&) { threw = true; }
    check(threw, "out-of-range target throws");
}

// --------------------------------------------------------------- 11
void test_n_steps_zero_yields_zeros() {
    Model m;
    m.add_layer(new Dense(3, 1));
    Tensor x(1, 3); x[0][0] = 1; x[0][1] = 2; x[0][2] = 3;
    Tensor base(1, 3); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 0);
    for (int j = 0; j < 3; ++j) {
        check(std::isfinite(a.attributions[0][j]), "no NaN from 0-step run");
        check_close(a.attributions[0][j], 0.0, 0.0, "0-step attribution is 0");
    }
}

// --------------------------------------------------------------- 12
void test_linear_model_ig_is_exact() {
    // For a linear f the gradient is constant along the path, so ANY Riemann
    // rule is exact: IG_i = (x_i - x'_i) * W_i.
    Model m = make_linear_model(2.0, 3.0, -1.0);
    Tensor x(1, 3); x[0][0] = 1.5; x[0][1] = -0.5; x[0][2] = 2.0;
    Tensor base(1, 3); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 7);
    check_close(a.attributions[0][0], 1.5 * 2.0, 1e-12, "linear IG x0");
    check_close(a.attributions[0][1], -0.5 * 3.0, 1e-12, "linear IG x1");
    check_close(a.attributions[0][2], 2.0 * -1.0, 1e-12, "linear IG x2");
    // Completeness must hold EXACTLY for a linear model (delta == 0).
    check_close(a.completeness_delta, 0.0, 1e-12, "linear completeness exact");
}

// --------------------------------------------------------------- 13
void test_completeness_axiom_holds() {
    Model m;
    m.add_layer(new Dense(4, 4));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(4, 1));

    Tensor x(1, 4);
    x[0][0] = 1.5; x[0][1] = -0.7; x[0][2] = 2.3; x[0][3] = 0.4;
    Tensor base(1, 4); base.fill(0.0);

    IGAttribution a = integrated_gradients(m, x, base, 0, 128);
    check(std::abs(a.completeness_delta) < 1e-3,
          "completeness axiom holds within 1e-3 on nonlinear MLP");
}

// --------------------------------------------------------------- 14
void test_more_steps_reduces_delta() {
    Model m;
    m.add_layer(new Dense(4, 4));
    m.add_layer(new Activation(ReLU()));
    m.add_layer(new Dense(4, 1));

    Tensor x(1, 4);
    x[0][0] = 1.5; x[0][1] = -0.7; x[0][2] = 2.3; x[0][3] = 0.4;
    Tensor base(1, 4); base.fill(0.0);

    IGAttribution few = integrated_gradients(m, x, base, 0, 4);
    IGAttribution many = integrated_gradients(m, x, base, 0, 256);
    check(std::abs(many.completeness_delta) <= std::abs(few.completeness_delta) + 1e-12,
          "completeness delta shrinks (or stays) as n_steps grows");
}

// --------------------------------------------------------------- 15
void test_attribution_antisymmetric_in_input_and_baseline() {
    // IG(x; x->x') == -IG(x'; x) — the path is traversed in reverse.
    Model m = make_linear_model(2.0, 3.0, -1.0);
    Tensor x(1, 3); x[0][0] = 1.5; x[0][1] = -0.5; x[0][2] = 2.0;
    Tensor base(1, 3); base.fill(0.0);

    IGAttribution fwd = integrated_gradients(m, x, base, 0, 16);
    IGAttribution rev = integrated_gradients(m, base, x, 0, 16);
    for (int j = 0; j < 3; ++j)
        check_close(fwd.attributions[0][j], -rev.attributions[0][j], 1e-12,
                    "antisymmetry under baseline swap");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_integrated_gradients\n";
    std::cout << "============================================================\n";

    test_input_gradient_matches_analytic();
    test_input_gradient_targets_requested_column();
    test_riemann_params_match_captum();
    test_unknown_method_throws();
    test_ig_does_not_pollute_parameter_gradients();
    test_ig_attribution_shape_matches_input();
    test_completeness_delta_computed();
    test_zero_baseline_overload_matches_explicit();
    test_shape_mismatch_throws();
    test_target_out_of_range_throws();
    test_n_steps_zero_yields_zeros();
    test_linear_model_ig_is_exact();
    test_completeness_axiom_holds();
    test_more_steps_reduces_delta();
    test_attribution_antisymmetric_in_input_and_baseline();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}