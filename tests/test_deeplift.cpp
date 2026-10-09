// test_deeplift.cpp — Tests for DeepLIFT with the Rescale rule
// (Shrikumar, Greenside, Kundaje, "Learning Important Features Through
// Propagating Activation Differences", ICML 2017, arXiv:1704.02685).
//
//   m_i       = (f(a_i) - f(a'_i)) / (a_i - a'_i)   per neuron, elementwise
//                                                       nonlinear layers only
//   G_in      = G_out ⊙ m                          nonlinear layers
//   G_in      = G_out                              linear layers (identity m)
//   contrib_l = (a_l - a'_l) ⊙ G_in,l               each layer's contribution
//   attribution = contrib_0                         the INPUT layer's — THE answer
//
// CLOSED-FORM ORACLE (why this is unusually testable). On a purely LINEAR
// network every multiplier is the identity, so DeepLIFT degenerates to
// multiplying the exact gradient by (x - x'):
//
//   attribution_i = (x_i - x'_i) · ∂F/∂x_i
//
// which is hand-computable from the weights with no tolerance at all. That is
// the headline test, and it pins the whole recursion end to end.
//
// COMPLETENESS is sum(attributions) == F(x) - F(x') — the INPUT layer's
// contribution alone, NOT the sum over all layers. Summing every layer counts
// the same output change once per layer: on a 5-layer Dense/ReLU net the input
// contribution sums to 0.575 == F(x)-F(x') while all five together sum to
// 2.875. (The implementation plan states the all-layers version in prose at
// :47 while its own pseudocode at :243 computes the correct one.)
//
// THREE PLACES WHERE THE OBVIOUS TEST IS VACUOUS, all established by
// measurement rather than assumption. Each is commented at its test.
//
//  1. "DeepLIFT != IG on a ReLU net" (plan test 11) is FALSE as written. When
//     no ReLU changes state along the straight-line path, F is exactly linear
//     on the path, the path integral EQUALS the endpoint secant, and the two
//     methods agree to ~1e-15. Measured over 300 random ReLU nets of depth
//     1..4: best relative separation 7.2e-15. They diverge only when the path
//     crosses a kink, or on a smooth activation. Test 11 asserts the correct
//     three-way statement.
//  2. ReLU cannot distinguish the Δout/Δin multiplier from f'(a) (plan test 3).
//     For every ReLU neuron the two are equal, or the attribution is 0 either
//     way, so no ReLU fixture can tell the branches apart. Tests 4-5 use Tanh,
//     where they differ by up to 1101x.
//  3. An ε-fallback fixture passes under ANY fallback value unless the adjoint
//     reaching the degenerate neuron is nonzero AND a linear layer mixes it
//     back onto an input element whose own delta_in is nonzero. Test 5's
//     fixture is built to that specification, and the resulting attribution is
//     compared against all three plausible fallback values.
//
// Coverage:
//   1.  linear closed-form oracle: attribution_i == (x_i - x'_i)·∂F/∂x_i exactly
//   2.  single-ReLU oracle: m == 1 above the crossing, 0 below it
//   3.  ReLU+Dense hand weights: completeness + per-neuron multipliers
//   4.  multiplier is Δout/Δin, NOT f'(a) (Tanh; the two differ by 1101x)
//   5.  ε-fallback is f'(a) — not 0, not 1 — with an OBSERVABLE adjoint
//   6.  whole-batch input == baseline: all zeros, no NaN/Inf, completeness 0
//   7.  completeness on a deep net with random NON-UNIFORM weights
//   8.  validation errors (shape, target, eps, empty model, empty baselines)
//   9.  no parameter-gradient residue (seeded nonzero first)
//  10.  determinism: three calls bit-identical
//  11.  DeepLIFT == IG on a kink-free ReLU path, != IG when the path crosses a
//      kink, and != IG on a smooth activation
//  12.  deep_lift_shap: K=3 == mean of 3 deep_lift calls; K=1 == deep_lift
//  13.  result metadata + zero-baseline overload

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

// WEIGHT CONVENTION (verified empirically, not assumed). Dense::forward
// computes `input * weights.transpose()` with `weights` shaped
// (out_features, in_features), so with W written in the natural (in, out)
// human layout,
//
//   f(x)[j] = Σ_i W[i][j] · x[i]
//
// checked with W = [[2], [3]] on x = [10, 100] => f = 320. Every fixture below
// is written in that (in, out) layout and transposed on the way in.
//
// A fixture written with the wrong shape is a silent HEAP OVERWRITE, not a
// compile error, because set_dense indexes weights[j][i] directly. The helper
// therefore checks its argument against the layer's own (in, out) pair.
static void set_dense(Dense* d, const std::vector<std::vector<double>>& W,
                      const std::vector<double>& b) {
    const size_t in_dim = d->weights.cols;    // Dense stores (out, in)
    const size_t out_dim = d->weights.rows;
    if (W.size() != in_dim || W.empty() || W[0].size() != out_dim) {
        std::cout << "  FAIL: set_dense got a " << W.size() << "x" << W[0].size()
                  << " matrix, layer wants " << in_dim << "x" << out_dim
                  << std::endl;
        failed++;
        return;
    }
    d->weights.fill(0.0);
    for (size_t i = 0; i < in_dim; ++i)
        for (size_t j = 0; j < out_dim; ++j) d->weights[j][i] = W[i][j];
    d->bias.fill(0.0);
    for (size_t j = 0; j < out_dim && j < b.size(); ++j) d->bias[0][j] = b[j];
}

// --- 1: the linear closed-form oracle (the headline test) --------------------

static void test_linear_closed_form_oracle() {
    // F(x) = W2 · (W1 · x + b1) + b2,  x ∈ R^2,  x' = 0
    //   W1 = [[1, 2], [3, -1]]  (2 -> 2), b1 = [0.5, -0.5]
    //   W2 = [[2], [1]]        (2 -> 1), b2 = [0.25]
    // Hand-set, so every expected value below is exact rational arithmetic:
    //   h0 = 1·x0 + 3·x1 + 0.5        h1 = 2·x0 - x1 - 0.5
    //   F  = 2·h0 + 1·h1 + 0.25 = 4·x0 + 5·x1 + 0.75
    Model m;
    Dense* d1 = new Dense(2, 2);
    set_dense(d1, {{1.0, 2.0}, {3.0, -1.0}}, {0.5, -0.5});
    m.add_layer(d1);
    Dense* d2 = new Dense(2, 1);
    set_dense(d2, {{2.0}, {1.0}}, {0.25});
    m.add_layer(d2);

    const Tensor x = make_row({1.5, -0.5});
    const Tensor xb = Tensor::zeros(1, 2);

    // ∂F/∂x by hand, cross-checked against the shipped analytic path so the
    // oracle value is not merely "whatever the implementation produces".
    const Tensor grad = input_gradient(m, x, 0);
    check_close(grad[0][0], 4.0, 1e-12, "oracle precondition: dF/dx_0 == 4");
    check_close(grad[0][1], 5.0, 1e-12, "oracle precondition: dF/dx_1 == 5");

    DeepLiftResult r = deep_lift(m, x, xb, 0);
    check_close(r.attributions[0][0], 1.5 * 4.0, 1e-12,
                "linear oracle [0][0] == (x_0 - x'_0)·dF/dx_0");
    check_close(r.attributions[0][1], (-0.5) * 5.0, 1e-12,
                "linear oracle [0][1] == (x_1 - x'_1)·dF/dx_1");

    // F(x) = 4(1.5) + 5(-0.5) + 0.75 = 4.25;  F(0) = 0.75  =>  delta 3.5
    const Tensor fx = m.forward(x), f0 = m.forward(xb);
    check_close(fx[0][0] - f0[0][0], 3.5, 1e-12, "oracle precondition: F(x)-F(0) == 3.5");
    check_close(r.attributions.sum(), 3.5, 1e-12, "linear oracle: sum == F(x)-F(x')");
    check_close(r.completeness_delta, 0.0, 1e-12, "linear oracle: completeness_delta ~ 0");
    check(r.n_layers == 2, "linear oracle: n_layers == 2");
}

// --- 2: the single-ReLU oracle ----------------------------------------------

static void test_single_relu_oracle() {
    // Model = Activation<ReLU> only, one neuron at a time so `target` never
    // masks. For each (x, x') the multiplier is the secant of ReLU:
    //   0 <= x' < x        f is x on both  => m = 1        => attr = x - x'
    //   x' < 0 < x         f is x and 0    => m = x/(x-x') => attr = x - 0 = x
    //   x, x' both <= 0    f is 0 on both  => m = 0        => attr = 0
    struct Case { double x, xp, expect; const char* note; };
    const Case cases[] = {
        { 2.0,  1.0,  1.0,  "both positive: m == 1" },
        { 3.0,  1.0,  2.0,  "both positive: m == 1" },
        { 0.5,  0.25, 0.25, "both positive, fractional: m == 1" },
        { 2.0, -1.0,  2.0,  "crossing: m == 2/3, attr == ReLU(x)-ReLU(x')" },
        { 1.0, -3.0,  1.0,  "crossing from far below" },
        {-3.0, -1.0,  0.0,  "both negative: m == 0" },
        {-1.0, -2.0,  0.0,  "both negative, x > x'" },
        {-1.0, -1.0,  0.0,  "x == x': eps-branch, still exactly 0" },
    };
    for (const Case& c : cases) {
        Model m;
        m.add_layer(new Activation<ReLU>(ReLU()));
        DeepLiftResult r = deep_lift(m, make_row({c.x}), make_row({c.xp}), 0);
        check_close(r.attributions[0][0], c.expect, 1e-15,
                    std::string("single-ReLU oracle (") + c.note + ")");
        check_close(r.completeness_delta, 0.0, 1e-15,
                    std::string("single-ReLU oracle: completeness (") + c.note + ")");
    }

    // `target` must select the column: a 2-neuron ReLU explains only target.
    Model m2;
    m2.add_layer(new Activation<ReLU>(ReLU()));
    DeepLiftResult t0 = deep_lift(m2, make_row({3.0, 0.5}), make_row({1.0, 0.25}), 0);
    DeepLiftResult t1 = deep_lift(m2, make_row({3.0, 0.5}), make_row({1.0, 0.25}), 1);
    check_close(t0.attributions[0][0], 2.0, 1e-15, "target 0: neuron 0 attributed");
    check_close(t0.attributions[0][1], 0.0, 1e-15, "target 0: neuron 1 NOT attributed");
    check_close(t1.attributions[0][0], 0.0, 1e-15, "target 1: neuron 0 NOT attributed");
    check_close(t1.attributions[0][1], 0.25, 1e-15, "target 1: neuron 1 attributed");
}

// --- 3: ReLU + Dense, hand weights ------------------------------------------

static void test_relu_dense_hand_weights() {
    // ReLU(2) -> Dense(2 -> 1), W = [[2], [1]], b = 0.5
    //   x  = [2, -3]  =>  h = [2, 0]
    //   x' = [1, -1]  =>  h' = [1, 0]
    // delta_in = [1, -2].  Neuron 1 is dead on both sides so m == 0 there and
    // contributes 0 regardless; neuron 0 carries the whole change.
    Model m;
    m.add_layer(new Activation<ReLU>(ReLU()));
    Dense* d = new Dense(2, 1);
    set_dense(d, {{2.0}, {1.0}}, {0.5});
    m.add_layer(d);

    const Tensor x = make_row({2.0, -3.0});
    const Tensor xb = make_row({1.0, -1.0});

    DeepLiftResult r = deep_lift(m, x, xb, 0);
    check_close(r.attributions[0][0], 2.0, 1e-12, "ReLU+Dense: attribution_0 == 2");
    check_close(r.attributions[0][1], 0.0, 1e-12, "ReLU+Dense: attribution_1 == 0");

    // Completeness: F(x) = 2·2 + 1·0 + 0.5 = 4.5;  F(x') = 2·1 + 0 + 0.5 = 2.5
    check_close(m.forward(x)[0][0] - m.forward(xb)[0][0], 2.0, 1e-12,
                "ReLU+Dense precondition: F(x)-F(x') == 2");
    check_close(r.attributions.sum(), 2.0, 1e-12, "ReLU+Dense: completeness holds");
    check_close(r.completeness_delta, 0.0, 1e-12, "ReLU+Dense: completeness_delta ~ 0");
    check(r.n_layers == 2, "ReLU+Dense: n_layers == 2");
}

// --- 4: the multiplier is Δout/Δin, NOT f'(a) -------------------------------

static void test_multiplier_is_secant_not_derivative() {
    // ReLU cannot make this distinction (m == f' for every ReLU neuron), so
    // this fixture uses a saturating nonlinearity. Single Tanh, x = 5, x' = 0:
    //   m     = (tanh(5) - tanh(0)) / 5      = 0.199981840852519
    //   f'(5) = 1 - tanh(5)^2                = 0.000181583230943860
    // The two differ by a factor of 1101, so the branches are not confusable
    // and a wrong branch is off by three orders of magnitude.
    Model m;
    m.add_layer(new Activation<Tanh>(Tanh()));

    const Tensor x = make_row({5.0});
    const Tensor xb = Tensor::zeros(1, 1);
    DeepLiftResult r = deep_lift(m, x, xb, 0);

    const double secant = std::tanh(5.0) / 5.0;
    const double deriv  = 1.0 - std::tanh(5.0) * std::tanh(5.0);
    const double expect = 5.0 * secant;          // == (x - x') · m

    check(std::fabs(secant / deriv) > 100.0,
          "fixture precondition: m and f'(a) differ by >100x (measured 1101x)");
    check_close(r.attributions[0][0], expect, 1e-12,
                "Tanh: attribution == (x-x')·Δout/Δin (the secant)");
    // The derivative branch would give a different, observably wrong answer.
    check(std::fabs(r.attributions[0][0] - 5.0 * deriv) > 0.5,
          "Tanh: attribution is NOT 5·f'(a) (the derivative branch is excluded)");
    check_close(r.completeness_delta, 0.0, 1e-15, "Tanh: completeness_delta ~ 0");
}

// --- 5: the ε-fallback is the derivative, with an OBSERVABLE adjoint --------

static void test_eps_fallback_is_the_derivative() {
    // THE discriminating fixture, and the one the plan does not supply.
    //
    //   Dense(2 -> 2), W (in, out) = [[1, 2], [1, 3]], b = 0
    //   Tanh
    //   x  = [1, 0]     x' = [0, 1]
    //
    // Tanh's pre-activations:
    //   a  = [1·1 + 1·0,  2·1 + 3·0] = [1, 2]
    //   a' = [1·0 + 1·1,  2·0 + 3·1] = [1, 3]
    //   delta_in = [0, -1]        <-- neuron 0 is EXACTLY degenerate
    //
    // WHY THIS FIXTURE HAS REAL POWER (this is the part that is easy to get
    // wrong). The degenerate element's own delta_in is 0, so naively the
    // fallback looks unobservable — the attribution for that element would be
    // 0 under ANY fallback. It is observable because the DENSE layer mixes:
    // for target 0, G_out = [1, 0] enters the Tanh, and
    //     G_in = G_out ⊙ m = [1 · f'(1),  0 · secant_1]
    // then the first Dense propagates it back as
    //     G[0] = G[0]·W[0][0] + G[1]·W[0][1] = f'(1)·1 + 0·2 = f'(1)
    //     G[1] = G[0]·W[1][0] + G[1]·W[1][1] = f'(1)·1 + 0·3 = f'(1)
    // and the INPUT layer's delta is [1, -1] — nonzero! So
    //     attribution = [1·f'(1), -1·f'(1)]
    // which changes with the fallback value. Measured separations:
    //     f'(a) fallback -> [ 0.419974341614026, -0.419974341614026]
    //     0       fallback -> [ 0.0,                  0.0                ]
    //     1       fallback -> [ 1.0,                 -1.0                ]
    Model m;
    Dense* d1 = new Dense(2, 2);
    set_dense(d1, {{1.0, 2.0}, {1.0, 3.0}}, {0.0, 0.0});
    m.add_layer(d1);
    m.add_layer(new Activation<Tanh>(Tanh()));

    const Tensor x = make_row({1.0, 0.0});
    const Tensor xb = make_row({0.0, 1.0});

    // Preconditions on the fixture itself.
    const Tensor a = d1->forward(x), ap = d1->forward(xb);
    check_close(a[0][0], 1.0, 1e-15, "fixture: a_0 == 1");
    check_close(ap[0][0], 1.0, 1e-15, "fixture: a'_0 == 1");
    check_close(a[0][0] - ap[0][0], 0.0, 1e-15,
                "fixture: a_0 == a'_0 EXACTLY, so the eps-branch fires");
    check_close(a[0][1] - ap[0][1], -1.0, 1e-15, "fixture: delta_in_1 == -1");
    // The dense mixing must carry the degenerate neuron's fallback onto an
    // input element whose own delta_in is nonzero — otherwise the test below
    // would be vacuous.
    check(std::fabs(d1->weights[0][0]) > 0.5,
          "fixture: the dense mixes the degenerate neuron's adjoint onto "
          "input 0 (W[0][0] != 0)");
    check_close(x[0][0] - xb[0][0], 1.0, 1e-15,
          "fixture: the INPUT element 0 has a nonzero delta (so the fallback "
          "is observable)");

    const double tanh_d1  = 1.0 - std::tanh(1.0) * std::tanh(1.0);
    const double secant1 = (std::tanh(a[0][1]) - std::tanh(ap[0][1])) /
                           (a[0][1] - ap[0][1]);
    check_close(tanh_d1, 0.419974341614026, 1e-12, "fixture: tanh'(1) value");
    check_close(secant1, 0.0310271736109136, 1e-12, "fixture: neuron 1 secant value");

    // target 0: attribution == [f'(1), -f'(1)]
    DeepLiftResult r0 = deep_lift(m, x, xb, 0);
    check_close(r0.attributions[0][0], tanh_d1, 1e-12,
                "eps-branch: fallback == f'(a), exactly");
    check_close(r0.attributions[0][1], -tanh_d1, 1e-12,
                "eps-branch: the mixed adjoint carries f'(a) to input 1");
    // Explicitly exclude the two wrong fallbacks.
    check(std::fabs(r0.attributions[0][0]) > 0.1,
          "eps-branch: NOT the 0-fallback (which would give [0, 0])");
    check(std::fabs(r0.attributions[0][0] - 1.0) > 0.5,
          "eps-branch: NOT the 1-fallback (which would give [1, -1])");

    // target 1: G_out = [0, 1] => G_in = [0·f'(1), 1·secant_1]
    //           dense: G[0] = 0·1 + secant_1·2 = 2·secant_1
    //                  G[1] = 0·1 + secant_1·3 = 3·secant_1
    //           attribution = [1·2·secant_1, -1·3·secant_1]
    // The degenerate neuron does NOT participate here (its adjoint is 0), which
    // is what makes this the SECANT-only view of the same fixture.
    DeepLiftResult r1 = deep_lift(m, x, xb, 1);
    check_close(r1.attributions[0][0], 2.0 * secant1, 1e-12,
                "eps-branch (target 1): neuron 1 uses the secant branch");
    check_close(r1.attributions[0][1], -3.0 * secant1, 1e-12,
                "eps-branch (target 1): neuron 1 scaled by adjoint 3");
    check(std::fabs(r1.completeness_delta) < 1e-12,
          "eps-branch (target 1): completeness holds");

    // The degenerate-neuron case where BOTH inputs are equal: every neuron
    // hits the ε-branch at once. The answer must be exactly zero and finite.
    DeepLiftResult rz = deep_lift(m, make_row({0.7, -0.3}), make_row({0.7, -0.3}), 0);
    check_close(rz.attributions[0][0], 0.0, 1e-15,
                "eps-branch: x == x' gives exactly 0 attribution");
    check(std::isfinite(rz.attributions[0][1]),
          "eps-branch: x == x' stays finite (an unguarded 0/0 would be NaN)");
}

// --- 6: whole-batch input == baseline ---------------------------------------

static void test_input_equals_baseline() {
    // The perfectly ordinary case that an unguarded delta_out/delta_in turns
    // into NaN. Every neuron hits the ε-branch simultaneously.
    Model m;
    Dense* d1 = new Dense(3, 4);
    d1->weights = Tensor::random(4, 3, 0.5);
    d1->bias = Tensor::random(1, 4, 0.5);
    m.add_layer(d1);
    m.add_layer(new Activation<Tanh>(Tanh()));
    Dense* d2 = new Dense(4, 2);
    d2->weights = Tensor::random(2, 4, 0.5);
    d2->bias = Tensor::random(1, 2, 0.5);
    m.add_layer(d2);
    m.add_layer(new Activation<ReLU>(ReLU()));

    const Tensor x = Tensor::random(2, 3, 0.8);
    DeepLiftResult r = deep_lift(m, x, x.clone(), 1);

    bool all_zero = true, all_finite = true;
    for (size_t i = 0; i < r.attributions.rows; ++i) {
        for (size_t j = 0; j < r.attributions.cols; ++j) {
            const double v = r.attributions[i][j];
            if (v != 0.0) all_zero = false;
            if (!std::isfinite(v)) all_finite = false;
        }
    }
    check(all_zero, "input == baseline: every attribution is EXACTLY zero");
    check(all_finite, "input == baseline: no NaN and no Inf (this is what an "
                      "unguarded delta_out/delta_in breaks)");
    check_close(r.completeness_delta, 0.0, 1e-15,
                "input == baseline: completeness_delta == 0");
}

// --- 7: completeness on a deep net, random NON-UNIFORM weights --------------

static void test_completeness_deep_net() {
    // Random non-uniform init is MANDATORY here (the plan's anti-pattern
    // section): a uniform matrix has row sums == column sums by construction,
    // so a row-vs-column mix-up would satisfy completeness vacuously.
    Model m;
    const size_t dims[5][2] = {{4, 4}, {4, 5}, {5, 3}, {3, 3}, {3, 2}};
    for (int i = 0; i < 5; ++i) {
        Dense* d = new Dense(dims[i][0], dims[i][1]);
        d->weights = Tensor::random(dims[i][1], dims[i][0], 0.3);
        d->bias = Tensor::random(1, dims[i][1], 0.3);
        m.add_layer(d);
        if (i < 4) m.add_layer(new Activation<ReLU>(ReLU()));
    }

    const Tensor x = Tensor::random(1, 4, 0.3);
    const Tensor xb = Tensor::random(1, 4, 0.3);
    DeepLiftResult r = deep_lift(m, x, xb, 1);

    check(r.n_layers == 9, "deep net: n_layers == 9 (5 Dense + 4 ReLU)");
    check(std::fabs(r.completeness_delta) < 1e-9,
          "deep net: |completeness_delta| < 1e-9");

    // Preconditions on the fixture, asserted rather than assumed: the weights
    // must NOT be uniform, or the completeness check above would pass
    // vacuously under a row/column mix-up.
    Dense* first = dynamic_cast<Dense*>(m.layers[0].get());
    double row_sums[4], col_sums[4];
    for (int i = 0; i < 4; ++i) { row_sums[i] = 0.0; col_sums[i] = 0.0; }
    for (size_t i = 0; i < first->weights.rows; ++i)
        for (size_t j = 0; j < first->weights.cols; ++j) {
            row_sums[i] += first->weights[i][j];
            col_sums[j] += first->weights[i][j];
        }
    bool non_uniform = false;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            if (std::fabs(row_sums[i] - col_sums[j]) > 1e-9) non_uniform = true;
    check(non_uniform, "deep net fixture: weights are non-uniform (row sums != column sums)");

    // A square matrix has row sums == column sums as a SET only if it is
    // symmetric, which random weights are not — assert that directly too.
    bool symmetric = true;
    for (size_t i = 0; i < first->weights.rows; ++i)
        for (size_t j = 0; j < first->weights.cols; ++j)
            if (std::fabs(first->weights[i][j] - first->weights[j][i]) > 1e-12)
                symmetric = false;
    check(!symmetric, "deep net fixture: weights matrix is not symmetric "
                      "(so a transpose mix-up is observable)");

    check(std::fabs(deep_lift(m, x, xb, 0).completeness_delta) < 1e-9,
          "deep net: completeness holds for target 0 too");
}

// --- 8: validation ----------------------------------------------------------

static void test_validation() {
    Model ok;
    Dense* d = new Dense(2, 1);
    d->weights.fill(1.0);
    ok.add_layer(d);

    bool threw = false;
    try { deep_lift(ok, make_row({1.0, 2.0}), make_row({1.0, 2.0, 3.0}), 0); }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift throws std::invalid_argument on shape mismatch");

    threw = false;
    try { deep_lift(ok, make_row({1.0, 2.0}), Tensor::zeros(1, 2), 0, 0.0); }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift throws on eps == 0");

    threw = false;
    try { deep_lift(ok, make_row({1.0, 2.0}), Tensor::zeros(1, 2), 0, -1e-10); }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift throws on negative eps");

    threw = false;
    try { deep_lift(ok, make_row({1.0, 2.0}), Tensor::zeros(1, 2), 7); }
    catch (const std::out_of_range&) { threw = true; }
    check_throws(threw, "deep_lift throws std::out_of_range for a bad target");

    threw = false;
    try { Model empty; deep_lift(empty, make_row({1.0, 2.0}), Tensor::zeros(1, 2), 0); }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift throws on an empty model");

    threw = false;
    try { deep_lift_shap(ok, make_row({1.0, 2.0}), {}, 0); }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift_shap throws on an empty baseline list");

    threw = false;
    try {
        deep_lift_shap(ok, make_row({1.0, 2.0}),
                       {Tensor::zeros(1, 2), Tensor::zeros(1, 3)}, 0);
    }
    catch (const std::invalid_argument&) { threw = true; }
    check_throws(threw, "deep_lift_shap throws on a mismatched baseline");
}

// --- 9: no parameter-gradient residue ---------------------------------------

static void test_no_gradient_residue() {
    // Seed NONZERO parameter gradients first. A suite that merely asserted
    // "still zero" after the call would pass vacuously — the values would have
    // been zero to begin with. deep_lift DOES call Dense::backward on every
    // linear layer, and Dense::backward ACCUMULATES into grad_weights
    // (layer.cpp:78), so this is a real assertion.
    Model m;
    Dense* d1 = new Dense(3, 3);
    d1->weights.fill(0.5); d1->bias.fill(0.25);
    m.add_layer(d1);
    m.add_layer(new Activation<ReLU>(ReLU()));
    Dense* d2 = new Dense(3, 2);
    d2->weights.fill(0.5); d2->bias.fill(0.25);
    m.add_layer(d2);

    double seeded = 0.0;
    for (auto& lp : m.layers) {
        Dense* dp = dynamic_cast<Dense*>(lp.get());
        if (dp) { dp->grad_weights.fill(7.0); dp->grad_bias.fill(5.0); seeded = 7.0; }
    }
    check(seeded != 0.0, "residue fixture: gradients really were seeded nonzero");

    deep_lift(m, Tensor::random(1, 3, 0.7), Tensor::random(1, 3, 0.7), 1);

    bool all_zero = true;
    for (auto& lp : m.layers)
        for (auto* g : lp->gradients())
            for (size_t i = 0; i < g->rows; ++i)
                for (size_t j = 0; j < g->cols; ++j)
                    if (g->data[i * g->cols + j] != 0.0) all_zero = false;
    check(all_zero, "deep_lift zeroes pre-existing parameter gradients");
}

// --- 10: determinism --------------------------------------------------------

static void test_determinism() {
    Model m;
    Dense* d1 = new Dense(3, 4);
    d1->weights = Tensor::random(4, 3, 0.6);
    d1->bias = Tensor::random(1, 4, 0.6);
    m.add_layer(d1);
    m.add_layer(new Activation<Tanh>(Tanh()));
    Dense* d2 = new Dense(4, 2);
    d2->weights = Tensor::random(2, 4, 0.6);
    d2->bias = Tensor::random(1, 2, 0.6);
    m.add_layer(d2);

    const Tensor x = Tensor::random(1, 3, 0.6);
    const Tensor xb = Tensor::random(1, 3, 0.6);

    DeepLiftResult a = deep_lift(m, x, xb, 1);
    DeepLiftResult b = deep_lift(m, x, xb, 1);
    DeepLiftResult c = deep_lift(m, x, xb, 1);

    bool identical = true;
    for (size_t i = 0; i < a.attributions.rows; ++i)
        for (size_t j = 0; j < a.attributions.cols; ++j)
            if (a.attributions[i][j] != b.attributions[i][j] ||
                b.attributions[i][j] != c.attributions[i][j]) identical = false;
    check(identical, "deep_lift is bit-identical across three calls");
    check(a.completeness_delta == b.completeness_delta,
          "deep_lift completeness_delta is bit-identical across calls");
}

// --- 11: DeepLIFT vs Integrated Gradients -----------------------------------

static void test_vs_integrated_gradients() {
    // THREE assertions, all measured rather than assumed. See the header note
    // on why "they differ on a ReLU net" is false as stated.
    //
    // (a) KINK-FREE ReLU path: DeepLIFT == IG to ~1e-15. Correct, not a bug —
    //     F is exactly linear along the path, so the path integral equals the
    //     secant. Bias (5, 5) keeps both pre-activations positive all the way.
    Model rn;
    Dense* rd = new Dense(2, 2);
    set_dense(rd, {{1.0, 2.0}, {1.0, 3.0}}, {5.0, 5.0});
    rn.add_layer(rd);
    rn.add_layer(new Activation<ReLU>(ReLU()));
    Dense* rd2 = new Dense(2, 1);
    set_dense(rd2, {{1.0}, {2.0}}, {0.0});
    rn.add_layer(rd2);

    const Tensor x = make_row({6.0, 3.0});
    const Tensor xb = make_row({4.0, 2.0});

    // Fixture precondition: the path must not cross a ReLU kink.
    Dense* rda = dynamic_cast<Dense*>(rn.layers[0].get());
    double lo = 1e300;
    for (int t = 0; t <= 10; ++t) {
        Tensor p(1, 2);
        for (size_t j = 0; j < 2; ++j)
            p[0][j] = xb[0][j] + (t / 10.0) * (x[0][j] - xb[0][j]);
        Tensor h = rda->forward(p);
        for (size_t j = 0; j < 2; ++j) lo = std::min(lo, h[0][j]);
    }
    check(lo > 0.0, "fixture: no ReLU kink is crossed along the path (preact > 0)");

    DeepLiftResult rdl = deep_lift(rn, x, xb, 0);
    IGAttribution rig = integrated_gradients(rn, x, xb, 0, 512);
    double rel = 0.0, ref = 0.0;
    for (size_t j = 0; j < 2; ++j) {
        rel = std::max(rel, std::fabs(rdl.attributions[0][j] - rig.attributions[0][j]));
        ref = std::max(ref, std::fabs(rdl.attributions[0][j]));
    }
    check(rel / ref < 1e-12,
          "kink-free ReLU path: DeepLIFT == IG to <1e-12 relative "
          "(secant == path integral)");
    check(std::fabs(rdl.completeness_delta) < 1e-12,
          "kink-free ReLU path: DeepLIFT completeness is EXACT");

    // (b) KINK-CROSSED ReLU path: they genuinely differ. Bias 0 and x, x' on
    //     opposite sides of a zero pre-activation, so F is piecewise linear
    //     along the path and the path integral is strictly less than the
    //     endpoint secant.
    Model kn;
    Dense* kd = new Dense(2, 2);
    set_dense(kd, {{1.0, 2.0}, {1.0, 3.0}}, {0.0, 0.0});
    kn.add_layer(kd);
    kn.add_layer(new Activation<ReLU>(ReLU()));
    Dense* kd2 = new Dense(2, 1);
    set_dense(kd2, {{1.0}, {2.0}}, {0.0});
    kn.add_layer(kd2);

    const Tensor kx = make_row({1.5, 0.5});
    const Tensor kxb = make_row({-0.5, 0.25});
    DeepLiftResult kdl = deep_lift(kn, kx, kxb, 0);
    IGAttribution kig = integrated_gradients(kn, kx, kxb, 0, 512);
    rel = 0.0; ref = 0.0;
    for (size_t j = 0; j < 2; ++j) {
        rel = std::max(rel, std::fabs(kdl.attributions[0][j] - kig.attributions[0][j]));
        ref = std::max(ref, std::fabs(kdl.attributions[0][j]));
    }
    check(rel / ref > 1e-6,
          "kink-crossed ReLU path: DeepLIFT differs from IG by >1e-6 relative "
          "(the secant is an ENDPOINT value, IG integrates along the path)");
    check(std::fabs(kdl.completeness_delta) < std::fabs(kig.completeness_delta),
          "kink-crossed ReLU path: DeepLIFT's completeness error stays below IG's");

    // (c) SMOOTH activation: they diverge even without a kink, because the
    //     secant is not the average derivative. This is the assertion that
    //     would fail if DeepLIFT were an IG clone.
    Model tn;
    Dense* td = new Dense(2, 2);
    set_dense(td, {{1.0, 2.0}, {1.0, 3.0}}, {0.0, 0.0});
    tn.add_layer(td);
    tn.add_layer(new Activation<Tanh>(Tanh()));
    Dense* td2 = new Dense(2, 1);
    set_dense(td2, {{1.0}, {2.0}}, {0.0});
    tn.add_layer(td2);

    DeepLiftResult tdl = deep_lift(tn, kx, kxb, 0);
    IGAttribution tig = integrated_gradients(tn, kx, kxb, 0, 512);
    rel = 0.0; ref = 0.0;
    for (size_t j = 0; j < 2; ++j) {
        rel = std::max(rel, std::fabs(tdl.attributions[0][j] - tig.attributions[0][j]));
        ref = std::max(ref, std::fabs(tdl.attributions[0][j]));
    }
    check(rel / ref > 1e-7,
          "Tanh net: DeepLIFT differs from IG by >1e-7 relative (secant != "
          "averaged derivative — this is what distinguishes the two methods)");
    check(std::fabs(tdl.completeness_delta) < 1e-12,
          "Tanh net: DeepLIFT completeness is exact where IG's is not");
}

// --- 12: deep_lift_shap -----------------------------------------------------

static void test_deeplift_shap() {
    Model m;
    Dense* d1 = new Dense(3, 3);
    d1->weights = Tensor::random(3, 3, 0.4);
    d1->bias = Tensor::random(1, 3, 0.4);
    m.add_layer(d1);
    m.add_layer(new Activation<ReLU>(ReLU()));
    Dense* d2 = new Dense(3, 2);
    d2->weights = Tensor::random(2, 3, 0.4);
    d2->bias = Tensor::random(1, 2, 0.4);
    m.add_layer(d2);

    const Tensor x = Tensor::random(1, 3, 0.5);
    const std::vector<Tensor> bl = {
        Tensor::random(1, 3, 0.5), Tensor::random(1, 3, 0.5), Tensor::random(1, 3, 0.5)};

    DeepLiftResult s3 = deep_lift_shap(m, x, bl, 0);

    // K=3 must equal the mean of the three individual passes to 1e-15.
    Tensor manual = Tensor::zeros(1, 3);
    for (size_t k = 0; k < bl.size(); ++k) {
        DeepLiftResult one = deep_lift(m, x, bl[k], 0);
        for (size_t j = 0; j < 3; ++j) manual[0][j] += one.attributions[0][j] / 3.0;
    }
    double worst = 0.0;
    for (size_t j = 0; j < 3; ++j)
        worst = std::max(worst, std::fabs(manual[0][j] - s3.attributions[0][j]));
    check(worst < 1e-15, "deep_lift_shap K=3 == mean of 3 deep_lift calls to 1e-15");
    check(s3.n_layers == 3, "deep_lift_shap: n_layers echoes the model");
    check(std::fabs(s3.completeness_delta) < 1e-9,
          "deep_lift_shap: completeness_delta stays ~0 under averaging");

    // K=1 must equal plain deep_lift EXACTLY (bit-identical: multiplying by
    // 1.0 and adding 0.0 cannot change a double).
    DeepLiftResult s1 = deep_lift_shap(m, x, {bl[0]}, 0);
    DeepLiftResult d1only = deep_lift(m, x, bl[0], 0);
    bool bit_identical = true;
    for (size_t j = 0; j < 3; ++j)
        if (s1.attributions[0][j] != d1only.attributions[0][j]) bit_identical = false;
    check(bit_identical, "deep_lift_shap K=1 is bit-identical to plain deep_lift");
}

// --- 13: metadata + zero-baseline overload ----------------------------------

static void test_metadata_and_overload() {
    Model m;
    Dense* d = new Dense(3, 1);
    set_dense(d, {{1.0}, {2.0}, {3.0}}, {0.5});
    m.add_layer(d);

    const Tensor x = make_row({1.0, -1.0, 0.5});
    DeepLiftResult via_base = deep_lift(m, x, Tensor::zeros(1, 3), 0);
    DeepLiftResult via_over = deep_lift(m, x, 0);

    bool same = true;
    for (size_t j = 0; j < 3; ++j)
        if (via_base.attributions[0][j] != via_over.attributions[0][j]) same = false;
    check(same, "zero-baseline overload matches an explicit zero baseline");
    check(via_over.attributions.rows == x.rows &&
              via_over.attributions.cols == x.cols,
          "metadata: attributions have the input's shape");
    check(via_over.n_layers == 1, "metadata: n_layers == 1");
    check(std::isfinite(via_over.attributions.sum()),
          "metadata: attribution sum is finite");
    // attribution_i == x_i · c_i with c = [1, 2, 3]:
    //   [1·1, -1·2, 0.5·3] = [1, -2, 1.5],  sum 0.5
    // F(x) = 1·1 + 2·(-1) + 3·0.5 + 0.5 = 1;  F(0) = 0.5  =>  delta 0.5
    check_close(via_over.attributions.sum(), 0.5, 1e-12,
                "zero-baseline overload: completeness holds");
    check_close(via_over.attributions[0][0], 1.0, 1e-12, "metadata: attr_0 == 1");
    check_close(via_over.attributions[0][1], -2.0, 1e-12, "metadata: attr_1 == -2");
    check_close(via_over.attributions[0][2], 1.5, 1e-12, "metadata: attr_2 == 1.5");
}

int main() {
    std::cout << "\n============================================================\n";
    std::cout << "  test_deeplift\n";
    std::cout << "============================================================\n";

    test_linear_closed_form_oracle();
    test_single_relu_oracle();
    test_relu_dense_hand_weights();
    test_multiplier_is_secant_not_derivative();
    test_eps_fallback_is_the_derivative();
    test_input_equals_baseline();
    test_completeness_deep_net();
    test_validation();
    test_no_gradient_residue();
    test_determinism();
    test_vs_integrated_gradients();
    test_deeplift_shap();
    test_metadata_and_overload();

    std::cout << "\n============================================================\n";
    std::cout << "  PASSED: " << passed << "    FAILED: " << failed << "\n";
    std::cout << "============================================================\n";
    return failed == 0 ? 0 : 1;
}