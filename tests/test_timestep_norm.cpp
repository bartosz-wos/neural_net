// test_timestep_norm.cpp — Tests for TimestepNorm.
//
// Ma, Yang, Xiong, Chen, Yu, Zhang, Ma, Zhou, "MEGALODON: Efficient LLM
// Pretraining and Inference with Unlimited Context Length"
// (https://arxiv.org/abs/2404.08801), §3.2 (Eq. 4) + Appendix B.2.
//
// TimestepNorm = Group Normalization along the *timestep* axis, made causal
// for autoregressive modeling by using CUMULATIVE statistics: at step t only
// x_1..x_t enter the mean/variance, so no future information leaks.
//
//   mu_t[g]    = 1/(t*dg) * sum_{i<=t} sum_{j in g} x_ij
//   var_t[g]   = 1/(t*dg) * sum_{i<=t} sum_{j in g} (x_ij - mu_t[g])^2
//   y_t,j      = (gamma_j + 1) * (x_tj - mu_t[g(j)]) / sqrt(var + eps) + beta_j
//
// The "+1" on gamma is the paper's plus-1 reparameterization (Appendix B.2):
// gamma is initialized at 0 so weight decay keeps the effective scale near 1.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <memory>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

// Loss used by every FD check: L = 0.5 * sum (y - target)^2
static double sq_loss(const Tensor& y, const Tensor& target) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y(i, j) - target(i, j);
            L += 0.5 * d * d;
        }
    return L;
}

static Tensor make_target(size_t n, size_t d, unsigned seed) {
    Tensor t(n, d);
    srand(seed);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) t(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
    return t;
}

// =====================================================================
// Test 1: Constructor validation + accessors
// =====================================================================
static void test_constructor_validation() {
    cout << endl << "-- Test 1: TimestepNorm constructor validation --" << endl;

    bool threw_features = false;
    try { TimestepNorm t(0); } catch (...) { threw_features = true; }
    check("features=0 throws", threw_features);

    bool threw_eps = false;
    try { TimestepNorm t(4, 0, 0.0); } catch (...) { threw_eps = true; }
    check("eps=0 throws", threw_eps);

    bool threw_eps_neg = false;
    try { TimestepNorm t(4, 2, -1e-5); } catch (...) { threw_eps_neg = true; }
    check("eps<0 throws", threw_eps_neg);

    bool ok_construct = false;
    try { TimestepNorm t(8, 2, 1e-5); ok_construct = true; } catch (...) {}
    check("valid construct (features=8, groups=2)", ok_construct);

    bool ok_single = false;
    try { TimestepNorm t(8, 0, 1e-5); ok_single = true; } catch (...) {}
    check("num_groups=0 (single group) constructs", ok_single);

    bool threw_too_many = false;
    try { TimestepNorm t(4, 5, 1e-5); } catch (...) { threw_too_many = true; }
    check("num_groups > features throws", threw_too_many);

    bool threw_indivisible = false;
    try { TimestepNorm t(6, 4, 1e-5); } catch (...) { threw_indivisible = true; }
    check("features % num_groups != 0 throws", threw_indivisible);

    // k == features (group size 1) is legal — GroupNorm allows it.
    bool ok_per_feature = false;
    try { TimestepNorm t(4, 4, 1e-5); ok_per_feature = true; } catch (...) {}
    check("num_groups == features constructs (group size 1)", ok_per_feature);
}

// =====================================================================
// Test 2: Accessors
// =====================================================================
static void test_accessors() {
    cout << endl << "-- Test 2: accessors --" << endl;

    TimestepNorm t(8, 2, 1e-7);
    check("features() == 8", t.features() == 8);
    check("num_groups() == 2", t.num_groups() == 2);
    check("eps() == 1e-7", std::abs(t.eps() - 1e-7) < 1e-18);
    check("name() == \"TimestepNorm\"", t.name() == "TimestepNorm");

    // Single-group default
    TimestepNorm s(8);
    check("default num_groups() == 1", s.num_groups() == 1);

    // Plus-1 reparameterization: gamma init 0, so effective scale is 1.
    double gmax = 0.0;
    for (size_t j = 0; j < 8; ++j) gmax = std::max(gmax, std::abs(t.gamma(0, j)));
    check("gamma initialized to 0 (plus-1 reparam)", gmax < 1e-18);

    double bmax = 0.0;
    for (size_t j = 0; j < 8; ++j) bmax = std::max(bmax, std::abs(t.beta(0, j)));
    check("beta initialized to 0", bmax < 1e-18);
}

// =====================================================================
// Test 3: Parameter / gradient contract
// =====================================================================
static void test_parameter_contract() {
    cout << endl << "-- Test 3: parameter/gradient contract --" << endl;

    TimestepNorm t(6, 3, 1e-5);
    auto p = t.parameters();
    auto g = t.gradients();
    check("2 parameters (gamma, beta)", p.size() == 2);
    check("2 gradients", g.size() == 2);
    check("gamma shape (1, 6)", p[0]->rows == 1 && p[0]->cols == 6);
    check("beta shape (1, 6)", p[1]->rows == 1 && p[1]->cols == 6);
    check("grad gamma shape (1, 6)", g[0]->rows == 1 && g[0]->cols == 6);
    check("grad beta shape (1, 6)", g[1]->rows == 1 && g[1]->cols == 6);
}

// =====================================================================
// Test 4: Forward shape + finiteness
// =====================================================================
static void test_forward_shape() {
    cout << endl << "-- Test 4: forward shape + finiteness --" << endl;

    TimestepNorm t(8, 2, 1e-5);
    Tensor x(5, 8);
    srand(1234);
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 8; ++j) x(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;

    Tensor y = t.forward(x);
    check("forward shape (5,8) -> (5,8)", y.rows == 5 && y.cols == 8);

    bool finite = true;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j)
            if (!std::isfinite(y(i, j))) finite = false;
    check("forward finite", finite);

    // n=1 degenerate: single timestep, var == 0
    Tensor x1(1, 8);
    for (size_t j = 0; j < 8; ++j) x1(0, j) = 0.3 * (j + 1);
    Tensor y1 = t.forward(x1);
    check("n=1 forward finite", std::isfinite(y1(0, 0)));
}

// =====================================================================
// Test 5: Hand-derived forward reference (d=2, single group)
// =====================================================================
//
// With n=3, d=2, num_groups=1, gamma=0, beta=0, eps tiny:
//
//   t=0: mu = x[0,0], var = 0  ->  y = (x - mu)/sqrt(0+eps) = 0
//   t=1: mu = (x00+x01+x10+x11)/4
//        var = ((x00-mu)^2+(x01-mu)^2+(x10-mu)^2+(x11-mu)^2)/4
//        y[t,j] = (x[t,j]-mu)/sqrt(var+eps)
//   t=2: same over all 6 entries.
static void test_hand_derived_forward() {
    cout << endl << "-- Test 5: hand-derived forward reference --" << endl;

    const size_t n = 3, d = 2;
    const double eps = 1e-8;

    Tensor x(n, d);
    x(0, 0) = 1.0;  x(0, 1) = 2.0;
    x(1, 0) = 3.0;  x(1, 1) = -1.0;
    x(2, 0) = -2.0; x(2, 1) = 0.5;

    TimestepNorm t(d, 0, eps);
    Tensor y = t.forward(x);

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double cnt = (double)(i + 1);
        double mu = 0.0;
        for (size_t r = 0; r <= i; ++r)
            for (size_t j = 0; j < d; ++j) mu += x(r, j);
        mu /= cnt * d;

        double var = 0.0;
        for (size_t r = 0; r <= i; ++r)
            for (size_t j = 0; j < d; ++j) {
                double dd = x(r, j) - mu;
                var += dd * dd;
            }
        var /= cnt * d;

        double sigma = std::sqrt(var + eps);
        for (size_t j = 0; j < d; ++j) {
            double expect = (x(i, j) - mu) / sigma;
            max_err = std::max(max_err, std::abs(y(i, j) - expect));
        }
    }
    cout << "    max_err = " << scientific << setprecision(3) << max_err << defaultfloat << endl;
    check("hand-derived forward matches (< 1e-10)", max_err < 1e-10);

    // t=0 output is exactly 0 when each group has size 1: with a single
    // element in the group, mu == x[0][j] so (x - mu) vanishes identically.
    // NOTE: this does NOT hold for groups of size > 1 — with d=2 and one
    // group, var at t=0 is 0.25 (not 0), so y[0][0] = (1.0-1.5)/sqrt(0.25+eps)
    // ≈ -1. The t=0 row for the single-group case is already validated
    // element-by-element by the hand-derived reference above (max_err 2.2e-16).
    TimestepNorm per_feature(d, d, eps);  // d groups of size 1
    Tensor y_pf = per_feature.forward(x);
    check("t=0 output is exactly 0 for size-1 groups", std::abs(y_pf(0, 0)) < 1e-15 &&
                                                      std::abs(y_pf(0, 1)) < 1e-15);
}

// =====================================================================
// Test 6: Causality — y[s] must not depend on x[t] for t > s
// =====================================================================
static void test_causality() {
    cout << endl << "-- Test 6: causality (no future leakage) --" << endl;

    const size_t n = 5, d = 6;
    Tensor x(n, d);
    srand(77);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) x(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;

    TimestepNorm t(d, 2, 1e-5);
    Tensor y_ref = t.forward(x);

    // Perturb only the LAST timestep. y[0..n-2] must be bit-exact unchanged.
    Tensor x2 = x.clone();
    for (size_t j = 0; j < d; ++j) x2(n - 1, j) += 5.0;
    Tensor y2 = t.forward(x2);

    double max_diff = 0.0;
    for (size_t i = 0; i < n - 1; ++i)
        for (size_t j = 0; j < d; ++j)
            max_diff = std::max(max_diff, std::abs(y_ref(i, j) - y2(i, j)));
    check("perturbing x[n-1] leaves y[0..n-2] bit-exact", max_diff == 0.0);

    // y[n-1] itself MUST change, or the test is vacuous.
    double changed = 0.0;
    for (size_t j = 0; j < d; ++j) changed = std::max(changed, std::abs(y_ref(n - 1, j) - y2(n - 1, j)));
    check("perturbing x[n-1] does change y[n-1] (non-vacuous)", changed > 1e-6);
}

// =====================================================================
// Test 7: Group split — each group uses only its own features
// =====================================================================
static void test_group_isolation() {
    cout << endl << "-- Test 7: group isolation --" << endl;

    const size_t n = 4, d = 4, g = 2, dg = 2;
    Tensor x(n, d);
    srand(999);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) x(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;

    TimestepNorm t(d, g, 1e-8);
    Tensor y = t.forward(x);

    double max_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t gi = 0; gi < g; ++gi) {
            double cnt = (double)(i + 1);
            double mu = 0.0;
            for (size_t r = 0; r <= i; ++r)
                for (size_t j = gi * dg; j < (gi + 1) * dg; ++j) mu += x(r, j);
            mu /= cnt * dg;
            double var = 0.0;
            for (size_t r = 0; r <= i; ++r)
                for (size_t j = gi * dg; j < (gi + 1) * dg; ++j) {
                    double dd = x(r, j) - mu;
                    var += dd * dd;
                }
            var /= cnt * dg;
            double sigma = std::sqrt(var + 1e-8);
            for (size_t j = gi * dg; j < (gi + 1) * dg; ++j) {
                double expect = (x(i, j) - mu) / sigma;
                max_err = std::max(max_err, std::abs(y(i, j) - expect));
            }
        }
    }
    cout << "    max_err = " << scientific << setprecision(3) << max_err << defaultfloat << endl;
    check("hand-derived grouped forward matches (< 1e-10)", max_err < 1e-10);
}

// =====================================================================
// Test 8: Determinism
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 8: determinism --" << endl;

    Tensor x(6, 8);
    srand(31337);
    for (size_t i = 0; i < 6; ++i)
        for (size_t j = 0; j < 8; ++j) x(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;

    TimestepNorm t(8, 2, 1e-5);
    Tensor y1 = t.forward(x);
    Tensor y2 = t.forward(x);
    double md = 0.0;
    for (size_t i = 0; i < y1.rows; ++i)
        for (size_t j = 0; j < y1.cols; ++j) md = std::max(md, std::abs(y1(i, j) - y2(i, j)));
    check("two consecutive forwards bit-exact", md == 0.0);
}

int main() {
    cout << "==========================================" << endl;
    cout << "  TimestepNorm Tests" << endl;
    cout << "  MEGALODON §3.2 (Eq. 4) + Appendix B.2" << endl;
    cout << "==========================================" << endl;

    test_constructor_validation();
    test_accessors();
    test_parameter_contract();
    test_forward_shape();
    test_hand_derived_forward();
    test_causality();
    test_group_isolation();
    test_determinism();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
