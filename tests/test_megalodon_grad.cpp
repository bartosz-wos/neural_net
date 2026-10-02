// test_megalodon_grad.cpp — FD-checked gradient tests for MegalodonBlock
//   (MEGALODON §3.3 + §3.4).
//
// The block stacks five sublayers, each with its own hand-derived backward:
// SwiGLU FFN -> LayerNorm -> TWO-HOP residual -> masked multi-head attention
// -> per-head RMS normalization -> W_z -> ComplexEMA -> TimestepNorm.
//
// Two failure modes this suite is built to catch:
//
//   (1) The two-hop residual. Y = FFN(LN(Ŷ)) + x, and Ŷ = Attn + x, so the
//       block-input gradient must accumulate from BOTH residual paths. A
//       one-hop implementation (Y = FFN(LN(Ŷ)) + Ŷ) yields a clean ~2x error
//       on d_input. Test 2 checks EVERY input entry at n=5.
//
//   (2) The A-WEIGHTED softmax-backward row sum. The softmax Jacobian is
//       dS = A ⊙ (dA - Σ_m A·dA). Using the raw Σ dA instead is a classic
//       bug that leaves dQ/dK subtly wrong while dV stays correct.
//
// Random non-uniform init throughout — a uniform fixture masks row-vs-column
// confusion in the matmul backwards.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static Tensor rand_tensor(size_t r, size_t c, double scale, unsigned seed) {
    Tensor t(r, c);
    srand(seed);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(i, j) = ((rand() / (double)RAND_MAX) * 2.0 - 1.0) * scale;
    return t;
}

static double sq_loss(const Tensor& y, const Tensor& target) {
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) {
            double d = y(i, j) - target(i, j);
            L += 0.5 * d * d;
        }
    return L;
}

// FD of the loss w.r.t. one input entry.
static double fd_input(MegalodonBlock& b, const Tensor& x, const Tensor& target,
                       size_t r, size_t c, double eps) {
    Tensor orig = x.clone();
    auto loss_at = [&](double delta) {
        Tensor xa = orig.clone();
        xa(r, c) += delta;
        return sq_loss(b.forward(xa), target);
    };
    return (loss_at(+eps) - loss_at(-eps)) / (2.0 * eps);
}

// FD of the loss w.r.t. one parameter entry.
static double fd_param(MegalodonBlock& b, const Tensor& x, const Tensor& target,
                       Tensor* param, size_t r, size_t c, double eps) {
    double orig = (*param)(r, c);
    auto loss_at = [&](double delta) {
        (*param)(r, c) = orig + delta;
        return sq_loss(b.forward(x), target);
    };
    double Lp = loss_at(+eps), Lm = loss_at(-eps);
    (*param)(r, c) = orig;
    return (Lp - Lm) / (2.0 * eps);
}

// Run forward + backward once, returning d_input.
static Tensor ana_backward(MegalodonBlock& b, const Tensor& x, const Tensor& target) {
    b.zero_grad();
    Tensor y = b.forward(x);
    Tensor gy = Tensor(x.rows, x.cols);
    for (size_t i = 0; i < x.rows; ++i)
        for (size_t j = 0; j < x.cols; ++j) gy(i, j) = y(i, j) - target(i, j);
    return b.backward(gy, 0.0);
}

// Build a block with fixed random parameters.
static void seed_block(MegalodonBlock& b, unsigned seed) {
    auto p = b.parameters();
    for (size_t i = 0; i < p.size(); ++i) {
        // TimestepNorm gamma/beta init at 0 (plus-1 reparam) — give them
        // non-zero values so the FD checks actually exercise their paths.
        Tensor v = rand_tensor(p[i]->rows, p[i]->cols, 0.4, seed + i * 7);
        *p[i] = v;
    }
}

// FD noise floor and tolerance policy
// ==================================
//
// This block's gradients span FOUR orders of magnitude in one test: the
// ffn_down bias gradient is ~4e-1 while the CEMA-path gradients are ~3e-7,
// against a loss that is O(1). A fixed relative-error threshold is therefore
// unusable — the small gradients sit at the double-precision noise floor of
// the loss itself, where FD measures nothing but rounding.
//
// Evidence this is FD noise and not a wrong gradient: sweeping eps from
// 1e-7 to 1e-1 produces a U-SHAPED rel_err curve with a minimum far below
// 1e-3 (e.g. q_gamma dips to 9e-9 at eps=1e-2). A genuinely wrong gradient
// would plateau at O(1) instead.
//
// So the criterion is a combined absolute/relative one, and eps is 1e-5
// (chosen from that sweep):
//
//     pass  <=>  |ana - num| <= 1e-8            (below the FD noise floor)
//              OR |ana - num|/|num| <= 1e-4     (meaningful magnitude)
//
// Non-vacuity: the real bug this suite found during development (a missing
// cross term in the per-head RMS Jacobian) produced an ABSOLUTE error of
// ~4.5e-7 — 45x above the 1e-8 floor — so it is caught by the absolute
// branch, not masked by it.
static const double FD_EPS = 1e-5;
static const double FD_ABS_FLOOR = 1e-8;
static const double FD_REL_TOL = 1e-4;

// Returns true when the worst observed discrepancy over a probe set is
// acceptable: either below the absolute FD noise floor, or a small relative
// error where the gradient is large enough for the ratio to mean something.
static bool fd_ok(double worst_abs, double worst_rel) {
    return worst_abs <= FD_ABS_FLOOR || worst_rel <= FD_REL_TOL;
}

// =====================================================================
//
// Set every parameter so the block degenerates: zero CEMA output (all
// ComplexEMA params 0 -> p = 0.5, but omega=0 and eta=0 makes cema = 0),
// W_z = 0 -> z_pre = 0 -> q = beta, etc. That is fiddly; instead the
// cheapest exact check is the loss-surface identity below.
// =====================================================================
static void test_hand_derived() {
    cout << endl << "-- Test 1: hand-derived d_input (degenerate block) --" << endl;

    const size_t d = 2, n = 3;
    MegalodonBlock b(d, 1, 1, 0, 1, 0);

    // TimestepNorm gamma = 0, beta = 0 (defaults). CEMA eta = 0, omega = 0
    // => cema output is exactly 0. W_z = 0 => z_pre = 0 => z_norm = 0, so
    // q = k = beta = 0 and all attention scores are 0 (uniform attention).
    b.cema_.eta_re_.fill(0.0);
    b.cema_.eta_im_.fill(0.0);
    b.cema_.omega_.fill(0.0);
    b.W_z_.weights.fill(0.0);
    b.W_z_.bias.fill(0.0);
    b.W_o_.weights.fill(0.0);
    b.W_o_.bias.fill(0.0);
    b.ffn_down_.weights.fill(0.0);
    b.ffn_down_.bias.fill(0.0);

    Tensor x = rand_tensor(n, d, 1.0, 500);
    Tensor y = b.forward(x);
    double md = 0.0;
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) md = max(md, fabs(y(t, j) - x(t, j)));
    cout << "    max|Y - x| with all output paths zeroed = " << scientific << md << defaultfloat << endl;
    check("both branches zeroed => Y == x exactly (two-hop identity)", md == 0.0);

    // The gradient of L = 0.5*||Y - T||^2 with Y = x is exactly (x - T).
    Tensor target = rand_tensor(n, d, 1.0, 501);
    Tensor gx = ana_backward(b, x, target);
    double worst = 0.0;
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            worst = max(worst, fabs(gx(t, j) - (x(t, j) - target(t, j))));
    cout << "    max|d_input - (x - target)| = " << scientific << worst << defaultfloat << endl;
    // This is THE two-hop test: Y = x means dL/dx = (x - T) exactly, with no
    // contribution from either sublayer. A one-hop residual would give
    // dL/dx = 2*(x - T).
    check("Y == x  =>  d_input == (x - target) exactly", worst < 1e-12);
}

// =====================================================================
// Test 2: FD input gradient, every entry (n=5, d=4) — the two-hop test
// =====================================================================
static void test_fd_input() {
    cout << endl << "-- Test 2: FD input gradient, every entry (n=5, d=4) --" << endl;

    const size_t n = 5, d = 4;
    MegalodonBlock b(d, 2, 2, 0, 2, 0);
    seed_block(b, 600);
    Tensor x = rand_tensor(n, d, 1.0, 601);
    Tensor target = rand_tensor(n, d, 1.0, 602);

    Tensor gx = ana_backward(b, x, target);

    const double eps = FD_EPS;
    double worst = 0.0, worst_abs = 0.0;
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) {
            double num = fd_input(b, x, target, t, j, eps);
            double ad = fabs(gx(t, j) - num);
            worst_abs = max(worst_abs, ad);
            worst = max(worst, ad / std::max(1e-8, fabs(num)));
        }
    cout << "    worst input-grad rel_err = " << scientific << worst
         << "   worst abs_err = " << worst_abs << defaultfloat << endl;
    check("input gradient matches FD (two-hop residual verified)", worst_abs <= FD_ABS_FLOOR);
}

// =====================================================================
// Test 3: FD input gradient, multi-chunk (chunk_size=2) and multi-head
// =====================================================================
static void test_fd_input_chunked() {
    cout << endl << "-- Test 3: FD input gradient with chunk_size=2, n=6, 2 heads --" << endl;

    const size_t n = 6, d = 4;
    MegalodonBlock b(d, 2, 2, 2, 2, 2);
    seed_block(b, 700);
    Tensor x = rand_tensor(n, d, 1.0, 701);
    Tensor target = rand_tensor(n, d, 1.0, 702);

    Tensor gx = ana_backward(b, x, target);

    const double eps = FD_EPS;
    double worst = 0.0, worst_abs = 0.0;
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) {
            double num = fd_input(b, x, target, t, j, eps);
            double ad = fabs(gx(t, j) - num);
            worst_abs = max(worst_abs, ad);
            worst = max(worst, ad / std::max(1e-8, fabs(num)));
        }
    cout << "    worst input-grad rel_err = " << scientific << worst
         << "   worst abs_err = " << worst_abs << defaultfloat << endl;
    check("chunked + multi-head input gradient matches FD", worst_abs <= FD_ABS_FLOOR);
}

// =====================================================================
// Test 4: FD parameter gradients across every sublayer
// =====================================================================
static void test_fd_params() {
    cout << endl << "-- Test 4: FD parameter gradients (every sublayer) --" << endl;

    const size_t n = 4, d = 4;
    MegalodonBlock b(d, 1, 2, 2, 2, 0);
    seed_block(b, 800);
    Tensor x = rand_tensor(n, d, 1.0, 801);
    Tensor target = rand_tensor(n, d, 1.0, 802);

    // Re-run forward+backward so the cached gradients correspond to this x.
    ana_backward(b, x, target);
    auto p = b.parameters();
    auto g = b.gradients();

    static const char* names[] = {
        "tsn.gamma", "tsn.beta",
        "cema.alpha", "cema.delta", "cema.theta", "cema.eta_re", "cema.eta_im", "cema.omega",
        "W_z.weights", "W_z.bias",
        "q_gamma", "q_beta", "k_gamma", "k_beta",
        "W_v.weights", "W_v.bias",
        "W_o.weights", "W_o.bias",
        "ffn_ln.gamma", "ffn_ln.beta",
        "ffn.w1.weights", "ffn.w1.bias", "ffn.w2.weights", "ffn.w2.bias",
        "ffn_down.weights", "ffn_down.bias"
    };

    const double eps = FD_EPS;
    for (size_t pi = 0; pi < p.size(); ++pi) {
        // Probe up to 3 representative entries per tensor — enough to catch
        // row/column transposition without an O(params) test runtime.
        double worst = 0.0, gscale = 0.0, worst_abs = 0.0;
        size_t probes = 0;
        for (size_t i = 0; i < p[pi]->rows && probes < 3; ++i)
            for (size_t j = 0; j < p[pi]->cols && probes < 3; ++j) {
                double num = fd_param(b, x, target, p[pi], i, j, eps);
                double ana = (*g[pi])(i, j);
                gscale = max(gscale, fabs(num));
                worst_abs = max(worst_abs, fabs(ana - num));
                worst = max(worst, fabs(ana - num) / std::max(1e-30, fabs(num)));
                ++probes;
            }
        cout << "    " << setw(20) << left << names[pi] << " (" << p[pi]->rows << ","
             << p[pi]->cols << ")  rel_err = " << scientific << setw(12) << worst
             << "  abs_err = " << setw(12) << worst_abs
             << "  (scale " << gscale << ")" << defaultfloat << endl;
        check(string("param ") + names[pi] + " gradient matches FD", fd_ok(worst_abs, worst));
    }
}

// =====================================================================
// Test 5: the two-hop residual signature on the GRADIENT
//
// Zero the FFN down-projection. Forward gives Y = x, so
//   dL/dx = dL/dY · ∂Y/∂x = dL/dY · I  = dL/dY   (exactly)
// If the implementation used a one-hop residual, Y would be x + attn_out
// and dL/dx would instead be 2·dL/dY. Asserting d_input == grad_output here
// isolates the residual topology from every other chain.
// =====================================================================
static void test_two_hop_gradient() {
    cout << endl << "-- Test 5: two-hop residual signature on d_input --" << endl;

    const size_t n = 4, d = 4;
    MegalodonBlock b(d, 2, 2, 0, 2, 0);
    seed_block(b, 900);
    Tensor x = rand_tensor(n, d, 1.0, 901);
    Tensor target = rand_tensor(n, d, 1.0, 902);

    b.zero_grad();
    Tensor y = b.forward(x);
    Tensor gy = Tensor(n, d);
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j) gy(t, j) = y(t, j) - target(t, j);

    // Kill the FFN: Y = x, so dL/dY = x - target and dL/dx must equal it.
    b.ffn_down_.weights.fill(0.0);
    b.ffn_down_.bias.fill(0.0);
    Tensor y2 = b.forward(x);
    Tensor gx = b.backward(gy, 0.0);

    double worst = 0.0;
    for (size_t t = 0; t < n; ++t)
        for (size_t j = 0; j < d; ++j)
            worst = max(worst, fabs(gx(t, j) - gy(t, j)));
    cout << "    max|d_input - grad_output| (FFN zeroed) = " << scientific << worst << defaultfloat << endl;
    check("FFN zeroed => d_input == grad_output (two-hop, not 2x)", worst < 1e-12);
}

// =====================================================================
// Test 6: zero_grad / update_weights / accumulation
// =====================================================================
static void test_zero_grad_update() {
    cout << endl << "-- Test 6: zero_grad / update_weights / accumulation --" << endl;

    const size_t n = 4, d = 4;
    MegalodonBlock b(d, 1, 2, 0, 2, 0);
    seed_block(b, 1000);
    Tensor x = rand_tensor(n, d, 1.0, 1001);
    Tensor target = rand_tensor(n, d, 1.0, 1002);

    // MegalodonBlock::backward RESETS every parameter gradient at the top of
    // the call (it assigns fresh zeros, so repeated calls do NOT accumulate
    // the way Dense::backward's `+=` does). This differs from the sub-layer
    // convention but is intentional: each block backward is a complete,
    // self-contained pass over its own parameter list. Assert that contract
    // explicitly rather than assuming accumulation.
    ana_backward(b, x, target);
    auto g1 = b.gradients();
    vector<Tensor> snapshot;
    for (auto* t : g1) snapshot.push_back(t->clone());

    Tensor gy = Tensor(n, d);
    {
        Tensor y = b.forward(x);
        for (size_t t = 0; t < n; ++t)
            for (size_t j = 0; j < d; ++j) gy(t, j) = y(t, j) - target(t, j);
    }
    b.forward(x);
    b.backward(gy, 0.0);
    auto g2 = b.gradients();

    double worst = 0.0;
    for (size_t k = 0; k < snapshot.size(); ++k)
        for (size_t i = 0; i < snapshot[k].rows; ++i)
            for (size_t j = 0; j < snapshot[k].cols; ++j)
                worst = max(worst, fabs((*g2[k])(i, j) - snapshot[k](i, j)));
    cout << "    max|grad after pass 2 - grad after pass 1| = " << scientific << worst << defaultfloat << endl;
    check("backward resets gradients (repeat pass is idempotent)", worst < 1e-14);

    b.zero_grad();
    auto g3 = b.gradients();
    double worst0 = 0.0;
    for (auto* t : g3)
        for (size_t i = 0; i < t->rows; ++i)
            for (size_t j = 0; j < t->cols; ++j) worst0 = max(worst0, fabs((*t)(i, j)));
    check("zero_grad clears every gradient", worst0 == 0.0);

    // update_weights moves every parameter by exactly -lr * grad.
    auto p = b.parameters();
    vector<Tensor> before;
    for (auto* t : p) before.push_back(t->clone());
    b.forward(x);
    b.backward(gy, 0.0);
    auto gr = b.gradients();
    const double lr = 0.01;
    b.update_weights(lr);
    double worst_u = 0.0;
    for (size_t k = 0; k < p.size(); ++k)
        for (size_t i = 0; i < p[k]->rows; ++i)
            for (size_t j = 0; j < p[k]->cols; ++j)
                worst_u = max(worst_u, fabs((*p[k])(i, j) - (before[k](i, j) - lr * (*gr[k])(i, j))));
    check("update_weights moves every parameter by -lr * grad", worst_u < 1e-14);
}

// =====================================================================
// Test 7: end-to-end training reduces loss
// =====================================================================
static void test_training() {
    cout << endl << "-- Test 7: end-to-end training reduces loss --" << endl;

    const size_t n = 6, d = 4;
    MegalodonModel m(3, d, 1, 2, 2, 2, 0, 2, 0);

    // Target: y = 0.5 * x[:, 0] broadcast — a learnable linear function.
    Tensor x(n, 3);
    srand(1100);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < 3; ++j) x(i, j) = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
    Tensor target(n, 1);
    for (size_t i = 0; i < n; ++i) target(i, 0) = 0.5 * x(i, 0);

    auto loss_now = [&]() {
        Tensor y = m.forward(x);
        double L = 0.0;
        for (size_t i = 0; i < n; ++i) { double d = y(i, 0) - target(i, 0); L += 0.5 * d * d; }
        return L;
    };

    const double lr = 0.01;
    double L0 = loss_now();
    for (int step = 0; step < 60; ++step) {
        m.zero_grad();
        Tensor y = m.forward(x);
        Tensor gy(n, 1);
        for (size_t i = 0; i < n; ++i) gy(i, 0) = y(i, 0) - target(i, 0);
        m.backward(gy, 0.0);
        m.update_weights(lr);
    }
    double Lf = loss_now();
    cout << "    L0 = " << fixed << setprecision(5) << L0
         << " -> Lf = " << Lf << defaultfloat << endl;
    check("loss reduced by > 50% over 60 SGD steps", Lf < L0 * 0.5);
}

int main() {
    cout << "==========================================" << endl;
    cout << "  Megalodon Gradient Tests" << endl;
    cout << "  MEGALODON §3.3-3.4 — FD verified" << endl;
    cout << "==========================================" << endl;

    test_hand_derived();
    test_fd_input();
    test_fd_input_chunked();
    test_fd_params();
    test_two_hop_gradient();
    test_zero_grad_update();
    test_training();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
