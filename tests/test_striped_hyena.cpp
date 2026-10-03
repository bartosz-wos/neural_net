// Tests for StripedHyena — the striped (interleaved) hybrid of the Hyena
// operator and multi-head attention.
//
// Poli, Massaro, Muntoni, Singhal, Beigi et al., "Mechanistic Design and
// Scaling of Hybrid Architectures", arXiv:2403.17844v2 §3.3 (Topology) and
// Appendix C.2 (StripedHyena: "1A:1H, 1A:3H, 1A:11H" striping schedules).
// The underlying Hyena_N operator is Poli et al. arXiv:2302.10866v3 Def. 3.1
// and is already shipped in layers/architectures/hyena.h.
//
// Conventions: sequences are (L, d_model) row-major Tensors.

#include "nn/nn.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const std::string& name) {
    tests_run++;
    if (cond) {
        tests_passed++;
        std::printf("  [PASS] %s\n", name.c_str());
    } else {
        tests_failed++;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            double d = std::fabs(a[i][j] - b[i][j]);
            if (d > m) m = d;
        }
    return m;
}

static double max_abs(const Tensor& a) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            double v = std::fabs(a[i][j]);
            if (v > m) m = v;
        }
    return m;
}

static bool all_finite(const Tensor& a) {
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (!std::isfinite(a[i][j])) return false;
    return true;
}

static Tensor random_tensor(size_t r, size_t c, double scale) {
    return Tensor::random(r, c, scale);
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor_validation() {
    std::printf("Test 1: constructor validation\n");

    bool threw = false;
    try { StripedHyenaModel m(0, 8, 4, 2, 4, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on d_model=0");

    threw = false;
    try { StripedHyenaModel m(4, 8, 0, 2, 4, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on num_layers=0");

    threw = false;
    try { StripedHyenaModel m(4, 8, 4, 0, 4, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on output_dim=0");

    threw = false;
    try { StripedHyenaModel m(4, 8, 4, 2, 4, 0); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on hyena_ratio=0");

    threw = false;
    try { StripedHyenaModel m(4, 8, 4, 2, 0, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on seq_len=0");

    threw = false;
    try { StripedHyenaModel m(4, 8, 4, 2, 4, 0); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on num_heads=0");

    threw = false;
    try { StripedHyenaModel m(4, 9, 4, 2, 4, 2); } catch (...) { threw = true; }
    check(threw, "StripedHyenaModel throws on d_model%num_heads!=0");

    threw = false;
    try { StripedHyenaBlock b(0, 8, true, 4, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaBlock throws on d_model=0");

    threw = false;
    try { StripedHyenaBlock b(8, 0, true, 2, 1); } catch (...) { threw = true; }
    check(threw, "StripedHyenaBlock throws on seq_len=0");

    threw = false;
    try { StripedHyenaBlock b(8, 4, true, 2, 0); } catch (...) { threw = true; }
    check(threw, "StripedHyenaBlock throws on num_heads=0");

    threw = false;
    try { StripedHyenaBlock b(9, 4, true, 2, 2); } catch (...) { threw = true; }
    check(threw, "StripedHyenaBlock throws on d_model%num_heads!=0");

    // Valid constructs.
    StripedHyenaModel m1(4, 8, 4, 2, 4, 1);
    check(m1.d_model() == 8, "valid model constructs (d_model=8)");
    StripedHyenaModel m2(4, 8, 4, 2, 4, 1, 3);
    check(m2.hyena_ratio() == 3, "valid model constructs with hyena_ratio=3");
    StripedHyenaBlock b1(8, 4, false, 2, 2);
    check(!b1.is_attention(), "valid Hyena block constructs");
    StripedHyenaBlock b2(8, 4, true, 2, 2);
    check(b2.is_attention(), "valid attention block constructs");
}

// ---------------------------------------------------------------------------
// Test 2: striping schedule (arXiv:2403.17844 Appendix C.2)
//
//   "We use 3 striping schedule ratios: 1A:1H, 1A:3H, 1A:11H, where A =
//    Attention and H = Hyena along model depth. In instances where the number
//    of layers is not a multiple of the schedule, the ratio is repeated
//    until the target depth is reached."
// ---------------------------------------------------------------------------
static void test_striping_schedule() {
    std::printf("Test 2: striping schedule\n");

    // 1A:1H at depth 4 -> A H A H
    {
        StripedHyenaModel m(4, 8, 4, 2, 4, 1, 1);
        check(m.stripe_schedule() == std::vector<bool>({true, false, true, false}),
              "1A:1H depth 4 -> [A H A H]");
    }
    // 1A:1H at depth 5 -> A H A H A (truncation keeps the leading pattern)
    {
        StripedHyenaModel m(4, 8, 5, 2, 4, 1, 1);
        check(m.stripe_schedule() == std::vector<bool>({true, false, true, false, true}),
              "1A:1H depth 5 -> [A H A H A] (repeat, not truncate)");
    }
    // 1A:3H at depth 8 -> A H H H A H H H  (exactly 2 full patterns)
    {
        StripedHyenaModel m(4, 8, 8, 2, 4, 1, 3);
        check(m.stripe_schedule() == std::vector<bool>(
                  {true, false, false, false, true, false, false, false}),
              "1A:3H depth 8 -> [A H H H A H H H]");
    }
    // 1A:3H at depth 5 -> A H H H A (truncated mid-pattern, but ATTENTION-first)
    {
        StripedHyenaModel m(4, 8, 5, 2, 4, 1, 3);
        check(m.stripe_schedule() == std::vector<bool>(
                  {true, false, false, false, true}),
              "1A:3H depth 5 -> [A H H H A] (partial pattern keeps leading A)");
    }
    // 1A:11H at depth 12 -> A + 11 H (exactly one pattern)
    {
        StripedHyenaModel m(4, 8, 12, 2, 4, 1, 11);
        std::vector<bool> expect(12, false);
        expect[0] = true;
        check(m.stripe_schedule() == expect, "1A:11H depth 12 -> [A H H H H H H H H H H H]");
    }
    // 1A:11H at depth 13 -> two attention blocks, the second at index 12
    {
        StripedHyenaModel m(4, 8, 13, 2, 4, 1, 11);
        check(m.stripe_schedule()[0]  == true,  "1A:11H depth 13: block 0 is attention");
        check(m.stripe_schedule()[12] == true,  "1A:11H depth 13: block 12 is attention");
        check(m.stripe_schedule()[1]  == false, "1A:11H depth 13: block 1 is Hyena");
        check(m.stripe_schedule()[11] == false, "1A:11H depth 13: block 11 is Hyena");
    }
    // schedule length always equals depth
    {
        for (size_t depth = 1; depth <= 15; ++depth) {
            for (size_t r : {1u, 3u, 11u}) {
                StripedHyenaModel m(4, 8, depth, 2, 4, 1, r);
                if (m.stripe_schedule().size() != depth) {
                    check(false, "schedule length == depth for all (depth, ratio)");
                    return;
                }
                if (m.stripe_schedule()[0] != true) {
                    check(false, "schedule always begins with attention");
                    return;
                }
            }
        }
        check(true, "schedule length == depth and starts with attention for all (depth<=15, ratio in {1,3,11})");
    }
    // model blocks agree with the schedule
    {
        StripedHyenaModel m(4, 8, 7, 2, 4, 1, 3);
        bool agree = true;
        for (size_t i = 0; i < 7; ++i)
            if (m.blocks[i].is_attention() != m.stripe_schedule()[i]) agree = false;
        check(agree, "model block kinds match stripe_schedule entry-for-entry");
    }
}

// ---------------------------------------------------------------------------
// Test 3: forward — shape, finiteness, non-degeneracy
// ---------------------------------------------------------------------------
static void test_forward_shapes() {
    std::printf("Test 3: forward shapes\n");

    // Model forward, both kinds present.
    {
        StripedHyenaModel m(4, 8, 6, 3, 5, 1, 1);
        Tensor x = random_tensor(5, 4, 0.5);
        Tensor y = m.forward(x);
        check(y.rows == 1 && y.cols == 3, "model forward shape (1, out=3) after mean-pool");
        check(all_finite(y), "model output finite");
        check(max_abs(y) > 1e-9, "model output non-zero");
    }
    // 1A:11H model — the paper's compute-optimal-ish schedule.
    {
        StripedHyenaModel m(4, 8, 12, 2, 6, 1, 11);
        Tensor x = random_tensor(6, 4, 0.5);
        Tensor y = m.forward(x);
        check(y.rows == 1 && y.cols == 2, "1A:11H model forward shape (1, out=2) after mean-pool");
        check(all_finite(y), "1A:11H model output finite");
    }
    // Block forward, both kinds.
    {
        StripedHyenaBlock hb(8, 4, false, 2, 2);   // Hyena
        Tensor x = random_tensor(4, 8, 0.5);
        Tensor y = hb.forward(x);
        check(y.rows == 4 && y.cols == 8, "Hyena block forward shape");
        check(all_finite(y), "Hyena block output finite");
        check(max_abs(y) > 1e-9, "Hyena block output non-zero");
    }
    {
        StripedHyenaBlock ab(8, 4, true, 2, 2);    // attention
        Tensor x = random_tensor(4, 8, 0.5);
        Tensor y = ab.forward(x);
        check(y.rows == 4 && y.cols == 8, "attention block forward shape");
        check(all_finite(y) && max_abs(y) > 1e-9, "attention block output finite and non-zero");
    }
    // ffn_mult=0 skips the channel mixer (isolates the sequence mixer).
    {
        StripedHyenaModel m(4, 8, 4, 2, 4, 1, 1, 0);
        Tensor x = random_tensor(4, 4, 0.5);
        Tensor y = m.forward(x);
        // The model mean-pools over the sequence axis, so the output is
        // (1, output_dim) regardless of the input length.
        check(y.rows == 1 && y.cols == 2 && all_finite(y),
              "ffn_mult=0 (no channel mixer) forward works");
    }
    // Wrong input width is rejected.
    {
        StripedHyenaModel m(4, 8, 4, 2, 4, 1, 1);
        bool threw = false;
        try { m.forward(random_tensor(4, 5, 0.5)); } catch (...) { threw = true; }
        check(threw, "model throws on wrong input width");
    }
    // L=1 degenerate sequence length.
    {
        StripedHyenaBlock b(8, 1, false, 1, 1);
        Tensor x = random_tensor(1, 8, 0.5);
        Tensor y = b.forward(x);
        check(y.rows == 1 && y.cols == 8 && all_finite(y), "L=1 Hyena block finite");
    }
}

// ---------------------------------------------------------------------------
// Test 4: the two mixers are genuinely different
// ---------------------------------------------------------------------------
static void test_mixers_differ() {
    std::printf("Test 4: mixer kinds are distinct\n");

    Tensor x = random_tensor(4, 8, 0.5);
    StripedHyenaBlock hb(8, 4, false, 2, 1);
    StripedHyenaBlock ab(8, 4, true, 2, 1);

    Tensor yh = hb.forward(x);
    Tensor ya = ab.forward(x);
    check(max_abs_diff(yh, ya) > 1e-9, "Hyena block and attention block give different outputs");

    // With EVERY parameter zeroed both blocks collapse to the identity, because
    // the mixer and the channel mixer both contribute 0 and only the residual
    // carries through: out = x. That is a real invariant, so assert it — and
    // it is the strongest available proof that the two block kinds are wired
    // through the same residual structure rather than one aliasing the other.
    Tensor x0 = random_tensor(4, 8, 0.5);
    {
        StripedHyenaBlock hb2(8, 4, false, 2, 1);
        StripedHyenaBlock ab2(8, 4, true, 2, 1);
        for (auto* p : hb2.parameters()) for (size_t i=0;i<p->rows;++i) for (size_t j=0;j<p->cols;++j) (*p)[i][j] = 0.0;
        for (auto* p : ab2.parameters()) for (size_t i=0;i<p->rows;++i) for (size_t j=0;j<p->cols;++j) (*p)[i][j] = 0.0;
        Tensor zh = hb2.forward(x0), za = ab2.forward(x0);
        check(max_abs_diff(zh, x0) == 0.0, "zeroed Hyena block is exactly the identity (out == x)");
        check(max_abs_diff(za, x0) == 0.0, "zeroed attention block is exactly the identity (out == x)");
    }
}

// ---------------------------------------------------------------------------
// Test 5: determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::printf("Test 5: determinism\n");

    StripedHyenaModel m(4, 8, 6, 2, 5, 1, 1);
    Tensor x = random_tensor(5, 4, 0.5);
    Tensor y1 = m.forward(x);
    Tensor y2 = m.forward(x);
    check(max_abs_diff(y1, y2) == 0.0, "model forward bit-exact across two calls");

    StripedHyenaBlock b(8, 4, false, 2, 1);
    Tensor z = random_tensor(4, 8, 0.5);
    check(max_abs_diff(b.forward(z), b.forward(z)) == 0.0, "block forward bit-exact across two calls");
}

// ---------------------------------------------------------------------------
// Test 6: parameter / gradient contract
// ---------------------------------------------------------------------------
static void test_parameter_contract() {
    std::printf("Test 6: parameter and gradient contract\n");

    StripedHyenaBlock hb(8, 4, false, 2, 1);   // Hyena block, ffn_mult=1
    auto p = hb.parameters();
    auto g = hb.gradients();
    check(p.size() == g.size(), "block parameters().size() == gradients().size()");
    check(p.size() > 0, "block exposes parameters");
    bool shapes_match = true;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes_match = false;
    check(shapes_match, "every param has a shape-matched grad");

    StripedHyenaBlock ab(8, 4, true, 2, 1);
    check(ab.parameters().size() == ab.gradients().size(), "attention block param/grad counts agree");

    StripedHyenaModel m(4, 8, 6, 2, 5, 1, 1);
    check(m.parameters().size() == m.gradients().size(), "model param/grad counts agree");
    check(m.parameters().size() > 0, "model exposes parameters");

    // zero_grad clears everything.
    Tensor x = random_tensor(5, 4, 0.5);
    Tensor y = m.forward(x);
    Tensor gy = Tensor::zeros(y.rows, y.cols);
    gy(0, 0) = 1.0;
    m.zero_grad();
    m.backward(gy, 0.0);
    double before = 0.0;
    for (auto* gr : m.gradients())
        for (size_t i = 0; i < gr->rows; ++i)
            for (size_t j = 0; j < gr->cols; ++j) before += std::fabs((*gr)[i][j]);
    check(before > 0.0, "backward produces non-zero gradients");

    m.zero_grad();
    double after = 0.0;
    for (auto* gr : m.gradients())
        for (size_t i = 0; i < gr->rows; ++i)
            for (size_t j = 0; j < gr->cols; ++j) after += std::fabs((*gr)[i][j]);
    check(after == 0.0, "zero_grad clears every gradient");
}

// ---------------------------------------------------------------------------
// Test 7: FD input gradient — the core gradient check
// Random non-uniform init is mandatory: a uniform init makes
// sum_k W[k,j] == sum_k W[j,k] by construction, so a row/column confusion in
// the matmul backward would pass vacuously.
// Loss: L = 0.5 * sum(out^2), so dL/dout = out.
// ---------------------------------------------------------------------------
static double fd_input_grad(StripedHyenaModel& m, const Tensor& x, size_t i, size_t j,
                            double eps, bool subtract_one) {
    Tensor xp = x.clone(), xm = x.clone();
    double base = 0.0;
    for (size_t r = 0; r < xp.rows; ++r)
        for (size_t c = 0; c < xp.cols; ++c) base += 0.5 * m.forward(xp)[r][c] * m.forward(xp)[r][c];
    xp[i][j] += eps; xm[i][j] -= eps;
    Tensor yp = m.forward(xp), ym = m.forward(xm);
    double lp = 0.0, lm = 0.0;
    for (size_t r = 0; r < yp.rows; ++r) {
        for (size_t c = 0; c < yp.cols; ++c) {
            lp += 0.5 * yp[r][c] * yp[r][c];
            lm += 0.5 * ym[r][c] * ym[r][c];
        }
    }
    (void)base; (void)subtract_one;
    return (lp - lm) / (2.0 * eps);
}

static void test_fd_input_gradient() {
    std::printf("Test 7: FD input gradient\n");

    // Hyena-heavy model (1A:3H, depth 5): mostly Hyena path.
    StripedHyenaModel m(4, 8, 5, 2, 4, 1, 3);
    Tensor x = random_tensor(4, 4, 0.6);

    Tensor y = m.forward(x);
    Tensor gy = Tensor::zeros(y.rows, y.cols);
    for (size_t r = 0; r < y.rows; ++r)
        for (size_t c = 0; c < y.cols; ++c) gy[r][c] = y[r][c];   // dL/dy = y
    m.zero_grad();
    Tensor dx = m.backward(gy, 0.0);

    double worst = 0.0;
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c) {
            double num = fd_input_grad(m, x, r, c, 1e-5, false);
            double ana = dx[r][c];
            double denom = std::max(std::fabs(ana), std::fabs(num));
            if (denom < 1e-9) denom = 1.0;   // floor: avoid rel_err ~1 at 1e-17
            double rel = std::fabs(ana - num) / denom;
            if (rel > worst) worst = rel;
        }
    check(worst < 1e-3, "model FD input gradient rel_err < 1e-3 (1A:3H) worst=" +
          std::to_string(worst));

    // Attention-heavy model (1A:1H): exercises the attention backward path too.
    StripedHyenaModel m2(4, 8, 4, 2, 4, 1, 1);
    Tensor x2 = random_tensor(4, 4, 0.6);
    Tensor y2 = m2.forward(x2);
    Tensor gy2 = Tensor::zeros(y2.rows, y2.cols);
    for (size_t r = 0; r < y2.rows; ++r)
        for (size_t c = 0; c < y2.cols; ++c) gy2[r][c] = y2[r][c];
    m2.zero_grad();
    Tensor dx2 = m2.backward(gy2, 0.0);

    double worst2 = 0.0;
    for (size_t r = 0; r < x2.rows; ++r)
        for (size_t c = 0; c < x2.cols; ++c) {
            double num = fd_input_grad(m2, x2, r, c, 1e-5, false);
            double ana = dx2[r][c];
            double denom = std::max(std::fabs(ana), std::fabs(num));
            if (denom < 1e-9) denom = 1.0;
            double rel = std::fabs(ana - num) / denom;
            if (rel > worst2) worst2 = rel;
        }
    check(worst2 < 1e-3, "model FD input gradient rel_err < 1e-3 (1A:1H) worst=" +
          std::to_string(worst2));
}

// ---------------------------------------------------------------------------
// Test 8: FD parameter gradients on a single block
// ---------------------------------------------------------------------------
static void test_fd_parameter_gradient() {
    std::printf("Test 8: FD block parameter gradient\n");

    // Hyena block, ffn_mult=0 so the channel mixer is out of the picture.
    // Args: (d_model=4, seq_len=3, is_attention=false, ffn_mult=0, num_heads=1)
    StripedHyenaBlock b(4, 3, false, 0, 1);
    Tensor x = random_tensor(3, 4, 0.6);

    Tensor y = b.forward(x);
    Tensor gy = Tensor::zeros(y.rows, y.cols);
    for (size_t r = 0; r < y.rows; ++r)
        for (size_t c = 0; c < y.cols; ++c) gy[r][c] = y[r][c];
    b.zero_grad();
    b.backward(gy, 0.0);

    // Find the Hyena in_proj weight tensor and FD-check element [0][0].
    Tensor* W = nullptr;
    for (auto* p : b.parameters()) {
        if (p->rows == 12 && p->cols == 4) { W = p; break; }   // (order+1)*D = 3*4
    }
    if (W == nullptr) {
        check(false, "located Hyena in_proj weights (12, 4)");
        return;
    }
    check(true, "located Hyena in_proj weights (12, 4)");

    auto loss = [&](const Tensor& in) {
        Tensor o = b.forward(in);
        double L = 0.0;
        for (size_t r = 0; r < o.rows; ++r)
            for (size_t c = 0; c < o.cols; ++c) L += 0.5 * o[r][c] * o[r][c];
        return L;
    };

    double eps = 1e-5;
    double ana = 0.0;
    for (auto* gr : b.gradients())
        if (gr->rows == 12 && gr->cols == 4) { ana = (*gr)[0][0]; break; }

    double orig = (*W)[0][0];
    (*W)[0][0] = orig + eps; double lp = loss(x);
    (*W)[0][0] = orig - eps; double lm = loss(x);
    (*W)[0][0] = orig;
    double num = (lp - lm) / (2.0 * eps);

    double denom = std::max(std::fabs(ana), std::fabs(num));
    if (denom < 1e-9) denom = 1.0;
    check(std::fabs(ana - num) / denom < 1e-3,
          "Hyena in_proj W[0][0] FD rel_err < 1e-3 (ana=" + std::to_string(ana) +
          " num=" + std::to_string(num) + ")");
}

// ---------------------------------------------------------------------------
// Test 9: training reduces loss
// ---------------------------------------------------------------------------
static void test_training_reduces_loss() {
    std::printf("Test 9: end-to-end training\n");

    StripedHyenaModel m(4, 8, 6, 1, 4, 1, 1);
    srand(1234);

    // Target: the identity map, mean-pooled.
    Tensor x = random_tensor(4, 4, 0.8);
    Tensor y = x.clone();

    auto loss = [&]() {
        Tensor o = m.forward(x);
        double L = 0.0;
        for (size_t r = 0; r < o.rows; ++r)
            for (size_t c = 0; c < o.cols; ++c) {
                double d = o[r][c] - y[r][c];
                L += 0.5 * d * d;
            }
        return L;
    };

    double l0 = loss();
    for (int step = 0; step < 120; ++step) {
        Tensor o = m.forward(x);
        Tensor gy = Tensor::zeros(o.rows, o.cols);
        for (size_t r = 0; r < o.rows; ++r)
            for (size_t c = 0; c < o.cols; ++c) gy[r][c] = o[r][c] - y[r][c];
        m.zero_grad();
        m.backward(gy, 0.0);
        m.update_weights(0.05);
    }
    double l1 = loss();
    check(l1 < l0 * 0.5, "training reduces loss > 50% (" + std::to_string(l0) +
          " -> " + std::to_string(l1) + ")");
}

int main() {
    std::printf("========================================\n");
    std::printf("  StripedHyena Tests\n");
    std::printf("========================================\n");

    test_constructor_validation();
    test_striping_schedule();
    test_forward_shapes();
    test_mixers_differ();
    test_determinism();
    test_parameter_contract();
    test_fd_input_gradient();
    test_fd_parameter_gradient();
    test_training_reduces_loss();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
