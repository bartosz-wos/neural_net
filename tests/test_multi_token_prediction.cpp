// Multi-Token Prediction Head — tests
//   Gloeckle et al. (Meta, 2024), https://arxiv.org/abs/2404.19737
//
// Tests:
//   1.  Constructor validation (d_model=0 / vocab=0 / n_future=0 all throw)
//   2.  Forward shape — per-head logits (N, vocab); summed loss is finite scalar
//   3.  Independent-head signature — perturbing head[k] leaves head[j≠k] bit-exact
//   4a. Hand-derived N=1 single-head d_logits — N=1, vocab=3, y=0, logit=[2,0,1]
//   4b. Hand-derived trunk grad = SUM (n_future=2, identical W/b, distinct targets)
//   5.  FD every W_k (n_future=3, all entries) — rel_err <= 1%
//   6.  FD every b_k (n_future=3, all entries) — rel_err <= 1%
//   7.  FD trunk grad (all (t, c) entries) — rel_err <= 1%
//   8.  update_weights moves params; loss decreases over 1 step (vacuity gate)
//   9.  zero_grad clears every grad tensor (max abs = 0)
//   10. parameters()/gradients() shape contract (2*n_future entries each, same order)
//   11. MultiTokenPredictionModel forward shape (N=4, input_dim=2) -> (4, vocab); finite
//   12. MultiTokenPredictionModel training reduces loss (summed CE on shifted targets)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm>

#include "nn/layers/architectures/multi_token_prediction.h"
#include "nn/core/tensor.h"

using namespace std;

// ============================================================================
// Helpers
// ============================================================================

static inline double absd(double x) { return std::fabs(x); }

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return -1.0;
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            m = std::max(m, std::fabs(a(i, j) - b(i, j)));
    return m;
}

static double sum_loss_for_targets(MultiTokenPredictionHead& head,
                                   const Tensor& h, const Tensor& targets) {
    return head.forward(h, targets);
}

static int total = 0, passed = 0;
#define CHECK(expr)                                                        \
    do {                                                                    \
        ++total;                                                            \
        if (expr) { ++passed; cout << "[PASS] " #expr "\n"; }              \
        else      { cout << "[FAIL] " #expr "  at line " << __LINE__ << "\n"; } \
    } while (0)

// Forward + softmax + per-position CE in pure-tensor form (for the
// hand-derived N=1 single-head test).
static double nll_from_logits(double a, double b, double c, int target) {
    double m = std::max({a, b, c});
    double ea = std::exp(a - m), eb = std::exp(b - m), ec = std::exp(c - m);
    double z = ea + eb + ec;
    double pa = ea / z, pb = eb / z, pc = ec / z;
    double logp = (target == 0) ? std::log(pa)
              : (target == 1) ? std::log(pb)
              :                 std::log(pc);
    return -logp;
}

int main() {
    cout << fixed << setprecision(6);

    // -----------------------------------------------------------------------
    // Test 1: constructor validation
    // -----------------------------------------------------------------------
    cout << "\n--- Test 1: constructor validation ---\n";
    {
        bool th_dm   = false, th_v = false, th_nf = false;
        try { MultiTokenPredictionHead(0, 4, 2); } catch (const std::invalid_argument&) { th_dm = true; }
        try { MultiTokenPredictionHead(3, 0, 2); } catch (const std::invalid_argument&) { th_v  = true; }
        try { MultiTokenPredictionHead(3, 4, 0); } catch (const std::invalid_argument&) { th_nf = true; }
        bool valid_doesnt_throw = true;
        try { MultiTokenPredictionHead h_valid(3, 4, 2); (void)h_valid; }
        catch (...) { valid_doesnt_throw = false; }
        CHECK(th_dm && th_v && th_nf && valid_doesnt_throw);
    }

    // -----------------------------------------------------------------------
    // Test 2: forward shape / finite loss
    // -----------------------------------------------------------------------
    cout << "\n--- Test 2: forward shape / finite loss ---\n";
    {
        const size_t N = 5, d = 3, V = 4, K = 3;
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d), targets(N, K);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = 0.2 * t - 0.1 * j;
        for (size_t t = 0; t < N; ++t)
            for (size_t k = 0; k < K; ++k) targets(t, k) = (t + 2 * k) % V;

        double L = head.forward(h, targets);
        bool finite = std::isfinite(L);
        bool pos    = (L > 0.0);
        const auto& logits = head.last_logits();
        bool shape_ok = (logits.size() == K);
        if (shape_ok) for (size_t k = 0; k < K; ++k)
            shape_ok = shape_ok && (logits[k].rows == N) && (logits[k].cols == V);
        CHECK(finite && pos && shape_ok);
    }

    // -----------------------------------------------------------------------
    // Test 3: independent-head signature (perturb W_0, only head 0 changes)
    // -----------------------------------------------------------------------
    cout << "\n--- Test 3: independent-head signature ---\n";
    {
        const size_t N = 5, d = 3, V = 4, K = 3;
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d), targets(N, K);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = 0.15 * t + 0.05 * j;
        for (size_t t = 0; t < N; ++t)
            for (size_t k = 0; k < K; ++k) targets(t, k) = (t + k) % V;

        head.forward(h, targets);
        auto L_before = head.last_logits();          // (K, N, V)

        // Snapshot heads 1 and 2 — must be bit-exact unchanged after perturbation.
        Tensor L1_before = L_before[1].clone();
        Tensor L2_before = L_before[2].clone();

        // Perturb head 0: change W_0[1, 2] by +0.5 (enough to change logits significantly).
        head.weights()[0](1, 2) += 0.5;

        // Re-run forward to refresh last_logits_.
        head.forward(h, targets);
        auto L_after = head.last_logits();

        // Head 0 changed significantly.
        double d0 = max_abs_diff(L_before[0], L_after[0]);
        // Heads 1 and 2 are bit-exact unchanged.
        double d1 = max_abs_diff(L1_before, L_after[1]);
        double d2 = max_abs_diff(L2_before, L_after[2]);

        cout << "  d(head 0)=" << d0 << "  d(head 1)=" << d1 << "  d(head 2)=" << d2 << "\n";
        CHECK(d0 > 1e-4 && d1 == 0.0 && d2 == 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 4a: hand-derived N=1, single-head d_logits
    //         h = [[1, -2]], logit = [2, 0, 1] (forced via W_0, b_0), y = 0
    //         Expected d_logits = (1/N) * (softmax - one_hot(y)) = [-0.2452, 0.1022, 0.2778]
    // -----------------------------------------------------------------------
    cout << "\n--- Test 4a: hand-derived N=1 single-head d_logits ---\n";
    {
        const size_t N = 1, d = 2, V = 3, K = 1;
        MultiTokenPredictionHead head(d, V, K);
        // Force W_0 = [[1, 0], [0, 1], [0, 0]], b_0 = [2, 0, 1]
        head.weights()[0].fill(0.0);
        head.weights()[0](0, 0) = 1.0;
        head.weights()[0](1, 1) = 1.0;
        head.biases()[0](0, 0) = 2.0;
        head.biases()[0](0, 1) = 0.0;
        head.biases()[0](0, 2) = 1.0;

        Tensor h(N, d); h(0, 0) = 1.0; h(0, 1) = -2.0;
        Tensor targets(N, K); targets(0, 0) = 0;

        // Expected softmax values:
        double m = std::max({2.0 + 1.0*1.0 + 0.0*-2.0,   // logit_0 = 3
                             0.0 + 0.0*1.0 + 1.0*-2.0,  // logit_1 = -2
                             1.0 + 0.0*1.0 + 0.0*-2.0}); // logit_2 = 1
        // For stability we let the impl compute it; but write the expected values
        // in case m is the max. Easier: just compute from a direct softmax on logits.
        double lg0 = 2.0 + 1.0*1.0 + 0.0*-2.0;        // = 3
        double lg1 = 0.0 + 0.0*1.0 + 1.0*-2.0;        // = -2
        double lg2 = 1.0 + 0.0*1.0 + 0.0*-2.0;        // = 1
        double L_nll = nll_from_logits(lg0, lg1, lg2, /*target*/0);
        (void)m; (void)L_nll;

        double L = head.forward(h, targets);
        // Note: logit at (0, 0) is computed by my impl as sum_c h(0,c) * W(0,c) + b(0,0) = 1*1 + (-2)*0 + 2 = 3
        // Forward returns the SUM of per-head L_k. With K=1 and N_valid=1, that's just L_nll.

        Tensor d_trunk = head.backward(/*lr*/0.0);
        // grad_W_0[0, 0] = d_logits[0, 0] * h[0, 0]
        // grad_W_0[0, 1] = d_logits[0, 0] * h[0, 1]
        // grad_W_0[1, 1] = d_logits[0, 1] * h[0, 1]
        // grad_W_0[2, 2] = d_logits[0, 2] * h[0, 2] but h only has 2 cols...
        // Easier: check d_logits itself from d_trunk decomposition.

        // Expected softmax and d_logits:
        double sm_m = std::max({lg0, lg1, lg2});
        double ez0 = std::exp(lg0 - sm_m), ez1 = std::exp(lg1 - sm_m), ez2 = std::exp(lg2 - sm_m);
        double z = ez0 + ez1 + ez2;
        double p0 = ez0 / z, p1 = ez1 / z, p2 = ez2 / z;
        double dl0 = (1.0 / N) * (p0 - 1.0);
        double dl1 = (1.0 / N) * (p1 - 0.0);
        double dl2 = (1.0 / N) * (p2 - 0.0);
        cout << "  expected d_logits = [" << dl0 << ", " << dl1 << ", " << dl2 << "]\n";
        cout << "  L (forward)       = " << L << "\n";
        cout << "  L (hand-derived)  = " << L_nll << "\n";

        // Recover d_logits from grad_W (since grad_W[v, c] = d_logits[v] * h[c]):
        // grad_W[0, 0] / h[0, 0] should equal dl0.
        double got_dl0 = head.grad_weights()[0](0, 0) / h(0, 0);
        double got_dl1 = head.grad_weights()[0](1, 1) / h(0, 1);
        cout << "  d_logits[0]      got=" << got_dl0 << "  expected=" << dl0 << "\n";
        cout << "  d_logits[1]      got=" << got_dl1 << "  expected=" << dl1 << "\n";

        bool ok_close = std::fabs(got_dl0 - dl0) < 1e-6 && std::fabs(got_dl1 - dl1) < 1e-6;
        // For an extra check: d_trunk = d_logits . W (W is identity-ish in cols 0,1 of rows 0,1).
        // d_trunk[0, 0] = dl0 * W[0, 0] + dl1 * W[1, 0] + dl2 * W[2, 0]
        //               = dl0 * 1.0 + dl1 * 0.0 + dl2 * 0.0 = dl0
        // d_trunk[0, 1] = dl0 * 0 + dl1 * 1 + dl2 * 0 = dl1.
        bool trunk_ok = std::fabs(d_trunk(0, 0) - dl0) < 1e-6
                     && std::fabs(d_trunk(0, 1) - dl1) < 1e-6;
        cout << "  d_trunk[0, 0]    got=" << d_trunk(0, 0) << "  expected=" << dl0 << "\n";
        cout << "  d_trunk[0, 1]    got=" << d_trunk(0, 1) << "  expected=" << dl1 << "\n";
        CHECK(ok_close && trunk_ok);
    }

    // -----------------------------------------------------------------------
    // Test 4b: hand-derived trunk-grad = SUM, n_future=2 identical heads, distinct targets
    // -----------------------------------------------------------------------
    cout << "\n--- Test 4b: hand-derived trunk grad = SUM (n_future=2) ---\n";
    {
        const size_t N = 2, d = 2, V = 3, K = 2;
        MultiTokenPredictionHead head(d, V, K);

        // Force identical W and b for both heads:
        Tensor W_test(V, d);
        W_test.fill(0.0);
        W_test(0, 0) = 1.0; W_test(1, 1) = 1.0; W_test(2, 0) = 0.7; W_test(2, 1) = -0.3;
        Tensor b_test(1, V);
        b_test.fill(0.0);
        b_test(0, 0) = 0.5; b_test(0, 1) = 1.0; b_test(0, 2) = -0.5;
        for (size_t k = 0; k < K; ++k) {
            for (size_t i = 0; i < V; ++i) for (size_t j = 0; j < d; ++j)
                head.weights()[k](i, j) = W_test(i, j);
            for (size_t v = 0; v < V; ++v) head.biases()[k](0, v) = b_test(0, v);
        }

        Tensor h(N, d);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = 0.3 * (t + 1) - 0.1 * j;

        // Targets chosen so the two heads produce DIFFERENT d_logits:
        Tensor targets(N, K);
        targets(0, 0) = 0; targets(0, 1) = 1;
        targets(1, 0) = 2; targets(1, 1) = 2;

        // Compute logits for each head and the softmax-based d_logits expected.
        // For each (t, k) with t + k < N (which is always true for K=2, N=2 — t+k < 2 iff k=0 or (t=0,k=1)):
        // mask_k[t] = 1 iff t + k < N.
        // So row t=0: head 0 valid (0+0=0<2), head 1 valid (0+1=1<2) — both contribute.
        // Row t=1: head 0 valid (1+0=1<2), head 1 INVALID (1+1=2 not <2) — only head 0 contributes.
        // N_valid_0 = 2, N_valid_1 = 1.
        // For t=1 (head 1 masked), d_logits_1[1, v] = 0.

        double L = head.forward(h, targets);
        Tensor d_trunk = head.backward(0.0);

        // Compute expected d_logits by hand (per head, per row).
        std::vector<std::vector<double>> dl_expected(K, std::vector<double>(V * N, 0.0));
        std::vector<int> mask = {1, 1, 0};   // for t=0..1: head 0 always, head 1 valid only for t=0
        // Actually need per-(t, k). I'll recompute.
        // mask_0[t] = 1 if t < N (always for K=2, N=2, k=0)
        // mask_1[t] = 1 if t + 1 < N (true only for t=0)
        int mask_0[2] = {1, 1};
        int mask_1[2] = {1, 0};
        int N_valid[2] = {2, 1};

        double d_trunk_expected[2][2] = {{0, 0}, {0, 0}};
        for (size_t k = 0; k < K; ++k) {
            for (size_t t = 0; t < N; ++t) {
                int valid = (k == 0) ? mask_0[t] : mask_1[t];
                if (!valid) continue;
                // Compute logits[t, v] = sum_c h[t, c] * W[v, c] + b[v]
                double lg[V];
                for (size_t v = 0; v < V; ++v) {
                    double s = b_test(0, v);
                    for (size_t c = 0; c < d; ++c) s += h(t, c) * W_test(v, c);
                    lg[v] = s;
                }
                double m = std::max({lg[0], lg[1], lg[2]});
                double ez[V];
                double zz = 0;
                for (size_t v = 0; v < V; ++v) { ez[v] = std::exp(lg[v] - m); zz += ez[v]; }
                double p[V];
                for (size_t v = 0; v < V; ++v) p[v] = ez[v] / zz;
                double target_idx = targets(t, k);
                double scale = 1.0 / N_valid[k];
                for (size_t v = 0; v < V; ++v) {
                    double dlv = scale * (p[v] - (v == target_idx ? 1.0 : 0.0));
                    d_trunk_expected[t][0] += dlv * W_test(v, 0);
                    d_trunk_expected[t][1] += dlv * W_test(v, 1);
                }
            }
        }

        cout << "  L (forward)  = " << L << "\n";
        cout << "  d_trunk expected = [[" << d_trunk_expected[0][0] << ", " << d_trunk_expected[0][1] << "], ["
                                          << d_trunk_expected[1][0] << ", " << d_trunk_expected[1][1] << "]]\n";
        cout << "  d_trunk got      = [[" << d_trunk(0, 0) << ", " << d_trunk(0, 1) << "], ["
                                          << d_trunk(1, 0) << ", " << d_trunk(1, 1) << "]]\n";
        bool ok = true;
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j)
                ok = ok && std::fabs(d_trunk(t, j) - d_trunk_expected[t][j]) < 1e-9;
        CHECK(ok);
    }

    // -----------------------------------------------------------------------
    // Test 5: FD every W_k  (n_future=3, vocab=3, d=3, N=3)
    // -----------------------------------------------------------------------
    cout << "\n--- Test 5: FD every W_k (n_future=3, all entries) ---\n";
    {
        const size_t N = 3, d = 3, V = 3, K = 3;
        std::mt19937 rng(123);
        std::uniform_real_distribution<double> u(-0.5, 0.5);

        for (size_t k = 0; k < K; ++k) {
            // fresh head per k so perturbations only affect this k's slice
            MultiTokenPredictionHead head(d, V, K);
            // perturb weights/biases of head k to make FD nontrivial
            for (size_t i = 0; i < V; ++i)
                for (size_t j = 0; j < d; ++j) head.weights()[k](i, j) = u(rng);

            Tensor h(N, d);
            for (size_t t = 0; t < N; ++t)
                for (size_t j = 0; j < d; ++j) h(t, j) = u(rng);
            Tensor targets(N, K);
            std::uniform_int_distribution<int> u_int(0, (int)V - 1);
            for (size_t t = 0; t < N; ++t)
                for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = u_int(rng);

            head.forward(h, targets);
            head.backward(0.0);

            double max_err = 0.0;
            const double eps = 1e-5;
            for (size_t i = 0; i < V; ++i) {
                for (size_t j = 0; j < d; ++j) {
                    double orig = head.weights()[k](i, j);
                    // L+:
                    head.weights()[k](i, j) = orig + eps;
                    double Lp = head.forward(h, targets);
                    head.weights()[k](i, j) = orig - eps;
                    double Lm = head.forward(h, targets);
                    head.weights()[k](i, j) = orig;
                    double num = (Lp - Lm) / (2.0 * eps);
                    head.forward(h, targets);
                    head.backward(0.0);
                    double ana = head.grad_weights()[k](i, j);
                    double denom = std::max(std::fabs(num), std::fabs(ana));
                    double rel = denom < 1e-12 ? std::fabs(num - ana)
                                                : std::fabs(num - ana) / denom;
                    max_err = std::max(max_err, rel);
                }
            }
            cout << "  W[" << k << "] max rel_err = " << max_err << "\n";
            CHECK(max_err < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Test 6: FD every b_k
    // -----------------------------------------------------------------------
    cout << "\n--- Test 6: FD every b_k (n_future=3, all entries) ---\n";
    {
        const size_t N = 3, d = 3, V = 3, K = 3;
        std::mt19937 rng(456);
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = u(rng);
        Tensor targets(N, K);
        std::uniform_int_distribution<int> u_int(0, (int)V - 1);
        for (size_t t = 0; t < N; ++t)
            for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = u_int(rng);

        for (size_t k = 0; k < K; ++k) {
            double max_err = 0.0;
            const double eps = 1e-5;
            for (size_t v = 0; v < V; ++v) {
                double orig = head.biases()[k](0, v);
                head.biases()[k](0, v) = orig + eps;
                double Lp = head.forward(h, targets);
                head.biases()[k](0, v) = orig - eps;
                double Lm = head.forward(h, targets);
                head.biases()[k](0, v) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                head.forward(h, targets);
                head.backward(0.0);
                double ana = head.grad_biases()[k](0, v);
                double denom = std::max(std::fabs(num), std::fabs(ana));
                double rel = denom < 1e-12 ? std::fabs(num - ana)
                                            : std::fabs(num - ana) / denom;
                max_err = std::max(max_err, rel);
            }
            cout << "  b[" << k << "] max rel_err = " << max_err << "\n";
            CHECK(max_err < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // Test 7: FD trunk gradient
    // -----------------------------------------------------------------------
    cout << "\n--- Test 7: FD trunk gradient (all entries) ---\n";
    {
        const size_t N = 4, d = 3, V = 4, K = 3;
        std::mt19937 rng(789);
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = u(rng);
        Tensor targets(N, K);
        std::uniform_int_distribution<int> u_int(0, (int)V - 1);
        for (size_t t = 0; t < N; ++t)
            for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = u_int(rng);

        head.forward(h, targets);
        Tensor d_trunk = head.backward(0.0);

        double max_err = 0.0;
        const double eps = 1e-5;
        for (size_t t = 0; t < N; ++t) {
            for (size_t j = 0; j < d; ++j) {
                double orig = h(t, j);
                h(t, j) = orig + eps;
                double Lp = head.forward(h, targets);
                h(t, j) = orig - eps;
                double Lm = head.forward(h, targets);
                h(t, j) = orig;
                double num = (Lp - Lm) / (2.0 * eps);
                double ana = d_trunk(t, j);
                double denom = std::max(std::fabs(num), std::fabs(ana));
                double rel = denom < 1e-12 ? std::fabs(num - ana)
                                            : std::fabs(num - ana) / denom;
                max_err = std::max(max_err, rel);
            }
        }
        cout << "  d_trunk max rel_err = " << max_err << "\n";
        CHECK(max_err < 0.01);
    }

    // -----------------------------------------------------------------------
    // Test 8: update_weights moves params; loss decreases over a step (vacuity gate)
    // -----------------------------------------------------------------------
    cout << "\n--- Test 8: update_weights moves params ---\n";
    {
        const size_t N = 3, d = 2, V = 3, K = 2;
        MultiTokenPredictionHead head(d, V, K);
        std::mt19937 rng(11);
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        Tensor h(N, d);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = u(rng);
        Tensor targets(N, K);
        std::uniform_int_distribution<int> u_int(0, (int)V - 1);
        for (size_t t = 0; t < N; ++t)
            for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = u_int(rng);

        double L0 = head.forward(h, targets);
        head.backward(0.0);
        // Snapshot one parameter
        double W0_before = head.weights()[0](0, 0);
        head.update_weights(0.1);
        double W0_after = head.weights()[0](0, 0);
        double updated = head.forward(h, targets);
        head.update_weights(0.1);
        double down = head.forward(h, targets);
        cout << "  L0=" << L0 << "  L_after_1_step=" << updated << "  L_after_2_step=" << down << "\n";
        bool moved = std::fabs(W0_after - W0_before) > 1e-10;
        CHECK(moved && updated < L0 + 1e-9);   // After 1 SG step the loss is lower (random init is high).
    }

    // -----------------------------------------------------------------------
    // Test 9: zero_grad clears every grad
    // -----------------------------------------------------------------------
    cout << "\n--- Test 9: zero_grad clears all grads ---\n";
    {
        const size_t N = 3, d = 2, V = 3, K = 2;
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d), targets(N, K);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = 0.1 * t;
        for (size_t t = 0; t < N; ++t)
            for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = (t + kk) % V;
        head.forward(h, targets);
        head.backward(0.0);
        head.zero_grad();
        double max_g = 0.0;
        for (size_t k = 0; k < K; ++k) {
            max_g = std::max(max_g, head.grad_weights()[k].max());
            max_g = std::max(max_g, std::fabs(head.grad_biases()[k].max()));
            for (size_t i = 0; i < head.grad_weights()[k].rows; ++i)
                for (size_t j = 0; j < head.grad_weights()[k].cols; ++j)
                    max_g = std::max(max_g, std::fabs(head.grad_weights()[k](i, j)));
            for (size_t v = 0; v < head.grad_biases()[k].cols; ++v)
                max_g = std::max(max_g, std::fabs(head.grad_biases()[k](0, v)));
        }
        cout << "  max grad after zero_grad = " << max_g << "\n";
        CHECK(max_g == 0.0);
    }

    // -----------------------------------------------------------------------
    // Test 10: parameters()/gradients() shape contract
    // -----------------------------------------------------------------------
    cout << "\n--- Test 10: parameters()/gradients() contract ---\n";
    {
        const size_t N = 2, d = 3, V = 3, K = 3;
        MultiTokenPredictionHead head(d, V, K);
        Tensor h(N, d), targets(N, K);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < d; ++j) h(t, j) = 0.1;
        for (size_t t = 0; t < N; ++t)
            for (size_t kk = 0; kk < K; ++kk) targets(t, kk) = 0;

        head.forward(h, targets);
        head.backward(0.0);
        auto ps = head.parameters();
        auto gs = head.gradients();
        cout << "  |parameters|=" << ps.size() << "  |gradients|=" << gs.size() << "\n";
        bool same_count = (ps.size() == gs.size()) && (ps.size() == 2 * K);
        bool same_shape = true;
        if (same_count) {
            for (size_t i = 0; i < ps.size(); ++i) {
                same_shape = same_shape && (ps[i]->rows == gs[i]->rows && ps[i]->cols == gs[i]->cols);
            }
        }
        CHECK(same_count && same_shape);
    }

    // -----------------------------------------------------------------------
    // Test 11: MultiTokenPredictionModel forward shape
    // -----------------------------------------------------------------------
    cout << "\n--- Test 11: MultiTokenPredictionModel forward shape ---\n";
    {
        const size_t N = 4, id = 2, d = 3, V = 4, K = 3;
        MultiTokenPredictionModel model(id, d, V, K);
        Tensor input(N, id);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < id; ++j) input(t, j) = 0.1 * t - 0.2 * j;
        Tensor out = model.forward(input);
        bool shape_ok = (out.rows == N) && (out.cols == V);
        bool finite = true;
        for (size_t i = 0; i < out.rows; ++i)
            for (size_t j = 0; j < out.cols; ++j)
                finite = finite && std::isfinite(out(i, j));
        cout << "  out shape = " << out.rows << "x" << out.cols
             << "  finite=" << finite << "\n";
        CHECK(shape_ok && finite);
    }

    // -----------------------------------------------------------------------
    // Test 12: MultiTokenPredictionModel training reduces loss
    // -----------------------------------------------------------------------
    cout << "\n--- Test 12: MultiTokenPredictionModel training reduces loss (regression) ---\n";
    {
        const size_t N = 3, id = 2, d = 4, V = 5, K = 3;
        MultiTokenPredictionModel model(id, d, V, K);
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> u(-0.5, 0.5);
        Tensor input(N, id);
        for (size_t t = 0; t < N; ++t)
            for (size_t j = 0; j < id; ++j) input(t, j) = u(rng);

        // Build shifted targets: row t = ground truth at position t+k, else 0.
        // Toy class indices in [0, V).
        Tensor gt(N, 1);
        for (size_t t = 0; t < N; ++t) gt(t, 0) = static_cast<double>(t % V);
        Tensor targets(N, K);
        for (size_t t = 0; t < N; ++t) {
            for (size_t k = 0; k < K; ++k) {
                size_t pos = t + k;
                targets(t, k) = (pos < N) ? gt(pos, 0) : 0.0;
            }
        }

        // Forward just to populate head internals with dummy targets.
        model.forward(input);

        // Compute initial loss:
        Tensor trunk_logits = /* input_proj applied to input */ Tensor(N, d);
        // Recompute the trunk grad by running head with the real targets:
        // We don't have a public accessor for trunk_; emulate via input_proj + head.
        // Compute initial summed CE.
        auto compute_loss = [&]() {
            // Run forward through input_proj manually:
            const auto& Wp = model.input_proj().weights;
            const auto& bp = model.input_proj().bias;
            Tensor trunk(N, d);
            for (size_t t = 0; t < N; ++t)
                for (size_t j = 0; j < d; ++j) {
                    double s = bp(0, j);
                    for (size_t i = 0; i < id; ++i) s += input(t, i) * Wp(j, i);
                    trunk(t, j) = s;
                }
            return sum_loss_for_targets(model.head(), trunk, targets);
        };

        double L0 = compute_loss();
        // Train via input_proj.backward + head.backward chain.
        const double lr = 0.05;
        for (size_t step = 0; step < 80; ++step) {
            // Compute trunk manually (since the Model doesn't store it), then head.backward() updates the head grads.
            const auto& Wp = model.input_proj().weights;
            const auto& bp = model.input_proj().bias;
            Tensor trunk(N, d);
            for (size_t t = 0; t < N; ++t)
                for (size_t j = 0; j < d; ++j) {
                    double s = bp(0, j);
                    for (size_t i = 0; i < id; ++i) s += input(t, i) * Wp(j, i);
                    trunk(t, j) = s;
                }
            // run head forward & backward so head grads are populated
            double L_step = sum_loss_for_targets(model.head(), trunk, targets);
            Tensor d_trunk = model.head().backward(lr);
            // input_proj.backward for the trunk grad -> input proj grad
            model.input_proj().forward(input);             // refresh last_input for backward
            model.input_proj().backward(d_trunk, lr);
            // update
            model.input_proj().update_weights(lr);
            model.head().update_weights(lr);
            model.input_proj().zero_grad();
            model.head().zero_grad();
            (void)L_step;
        }
        double L1 = compute_loss();
        cout << "  L0=" << L0 << "  L80=" << L1 << "  ratio=" << (L0 > 0 ? L1 / L0 : 0) << "\n";
        CHECK(L1 < L0 - 1e-4);
    }

    cout << "\n=== Summary: " << passed << " of " << total << " tests passed ===\n";
    return (passed == total) ? 0 : 1;
}
