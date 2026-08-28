// Forgetting Transformer (FoX) — Lin, Yang, Sun et al., ICLR 2025
//   https://arxiv.org/abs/2503.02130
//
// Causal softmax attention with a learnable data-dependent forget gate entering
// as an additive log-decay bias:
//
//   f[t, h]    = sigmoid((X W_f)[t, h])
//   D[t, h]    = sum_{i <= t} log f[i, h]
//   bias[t, s] = D[t, h] - D[s, h]                       (<= 0 for s <= t)
//   scores     = Q_h K_h^T * scale + bias  (causal), A = row_softmax(scores)
//
// Tests:
//   1.  Constructor validation (3 invalid inputs throw, valid does not)
//   2.  Forward shape + finite + nonzero
//   3.  Forget gate f in (0, 1)
//   4.  bias[t, t] == 0 and bias[t, s] <= 0 for s < t (monotone decay)
//   5.  Attention rows sum to 1 and are strictly causal (A[t, s] == 0 for s > t)
//   6.  Causality: perturbing x[s] for s > t leaves y[t] bit-exact
//   7.  Input gradient FD check
//   8.  W_q / W_k / W_v / W_o gradient FD checks
//   9.  W_f gradient FD check (the FoX signature parameter)
//   10. Large positive W_f (f -> 1) reduces to vanilla causal attention
//       (compared against a bias-free reference computed in the test)
//   11. Strongly negative W_f makes attention local (diagonal dominant)
//   12. Determinism: copied params -> bit-exact forward
//   13. zero_grad clears all 5 gradients; update_weights moves all 5 params
//   14. parameters()/gradients() contract (5 each, shape-matched)
//   15. FoXBlock forward shape + input gradient FD check
//   16. FoXModel forward shape + training reduces loss
//   17. Multi-head (H=2) input gradient FD check
//   18. T=1 edge case (bias identically 0, attention is trivially [1])
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/attention/fox.h"

using namespace std;

static int passed = 0, total = 0;
static void check(bool cond, const string& what) {
    ++total;
    if (cond) { ++passed; cout << "[PASS] " << what << "\n"; }
    else      { cout << "[FAIL] " << what << "\n"; }
}

static double l2_loss_value(const Tensor& out, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& out, const Tensor& target) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i) g.data[i] = out.data[i] - target.data[i];
    return g;
}

static Tensor make_input(size_t n, size_t d, unsigned seed, double scale = 0.6) {
    mt19937 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);
    Tensor t(n, d);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dist(rng) * scale;
    return t;
}

// Scale-normalized FD error for a whole tensor.
//
// A per-element `|a-n| / max(|a|,|n|)` ratio is meaningless at the
// double-precision noise floor: for a tensor whose largest gradient is ~1e-6,
// an element of magnitude 1e-10 that agrees to 4 significant digits still
// reports rel_err ~1e-3, and the reported error GROWS as eps shrinks (the
// fingerprint of FD noise rather than a gradient bug). We therefore normalize
// the worst absolute discrepancy by the tensor's own gradient scale. This
// remains sensitive to real bugs — including global constant-factor bugs,
// since the FD estimate measures the true loss and does not shrink with a
// wrong analytical implementation.
static double normalized_fd_error(const std::vector<double>& ana,
                                  const std::vector<double>& num) {
    double worst_abs = 0.0, scale = 0.0;
    for (size_t i = 0; i < ana.size(); ++i) {
        worst_abs = max(worst_abs, fabs(ana[i] - num[i]));
        scale = max(scale, max(fabs(ana[i]), fabs(num[i])));
    }
    return worst_abs / max(scale, 1e-12);
}

// Centered-FD check on every element of the input.
template <typename LayerT>
static double check_input_grad(LayerT& layer, const Tensor& input, const Tensor& target) {
    Tensor out = layer.forward(input);
    Tensor g = l2_loss_grad(out, target);
    Tensor ana = layer.backward(g, 0.0);

    const double eps = 1e-6;
    vector<double> a, nvals;
    for (size_t i = 0; i < input.data.size(); ++i) {
        Tensor xp = input.clone(); xp.data[i] += eps;
        Tensor xm = input.clone(); xm.data[i] -= eps;
        double lp = l2_loss_value(layer.forward(xp), target);
        double lm = l2_loss_value(layer.forward(xm), target);
        a.push_back(ana.data[i]);
        nvals.push_back((lp - lm) / (2 * eps));
    }
    return normalized_fd_error(a, nvals);
}

// Centered-FD check on every element of a parameter tensor.
template <typename LayerT>
static double check_param_grad(LayerT& layer, Tensor& param, const Tensor& grad,
                               const Tensor& input, const Tensor& target) {
    layer.zero_grad();
    Tensor out = layer.forward(input);
    layer.backward(l2_loss_grad(out, target), 0.0);
    Tensor ana = grad.clone();

    const double eps = 1e-6;
    vector<double> a, nvals;
    for (size_t i = 0; i < param.data.size(); ++i) {
        double orig = param.data[i];
        param.data[i] = orig + eps;
        double lp = l2_loss_value(layer.forward(input), target);
        param.data[i] = orig - eps;
        double lm = l2_loss_value(layer.forward(input), target);
        param.data[i] = orig;
        a.push_back(ana.data[i]);
        nvals.push_back((lp - lm) / (2 * eps));
    }
    return normalized_fd_error(a, nvals);
}

int main() {
    cout << fixed << setprecision(6);
    cout << "=== Forgetting Transformer (FoX) Tests ===\n\n";

    // ---- Test 1: constructor validation ----
    cout << "-- Test 1: constructor validation\n";
    {
        bool threw = false;
        try { FoXAttention a(0, 1); (void)a; } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "d_model=0 throws");

        threw = false;
        try { FoXAttention a(4, 0); (void)a; } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "num_heads=0 throws");

        threw = false;
        try { FoXAttention a(6, 4); (void)a; } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "d_model not divisible by num_heads throws");

        threw = false;
        try { FoXAttention a(8, 2); (void)a; } catch (...) { threw = true; }
        check(!threw, "valid (8, 2) constructs");
    }

    // ---- Test 2: forward shape / finite / nonzero ----
    cout << "\n-- Test 2: forward shape, finiteness, nonzero\n";
    {
        srand(1234);
        FoXAttention a(8, 2);
        Tensor x = make_input(5, 8, 11);
        Tensor y = a.forward(x);
        check(y.rows == 5 && y.cols == 8, "forward shape (5,8) -> (5,8)");
        bool finite = true, nonzero = false;
        for (double v : y.data) { if (!std::isfinite(v)) finite = false; if (fabs(v) > 1e-12) nonzero = true; }
        check(finite, "forward output finite");
        check(nonzero, "forward output nonzero");
        check(a.head_dim() == 4, "head_dim == d_model / num_heads");
    }

    // ---- Test 3: forget gate in (0, 1) ----
    cout << "\n-- Test 3: forget gate f in (0, 1)\n";
    {
        srand(99);
        FoXAttention a(8, 2);
        Tensor x = make_input(6, 8, 12, 3.0);   // large scale -> extreme logits
        a.forward(x);
        const Tensor& f = a.get_last_forget();
        check(f.rows == 6 && f.cols == 2, "forget gate shape (n, num_heads)");
        bool ok = true;
        for (double v : f.data) if (!(v > 0.0 && v < 1.0)) ok = false;
        check(ok, "all forget gate values strictly in (0, 1)");
    }

    // ---- Test 4: bias diagonal zero, monotone non-positive ----
    cout << "\n-- Test 4: forget bias structure\n";
    {
        srand(7);
        FoXAttention a(8, 2);
        Tensor x = make_input(6, 8, 13);
        a.forward(x);
        bool diag_zero = true, nonpos = true, monotone = true;
        for (size_t h = 0; h < 2; ++h) {
            for (size_t t = 0; t < 6; ++t) {
                if (fabs(a.forget_bias(h, t, t)) > 1e-15) diag_zero = false;
                for (size_t s = 0; s < t; ++s) {
                    if (a.forget_bias(h, t, s) > 1e-15) nonpos = false;
                    if (s + 1 <= t && a.forget_bias(h, t, s) > a.forget_bias(h, t, s + 1) + 1e-15)
                        monotone = false;
                }
            }
        }
        check(diag_zero, "bias[t, t] == 0 (empty sum)");
        check(nonpos, "bias[t, s] <= 0 for s < t");
        check(monotone, "bias decreases monotonically as key gets older");
    }

    // ---- Test 5: attention rows sum to 1 and are causal ----
    cout << "\n-- Test 5: causal row-stochastic attention\n";
    {
        srand(21);
        FoXAttention a(8, 2);
        size_t n = 5;
        Tensor x = make_input(n, 8, 14);
        a.forward(x);
        const Tensor& A = a.get_last_attn();
        check(A.rows == 2 * n && A.cols == n, "attention cache shape (H*n, n)");
        double worst_sum = 0.0;
        bool causal = true;
        for (size_t h = 0; h < 2; ++h)
            for (size_t t = 0; t < n; ++t) {
                double s = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    s += A(h * n + t, k);
                    if (k > t && fabs(A(h * n + t, k)) > 0.0) causal = false;
                }
                worst_sum = max(worst_sum, fabs(s - 1.0));
            }
        cout << "   max |row_sum - 1| = " << scientific << worst_sum << fixed << "\n";
        check(worst_sum < 1e-12, "attention rows sum to 1");
        check(causal, "attention is strictly causal (A[t, s] == 0 for s > t)");
    }

    // ---- Test 6: causality via perturbation ----
    cout << "\n-- Test 6: future tokens do not affect past outputs\n";
    {
        srand(33);
        FoXAttention a(8, 2);
        size_t n = 4;
        Tensor x = make_input(n, 8, 15);
        Tensor y0 = a.forward(x);
        Tensor xp = x.clone();
        for (size_t j = 0; j < 8; ++j) xp(3, j) += 1.0;   // perturb LAST token only
        Tensor y1 = a.forward(xp);
        double worst = 0.0;
        for (size_t t = 0; t < 3; ++t)
            for (size_t j = 0; j < 8; ++j)
                worst = max(worst, fabs(y0(t, j) - y1(t, j)));
        cout << "   max |dy| for t < 3 = " << scientific << worst << fixed << "\n";
        check(worst == 0.0, "rows 0..2 bit-exact unchanged when token 3 is perturbed");
    }

    // ---- Test 7: input gradient FD ----
    cout << "\n-- Test 7: input gradient FD check\n";
    {
        srand(41);
        FoXAttention a(4, 1);
        Tensor x = make_input(4, 4, 16);
        Tensor tgt = make_input(4, 4, 17, 0.3);
        double err = check_input_grad(a, x, tgt);
        cout << "   input grad rel_err = " << scientific << err << fixed << "\n";
        check(err < 1e-4, "input gradient matches FD (single head)");
    }

    // ---- Tests 8/9: parameter gradient FD checks ----
    cout << "\n-- Tests 8/9: parameter gradient FD checks\n";
    {
        srand(55);
        FoXAttention a(4, 1);
        Tensor x = make_input(4, 4, 18);
        Tensor tgt = make_input(4, 4, 19, 0.3);

        struct P { const char* nm; Tensor* w; Tensor* g; };
        P ps[] = {
            {"W_q", &a.W_q, &a.grad_W_q},
            {"W_k", &a.W_k, &a.grad_W_k},
            {"W_v", &a.W_v, &a.grad_W_v},
            {"W_o", &a.W_o, &a.grad_W_o},
            {"W_f", &a.W_f, &a.grad_W_f},
        };
        for (auto& p : ps) {
            double err = check_param_grad(a, *p.w, *p.g, x, tgt);
            cout << "   " << p.nm << " grad rel_err = " << scientific << err << fixed << "\n";
            check(err < 1e-4, string(p.nm) + " gradient matches FD");
        }
    }

    // ---- Test 10: f -> 1 reduces to vanilla causal attention ----
    cout << "\n-- Test 10: f -> 1 degenerates to vanilla causal attention\n";
    {
        srand(61);
        size_t n = 4, d = 4;
        FoXAttention a(d, 1);
        // Force the forget gate saturated at 1: huge positive logits.
        a.W_f.fill(0.0);
        // Bias-free: with W_f == 0, z == 0 -> f = 0.5, so bias != 0. Instead make
        // the gate saturate by adding a large constant column via W_f and a
        // strictly-positive input.
        Tensor x(n, d);
        for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = 1.0;
        for (size_t k = 0; k < d; ++k) a.W_f(k, 0) = 50.0;   // z = 200 -> f ~ 1
        Tensor y = a.forward(x);

        // Reference: vanilla causal attention with the SAME W_q/W_k/W_v/W_o.
        double scale = 1.0 / sqrt(static_cast<double>(d) + 1e-9);
        Tensor Q(n, d), K(n, d), V(n, d);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < d; ++j) {
                double q = 0, k = 0, v = 0;
                for (size_t c = 0; c < d; ++c) {
                    q += x(i, c) * a.W_q(c, j);
                    k += x(i, c) * a.W_k(c, j);
                    v += x(i, c) * a.W_v(c, j);
                }
                Q(i, j) = q; K(i, j) = k; V(i, j) = v;
            }
        Tensor ref(n, d); ref.fill(0.0);
        for (size_t t = 0; t < n; ++t) {
            vector<double> sc(t + 1, 0.0);
            double m = -1e30;
            for (size_t s = 0; s <= t; ++s) {
                double v = 0.0;
                for (size_t c = 0; c < d; ++c) v += Q(t, c) * K(s, c);
                sc[s] = v * scale;
                m = max(m, sc[s]);
            }
            double l = 0.0;
            for (size_t s = 0; s <= t; ++s) { sc[s] = exp(sc[s] - m); l += sc[s]; }
            for (size_t s = 0; s <= t; ++s) sc[s] /= l;
            vector<double> ho(d, 0.0);
            for (size_t s = 0; s <= t; ++s)
                for (size_t c = 0; c < d; ++c) ho[c] += sc[s] * V(s, c);
            for (size_t j = 0; j < d; ++j) {
                double o = 0.0;
                for (size_t c = 0; c < d; ++c) o += ho[c] * a.W_o(c, j);
                ref(t, j) = o;
            }
        }
        double worst = 0.0;
        for (size_t i = 0; i < y.data.size(); ++i)
            worst = max(worst, fabs(y.data[i] - ref.data[i]));
        cout << "   max |FoX(f~1) - vanilla causal| = " << scientific << worst << fixed << "\n";
        check(worst < 1e-9, "f -> 1 reproduces vanilla causal attention");
    }

    // ---- Test 11: f -> 0 makes attention local ----
    cout << "\n-- Test 11: f -> 0 makes attention diagonal-dominant\n";
    {
        srand(71);
        size_t n = 5, d = 4;
        FoXAttention a(d, 1);
        Tensor x(n, d);
        for (size_t i = 0; i < x.data.size(); ++i) x.data[i] = 1.0;
        for (size_t k = 0; k < d; ++k) a.W_f(k, 0) = -50.0;  // z = -200 -> f ~ 0
        a.forward(x);
        const Tensor& A = a.get_last_attn();
        bool diag_dom = true;
        for (size_t t = 1; t < n; ++t) if (A(t, t) < 0.99) diag_dom = false;
        cout << "   A[4][4] = " << A(4, 4) << ", A[4][3] = " << A(4, 3) << "\n";
        check(diag_dom, "with f ~ 0 each query attends (almost) only to itself");
    }

    // ---- Test 12: determinism with copied params ----
    cout << "\n-- Test 12: determinism\n";
    {
        srand(81);
        FoXAttention a(8, 2);
        srand(9999);
        FoXAttention b(8, 2);
        auto pa = a.parameters();
        auto pb = b.parameters();
        for (size_t i = 0; i < pa.size(); ++i) *pb[i] = pa[i]->clone();
        Tensor x = make_input(5, 8, 20);
        Tensor ya = a.forward(x);
        Tensor yb = b.forward(x);
        double worst = 0.0;
        for (size_t i = 0; i < ya.data.size(); ++i)
            worst = max(worst, fabs(ya.data[i] - yb.data[i]));
        check(worst == 0.0, "copied params produce bit-exact forward (max diff 0)");
    }

    // ---- Tests 13/14: contract ----
    cout << "\n-- Tests 13/14: parameter/gradient contract\n";
    {
        srand(91);
        FoXAttention a(8, 2);
        auto ps = a.parameters();
        auto gs = a.gradients();
        check(ps.size() == 5 && gs.size() == 5, "parameters() and gradients() both return 5");
        bool shapes_ok = true;
        for (size_t i = 0; i < ps.size(); ++i)
            if (ps[i]->rows != gs[i]->rows || ps[i]->cols != gs[i]->cols) shapes_ok = false;
        check(shapes_ok, "all 5 param/grad pairs shape-matched");
        check(a.W_f.rows == 8 && a.W_f.cols == 2, "W_f shape (d_model, num_heads)");

        Tensor x = make_input(4, 8, 21);
        Tensor tgt = make_input(4, 8, 22, 0.3);
        a.backward(l2_loss_grad(a.forward(x), tgt), 0.0);
        double gn = 0.0;
        for (auto* g : gs) for (double v : g->data) gn += fabs(v);
        check(gn > 0.0, "gradients nonzero after backward");
        a.zero_grad();
        double gn2 = 0.0;
        for (auto* g : gs) for (double v : g->data) gn2 += fabs(v);
        check(gn2 == 0.0, "zero_grad clears all 5 gradients");

        // update_weights moves every parameter
        a.backward(l2_loss_grad(a.forward(x), tgt), 0.0);
        vector<Tensor> before;
        for (auto* p : ps) before.push_back(p->clone());
        a.update_weights(0.1);
        bool all_moved = true;
        for (size_t i = 0; i < ps.size(); ++i) {
            bool moved = false;
            for (size_t j = 0; j < ps[i]->data.size(); ++j)
                if (fabs(ps[i]->data[j] - before[i].data[j]) > 0.0) moved = true;
            if (!moved) all_moved = false;
        }
        check(all_moved, "update_weights moves all 5 parameters");
    }

    // ---- Test 15: FoXBlock ----
    cout << "\n-- Test 15: FoXBlock\n";
    {
        srand(101);
        FoXBlock blk(4, 1, 8);
        Tensor x = make_input(4, 4, 23);
        Tensor y = blk.forward(x);
        check(y.rows == 4 && y.cols == 4, "FoXBlock forward shape (4,4) -> (4,4)");
        bool finite = true;
        for (double v : y.data) if (!std::isfinite(v)) finite = false;
        check(finite, "FoXBlock forward finite");

        Tensor tgt = make_input(4, 4, 24, 0.3);
        double err = check_input_grad(blk, x, tgt);
        cout << "   FoXBlock input grad rel_err = " << scientific << err << fixed << "\n";
        check(err < 1e-3, "FoXBlock input gradient matches FD");
    }

    // ---- Test 16: FoXModel ----
    cout << "\n-- Test 16: FoXModel forward + training\n";
    {
        srand(111);
        FoXModel m(3, 4, 2, 2, 1, 8);
        Tensor x = make_input(4, 3, 25);
        Tensor y = m.forward(x);
        check(y.rows == 4 && y.cols == 2, "FoXModel forward shape (4,3) -> (4,2)");
        check(m.num_blocks() == 2, "FoXModel has 2 blocks");

        Tensor tgt = make_input(4, 2, 26, 0.3);
        double l0 = l2_loss_value(m.forward(x), tgt);
        for (int step = 0; step < 120; ++step) {
            m.zero_grad();
            Tensor out = m.forward(x);
            m.backward(l2_loss_grad(out, tgt), 0.0);
            m.update_weights(0.02);
        }
        double lf = l2_loss_value(m.forward(x), tgt);
        cout << "   loss " << l0 << " -> " << lf << "\n";
        check(std::isfinite(lf) && lf < l0 * 0.7, "FoXModel training reduces loss by >30%");
    }

    // ---- Test 17: multi-head input gradient ----
    cout << "\n-- Test 17: multi-head (H=2) input gradient FD check\n";
    {
        srand(121);
        FoXAttention a(6, 2);
        Tensor x = make_input(5, 6, 27);
        Tensor tgt = make_input(5, 6, 28, 0.3);
        double err = check_input_grad(a, x, tgt);
        cout << "   H=2 input grad rel_err = " << scientific << err << fixed << "\n";
        check(err < 1e-4, "multi-head input gradient matches FD");

        double errf = check_param_grad(a, a.W_f, a.grad_W_f, x, tgt);
        cout << "   H=2 W_f grad rel_err = " << scientific << errf << fixed << "\n";
        check(errf < 1e-4, "multi-head W_f gradient matches FD");
    }

    // ---- Test 18: T=1 edge case ----
    cout << "\n-- Test 18: T=1 edge case\n";
    {
        srand(131);
        FoXAttention a(4, 1);
        Tensor x = make_input(1, 4, 29);
        Tensor y = a.forward(x);
        check(y.rows == 1 && y.cols == 4, "T=1 forward shape");
        check(fabs(a.get_last_attn()(0, 0) - 1.0) < 1e-15, "T=1 attention is exactly [1]");
        Tensor tgt = make_input(1, 4, 30, 0.3);
        double err = check_input_grad(a, x, tgt);
        cout << "   T=1 input grad rel_err = " << scientific << err << fixed << "\n";
        check(err < 1e-4, "T=1 input gradient matches FD");
    }

    cout << "\n=== Results: " << passed << "/" << total << " tests passed ===" << endl;
    return (passed == total) ? 0 : 1;
}
