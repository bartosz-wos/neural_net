// nGPT — Normalized Transformer on the Hypersphere
//   Loshchilov, Hsieh, Sun, Ginsburg (NVIDIA 2024), https://arxiv.org/abs/2410.01131
//
// Every representation and every projection-matrix ROW is constrained to unit L2
// norm, so all vectors live on a hypersphere. LayerNorm/RMSNorm are removed
// entirely; residual updates become a normalized LERP toward the sublayer output
// with learnable per-dimension "eigen learning rates" alpha:
//
//   h <- Norm( h + alpha * (Norm(f(h)) - h) )
//
// Tests:
//   1.  HypersphereLinear constructor validation
//   2.  HypersphereLinear forward shape + finite + nonzero
//   3.  Row-unit-norm invariant on normalized_weights()
//   4.  W gradient FD check (the row-normalization Jacobian — the crux)
//   5.  Hand-derived 1-row Jacobian case
//   6.  s (per-output scale) gradient FD check
//   7.  Input gradient FD check
//   8.  Unit-norm invariant survives update_weights (and W actually moved)
//   9.  zero_grad clears both gradients
//   10. parameters()/gradients() contract
//   11. NGPTBlock constructor validation
//   12. NGPTBlock forward shape + finite
//   13. NGPTBlock output rows are unit-norm (the hypersphere invariant)
//   14. alpha = 0 -> identity on a unit-norm input; alpha = 1 -> Norm(f(h))
//   15. NGPTBlock input gradient FD check
//   16. alpha_attn gradient FD check (eigen learning rate)
//   17. alpha_mlp gradient FD check
//   18. s_qk gradient FD check
//   19. W_q gradient FD check
//   20. W_k gradient FD check
//   21. W_v gradient FD check
//   22. W_o gradient FD check
//   23. MLP fc1 gradient FD check
//   24. MLP fc2 gradient FD check
//   25. NGPTModel forward shape
//   26. NGPTModel training reduces loss
//   27. NGPTModel parameters()/gradients() count contract
//   28. NGPTModel determinism with copied params
//   29. Multi-head (H=2) block input gradient FD check
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/architectures/ngpt.h"

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

// Scale-normalized FD error (same helper as tests/test_fox.cpp): worst absolute
// discrepancy divided by the tensor's own gradient scale. A per-element relative
// error is meaningless at the double-precision noise floor.
static double normalized_fd_error(const vector<double>& ana, const vector<double>& num) {
    double worst_abs = 0.0, scale = 0.0;
    for (size_t i = 0; i < ana.size(); ++i) {
        worst_abs = max(worst_abs, fabs(ana[i] - num[i]));
        scale = max(scale, max(fabs(ana[i]), fabs(num[i])));
    }
    return worst_abs / max(scale, 1e-12);
}

template <typename LayerT>
static double check_input_grad(LayerT& layer, const Tensor& input, const Tensor& target) {
    layer.zero_grad();
    Tensor out = layer.forward(input);
    Tensor ana = layer.backward(l2_loss_grad(out, target), 0.0);

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
    cout << fixed << setprecision(9);
    cout << "=== nGPT (Normalized Transformer on the Hypersphere) Tests ===\n\n";

    // ---------------------------------------------------------------- Test 1
    cout << "-- Test 1: HypersphereLinear constructor validation\n";
    {
        bool t1 = false, t2 = false, ok = true;
        try { HypersphereLinear h(0, 4); } catch (const invalid_argument&) { t1 = true; }
        try { HypersphereLinear h(4, 0); } catch (const invalid_argument&) { t2 = true; }
        try { HypersphereLinear h(4, 3); } catch (...) { ok = false; }
        check(t1, "in_features=0 throws");
        check(t2, "out_features=0 throws");
        check(ok, "valid (4, 3) constructs");
    }

    // ---------------------------------------------------------------- Test 2
    cout << "\n-- Test 2: HypersphereLinear forward shape + finite + nonzero\n";
    {
        HypersphereLinear h(4, 3);
        Tensor x = make_input(5, 4, 11);
        Tensor y = h.forward(x);
        check(y.rows == 5 && y.cols == 3, "forward shape (5,4) -> (5,3)");
        bool finite = true, nonzero = false;
        for (double v : y.data) { if (!isfinite(v)) finite = false; if (fabs(v) > 1e-12) nonzero = true; }
        check(finite, "forward output finite");
        check(nonzero, "forward output nonzero");
    }

    // ---------------------------------------------------------------- Test 3
    cout << "\n-- Test 3: row-unit-norm invariant\n";
    {
        HypersphereLinear h(5, 4);
        Tensor W = h.normalized_weights();
        double worst = 0.0;
        for (size_t o = 0; o < W.rows; ++o) {
            double n2 = 0.0;
            for (size_t c = 0; c < W.cols; ++c) n2 += W(o, c) * W(o, c);
            worst = max(worst, fabs(sqrt(n2) - 1.0));
        }
        cout << "   worst |row_norm - 1| = " << worst << "\n";
        check(worst < 1e-12, "every normalized_weights() row has unit L2 norm");
    }

    // ---------------------------------------------------------------- Test 4
    cout << "\n-- Test 4: W gradient FD (row-normalization Jacobian)\n";
    {
        // Non-square (3 -> 2) with random non-uniform init so a row-vs-column
        // transposition in the Jacobian cannot pass vacuously.
        HypersphereLinear h(3, 2);
        Tensor x = make_input(4, 3, 21);
        Tensor tgt = make_input(4, 2, 22);
        double e = check_param_grad(h, h.W, h.grad_W, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "W gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 5
    cout << "\n-- Test 5: hand-derived 1-row Jacobian\n";
    {
        // W = [[3, 4]] -> ||w|| = 5, w_hat = [0.6, 0.8].  x = [[1, 0]], s = 1.
        // y = 0.6.  With dy = 1: dW_hat = [[1, 0]], r = 0.6,
        // dW = ([1, 0] - [0.6, 0.8] * 0.6) / 5 = [0.128, -0.096].
        HypersphereLinear h(2, 1);
        h.W(0, 0) = 3.0; h.W(0, 1) = 4.0;
        h.s(0, 0) = 1.0;
        Tensor x(1, 2); x(0, 0) = 1.0; x(0, 1) = 0.0;
        Tensor y = h.forward(x);
        cout << "   y = " << y(0, 0) << " (expect 0.6)\n";
        check(fabs(y(0, 0) - 0.6) < 1e-12, "forward y == 0.6");

        h.zero_grad();
        Tensor dy(1, 1); dy(0, 0) = 1.0;
        h.backward(dy, 0.0);
        cout << "   dW = [" << h.grad_W(0, 0) << ", " << h.grad_W(0, 1) << "]"
             << " (expect [0.128, -0.096])\n";
        check(fabs(h.grad_W(0, 0) - 0.128) < 1e-12, "dW[0,0] == 0.128");
        check(fabs(h.grad_W(0, 1) + 0.096) < 1e-12, "dW[0,1] == -0.096");
        // The projector kills the radial component: dW . w_hat == 0.
        double radial = h.grad_W(0, 0) * 0.6 + h.grad_W(0, 1) * 0.8;
        cout << "   dW . w_hat = " << radial << " (expect 0)\n";
        check(fabs(radial) < 1e-15, "gradient is tangent to the sphere (dW . w_hat == 0)");
    }

    // ---------------------------------------------------------------- Test 6
    cout << "\n-- Test 6: s (per-output scale) gradient FD\n";
    {
        HypersphereLinear h(3, 2);
        Tensor x = make_input(4, 3, 31);
        Tensor tgt = make_input(4, 2, 32);
        double e = check_param_grad(h, h.s, h.grad_s, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "s gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 7
    cout << "\n-- Test 7: HypersphereLinear input gradient FD\n";
    {
        HypersphereLinear h(3, 2);
        Tensor x = make_input(4, 3, 41);
        Tensor tgt = make_input(4, 2, 42);
        double e = check_input_grad(h, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "input gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 8
    cout << "\n-- Test 8: unit-norm invariant survives update_weights\n";
    {
        HypersphereLinear h(4, 3);
        Tensor x = make_input(3, 4, 51);
        Tensor tgt = make_input(3, 3, 52);
        Tensor W_before = h.W.clone();

        h.zero_grad();
        Tensor out = h.forward(x);
        h.backward(l2_loss_grad(out, tgt), 0.0);
        h.update_weights(0.5);

        double moved = 0.0;
        for (size_t i = 0; i < h.W.data.size(); ++i)
            moved = max(moved, fabs(h.W.data[i] - W_before.data[i]));
        cout << "   max |W - W_before| = " << moved << "\n";
        check(moved > 1e-9, "update_weights actually moved W (test is not vacuous)");

        Tensor Wn = h.normalized_weights();
        double worst = 0.0;
        for (size_t o = 0; o < Wn.rows; ++o) {
            double n2 = 0.0;
            for (size_t c = 0; c < Wn.cols; ++c) n2 += Wn(o, c) * Wn(o, c);
            worst = max(worst, fabs(sqrt(n2) - 1.0));
        }
        cout << "   worst |row_norm - 1| after step = " << worst << "\n";
        check(worst < 1e-12, "rows still unit-norm after update_weights");

        // update_weights re-normalizes W in place, so the stored W is unit-norm too.
        double worst_raw = 0.0;
        for (size_t o = 0; o < h.W.rows; ++o) {
            double n2 = 0.0;
            for (size_t c = 0; c < h.W.cols; ++c) n2 += h.W(o, c) * h.W(o, c);
            worst_raw = max(worst_raw, fabs(sqrt(n2) - 1.0));
        }
        cout << "   worst |raw W row_norm - 1| after step = " << worst_raw << "\n";
        check(worst_raw < 1e-12, "raw W re-normalized in place by update_weights");
    }

    // ---------------------------------------------------------------- Test 9
    cout << "\n-- Test 9: zero_grad\n";
    {
        HypersphereLinear h(3, 2);
        Tensor x = make_input(3, 3, 61);
        Tensor tgt = make_input(3, 2, 62);
        Tensor out = h.forward(x);
        h.backward(l2_loss_grad(out, tgt), 0.0);
        bool had = false;
        for (double v : h.grad_W.data) if (fabs(v) > 1e-14) had = true;
        check(had, "gradients nonzero before zero_grad");
        h.zero_grad();
        double m = 0.0;
        for (double v : h.grad_W.data) m = max(m, fabs(v));
        for (double v : h.grad_s.data) m = max(m, fabs(v));
        check(m == 0.0, "zero_grad clears grad_W and grad_s");
    }

    // ---------------------------------------------------------------- Test 10
    cout << "\n-- Test 10: parameters()/gradients() contract\n";
    {
        HypersphereLinear h(3, 2);
        auto p = h.parameters();
        auto g = h.gradients();
        check(p.size() == 2 && g.size() == 2, "2 parameters and 2 gradients");
        bool shapes = true;
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes = false;
        check(shapes, "parameter/gradient shapes match");
        check(h.name() == "HypersphereLinear", "name() == HypersphereLinear");
    }

    // ---------------------------------------------------------------- Test 11
    cout << "\n-- Test 11: NGPTBlock constructor validation\n";
    {
        bool t1 = false, t2 = false, t3 = false, ok = true;
        try { NGPTBlock b(0, 1); } catch (const invalid_argument&) { t1 = true; }
        try { NGPTBlock b(8, 0); } catch (const invalid_argument&) { t2 = true; }
        try { NGPTBlock b(8, 3); } catch (const invalid_argument&) { t3 = true; }
        try { NGPTBlock b(8, 2); } catch (...) { ok = false; }
        check(t1, "d_model=0 throws");
        check(t2, "num_heads=0 throws");
        check(t3, "d_model % num_heads != 0 throws");
        check(ok, "valid (8, 2) constructs");
    }

    // ---------------------------------------------------------------- Test 12
    cout << "\n-- Test 12: NGPTBlock forward shape + finite\n";
    {
        NGPTBlock b(8, 2);
        Tensor x = make_input(4, 8, 71);
        Tensor y = b.forward(x);
        check(y.rows == 4 && y.cols == 8, "forward shape (4,8) -> (4,8)");
        bool finite = true;
        for (double v : y.data) if (!isfinite(v)) finite = false;
        check(finite, "forward output finite");
    }

    // ---------------------------------------------------------------- Test 13
    cout << "\n-- Test 13: NGPTBlock output rows are unit-norm\n";
    {
        NGPTBlock b(8, 2);
        Tensor x = make_input(5, 8, 81);
        Tensor y = b.forward(x);
        double worst = 0.0;
        for (size_t t = 0; t < y.rows; ++t) {
            double n2 = 0.0;
            for (size_t c = 0; c < y.cols; ++c) n2 += y(t, c) * y(t, c);
            worst = max(worst, fabs(sqrt(n2) - 1.0));
        }
        cout << "   worst |row_norm - 1| = " << worst << "\n";
        check(worst < 1e-12, "every output row lies on the unit hypersphere");
    }

    // ---------------------------------------------------------------- Test 14
    cout << "\n-- Test 14: alpha = 0 -> identity; alpha = 1 -> Norm(f(h))\n";
    {
        NGPTBlock b(8, 2);
        b.alpha_attn.fill(0.0);
        b.alpha_mlp.fill(0.0);
        // Unit-norm input rows, so Norm() is a no-op on the skip path.
        Tensor x = make_input(4, 8, 91);
        for (size_t t = 0; t < x.rows; ++t) {
            double n2 = 0.0;
            for (size_t c = 0; c < x.cols; ++c) n2 += x(t, c) * x(t, c);
            double inv = 1.0 / sqrt(n2);
            for (size_t c = 0; c < x.cols; ++c) x(t, c) *= inv;
        }
        Tensor y = b.forward(x);
        double worst = 0.0;
        for (size_t i = 0; i < y.data.size(); ++i) worst = max(worst, fabs(y.data[i] - x.data[i]));
        cout << "   alpha=0 max |y - x| = " << worst << "\n";
        check(worst < 1e-12, "alpha = 0 makes the block an identity on unit-norm input");

        // alpha = 1: the skip term vanishes, output depends on h ONLY through f(h).
        // Signature check that this is a LERP and not a plain (h + f(h)) residual:
        // with alpha = 1 the MLP-sublayer output equals Norm(f_mlp(.)) exactly, so
        // scaling the residual stream cannot shift the result the way a plain
        // additive residual would.
        NGPTBlock b1(8, 2);
        b1.alpha_attn.fill(1.0);
        b1.alpha_mlp.fill(1.0);
        Tensor y1 = b1.forward(x);
        Tensor ref = b1.last_mlp_normalized();
        double worst1 = 0.0;
        for (size_t i = 0; i < y1.data.size(); ++i) worst1 = max(worst1, fabs(y1.data[i] - ref.data[i]));
        cout << "   alpha=1 max |y - Norm(f_mlp)| = " << worst1 << "\n";
        check(worst1 < 1e-12, "alpha = 1 makes the output exactly Norm(f(h))");
    }

    // ---------------------------------------------------------------- Test 15
    cout << "\n-- Test 15: NGPTBlock input gradient FD\n";
    {
        NGPTBlock b(4, 1);
        Tensor x = make_input(3, 4, 101);
        Tensor tgt = make_input(3, 4, 102);
        double e = check_input_grad(b, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "block input gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Tests 16-24
    cout << "\n-- Tests 16-24: NGPTBlock parameter gradient FD checks\n";
    {
        NGPTBlock b(4, 1);
        Tensor x = make_input(3, 4, 111);
        Tensor tgt = make_input(3, 4, 112);

        struct Item { const char* nm; Tensor* p; Tensor* g; };
        vector<Item> items = {
            { "alpha_attn", &b.alpha_attn,   &b.grad_alpha_attn },
            { "alpha_mlp",  &b.alpha_mlp,    &b.grad_alpha_mlp  },
            { "s_qk",       &b.s_qk,         &b.grad_s_qk       },
            { "W_q",        &b.W_q.W,        &b.W_q.grad_W      },
            { "W_k",        &b.W_k.W,        &b.W_k.grad_W      },
            { "W_v",        &b.W_v.W,        &b.W_v.grad_W      },
            { "W_o",        &b.W_o.W,        &b.W_o.grad_W      },
            { "fc1",        &b.fc1.W,        &b.fc1.grad_W      },
            { "fc2",        &b.fc2.W,        &b.fc2.grad_W      },
        };
        for (auto& it : items) {
            double e = check_param_grad(b, *it.p, *it.g, x, tgt);
            cout << "   " << it.nm << " rel_err = " << e << "\n";
            check(e < 1e-4, string(it.nm) + " gradient matches centered FD");
        }
    }

    // ---------------------------------------------------------------- Test 25
    cout << "\n-- Test 25: NGPTModel forward shape\n";
    {
        NGPTModel m(3, 8, 2, 2, 2);
        Tensor x = make_input(4, 3, 121);
        Tensor y = m.forward(x);
        check(y.rows == 4 && y.cols == 2, "forward shape (4,3) -> (4,2)");
        bool finite = true;
        for (double v : y.data) if (!isfinite(v)) finite = false;
        check(finite, "forward output finite");
        check(m.num_blocks() == 2, "num_blocks() == 2");
    }

    // ---------------------------------------------------------------- Test 26
    cout << "\n-- Test 26: NGPTModel training reduces loss\n";
    {
        NGPTModel m(3, 8, 2, 2, 2);
        Tensor x = make_input(4, 3, 131);
        Tensor tgt = make_input(4, 2, 132);
        double l0 = l2_loss_value(m.forward(x), tgt);
        for (int step = 0; step < 60; ++step) {
            m.zero_grad();
            Tensor out = m.forward(x);
            m.backward(l2_loss_grad(out, tgt), 0.0);
            m.update_weights(0.05);
        }
        double lf = l2_loss_value(m.forward(x), tgt);
        cout << "   L0 = " << l0 << "  Lf = " << lf << "\n";
        check(isfinite(lf) && lf < l0, "training reduces loss over 60 SGD steps");
    }

    // ---------------------------------------------------------------- Test 27
    cout << "\n-- Test 27: NGPTModel parameters()/gradients() contract\n";
    {
        NGPTModel m(3, 8, 2, 2, 2);
        auto p = m.parameters();
        auto g = m.gradients();
        check(p.size() == g.size() && !p.empty(), "parameter and gradient counts agree");
        bool shapes = true;
        for (size_t i = 0; i < p.size(); ++i)
            if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes = false;
        check(shapes, "all parameter/gradient shapes match");
    }

    // ---------------------------------------------------------------- Test 28
    cout << "\n-- Test 28: NGPTModel determinism with copied params\n";
    {
        NGPTModel a(3, 8, 2, 1, 2), b(3, 8, 2, 1, 2);
        auto pa = a.parameters();
        auto pb = b.parameters();
        check(pa.size() == pb.size(), "same parameter count");
        for (size_t i = 0; i < pa.size(); ++i) *pb[i] = pa[i]->clone();
        Tensor x = make_input(4, 3, 141);
        Tensor ya = a.forward(x);
        Tensor yb = b.forward(x);
        double worst = 0.0;
        for (size_t i = 0; i < ya.data.size(); ++i) worst = max(worst, fabs(ya.data[i] - yb.data[i]));
        cout << "   max abs diff = " << worst << "\n";
        check(worst == 0.0, "copied params -> bit-exact forward");
    }

    // ---------------------------------------------------------------- Test 29
    cout << "\n-- Test 29: multi-head (H=2) block input gradient FD\n";
    {
        NGPTBlock b(6, 2);
        Tensor x = make_input(4, 6, 151);
        Tensor tgt = make_input(4, 6, 152);
        double e = check_input_grad(b, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "multi-head block input gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 30
    // Exercises the MODEL-level chain: classifier -> blocks -> the pre-block row
    // normalization -> input_proj. Without this, dropping the embed-normalization
    // Jacobian in NGPTModel::backward passes every other test vacuously (the
    // training test still converges because the direction is only mildly wrong).
    cout << "\n-- Test 30: NGPTModel input gradient FD (embed-normalization chain)\n";
    {
        NGPTModel m(3, 4, 2, 1, 1);
        Tensor x = make_input(3, 3, 161);
        Tensor tgt = make_input(3, 2, 162);
        double e = check_input_grad(m, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "model input gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 31
    cout << "\n-- Test 31: NGPTModel input_proj.W gradient FD\n";
    {
        NGPTModel m(3, 4, 2, 1, 1);
        Tensor x = make_input(3, 3, 171);
        Tensor tgt = make_input(3, 2, 172);
        double e = check_param_grad(m, m.input_proj.W, m.input_proj.grad_W, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "input_proj.W gradient matches centered FD");
    }

    // ---------------------------------------------------------------- Test 32
    cout << "\n-- Test 32: NGPTModel classifier.weights gradient FD\n";
    {
        NGPTModel m(3, 4, 2, 1, 1);
        Tensor x = make_input(3, 3, 181);
        Tensor tgt = make_input(3, 2, 182);
        double e = check_param_grad(m, m.classifier.weights, m.classifier.grad_weights, x, tgt);
        cout << "   rel_err = " << e << "\n";
        check(e < 1e-4, "classifier.weights gradient matches centered FD");
    }

    cout << "\n=== Summary: " << passed << " passed, " << (total - passed) << " failed ===\n";
    return (passed == total) ? 0 : 1;
}
