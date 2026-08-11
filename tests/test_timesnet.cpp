// TimesNet tests — Wu et al. 2023, ICLR
//   https://arxiv.org/abs/2210.02186
//
// Test categories (per TDD: must catch known-bad implementations):
//   1.  FFT helper correctness (known sinusoid peak)
//   2.  topk_periods sort order (deterministic amplitudes)
//   3.  DataEmbedding / Conv2DBlock / forward shapes + finiteness
//   4.  TimesBlock forward shape + finite
//   5.  TimesBlock INPUT GRADIENT vs FD (rel_err < 1e-3) — must be non-vacuous
//   6.  TimesBlock PARAM GRADIENTS vs FD (ln1 gamma, ffn1 W) — non-vacuous
//   7.  TimesNet forward shape + finite
//   8.  TimesNet INPUT GRADIENT vs FD (rel_err < 1e-3) — non-vacuous
//   9.  TimesNet training reduces loss (synthetic sine)
//  10.  TimesNet determinism (two fresh instances with COPIED params bit-exact)
//  11.  Multi-layer (e_layers > 1) forward shape + finite

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <complex>
#include <algorithm>
#include "nn/layers/architectures/timesnet.h"

using namespace std;

static double max_abs_err(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    size_t n = std::min(a.data.size(), b.data.size());
    for (size_t i = 0; i < n; ++i) m = max(m, fabs(a.data[i] - b.data[i]));
    return m;
}

static double max_rel_err_floor(const Tensor& a, const Tensor& b, double floor_v = 1e-5) {
    double m = 0.0;
    size_t n = std::min(a.data.size(), b.data.size());
    for (size_t i = 0; i < n; ++i) {
        double av = a.data[i], bv = b.data[i];
        double denom = max(max(fabs(av), fabs(bv)), floor_v);
        m = max(m, fabs(av - bv) / denom);
    }
    return m;
}

static double l2_loss_value(const Tensor& output, const Tensor& target) {
    double s = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        double d = output.data[i] - target.data[i];
        s += 0.5 * d * d;
    }
    return s;
}

static Tensor l2_loss_grad(const Tensor& output, const Tensor& target) {
    Tensor g(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        g.data[i] = output.data[i] - target.data[i];
    }
    return g;
}

static void fill_random(Tensor& t, mt19937& gen, double scale = 0.3) {
    normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = dis(gen);
}

// FD grad check on input via central differences.
template <typename OpT>
static Tensor finite_diff_grad_input(OpT& op, Tensor& input, const Tensor& target,
                                     double eps = 1e-3) {
    Tensor orig = input.clone();
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double v = orig(i, j);
            input(i, j) = v + eps;
            Tensor out_p = op.forward(input);
            double lp = l2_loss_value(out_p, target);
            input(i, j) = v - eps;
            Tensor out_m = op.forward(input);
            double lm = l2_loss_value(out_m, target);
            input(i, j) = v;
            grad(i, j) = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

int main() {
    cout << "=== TimesNet Tests ===" << endl;
    cout.setf(ios::unitbuf);
    int total = 0, passed = 0;
    auto check = [&](bool ok, const string& msg) {
        ++total;
        if (ok) { ++passed; cout << "[PASS] " << msg << "\n"; }
        else    { cout << "[FAIL] " << msg << "\n"; }
    };

    // -----------------------------------------------------------------------
    // Test 1: naive_dft peak location for a known sinusoid
    // -----------------------------------------------------------------------
    {
        vector<double> x(16);
        for (int i = 0; i < 16; ++i) x[i] = sin(2.0 * M_PI * 4.0 * i / 16.0);
        auto X = naive_dft(x);
        vector<double> amp(16);
        for (int k = 0; k < 16; ++k) amp[k] = abs(X[k]);
        int peak_lo = -1; double peak_amp = 0;
        for (int k = 1; k < 8; ++k) {
            if (amp[k] > peak_amp) { peak_amp = amp[k]; peak_lo = k; }
        }
        check(peak_lo == 4, "naive_dft sinusoid peak at k=N/period=4 (got k=" + to_string(peak_lo) + ")");
    }

    // -----------------------------------------------------------------------
    // Test 2: topk_periods returns periods sorted by amplitude
    // -----------------------------------------------------------------------
    {
        vector<double> amp(8, 0.0);
        amp[2] = 1.0;   // k=2 → period N/k = 4
        amp[4] = 3.0;   // k=4 → period 2 (largest amp)
        amp[6] = 0.5;
        auto p = topk_periods(amp, 3);
        check(p.size() == 3, "topk_periods returns K=3 entries");
        check(p[0] == 2, "topk_periods[0] == 2 (largest amp)");
        check(p[1] == 4, "topk_periods[1] == 4 (second-largest amp)");
    }

    // -----------------------------------------------------------------------
    // Test 3: Conv2DBlock forward shape + finite
    // -----------------------------------------------------------------------
    {
        const int in_ch = 2, hidden_ch = 4, out_ch = 2, H = 8, W = 8;
        Conv2DBlock blk(in_ch, hidden_ch, out_ch, H, W);
        Tensor x(2, in_ch * H * W);
        mt19937 gen(7);
        fill_random(x, gen, 0.3);
        Tensor y = blk.forward(x);
        check(y.rows == 2 && y.cols == (size_t)(out_ch * H * W),
             "Conv2DBlock forward shape (B, in_ch*H*W) -> (B, out_ch*H*W)");
        bool finite = true;
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!isfinite(y.data[i])) { finite = false; break; }
        check(finite, "Conv2DBlock output is finite");
    }

    // -----------------------------------------------------------------------
    // Test 4: TimesBlock forward shape (B*T, d_model) -> (B*T, d_model)
    // -----------------------------------------------------------------------
    {
        const int d_model = 4, seq_len = 8, top_k = 2, d_ff = 8;
        TimesBlock blk(d_model, seq_len, top_k, d_ff);
        Tensor x(2 * seq_len, d_model);
        mt19937 gen(11);
        fill_random(x, gen, 0.3);
        Tensor y = blk.forward(x);
        check(y.rows == (size_t)(2 * seq_len) && y.cols == (size_t)d_model,
             "TimesBlock forward shape (B*T, d_model) -> (B*T, d_model)");
        bool finite = true;
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!isfinite(y.data[i])) { finite = false; break; }
        check(finite, "TimesBlock output is finite");
    }

    // -----------------------------------------------------------------------
    // Test 5: TimesBlock INPUT GRADIENT vs FD (rel_err < 1e-3)
    //         Uses fixed periods so the loss is smooth.
    // -----------------------------------------------------------------------
    {
        const int d_model = 2, seq_len = 8, top_k = 2, d_ff = 4;
        TimesBlock blk(d_model, seq_len, top_k, d_ff);
        blk.set_fixed_periods({2, 4});
        Tensor x(1 * seq_len, d_model);
        mt19937 gen(19);
        fill_random(x, gen, 0.3);
        Tensor target(1 * seq_len, d_model);
        fill_random(target, gen, 0.1);

        Tensor y = blk.forward(x);
        Tensor grad_out = l2_loss_grad(y, target);
        blk.zero_grad();
        Tensor grad_in_ana = blk.backward(grad_out, 0.0);

        Tensor grad_in_fd = finite_diff_grad_input(blk, x, target);
        double rel = max_rel_err_floor(grad_in_ana, grad_in_fd, 1e-5);
        check(rel < 1e-3, "TimesBlock input grad rel_err < 1e-3 with fixed periods (got " + to_string(rel) + ")");
    }

    // -----------------------------------------------------------------------
    // Test 6: TimesBlock PARAMETER gradients vs FD (ln1 gamma + ffn1 W).
    //         FD on a single scalar element of the parameter tensor — perturb
    //         every element by ε in turn. Small param size for tractability.
    // -----------------------------------------------------------------------
    {
        const int d_model = 2, seq_len = 8, top_k = 2, d_ff = 4;
        TimesBlock blk(d_model, seq_len, top_k, d_ff);
        blk.set_fixed_periods({2, 4});
        Tensor x(seq_len, d_model);
        mt19937 gen(23);
        fill_random(x, gen, 0.3);
        Tensor target(seq_len, d_model);
        fill_random(target, gen, 0.1);

        // ---- (a) ln1 gamma gradient: FD against ln1_.grad_gamma_ ----
        // Re-run forward to populate caches, then perturb each gamma element.
        Tensor y = blk.forward(x);
        Tensor grad_out = l2_loss_grad(y, target);
        blk.zero_grad();
        blk.backward(grad_out, 0.0);
        Tensor ana_ln1 = blk.ln1_.grad_gamma_.clone();
        // FD: perturb each element of ln1_.gamma
        Tensor fd_ln1 = blk.ln1_.grad_gamma_.clone();  // same shape (1, d_model)
        fd_ln1.fill(0.0);
        double eps = 1e-3;
        Tensor orig_gamma = blk.ln1_.gamma.clone();
        for (size_t j = 0; j < blk.ln1_.gamma.data.size(); ++j) {
            double v = orig_gamma.data[j];
            blk.ln1_.gamma.data[j] = v + eps;
            Tensor yp = blk.forward(x);
            double lp = l2_loss_value(yp, target);
            blk.ln1_.gamma.data[j] = v - eps;
            Tensor ym = blk.forward(x);
            double lm = l2_loss_value(ym, target);
            blk.ln1_.gamma.data[j] = v;
            fd_ln1.data[j] = (lp - lm) / (2.0 * eps);
        }
        double rel_ln1 = max_rel_err_floor(ana_ln1, fd_ln1, 1e-5);
        check(rel_ln1 < 1e-3, "TimesBlock ln1 gamma grad rel_err < 1e-3 vs FD (got " + to_string(rel_ln1) + ")");

        // ---- (b) ffn1 weights gradient: FD against ffn1_.grad_weights ----
        blk.zero_grad();
        blk.forward(x);  // re-cache with restored gamma
        blk.backward(grad_out, 0.0);
        Tensor ana_ffn1 = blk.ffn1_.grad_weights.clone();
        Tensor fd_ffn1 = blk.ffn1_.grad_weights.clone();
        fd_ffn1.fill(0.0);
        Tensor orig_w = blk.ffn1_.weights.clone();
        // Sample a few representative elements (FD over every element of a
        // 4x2 matrix is 8 forwards — cheap).
        for (size_t j = 0; j < orig_w.data.size(); ++j) {
            double v = orig_w.data[j];
            blk.ffn1_.weights.data[j] = v + eps;
            Tensor yp = blk.forward(x);
            double lp = l2_loss_value(yp, target);
            blk.ffn1_.weights.data[j] = v - eps;
            Tensor ym = blk.forward(x);
            double lm = l2_loss_value(ym, target);
            blk.ffn1_.weights.data[j] = v;
            fd_ffn1.data[j] = (lp - lm) / (2.0 * eps);
        }
        double rel_ffn1 = max_rel_err_floor(ana_ffn1, fd_ffn1, 1e-5);
        check(rel_ffn1 < 1e-3, "TimesBlock ffn1 W grad rel_err < 1e-3 vs FD (got " + to_string(rel_ffn1) + ")");
    }

    // -----------------------------------------------------------------------
    // Test 7: TimesNet forward shape (B*T, in_dim) -> (B, pred_len*out_dim)
    // -----------------------------------------------------------------------
    {
        const int in_dim = 3, out_dim = 2, seq_len = 8, pred_len = 4;
        const int d_model = 4, e_layers = 2, top_k = 2;
        TimesNet net(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        Tensor x(2 * seq_len, in_dim);
        mt19937 gen(31);
        fill_random(x, gen, 0.3);
        Tensor y = net.forward(x);
        check(y.rows == 2 && y.cols == (size_t)(pred_len * out_dim),
             "TimesNet forward shape (B*T, in_dim) -> (B, pred_len*out_dim)");
        bool finite = true;
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!isfinite(y.data[i])) { finite = false; break; }
        check(finite, "TimesNet output is finite");
    }

    // -----------------------------------------------------------------------
    // Test 8: TimesNet INPUT GRADIENT vs FD (rel_err < 1e-3)
    // -----------------------------------------------------------------------
    {
        const int in_dim = 2, out_dim = 2, seq_len = 8, pred_len = 4;
        const int d_model = 4, e_layers = 2, top_k = 2;
        TimesNet net(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        net.set_fixed_periods({2, 4});
        Tensor x(seq_len, in_dim);  // B=1 for FD tractability
        mt19937 gen(37);
        fill_random(x, gen, 0.3);
        Tensor target(1, pred_len * out_dim);
        fill_random(target, gen, 0.1);

        Tensor y = net.forward(x);
        Tensor grad_out = l2_loss_grad(y, target);
        net.zero_grad();
        Tensor grad_in_ana = net.backward(grad_out, 0.0);

        Tensor grad_in_fd = finite_diff_grad_input(net, x, target);
        double rel = max_rel_err_floor(grad_in_ana, grad_in_fd, 1e-5);
        check(rel < 1e-3, "TimesNet input grad rel_err < 1e-3 (got " + to_string(rel) + ")");
    }

    // -----------------------------------------------------------------------
    // Test 9: TimesNet training reduces loss on a synthetic sine wave
    // -----------------------------------------------------------------------
    {
        const int in_dim = 1, out_dim = 1, seq_len = 16, pred_len = 4;
        const int d_model = 8, e_layers = 2, top_k = 2;
        TimesNet net(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        net.set_fixed_periods({2, 4, 8});  // periods 2, 4, 8 capture the period-8 sine
        const int B = 4;
        Tensor X((size_t)(B * seq_len), (size_t)in_dim);
        Tensor Y((size_t)B, (size_t)(pred_len * out_dim));
        mt19937 gen(43);
        for (int b = 0; b < B; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                X.data[(b * seq_len + t) * in_dim] = sin(2.0 * M_PI * (b * seq_len + t) / 8.0);
            }
            for (int t = 0; t < pred_len; ++t) {
                Y.data[b * pred_len + t] = sin(2.0 * M_PI * (b * seq_len + seq_len + t) / 8.0);
            }
        }
        const double lr = 0.001;
        Tensor y0 = net.forward(X);
        double l0 = l2_loss_value(y0, Y);
        for (int step = 0; step < 50; ++step) {
            net.zero_grad();
            Tensor yp = net.forward(X);
            Tensor go = l2_loss_grad(yp, Y);
            net.backward(go, lr);
            net.update_weights(lr);
        }
        Tensor yf = net.forward(X);
        double lf = l2_loss_value(yf, Y);
        check(std::isfinite(l0) && std::isfinite(lf),
              "TimesNet training loss is finite (l0=" + to_string(l0) + ", lf=" + to_string(lf) + ")");
        if (std::isfinite(l0) && std::isfinite(lf)) {
            check(lf < l0, "TimesNet training reduces loss (" + to_string(l0) + " -> " + to_string(lf) + ")");
        }
    }

    // -----------------------------------------------------------------------
    // Test 10: TimesNet determinism (two fresh instances with COPIED params)
    //          Forward bit-exact.
    // -----------------------------------------------------------------------
    {
        const int in_dim = 2, out_dim = 2, seq_len = 8, pred_len = 4;
        const int d_model = 4, e_layers = 2, top_k = 2;
        TimesNet net1(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        net1.set_fixed_periods({2, 4});
        Tensor x(seq_len, in_dim);
        mt19937 gen(47);
        fill_random(x, gen, 0.3);
        Tensor y1 = net1.forward(x);

        TimesNet net2(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        net2.set_fixed_periods({2, 4});
        // Copy params from net1 to net2
        auto p1 = net1.parameters();
        auto p2 = net2.parameters();
        check(p1.size() == p2.size(), "two TimesNets have same param count");
        bool sizes_match = true;
        for (size_t i = 0; i < p1.size(); ++i) {
            if (p1[i]->rows != p2[i]->rows || p1[i]->cols != p2[i]->cols) {
                sizes_match = false; break;
            }
        }
        check(sizes_match, "two TimesNets have same param shapes");
        for (size_t i = 0; i < p1.size(); ++i) {
            p2[i]->data = p1[i]->data;  // copy each param's data
        }
        Tensor y2 = net2.forward(x);
        double err = max_abs_err(y1, y2);
        check(err < 1e-12, "TimesNet determinism with copied params (max abs err=" + to_string(err) + ")");
    }

    // -----------------------------------------------------------------------
    // Test 11: Multi-layer (e_layers > 1) forward shape + finite
    // -----------------------------------------------------------------------
    {
        const int in_dim = 2, out_dim = 2, seq_len = 8, pred_len = 4;
        const int d_model = 4, e_layers = 3, top_k = 2;
        TimesNet net(in_dim, out_dim, seq_len, pred_len, d_model, e_layers, top_k);
        Tensor x(2 * seq_len, in_dim);
        mt19937 gen(59);
        fill_random(x, gen, 0.3);
        Tensor y = net.forward(x);
        check(y.rows == 2 && y.cols == (size_t)(pred_len * out_dim),
             "TimesNet (e_layers=3) forward shape");
        bool finite = true;
        for (size_t i = 0; i < y.data.size(); ++i)
            if (!isfinite(y.data[i])) { finite = false; break; }
        check(finite, "TimesNet (e_layers=3) output is finite");
    }

    cout << "=== Summary: " << passed << " passed, " << (total - passed) << " failed ===" << endl;
    return (passed == total) ? 0 : 1;
}
