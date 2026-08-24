// test_sliding_window.cpp — Gradient correctness tests for Sliding Window Attention
//   Mistral 7B (Jiang et al. 2023, https://arxiv.org/abs/2310.06825)
//   + Longformer-style global tokens (Beltagy et al. 2020, https://arxiv.org/abs/2004.05150)
//
// Tests:
//   1. Constructor validation (5 cases)
//   2. Forward shape (n=6, d=8, 2 heads, 1 kv-head GQA)
//   3. Forward shape (single-head, n=4, d=4, MHA mode)
//   4. Output finite + nonzero
//   5. Window mask correctness (causal): row i has nonzero attn only in [i-W+1, i]
//   6. Window mask correctness (non-causal): row i has nonzero attn only in [i-W/2, i+W/2]
//   7. Global tokens: rows 0..num_global attend to entire sequence
//   8. Input gradient FD check
//   9. W_q, W_k, W_v, W_o gradient FD checks
//  10. Multi-head GQA (num_heads=4, num_kv_heads=2) forward + input grad FD
//  11. SlidingWindowBlock forward shape
//  12. SlidingWindowBlock input gradient FD check
//  13. SlidingWindowBlock training reduces loss
//  14. SlidingWindowModel (2 blocks) forward shape
//  15. SlidingWindowModel training reduces loss
//  16. Mutation: zeroing the window mask produces a different output
//  17. Parameters / gradients shape consistency
//  18. Zero gradient doesn't crash
//  19. Determinism — two fresh SWA with copied params → bit-exact forward

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/attention/sliding_window.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static void check(const string& name, bool cond, double err = 0.0) {
    if (cond) {
        ++passed;
        cout << "  [PASS] " << name << "\n";
    } else {
        ++failed;
        cout << "  [FAIL] " << name << "  (err=" << err << ")\n";
    }
}

static double tensor_rel_err(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1.0;
    double num = 0, den = 0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        num = max(num, fabs(a.data[i] - b.data[i]));
        den = max(den, max(fabs(a.data[i]), fabs(b.data[i])));
    }
    if (den < 1e-12) return num;
    return num / den;
}

// ============================================================
// helpers
// ============================================================

static double l2_loss(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - tgt.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i) {
        g.data[i] = out.data[i] - tgt.data[i];
    }
    return g;
}

// Compute FD input gradient
static Tensor fd_input_grad(SlidingWindowAttention& attn,
                            Tensor& input,
                            const Tensor& target,
                            double eps = 1e-5)
{
    Tensor grad(input.rows, input.cols);
    Tensor out0 = attn.forward(input);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            Tensor outp = attn.forward(input);
            double lp = l2_loss(outp, target);
            input[i][j] = orig - eps;
            Tensor outm = attn.forward(input);
            double lm = l2_loss(outm, target);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// Compute FD param gradient (parameter idx: 0=W_q, 1=W_k, 2=W_v, 3=W_o)
static Tensor fd_param_grad(SlidingWindowAttention& attn,
                            size_t pidx,
                            Tensor& input,
                            const Tensor& target,
                            double eps = 1e-5)
{
    Tensor* p = attn.parameters()[pidx];
    Tensor grad(p->rows, p->cols);
    for (size_t i = 0; i < p->rows; ++i) {
        for (size_t j = 0; j < p->cols; ++j) {
            double orig = (*p)[i][j];
            (*p)[i][j] = orig + eps;
            Tensor outp = attn.forward(input);
            double lp = l2_loss(outp, target);
            (*p)[i][j] = orig - eps;
            Tensor outm = attn.forward(input);
            double lm = l2_loss(outm, target);
            (*p)[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// ============================================================
// 1. constructor validation
// ============================================================
static void test_constructor_validation() {
    cout << "\n[Test 1: constructor validation]\n";
    bool threw_d0 = false, threw_h0 = false, threw_w0 = false;
    bool threw_kvgt = false, threw_kv0 = false;
    try { SlidingWindowAttention attn(0, 2, 2, 4); }
    catch (const std::invalid_argument&) { threw_d0 = true; }
    try { SlidingWindowAttention attn(8, 0, 0, 4); }
    catch (const std::invalid_argument&) { threw_h0 = true; }
    try { SlidingWindowAttention attn(8, 2, 1, 0); }
    catch (const std::invalid_argument&) { threw_w0 = true; }
    try { SlidingWindowAttention attn(8, 2, 4, 4); }   // num_kv > num_query
    catch (const std::invalid_argument&) { threw_kvgt = true; }
    try { SlidingWindowAttention attn(8, 2, 0, 4); }
    catch (const std::invalid_argument&) { threw_kv0 = true; }
    check("d_model=0 throws", threw_d0);
    check("num_heads=0 throws", threw_h0);
    check("window=0 throws", threw_w0);
    check("num_kv>num_query throws", threw_kvgt);
    check("num_kv=0 throws", threw_kv0);
}

// ============================================================
// 2. forward shape (GQA mode)
// ============================================================
static void test_forward_shape_gqa() {
    cout << "\n[Test 2: forward shape GQA mode (n=6,d=8,h=2,kv=1)]\n";
    const size_t n = 6, d = 8;
    SlidingWindowAttention attn(d, 2, 1, 4);  // GQA: 2 query heads, 1 kv head
    Tensor input(n, d);
    std::mt19937 gen(0);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    cout << "  in " << input.rows << "x" << input.cols
         << " -> out " << out.rows << "x" << out.cols << "\n";
    check("forward shape correct", out.rows == n && out.cols == d);
}

// ============================================================
// 3. forward shape (MHA mode)
// ============================================================
static void test_forward_shape_mha() {
    cout << "\n[Test 3: forward shape MHA mode (n=4,d=4,h=1)]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 2);
    Tensor input(n, d);
    std::mt19937 gen(1);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    check("forward shape correct", out.rows == n && out.cols == d);
}

// ============================================================
// 4. output finite + nonzero
// ============================================================
static void test_output_finite_nonzero() {
    cout << "\n[Test 4: output finite + nonzero]\n";
    const size_t n = 6, d = 8;
    SlidingWindowAttention attn(d, 2, 1, 4);
    Tensor input(n, d);
    std::mt19937 gen(2);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    bool finite = true, has_nonzero = false;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (std::isnan(out.data[i]) || std::isinf(out.data[i])) { finite = false; break; }
        if (fabs(out.data[i]) > 1e-8) has_nonzero = true;
    }
    check("output finite",   finite);
    check("output nonzero",  has_nonzero);
}

// ============================================================
// 5. window mask correctness (causal)
// ============================================================
static void test_window_mask_causal() {
    cout << "\n[Test 5: window mask correctness (causal, W=2)]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 2, 0, /*causal=*/true);
    Tensor input(n, d);
    std::mt19937 gen(3);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    // attn head 0: (n, n) cache
    const Tensor& A = attn.last_attn_head(0);
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Causal W=2: in-window if j <= i (window includes current) AND j >= i - W + 1
            // = j in [max(0, i-W+1), i] (causal).
            int lo = (int)max((size_t)0, i - 1);     // i - W + 1 = i - 1
            int hi = (int)i;
            bool in_window = ((int)j >= lo) && ((int)j <= hi);
            bool nonzero = A[i][j] > 1e-9;
            if (in_window != nonzero) { ok = false; }
        }
    }
    check("causal W=2 mask: nonzero iff j in [i-1, i]", ok);
}

// ============================================================
// 6. window mask correctness (non-causal)
// ============================================================
static void test_window_mask_noncausal() {
    cout << "\n[Test 6: window mask correctness (non-causal, W=4)]\n";
    const size_t n = 6, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 4, 0, /*causal=*/false);
    Tensor input(n, d);
    std::mt19937 gen(4);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    const Tensor& A = attn.last_attn_head(0);
    bool ok = true;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Non-causal W=4: |i-j| <= W/2 = 2
            bool in_window = (int)abs((int)i - (int)j) <= 2;
            bool nonzero = A[i][j] > 1e-9;
            if (in_window != nonzero) { ok = false; }
        }
    }
    check("non-causal W=4 mask: nonzero iff |i-j| <= 2", ok);
}

// ============================================================
// 7. global tokens
// ============================================================
static void test_global_tokens() {
    cout << "\n[Test 7: global tokens (num_global=1, W=2)]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 2, /*num_global=*/1);
    Tensor input(n, d);
    std::mt19937 gen(5);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    const Tensor& A = attn.last_attn_head(0);
    // row 0 (global): every column should have nonzero attention
    bool row0_full = true;
    for (size_t j = 0; j < n; ++j) if (A[0][j] < 1e-9) { row0_full = false; break; }
    check("global row 0: nonzero attention everywhere", row0_full);
    // column 0: every row should have nonzero attention (key slot 0 is global)
    bool col0_full = true;
    for (size_t i = 0; i < n; ++i) if (A[i][0] < 1e-9) { col0_full = false; break; }
    check("global col 0: nonzero attention everywhere", col0_full);
    // row 2 (non-global, W=2 causal): window is j in [1, 2]. col 0 IS global,
    // so j=0 is allowed (the global-key effect from Longformer). j=3 is outside.
    bool row2_windowed = (A[2][0] > 1e-9) && (A[2][1] > 1e-9) && (A[2][2] > 1e-9) && (A[2][3] < 1e-9);
    check("row 2 still respects window (except for global col 0)", row2_windowed);
}

// ============================================================
// 8. input gradient FD check
// ============================================================
static void test_input_grad_fd() {
    cout << "\n[Test 8: input gradient FD check]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 2);
    Tensor input(n, d);
    std::mt19937 gen(6);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

    Tensor fd = fd_input_grad(attn, input, target, 1e-5);
    Tensor ana = l2_grad(attn.forward(input), target);
    // analytical backward
    attn.zero_grad();
    Tensor out = attn.forward(input);
    Tensor gout = l2_grad(out, target);
    attn.backward(gout, 0.0);
    Tensor ana_input_grad = attn.grad_input();

    double e1 = tensor_rel_err(fd, ana_input_grad);
    cout << "  fd vs ana input-grad rel_err = " << scientific << setprecision(2) << e1 << "\n";
    check("input gradient FD rel_err < 1e-5", e1 < 1e-5, e1);
}

// ============================================================
// 9. W_q, W_k, W_v, W_o gradient FD checks
// ============================================================
static void test_w_grad_fd() {
    cout << "\n[Test 9: W_q, W_k, W_v, W_o gradient FD checks]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 1, 1, 2);
    Tensor input(n, d);
    std::mt19937 gen(7);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

    for (size_t pidx = 0; pidx < 4; ++pidx) {
        Tensor fd = fd_param_grad(attn, pidx, input, target, 1e-5);
        attn.zero_grad();
        Tensor out = attn.forward(input);
        Tensor gout = l2_grad(out, target);
        attn.backward(gout, 0.0);
        Tensor* ana_g = attn.gradients()[pidx];

        double e = tensor_rel_err(fd, *ana_g);
        const char* nm[4] = {"W_q", "W_k", "W_v", "W_o"};
        cout << "  " << nm[pidx] << " fd vs ana rel_err = " << scientific << setprecision(2) << e << "\n";
        check(string(nm[pidx]) + " gradient FD rel_err < 1e-5", e < 1e-5, e);
    }
}

// ============================================================
// 10. multi-head GQA
// ============================================================
static void test_multi_head_gqa() {
    cout << "\n[Test 10: multi-head GQA (num_heads=4, num_kv=2)]\n";
    const size_t n = 5, d = 8;
    SlidingWindowAttention attn(d, 4, 2, 3);
    Tensor input(n, d);
    std::mt19937 gen(8);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    check("multi-head forward shape", out.rows == n && out.cols == d);

    // input grad FD
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;
    Tensor fd = fd_input_grad(attn, input, target, 1e-5);
    attn.zero_grad();
    Tensor out2 = attn.forward(input);
    attn.backward(l2_grad(out2, target), 0.0);
    Tensor ana = attn.grad_input();
    double e = tensor_rel_err(fd, ana);
    cout << "  multi-head input-grad rel_err = " << scientific << setprecision(2) << e << "\n";
    check("multi-head input grad FD rel_err < 1e-5", e < 1e-5, e);
}

// ============================================================
// 11. block forward shape
// ============================================================
static void test_block_forward_shape() {
    cout << "\n[Test 11: block forward shape]\n";
    const size_t n = 4, d = 4;
    SlidingWindowBlock block(d, 2, 1, 2, 0, true, 0);  // ffn_dim=0 → no FFN sub-layer
    Tensor input(n, d);
    std::mt19937 gen(9);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = block.forward(input);
    check("block forward shape", out.rows == n && out.cols == d);
}

// ============================================================
// 12. block input gradient FD
// ============================================================
static void test_block_input_grad_fd() {
    cout << "\n[Test 12: block input gradient FD check]\n";
    const size_t n = 4, d = 4;
    SlidingWindowBlock block(d, 2, 1, 2, 0, true, 0);
    Tensor input(n, d);
    std::mt19937 gen(10);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = -0.1;

    // FD
    Tensor fd(n, d);
    Tensor saved = input.clone();
    double eps = 1e-5;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            Tensor outp = block.forward(input);
            double lp = l2_loss(outp, target);
            input[i][j] = orig - eps;
            Tensor outm = block.forward(input);
            double lm = l2_loss(outm, target);
            input[i][j] = orig;
            fd[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    // analytical
    block.zero_grad();
    Tensor out = block.forward(input);
    block.backward(l2_grad(out, target), 0.0);
    Tensor ana = block.grad_input();
    double e = tensor_rel_err(fd, ana);
    cout << "  block input-grad rel_err = " << scientific << setprecision(2) << e << "\n";
    check("block input grad FD rel_err < 1e-5", e < 1e-5, e);
}

// ============================================================
// 13. block training reduces loss
// ============================================================
static void test_block_training() {
    cout << "\n[Test 13: block training reduces loss]\n";
    const size_t n = 4, d = 4;
    SlidingWindowBlock block(d, 2, 1, 2, 0, true, 8);   // ffn_dim=8 → full block
    Tensor input(n, d), target(n, d);
    std::mt19937 gen(11);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

    double lr = 0.05;
    Tensor out = block.forward(input);
    double l0 = l2_loss(out, target);
    for (size_t step = 0; step < 30; ++step) {
        Tensor out = block.forward(input);
        block.backward(l2_grad(out, target), 0.0);
        block.update_weights(lr);
    }
    out = block.forward(input);
    double l1 = l2_loss(out, target);
    cout << "  loss " << fixed << setprecision(4) << l0 << " -> " << l1 << "\n";
    check("block training reduces loss", l1 < l0, l1 - l0);
}

// ============================================================
// 14. model forward shape
// ============================================================
static void test_model_forward_shape() {
    cout << "\n[Test 14: 2-block model forward shape]\n";
    const size_t n = 4, d = 4, in_dim = 3, out_dim = 2;
    SlidingWindowModel model(in_dim, d, out_dim, 2, 1, 1, 2);
    Tensor input(n, in_dim);
    std::mt19937 gen(12);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = model.forward(input);
    check("model forward shape", out.rows == n && out.cols == out_dim);
}

// ============================================================
// 15. model training reduces loss
// ============================================================
static void test_model_training() {
    cout << "\n[Test 15: model training reduces loss]\n";
    const size_t n = 4, d = 4, in_dim = 3, out_dim = 2;
    SlidingWindowModel model(in_dim, d, out_dim, 2, 1, 1, 2);
    Tensor input(n, in_dim), target(n, out_dim);
    std::mt19937 gen(13);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.5;

    double lr = 0.05;
    Tensor out = model.forward(input);
    double l0 = l2_loss(out, target);
    for (size_t step = 0; step < 50; ++step) {
        Tensor out = model.forward(input);
        model.backward(l2_grad(out, target), 0.0);
        model.update_weights(lr);
    }
    out = model.forward(input);
    double l1 = l2_loss(out, target);
    cout << "  loss " << fixed << setprecision(4) << l0 << " -> " << l1 << "\n";
    check("model training reduces loss", l1 < l0, l1 - l0);
}

// ============================================================
// 16. mutation test — zeroing the window mask changes output
// ============================================================
static void test_mutation_mask() {
    cout << "\n[Test 16: mutation — zeroing window mask changes output]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn1(d, 1, 1, 2);
    SlidingWindowAttention attn2(d, 1, 1, 2);
    // Copy weights from attn1 to attn2
    attn2.W_q = attn1.W_q;
    attn2.W_k = attn1.W_k;
    attn2.W_v = attn1.W_v;
    attn2.W_o = attn1.W_o;

    Tensor input(n, d);
    std::mt19937 gen(14);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);

    Tensor out1 = attn1.forward(input);
    // Disable mask
    attn2.set_use_window_mask(false);
    // Copy weights from attn1 to attn2 (same pattern as test_determinism below).
    attn2.W_q = attn1.W_q;
    attn2.W_k = attn1.W_k;
    attn2.W_v = attn1.W_v;
    attn2.W_o = attn1.W_o;
    Tensor out2 = attn2.forward(input);

    double max_diff = 0;
    for (size_t i = 0; i < out1.data.size(); ++i)
        max_diff = max(max_diff, fabs(out1.data[i] - out2.data[i]));
    cout << "  max diff = " << scientific << setprecision(2) << max_diff << "\n";
    check("mask=off produces different output", max_diff > 1e-5, max_diff);
}

// ============================================================
// 17. parameters / gradients shape consistency
// ============================================================
static void test_param_grad_shapes() {
    cout << "\n[Test 17: parameters / gradients shape consistency]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 2, 1, 2);
    Tensor input(n, d);
    std::mt19937 gen(15);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.1;

    attn.zero_grad();
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    check("4 parameters", params.size() == 4);
    check("4 gradients",  grads.size()  == 4);
    for (size_t i = 0; i < 4; ++i) {
        check("param[" + to_string(i) + "] shape == grad shape",
              params[i]->rows == grads[i]->rows && params[i]->cols == grads[i]->cols);
    }
}

// ============================================================
// 18. zero gradient doesn't crash
// ============================================================
static void test_zero_grad_no_crash() {
    cout << "\n[Test 18: zero gradient doesn't crash]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn(d, 2, 1, 2);
    Tensor input(n, d);
    std::mt19937 gen(16);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = 0.0;  // zero target

    attn.zero_grad();
    Tensor out = attn.forward(input);
    Tensor gout = l2_grad(out, target);
    attn.backward(gout, 0.0);
    check("zero-gradient backward ran without crash", true);
}

// ============================================================
// 19. determinism — two fresh SWA with copied params → bit-exact forward
// ============================================================
static void test_determinism() {
    cout << "\n[Test 19: determinism]\n";
    const size_t n = 4, d = 4;
    SlidingWindowAttention attn1(d, 1, 1, 2);
    SlidingWindowAttention attn2(d, 1, 1, 2);
    attn2.W_q = attn1.W_q;
    attn2.W_k = attn1.W_k;
    attn2.W_v = attn1.W_v;
    attn2.W_o = attn1.W_o;

    Tensor input(n, d);
    std::mt19937 gen(17);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);

    Tensor out1 = attn1.forward(input);
    Tensor out2 = attn2.forward(input);
    double max_diff = 0;
    for (size_t i = 0; i < out1.data.size(); ++i)
        max_diff = max(max_diff, fabs(out1.data[i] - out2.data[i]));
    check("two fresh SWA with copied params → bit-exact forward", max_diff < 1e-12, max_diff);
}

// ============================================================
// main
// ============================================================
int main() {
    cout << "=== Sliding Window Attention Tests ===\n";
    test_constructor_validation();
    test_forward_shape_gqa();
    test_forward_shape_mha();
    test_output_finite_nonzero();
    test_window_mask_causal();
    test_window_mask_noncausal();
    test_global_tokens();
    test_input_grad_fd();
    test_w_grad_fd();
    test_multi_head_gqa();
    test_block_forward_shape();
    test_block_input_grad_fd();
    test_block_training();
    test_model_forward_shape();
    test_model_training();
    test_mutation_mask();
    test_param_grad_shapes();
    test_zero_grad_no_crash();
    test_determinism();

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
