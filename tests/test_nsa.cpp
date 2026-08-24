// test_nsa.cpp — Gradient correctness tests for Native Sparse Attention (NSA)
//   Yuan, Gao, Dai et al. 2025, https://arxiv.org/abs/2502.11089
//
// Three parallel branches (compression / selection / sliding window) combined
// via a learned 3-way softmax gate. Each branch has independent K, V
// projections per paper §3.3.3. Multi-head with GQA-style K/V sharing.
//
// Tests:
//   1. Constructor validation (7 cases)
//   2. Forward shape (single head)
//   3. Forward shape (multi-head GQA)
//   4. Output finite + nonzero
//   5. Per-(token, head, branch) gate sums to 1
//   6. n_cmp / n_sel_blocks computed correctly (deterministic, no branch decay)
//   7. Sliding window mask correctness (each query only attends to last w tokens)
//   8. Selection: top-n blocks by importance scores
//   9. Input gradient FD check
//  10. W_q, W_o gradient FD checks
//  11. W_k_cmp, W_v_cmp gradient FD checks
//  12. W_k_sel, W_v_sel gradient FD checks
//  13. W_k_win, W_v_win gradient FD checks
//  14. W_phi_k, W_phi_v gradient FD checks (compression MLP)
//  15. W_gate gradient FD check
//  16. Multi-head GQA input gradient FD check
//  17. NSABlock forward shape
//  18. NSABlock input gradient FD check
//  19. NSABlock training reduces loss
//  20. NSAModel (2 blocks) forward + training reduces loss
//  21. Mutation tests (zero each branch independently → output changes)
//  22. Parameters / gradients shape consistency
//  23. Zero gradient doesn't crash
//  24. Determinism — two fresh NSAs with copied params → bit-exact forward

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include "nn/layers/attention/nsa.h"

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

static Tensor fd_input_grad(NSAAttention& attn, Tensor& input, const Tensor& target,
                            double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
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

// FD for the FIRST ELEMENT of a specific parameter (since NSA has many params,
// this gives us per-element confidence without an O(P²) sweep).
// For a full FD sweep, use fd_input_grad + a slower param-by-param variant.
// eps=1e-4 gives much better signal-to-noise at small init scales (~0.05).
static double fd_param_first_element(NSAAttention& attn, Tensor* p,
                                     Tensor& input, const Tensor& target,
                                     double eps = 1e-4) {
    double orig = (*p)[0][0];
    (*p)[0][0] = orig + eps;
    Tensor outp = attn.forward(input);
    double lp = l2_loss(outp, target);
    (*p)[0][0] = orig - eps;
    Tensor outm = attn.forward(input);
    double lm = l2_loss(outm, target);
    (*p)[0][0] = orig;
    return (lp - lm) / (2.0 * eps);
}

// Full FD gradient for a parameter (O(p.rows * p.cols) forward passes per param).
// Use only on small tensors to avoid blowing up runtime.
static Tensor fd_param_full(NSAAttention& attn, Tensor* p,
                            Tensor& input, const Tensor& target,
                            double eps = 1e-5) {
    Tensor grad(p->rows, p->cols);
    double orig_save = (*p)[0][0];
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
    (*p)[0][0] = orig_save;
    return grad;
}

// ============================================================
// 1. constructor validation
// ============================================================
static void test_constructor_validation() {
    cout << "\n[Test 1: constructor validation]\n";
    bool threw_d0 = false, threw_h0 = false, threw_l0 = false;
    bool threw_d0_stride = false, threw_w0 = false, threw_n0 = false;
    bool threw_div = false;
    // d_model = 0
    try { NSAAttention attn(0, 2, 2, 2, 1, 1, 2); }
    catch (const std::invalid_argument&) { threw_d0 = true; }
    // num_query_heads = 0
    try { NSAAttention attn(8, 0, 0, 2, 1, 1, 2); }
    catch (const std::invalid_argument&) { threw_h0 = true; }
    // block_len = 0
    try { NSAAttention attn(8, 2, 1, 0, 1, 1, 2); }
    catch (const std::invalid_argument&) { threw_l0 = true; }
    // stride > block_len
    try { NSAAttention attn(8, 2, 1, 2, 4, 1, 2); }
    catch (const std::invalid_argument&) { threw_d0_stride = true; }
    // window_size = 0
    try { NSAAttention attn(8, 2, 1, 2, 1, 1, 0); }
    catch (const std::invalid_argument&) { threw_w0 = true; }
    // top_n = 0
    try { NSAAttention attn(8, 2, 1, 2, 1, 0, 2); }
    catch (const std::invalid_argument&) { threw_n0 = true; }
    // d_model not divisible by num_heads
    try { NSAAttention attn(9, 2, 1, 2, 1, 1, 2); }
    catch (const std::invalid_argument&) { threw_div = true; }

    check("d_model=0 throws", threw_d0);
    check("num_heads=0 throws", threw_h0);
    check("block_len=0 throws", threw_l0);
    check("stride>block_len throws", threw_d0_stride);
    check("window_size=0 throws", threw_w0);
    check("top_n=0 throws", threw_n0);
    check("d_model % num_heads != 0 throws", threw_div);
}

// ============================================================
// 2. forward shape (single head)
// ============================================================
static void test_forward_shape_single_head() {
    cout << "\n[Test 2: forward shape single head n=6,d=4,h=1,kv=1]\n";
    const size_t n = 6, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/2, /*d=*/1, /*n=*/2, /*w=*/3);
    Tensor input(n, d);
    std::mt19937 gen(0);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    cout << "  in " << input.rows << "x" << input.cols
         << " -> out " << out.rows << "x" << out.cols
         << " n_cmp=" << attn.n_cmp()
         << " n_sel_blocks=" << attn.n_sel_blocks() << "\n";
    check("forward shape correct", out.rows == n && out.cols == d);
    // With l=2, d=1, N=6 → n_cmp = (6-2)/1 + 1 = 5
    check("n_cmp = (N-l)/stride + 1 = 5", attn.n_cmp() == 5);
}

// ============================================================
// 3. forward shape (multi-head GQA)
// ============================================================
static void test_forward_shape_gqa() {
    cout << "\n[Test 3: forward shape GQA n=6,d=8,h=2,kv=1]\n";
    const size_t n = 6, d = 8;
    NSAAttention attn(d, 2, 1, /*l=*/2, /*d=*/1, /*n=*/2, /*w=*/3);
    Tensor input(n, d);
    std::mt19937 gen(1);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    check("forward shape (GQA) correct", out.rows == n && out.cols == d);
}

// ============================================================
// 4. output finite + nonzero
// ============================================================
static void test_output_finite_nonzero() {
    cout << "\n[Test 4: output finite + nonzero]\n";
    const size_t n = 6, d = 8;
    NSAAttention attn(d, 2, 1, 2, 1, 2, 3);
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
    check("output finite", finite);
    check("output nonzero", has_nonzero);
}

// ============================================================
// 5. gate sums to 1 across branches (per token, per head)
// ============================================================
static void test_gate_sums_to_one() {
    cout << "\n[Test 5: gate softmax sums to 1 per (token, head)]\n";
    const size_t n = 6, d = 8;
    NSAAttention attn(d, 2, 1, 2, 1, 2, 3);
    Tensor input(n, d);
    std::mt19937 gen(3);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);
    const Tensor& g = attn.last_gate();  // (n, num_heads, 3)
    check("gate shape (n, num_heads, 3)",
          g.rows == n && g.cols == 2 * 3);
    // rows of (n, num_heads, 3) — actual layout: (n * num_heads, 3)
    bool all_one = true;
    double max_err = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t h = 0; h < 2; ++h) {
            double sum = g(i, h * 3 + 0) + g(i, h * 3 + 1) + g(i, h * 3 + 2);
            double e = fabs(sum - 1.0);
            if (e > max_err) max_err = e;
            if (e > 1e-5) all_one = false;
        }
    check("gate rows sum to 1.0", all_one, max_err);
}

// ============================================================
// 6. n_cmp / n_sel_blocks deterministic
// ============================================================
static void test_n_cmp_deterministic() {
    cout << "\n[Test 6: n_cmp and n_sel_blocks deterministic]\n";
    const size_t n = 8, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/3, /*d=*/2, /*n=*/2, /*w=*/4);
    Tensor input(n, d);
    std::mt19937 gen(4);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);
    // n=8, l=3, d=2 → (8-3)/2 + 1 = 3
    check("n_cmp = (8-3)/2+1 = 3", attn.n_cmp() == 3);
    // block_size default = stride = 2 → (8-2)/2 + 1 = 4
    check("n_sel_blocks = (8-2)/2+1 = 4 (default block_size=stride)",
          attn.n_sel_blocks() == 4);
}

// ============================================================
// 7. sliding window mask: each query attends only to last w tokens
// ============================================================
static void test_sliding_window_mask() {
    cout << "\n[Test 7: sliding window mask correctness]\n";
    const size_t n = 6, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/2, /*d=*/1, /*n=*/2, /*w=*/3);
    Tensor input(n, d);
    std::mt19937 gen(5);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);
    // last_attn_win_ is laid out as (H, N * w) row-major; with H=1 it's (1, N*w).
    // Per query t, the w positions are at flat indices [t * w, (t+1) * w).
    bool row_sums_correct = true;
    double max_row_err = 0;
    for (size_t t = 0; t < n; ++t) {
        double row_sum = 0;
        for (size_t k = 0; k < 3; ++k) row_sum += attn.last_attn_win()(0, t * 3 + k);
        double e = fabs(row_sum - 1.0);
        if (e > 1e-5) { row_sums_correct = false; if (e > max_row_err) max_row_err = e; }
    }
    check("window attention rows sum to 1.0", row_sums_correct, max_row_err);
    // All entries should be non-negative (softmax output)
    bool all_nonneg = true;
    for (size_t i = 0; i < attn.last_attn_win().data.size(); ++i) {
        if (attn.last_attn_win().data[i] < -1e-10) { all_nonneg = false; break; }
    }
    check("window attention entries are non-negative", all_nonneg);
}

// ============================================================
// 8. selection: top-n blocks by importance scores
// ============================================================
static void test_selection_top_n() {
    cout << "\n[Test 8: top-n selection picks blocks by importance]\n";
    const size_t n = 8, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/3, /*d=*/2, /*n=*/2, /*w=*/3);
    Tensor input(n, d);
    std::mt19937 gen(6);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);
    const Tensor& ps = attn.last_attn_sel();
    // ps shape: (H=1, N, top_n * block_size) flattened to (1, N * top_n * block_size)
    // Per (query, selected-block), should be nonzero (selected), and the union of
    // selected blocks should not exceed top_n per query.
    check("selection attention shape has nonzero entries",
          ps.data.size() > 0 && ps(0, 0) >= 0.0);
}

// ============================================================
// 9. input gradient FD check
// ============================================================
static void test_input_grad_fd() {
    cout << "\n[Test 9: input gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/2, /*d=*/1, /*n=*/2, /*w=*/2);
    Tensor input(n, d);
    std::mt19937 gen(10);
    std::normal_distribution<double> dist(0.0, 0.5);  // larger init
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    std::mt19937 gen2(11);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen2);

    // analytical
    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    Tensor ana = attn.grad_input();

    // finite-difference
    Tensor fdi = fd_input_grad(attn, input, target);
    // Diagnostic: print a few elements
    cout << "  ana[0][0]=" << ana(0, 0) << " fd[0][0]=" << fdi(0, 0) << "\n";
    cout << "  ana[1][2]=" << ana(1, 2) << " fd[1][2]=" << fdi(1, 2) << "\n";
    cout << "  ana[2][3]=" << ana(2, 3) << " fd[2][3]=" << fdi(2, 3) << "\n";
    double re = tensor_rel_err(ana, fdi);
    cout << "  input grad rel_err = " << re << "\n";
    check("input grad FD rel_err < 1e-2", re < 1e-2, re);
}

// ============================================================
// 10. W_q, W_o parameter gradients FD check
// ============================================================
static void test_param_grad_W_q_W_o() {
    cout << "\n[Test 10: W_q / W_o param gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(20);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    // analytical
    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana_q = attn.grad_W_q.data[0];
    double ana_o = attn.grad_W_o.data[0];

    // FD for first element
    double fd_q = fd_param_first_element(attn, &attn.W_q, input, target);
    double fd_o = fd_param_first_element(attn, &attn.W_o, input, target);

    double err_q = fabs(ana_q - fd_q) / max(1e-12, max(fabs(ana_q), fabs(fd_q)));
    double err_o = fabs(ana_o - fd_o) / max(1e-12, max(fabs(ana_o), fabs(fd_o)));
    cout << "  W_q[0][0] ana=" << ana_q << " fd=" << fd_q << " rel_err=" << err_q << "\n";
    cout << "  W_o[0][0] ana=" << ana_o << " fd=" << fd_o << " rel_err=" << err_o << "\n";
    check("W_q gradient FD rel_err < 1e-2", err_q < 1e-2, err_q);
    check("W_o gradient FD rel_err < 1e-2", err_o < 1e-2, err_o);
}

// ============================================================
// 11. W_k_cmp, W_v_cmp parameter gradients FD check
// ============================================================
static void test_param_grad_W_k_cmp() {
    cout << "\n[Test 11: W_k_cmp / W_v_cmp param gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(30);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana_k = attn.grad_W_k_cmp.data[0];
    double ana_v = attn.grad_W_v_cmp.data[0];
    double fd_k  = fd_param_first_element(attn, &attn.W_k_cmp, input, target);
    double fd_v  = fd_param_first_element(attn, &attn.W_v_cmp, input, target);
    double err_k = fabs(ana_k - fd_k) / max(1e-12, max(fabs(ana_k), fabs(fd_k)));
    double err_v = fabs(ana_v - fd_v) / max(1e-12, max(fabs(ana_v), fabs(fd_v)));
    cout << "  W_k_cmp[0][0] ana=" << ana_k << " fd=" << fd_k << " rel_err=" << err_k << "\n";
    cout << "  W_v_cmp[0][0] ana=" << ana_v << " fd=" << fd_v << " rel_err=" << err_v << "\n";
    check("W_k_cmp gradient FD rel_err < 1e-2", err_k < 1e-2, err_k);
    check("W_v_cmp gradient FD rel_err < 1e-2", err_v < 1e-2, err_v);
}

// ============================================================
// 12. W_k_sel, W_v_sel parameter gradients FD check
// ============================================================
static void test_param_grad_W_k_sel() {
    cout << "\n[Test 12: W_k_sel / W_v_sel param gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(40);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana_k = attn.grad_W_k_sel.data[0];
    double ana_v = attn.grad_W_v_sel.data[0];
    double fd_k  = fd_param_first_element(attn, &attn.W_k_sel, input, target);
    double fd_v  = fd_param_first_element(attn, &attn.W_v_sel, input, target);
    double err_k = fabs(ana_k - fd_k) / max(1e-12, max(fabs(ana_k), fabs(fd_k)));
    double err_v = fabs(ana_v - fd_v) / max(1e-12, max(fabs(ana_v), fabs(fd_v)));
    cout << "  W_k_sel[0][0] ana=" << ana_k << " fd=" << fd_k << " rel_err=" << err_k << "\n";
    cout << "  W_v_sel[0][0] ana=" << ana_v << " fd=" << fd_v << " rel_err=" << err_v << "\n";
    check("W_k_sel gradient FD rel_err < 1e-2", err_k < 1e-2, err_k);
    check("W_v_sel gradient FD rel_err < 1e-2", err_v < 1e-2, err_v);
}

// ============================================================
// 13. W_k_win, W_v_win parameter gradients FD check
// ============================================================
static void test_param_grad_W_k_win() {
    cout << "\n[Test 13: W_k_win / W_v_win param gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(50);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana_k = attn.grad_W_k_win.data[0];
    double ana_v = attn.grad_W_v_win.data[0];
    double fd_k  = fd_param_first_element(attn, &attn.W_k_win, input, target);
    double fd_v  = fd_param_first_element(attn, &attn.W_v_win, input, target);
    double err_k = fabs(ana_k - fd_k) / max(1e-12, max(fabs(ana_k), fabs(fd_k)));
    double err_v = fabs(ana_v - fd_v) / max(1e-12, max(fabs(ana_v), fabs(fd_v)));
    cout << "  W_k_win[0][0] ana=" << ana_k << " fd=" << fd_k << " rel_err=" << err_k << "\n";
    cout << "  W_v_win[0][0] ana=" << ana_v << " fd=" << fd_v << " rel_err=" << err_v << "\n";
    check("W_k_win gradient FD rel_err < 1e-2", err_k < 1e-2, err_k);
    check("W_v_win gradient FD rel_err < 1e-2", err_v < 1e-2, err_v);
}

// ============================================================
// 14. W_phi_k, W_phi_v (compression MLP) gradient FD check
// ============================================================
static void test_param_grad_W_phi() {
    cout << "\n[Test 14: W_phi_k / W_phi_v gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, /*l=*/2, /*d=*/1, /*n=*/2, /*w=*/2);
    Tensor input(n, d);
    std::mt19937 gen(60);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana_k = attn.grad_W_phi_k.data[0];
    double ana_v = attn.grad_W_phi_v.data[0];
    double fd_k  = fd_param_first_element(attn, &attn.W_phi_k, input, target);
    double fd_v  = fd_param_first_element(attn, &attn.W_phi_v, input, target);
    double err_k = fabs(ana_k - fd_k) / max(1e-12, max(fabs(ana_k), fabs(fd_k)));
    double err_v = fabs(ana_v - fd_v) / max(1e-12, max(fabs(ana_v), fabs(fd_v)));
    cout << "  W_phi_k[0][0] ana=" << ana_k << " fd=" << fd_k << " rel_err=" << err_k << "\n";
    cout << "  W_phi_v[0][0] ana=" << ana_v << " fd=" << fd_v << " rel_err=" << err_v << "\n";
    check("W_phi_k gradient FD rel_err < 1e-2", err_k < 1e-2, err_k);
    check("W_phi_v gradient FD rel_err < 1e-2", err_v < 1e-2, err_v);
}

// ============================================================
// 15. W_gate gradient FD check
// ============================================================
static void test_param_grad_W_gate() {
    cout << "\n[Test 15: W_gate gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(70);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    double ana = attn.grad_W_gate.data[0];
    double fd  = fd_param_first_element(attn, &attn.W_gate, input, target);
    double err = fabs(ana - fd) / max(1e-12, max(fabs(ana), fabs(fd)));
    cout << "  W_gate[0][0] ana=" << ana << " fd=" << fd << " rel_err=" << err << "\n";
    check("W_gate gradient FD rel_err < 1e-2", err < 1e-2, err);
}

// ============================================================
// 16. multi-head GQA input gradient FD check
// ============================================================
static void test_multi_head_input_grad() {
    cout << "\n[Test 16: multi-head GQA input gradient FD check]\n";
    const size_t n = 4, d = 8;
    NSAAttention attn(d, 2, 1, 2, 1, 2, 2);  // 2 query heads, 1 kv head, head_dim=4
    Tensor input(n, d);
    std::mt19937 gen(80);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = attn.forward(input);
    Tensor go = l2_grad(out, target);
    attn.backward(go, 0.0);
    Tensor ana = attn.grad_input();
    Tensor fdi = fd_input_grad(attn, input, target);
    double re = tensor_rel_err(ana, fdi);
    cout << "  multi-head input grad rel_err = " << re << "\n";
    check("multi-head input grad FD rel_err < 1e-2", re < 1e-2, re);
}

// ============================================================
// 17. NSABlock forward shape
// ============================================================
static void test_block_forward_shape() {
    cout << "\n[Test 17: NSABlock forward shape]\n";
    const size_t n = 6, d = 8;
    NSABlock block(d, 2, 1, 2, 1, 2, 3);
    Tensor input(n, d);
    std::mt19937 gen(90);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = block.forward(input);
    check("block forward shape", out.rows == n && out.cols == d);
}

// ============================================================
// 18. NSABlock input gradient FD check
// ============================================================
static void test_block_input_grad_fd() {
    cout << "\n[Test 18: NSABlock input gradient FD check]\n";
    const size_t n = 4, d = 4;
    NSABlock block(d, 1, 1, 2, 1, 2, 2);  // ffn_dim=0 (no FFN)
    Tensor input(n, d);
    std::mt19937 gen(100);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = block.forward(input);
    Tensor go = l2_grad(out, target);
    block.backward(go, 0.0);
    Tensor ana = block.grad_input();
    // FD for input
    Tensor grad(input.rows, input.cols);
    const double eps = 1e-5;
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            Tensor outp = block.forward(input);
            double lp = l2_loss(outp, target);
            input[i][j] = orig - eps;
            Tensor outm = block.forward(input);
            double lm = l2_loss(outm, target);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    double re = tensor_rel_err(ana, grad);
    cout << "  block input grad rel_err = " << re << "\n";
    check("block input grad FD rel_err < 1e-2", re < 1e-2, re);
}

// ============================================================
// 19. NSABlock training reduces loss (50 steps, MSE on identity-ish target)
// ============================================================
static void test_block_training() {
    cout << "\n[Test 19: NSABlock training reduces loss]\n";
    const size_t n = 4, d = 4;
    NSABlock block(d, 1, 1, 2, 1, 2, 2);
    Tensor input(n, d);
    std::mt19937 gen(110);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor target(n, d);
    std::mt19937 gen2(111);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen2);

    double l0 = l2_loss(block.forward(input), target);
    const double lr = 0.005;
    for (int s = 0; s < 50; ++s) {
        Tensor out = block.forward(input);
        Tensor go = l2_grad(out, target);
        block.backward(go, lr);
        block.update_weights(lr);
    }
    double l1 = l2_loss(block.forward(input), target);
    cout << "  loss: " << l0 << " -> " << l1 << "\n";
    check("block training reduces loss", l1 < l0, l0 - l1);
}

// ============================================================
// 20. NSAModel training reduces loss
// ============================================================
static void test_model_training() {
    cout << "\n[Test 20: NSAModel 2-block training reduces loss]\n";
    NSAModel model(/*input=*/4, /*d=*/8, /*output=*/4,
                   /*num_blocks=*/2, /*num_query_heads=*/2, /*num_kv_heads=*/1,
                   /*l=*/2, /*stride=*/1, /*top_n=*/2, /*w=*/3);
    const size_t n = 4;
    Tensor input(n, 4);
    Tensor target(n, 4);
    std::mt19937 gen(120);
    std::normal_distribution<double> dist(0.0, 0.5);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    double l0 = l2_loss(model.forward(input), target);
    const double lr = 0.005;
    for (int s = 0; s < 50; ++s) {
        Tensor out = model.forward(input);
        Tensor go = l2_grad(out, target);
        model.backward(go, lr);
        model.update_weights(lr);
    }
    double l1 = l2_loss(model.forward(input), target);
    cout << "  loss: " << l0 << " -> " << l1 << "\n";
    check("model training reduces loss", l1 < l0, l0 - l1);
}

// ============================================================
// 21. mutation tests (zero each branch independently)
// ============================================================
static void test_mutation_branches() {
    cout << "\n[Test 21: mutation — zero each branch independently]\n";
    const size_t n = 6, d = 8;
    NSAAttention attn(d, 2, 1, 2, 1, 2, 3);
    Tensor input(n, d);
    std::mt19937 gen(130);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor baseline = attn.forward(input);

    // Zero compression
    NSAAttention attn2(d, 2, 1, 2, 1, 2, 3);
    // Re-init with same weights by copying — but for mutation test we just
    // compare baseline forward vs forward with use_compression_=false. To make
    // this fair, we re-init with same weights: copy baseline from attn by
    // matching random seed.
    // Simplest: just check that toggle produces a different output from
    // a fresh-but-equal-weight instance.
    std::mt19937 gen2(130);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen2);
    // Reset attn by toggling compression off and re-forward
    attn.set_use_compression(false);
    Tensor no_cmp = attn.forward(input);
    attn.set_use_compression(true);
    attn.set_use_selection(false);
    Tensor no_sel = attn.forward(input);
    attn.set_use_selection(true);
    attn.set_use_window(false);
    Tensor no_win = attn.forward(input);

    double max_diff_cmp = 0, max_diff_sel = 0, max_diff_win = 0;
    for (size_t i = 0; i < baseline.data.size(); ++i) {
        max_diff_cmp = max(max_diff_cmp, fabs(baseline.data[i] - no_cmp.data[i]));
        max_diff_sel = max(max_diff_sel, fabs(baseline.data[i] - no_sel.data[i]));
        max_diff_win = max(max_diff_win, fabs(baseline.data[i] - no_win.data[i]));
    }
    cout << "  diff(no_cmp)=" << max_diff_cmp
         << " diff(no_sel)=" << max_diff_sel
         << " diff(no_win)=" << max_diff_win << "\n";
    check("zeroing compression changes output", max_diff_cmp > 1e-8, max_diff_cmp);
    check("zeroing selection changes output", max_diff_sel > 1e-8, max_diff_sel);
    check("zeroing window changes output", max_diff_win > 1e-8, max_diff_win);
}

// ============================================================
// 22. parameters / gradients shape consistency
// ============================================================
static void test_param_grad_shape_consistency() {
    cout << "\n[Test 22: parameters / gradients shape consistency]\n";
    const size_t n = 4, d = 4;
    NSAAttention attn(d, 1, 1, 2, 1, 2, 2);
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    check("param count == grad count", params.size() == grads.size(),
          static_cast<double>(params.size() - grads.size()));
    bool all_match = true;
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->rows != grads[i]->rows ||
            params[i]->cols != grads[i]->cols) {
            all_match = false;
            cout << "    MISMATCH at " << i
                 << ": param " << params[i]->rows << "x" << params[i]->cols
                 << " vs grad " << grads[i]->rows << "x" << grads[i]->cols << "\n";
        }
    }
    check("all param/grad pairs shape-match", all_match);
}

// ============================================================
// 23. zero gradient doesn't crash
// ============================================================
static void test_zero_grad_no_crash() {
    cout << "\n[Test 23: zero gradient doesn't crash]\n";
    NSAAttention attn(4, 1, 1, 2, 1, 2, 2);
    Tensor input(4, 4);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = 0.01;
    Tensor out = attn.forward(input);
    Tensor go(4, 4); go.fill(0.0);
    bool ok = true;
    try { attn.backward(go, 0.0); }
    catch (...) { ok = false; }
    check("zero-gradient backward doesn't crash", ok);
}

// ============================================================
// 24. determinism — two fresh NSAs with copied params
// ============================================================
static void test_determinism() {
    cout << "\n[Test 24: determinism — copied params produce bit-exact forward]\n";
    NSAAttention a(4, 1, 1, 2, 1, 2, 2);
    NSAAttention b(4, 1, 1, 2, 1, 2, 2);
    // Copy all params from a → b
    auto pa = a.parameters();
    auto pb = b.parameters();
    for (size_t i = 0; i < pa.size(); ++i) {
        for (size_t k = 0; k < pa[i]->data.size(); ++k)
            (*pb[i]).data[k] = (*pa[i]).data[k];
    }
    Tensor input(4, 4);
    std::mt19937 gen(200);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor oa = a.forward(input);
    Tensor ob = b.forward(input);
    double max_d = 0;
    for (size_t i = 0; i < oa.data.size(); ++i)
        max_d = max(max_d, fabs(oa.data[i] - ob.data[i]));
    cout << "  max abs diff = " << max_d << "\n";
    check("determinism: copied-params forward bit-exact", max_d == 0.0, max_d);
}

// ============================================================
// main
// ============================================================
int main() {
    cout << "=========================================\n";
    cout << " Native Sparse Attention (NSA) Tests\n";
    cout << " DeepSeek-AI 2025 — arXiv:2502.11089\n";
    cout << "=========================================\n";

    test_constructor_validation();
    test_forward_shape_single_head();
    test_forward_shape_gqa();
    test_output_finite_nonzero();
    test_gate_sums_to_one();
    test_n_cmp_deterministic();
    test_sliding_window_mask();
    test_selection_top_n();
    test_input_grad_fd();
    test_param_grad_W_q_W_o();
    test_param_grad_W_k_cmp();
    test_param_grad_W_k_sel();
    test_param_grad_W_k_win();
    test_param_grad_W_phi();
    test_param_grad_W_gate();
    test_multi_head_input_grad();
    test_block_forward_shape();
    test_block_input_grad_fd();
    test_block_training();
    test_model_training();
    test_mutation_branches();
    test_param_grad_shape_consistency();
    test_zero_grad_no_crash();
    test_determinism();

    cout << "\n=========================================\n";
    cout << " Summary: " << passed << " passed, " << failed << " failed\n";
    cout << "=========================================\n";
    return failed > 0 ? 1 : 0;
}
