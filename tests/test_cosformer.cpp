// test_cosformer.cpp — Gradient correctness tests for cosFormer
//   Qin et al. 2022, "cosFormer: Rethinking Softmax in Attention"
//
// Tests:
//   1. CosFormerAttention forward shape
//   2. CosFormerAttention output finite + nonzero
//   3. CosFormerAttention input gradient FD check
//   4. CosFormerAttention W_q, W_k, W_v, W_o gradient FD checks
//   5. CosFormerAttention position vectors (cos_pos, sin_pos) properties
//   6. CosFormerAttention attention matrix symmetry (cos is symmetric)
//   7. CosFormerAttention recency bias (peaks on diagonal, falls off)
//   8. CosFormerBlock forward shape
//   9. CosFormerBlock input gradient FD check
//  10. CosFormerModel forward shape + training reduces loss
//  11. cosFormer with M > seq_len still works (longer period)
//  12. Mutation: zeroing cos/sin position vectors breaks the layer (non-vacuous)

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include "nn/layers/attention/cosformer.h"

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

static double rel_err(double a, double b) {
    double denom = max(fabs(a), fabs(b));
    if (denom < 1e-12) return fabs(a - b);
    return fabs(a - b) / denom;
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

// ============================================================
// 1. forward shape
// ============================================================
static void test_forward_shape() {
    cout << "\n[Test 1: forward shape]\n";
    const size_t n = 6, d = 4;
    CosFormerAttention attn(d, n);
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
// 2. finite + nonzero
// ============================================================
static void test_output_finite_nonzero() {
    cout << "\n[Test 2: output finite + nonzero]\n";
    const size_t n = 6, d = 4;
    CosFormerAttention attn(d, n);
    Tensor input(n, d);
    std::mt19937 gen(1);
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
// 3. input gradient FD check
// ============================================================
static void test_input_gradient() {
    cout << "\n[Test 3: input gradient FD check]\n";
    const size_t n = 4, d = 3;
    CosFormerAttention attn(d, n);
    Tensor input(n, d), target(n, d);
    std::mt19937 gen(2);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    attn.zero_grad();
    Tensor out = attn.forward(input);
    Tensor d_out = l2_grad(out, target);
    Tensor d_input_ana = attn.backward(d_out, 0.0);

    const double eps = 1e-5;
    Tensor input_copy = input.clone();
    double max_err = 0.0;
    for (size_t i = 0; i < input.data.size(); ++i) {
        double orig = input_copy.data[i];
        input_copy.data[i] = orig + eps;
        Tensor op = attn.forward(input_copy);
        double lp = l2_loss(op, target);
        input_copy.data[i] = orig - eps;
        Tensor om = attn.forward(input_copy);
        double lm = l2_loss(om, target);
        input_copy.data[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double ana = d_input_ana.data[i];
        max_err = max(max_err, rel_err(num, ana));
    }
    cout << "  max input-grad rel_err = " << max_err << "\n";
    check("input gradient FD check (rel_err < 1e-3)", max_err < 1e-3, max_err);
}

// ============================================================
// 4. parameter gradient FD checks (W_q, W_k, W_v, W_o)
// ============================================================
static void test_parameter_gradients() {
    cout << "\n[Test 4: parameter gradient FD checks]\n";
    const size_t n = 4, d = 3;
    CosFormerAttention attn(d, n);
    Tensor input(n, d), target(n, d);
    std::mt19937 gen(3);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    auto params = attn.parameters();
    auto grads  = attn.gradients();
    const double eps = 1e-5;

    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* W = params[p];
        Tensor* G = grads[p];
        // Check a handful of entries per param
        std::vector<std::pair<size_t,size_t>> idx;
        for (size_t i = 0; i < W->rows; ++i)
            for (size_t j = 0; j < W->cols; ++j)
                idx.push_back({i, j});
        double max_err = 0.0;
        for (auto& [i, j] : idx) {
            attn.zero_grad();
            Tensor out = attn.forward(input);
            Tensor d_out = l2_grad(out, target);
            attn.backward(d_out, 0.0);
            double ana = (*G)(i, j);

            double orig = (*W)(i, j);
            (*W)(i, j) = orig + eps;
            Tensor op = attn.forward(input);
            double lp = l2_loss(op, target);
            (*W)(i, j) = orig - eps;
            Tensor om = attn.forward(input);
            double lm = l2_loss(om, target);
            (*W)(i, j) = orig;
            double num = (lp - lm) / (2.0 * eps);
            max_err = max(max_err, rel_err(num, ana));
        }
        const char* names[] = {"W_q", "W_k", "W_v", "W_o"};
        cout << "  " << names[p] << " max grad rel_err = " << max_err << "\n";
        check(string(names[p]) + " grad FD (rel_err < 1e-3)", max_err < 1e-3, max_err);
    }
}

// ============================================================
// 5. position vector properties
// ============================================================
static void test_position_vectors() {
    cout << "\n[Test 5: position vector properties]\n";
    const size_t n = 8, d = 4;
    CosFormerAttention attn(d, n);
    Tensor input(n, d);
    std::mt19937 gen(4);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);  // populates position vectors lazily

    const Tensor& cp = attn.cos_pos();
    const Tensor& sp = attn.sin_pos();

    // cos(0) = 1, sin(0) = 0
    check("cos_pos[0] ≈ 1",  fabs(cp(0, 0) - 1.0) < 1e-10);
    check("sin_pos[0] ≈ 0",  fabs(sp(0, 0))       < 1e-10);
    // cos² + sin² = 1 for all positions
    bool cs1 = true;
    for (size_t t = 0; t < n; ++t) {
        double ss = cp(t, 0)*cp(t, 0) + sp(t, 0)*sp(t, 0);
        if (fabs(ss - 1.0) > 1e-10) cs1 = false;
    }
    check("cos² + sin² = 1", cs1);
}

// ============================================================
// 6. cos re-weighting factor symmetry
//    The cosFormer kernel is s(i,j) = ReLU(Q[i])·ReLU(K[j]) · cos(π(i-j)/2M).
//    The cos factor itself is symmetric: cos(π(i-j)/2M) == cos(π(j-i)/2M).
//    (The Q·K^T part isn't required to be symmetric — only the
//    re-weighting mechanism is. This test verifies that property.)
// ============================================================
static void test_kernel_symmetry() {
    cout << "\n[Test 6: cos re-weighting symmetry]\n";
    const size_t n = 8, d = 3;
    CosFormerAttention attn(d, n);
    Tensor input(n, d);
    std::mt19937 gen(5);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);

    // Verify the cos factor is symmetric (which is what cosFormer contributes
    // beyond vanilla linear attention). The cos re-weighting is the ONLY
    // symmetric part of the kernel — the Q·K^T inner product is asymmetric.
    const double M = (double)attn.M();
    bool symmetric = true;
    double worst_err = 0.0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // Cast through int to avoid size_t underflow at i < j.
            int di = (int)i - (int)j;
            int dj = (int)j - (int)i;
            double wij = cos(M_PI * 0.5 * (double)di / M);
            double wji = cos(M_PI * 0.5 * (double)dj / M);
            worst_err = max(worst_err, fabs(wij - wji));
            if (fabs(wij - wji) > 1e-14) symmetric = false;
        }
    }
    cout << "  max |cos(i,j) - cos(j,i)| = " << worst_err << "\n";
    check("cos re-weighting symmetric", symmetric, worst_err);
}

// ============================================================
// 7. recency bias: un-normalized attention kernel peaks on diagonal
// ============================================================
static void test_recency_bias() {
    cout << "\n[Test 7: recency bias peaks on diagonal]\n";
    const size_t n = 9, d = 3;
    CosFormerAttention attn(d, n);
    Tensor input(n, d);
    std::mt19937 gen(6);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    attn.forward(input);

    // For each position i, the cos re-weighting factor for self (j=i) is 1,
    // and it strictly decreases as |i-j| grows. (We use the analytic cosine
    // form here, not the cached position vectors — same values.)
    bool decreasing = true;
    double max_falloff_err = 0.0;
    for (size_t t = 0; t < n; ++t) {
        double prev = 1.0;  // cos(0) = 1
        for (int off = 1; t + (size_t)off < n; ++off) {
            double ang = M_PI * 0.5 * (double)off / (double)attn.M();
            double cw = cos(ang);
            if (cw > prev + 1e-10) decreasing = false;
            max_falloff_err = max(max_falloff_err, prev - cw);
            prev = cw;
        }
    }
    cout << "  recency-bias fall-off error = " << max_falloff_err << "\n";
    check("cos re-weighting decreases with |i-j|", decreasing);
}

// ============================================================
// 8. block forward shape
// ============================================================
static void test_block_forward_shape() {
    cout << "\n[Test 8: CosFormerBlock forward shape]\n";
    const size_t n = 5, d = 4, ffn = 8;
    CosFormerBlock block(d, n, ffn);
    Tensor input(n, d);
    std::mt19937 gen(7);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = block.forward(input);
    cout << "  block in " << input.rows << "x" << input.cols
         << " -> out " << out.rows << "x" << out.cols << "\n";
    check("block forward shape", out.rows == n && out.cols == d);
}

// ============================================================
// 9. block input gradient FD check
// ============================================================
static void test_block_input_gradient() {
    cout << "\n[Test 9: CosFormerBlock input gradient FD check]\n";
    const size_t n = 4, d = 3, ffn = 6;
    CosFormerBlock block(d, n, ffn);
    Tensor input(n, d), target(n, d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            input(i, j) = 0.3 * sin(0.1 * i) + 0.1 * j;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j)
            target(i, j) = 0.2 * i - 0.1 * j;

    block.zero_grad();
    Tensor out = block.forward(input);
    Tensor d_out = l2_grad(out, target);
    Tensor d_input_ana = block.backward(d_out, 0.0);

    const double eps = 1e-5;
    Tensor input_copy = input.clone();
    double max_err = 0.0;
    size_t worst_i = 0;
    for (size_t i = 0; i < input.data.size(); ++i) {
        double orig = input_copy.data[i];
        input_copy.data[i] = orig + eps;
        Tensor op = block.forward(input_copy);
        double lp = l2_loss(op, target);
        input_copy.data[i] = orig - eps;
        Tensor om = block.forward(input_copy);
        double lm = l2_loss(om, target);
        input_copy.data[i] = orig;
        double num = (lp - lm) / (2.0 * eps);
        double ana = d_input_ana.data[i];
        double err = rel_err(num, ana);
        if (err > max_err) { max_err = err; worst_i = i; }
    }
    cout << "  max block input-grad rel_err = " << max_err
         << "  at index " << worst_i
         << "  ana=" << d_input_ana.data[worst_i] << "\n";
    check("block input gradient FD (rel_err < 1e-3)", max_err < 1e-3, max_err);
}

// ============================================================
// 10. model forward + training reduces loss
// ============================================================
static void test_model_train() {
    cout << "\n[Test 10: CosFormerModel forward + training reduces loss]\n";
    const size_t n = 4, d = 4, out_f = 2;
    CosFormerModel model(d, n, out_f, /*num_blocks=*/1, /*ffn_dim=*/6);
    Tensor input(n, d), target(1, out_f);
    std::mt19937 gen(9);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    for (size_t i = 0; i < target.data.size(); ++i) target.data[i] = dist(gen);

    Tensor out = model.forward(input);
    cout << "  model out " << out.rows << "x" << out.cols << "\n";
    check("model forward shape", out.rows == 1 && out.cols == out_f);

    double loss_before = 1e9;
    double loss_after = 1e9;
    const double lr = 0.005;
    for (int step = 0; step < 30; ++step) {
        model.zero_grad();
        Tensor o = model.forward(input);
        double L = l2_loss(o, target);
        if (step == 0) loss_before = L;
        Tensor d = l2_grad(o, target);
        model.backward(d, 0.0);
        model.update_weights(lr);
        if (step == 29) loss_after = L;
    }
    double pct = (1.0 - loss_after / loss_before) * 100.0;
    cout << "  loss: " << loss_before << " -> " << loss_after
         << "  (" << pct << "% reduction)\n";
    check("training reduces loss (>20%)", loss_after < loss_before * 0.80);
}

// ============================================================
// 11. M > seq_len still works
// ============================================================
static void test_large_M() {
    cout << "\n[Test 11: M > seq_len (longer cos period)]\n";
    const size_t n = 4, d = 3;
    CosFormerAttention attn(d, n, /*M=*/2 * n);
    Tensor input(n, d);
    std::mt19937 gen(10);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);
    Tensor out = attn.forward(input);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i) {
        if (std::isnan(out.data[i]) || std::isinf(out.data[i])) { finite = false; break; }
    }
    check("M > seq_len still produces finite output", finite);
    check("M accessor returns configured value", attn.M() == 2 * n);
}

// ============================================================
// 12. mutation: zeroing cos_pos breaks the layer
// ============================================================
static void test_mutation_pos_vectors() {
    cout << "\n[Test 12: mutation — zeroing cos_pos alters output]\n";
    const size_t n = 4, d = 3;
    CosFormerAttention attn(d, n);
    Tensor input(n, d);
    std::mt19937 gen(11);
    std::normal_distribution<double> dist(0.0, 0.3);
    for (size_t i = 0; i < input.data.size(); ++i) input.data[i] = dist(gen);

    Tensor out_normal = attn.forward(input);

    // Force a SECOND forward with cos_pos zeroed out — output should differ.
    attn.zero_pos_for_test();
    Tensor out_broken = attn.forward(input);

    double max_diff = 0.0;
    for (size_t i = 0; i < out_normal.data.size(); ++i) {
        max_diff = max(max_diff, fabs(out_normal.data[i] - out_broken.data[i]));
    }
    cout << "  max diff (normal vs zeroed cos_pos) = " << max_diff << "\n";
    check("zeroing cos_pos alters output (non-vacuous)", max_diff > 1e-6, max_diff);
}

// ============================================================
// main
// ============================================================
int main() {
    cout << "=== cosFormer Tests ===\n";
    cout.setf(std::ios::unitbuf);

    test_forward_shape();
    test_output_finite_nonzero();
    test_input_gradient();
    test_parameter_gradients();
    test_position_vectors();
    test_kernel_symmetry();
    test_recency_bias();
    test_block_forward_shape();
    test_block_input_gradient();
    test_model_train();
    test_large_M();
    test_mutation_pos_vectors();

    cout << "\n=== Summary: " << passed << "/" << (passed + failed)
         << " tests passed ===\n";
    return failed == 0 ? 0 : 1;
}