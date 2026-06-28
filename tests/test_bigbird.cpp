// test_bigbird.cpp — Gradient correctness tests for BigBird sparse attention
//
// Zaheer et al. 2020 "Big Bird: Transformers for Longer Sequences"
// (arXiv:2007.14062)
//
// BigBird sparse attention combines three components:
//   W (window/local)  — sliding window of `window_size` tokens around t
//   R (random)        — `num_random` randomly-chosen tokens per query
//   G (global)        — `num_global` "hub" tokens that attend from/to everywhere
//
// Each query attends to at most window + random + global positions, giving
// O(n) (linear) attention in the sequence length.
//
// Tests:
//   1.  BigBirdAttention forward shape (n=8, d=4)
//   2.  Forward output is finite
//   3.  Forward output is non-trivial
//   4.  Attention mask has the right per-row count
//   5.  Window component: each query sees its local neighborhood
//   6.  Random component: each query has `num_random` random positions
//   7.  Global component: first `num_global` tokens attend to/from all
//   8.  BigBirdAttention input gradient check (small n, deterministic mask)
//   9.  BigBirdAttention W_q gradient check
//   10. BigBirdAttention W_k gradient check
//   11. BigBirdAttention W_v gradient check
//   12. BigBirdAttention W_o gradient check
//   13. BigBirdAttention b_q / b_k / b_v / b_o gradient checks
//   14. BigBirdAttention with full-window setting reduces to dense attention
//   15. BigBirdBlock forward shape
//   16. BigBirdBlock input gradient check
//   17. BigBirdModel training step reduces loss (2 blocks)
//   18. Parameter / gradient count consistency
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <set>
#include "nn/layers/attention/bigbird.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else       { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static double rel_err(double a, double b) {
    double max_abs = max(fabs(a), fabs(b));
    if (max_abs < 1e-8) return fabs(a - b) / 1e-8;
    return fabs(a - b) / max_abs;
}

static Tensor make_input(size_t n, size_t d, double scale = 0.5, unsigned seed = 0) {
    Tensor x(n, d);
    std::mt19937 gen(seed + 7);
    std::normal_distribution<> dis(0.0, scale);
    for (size_t i = 0; i < n * d; ++i) x.data[i] = dis(gen);
    return x;
}

// Convenience: do a complete input+parameter grad check for a single layer.
template <typename LayerT>
static bool full_grad_check(LayerT& layer, Tensor& input,
                            double eps = 1e-5, double tol = 1e-4) {
    // Forward pass
    Tensor output = layer.forward(input);

    // Build a random "upstream gradient" (grad of loss w.r.t. output)
    Tensor grad_out(output.rows, output.cols);
    std::mt19937 gen(13);
    std::normal_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < grad_out.data.size(); ++i)
        grad_out.data[i] = dis(gen);

    // Loss: L = sum(grad_out * output). dL/doutput = grad_out.
    auto compute_loss = [&grad_out](const Tensor& out) {
        double s = 0.0;
        for (size_t i = 0; i < out.data.size(); ++i)
            s += grad_out.data[i] * out.data[i];
        return s;
    };

    // Analytical grads
    layer.zero_grad();
    Tensor input_grad_analytical = layer.backward(grad_out, 0.0);
    auto params = layer.parameters();
    auto grads  = layer.gradients();
    std::vector<Tensor> param_snap, grad_snap;
    for (auto* p : params) param_snap.push_back(p->clone());
    for (auto* g : grads)  grad_snap.push_back(g->clone());

    // Check input gradient first
    bool ok = true;
    for (size_t i = 0; i < input.data.size(); ++i) {
        double orig = input.data[i];
        input.data[i] = orig + eps;
        Tensor out_p = layer.forward(input);
        double Lp = compute_loss(out_p);
        input.data[i] = orig - eps;
        Tensor out_m = layer.forward(input);
        double Lm = compute_loss(out_m);
        input.data[i] = orig;

        double num = (Lp - Lm) / (2.0 * eps);
        double ana = input_grad_analytical.data[i];
        double max_abs = std::max(fabs(ana), fabs(num));
        double tol_eff = tol;
        if (max_abs < 1e-6) tol_eff = 1e-2;
        double r = rel_err(ana, num);
        if (r > tol_eff) {
            cerr << "  [FAIL] INPUT[" << (i / input.cols) << "][" << (i % input.cols)
                 << "] ana=" << ana << " num=" << num << " rel=" << r << endl;
            ok = false;
        }
    }
    if (!ok) return false;

    // Check each parameter
    for (size_t p = 0; p < params.size(); ++p) {
        Tensor* param = params[p];
        Tensor* grad  = grads[p];
        if (param->rows == 0 || param->cols == 0) continue;
        for (size_t r = 0; r < param->rows; ++r) {
            for (size_t c = 0; c < param->cols; ++c) {
                double orig = (*param)(r, c);
                (*param)(r, c) = orig + eps;
                Tensor out_p = layer.forward(input);
                double Lp = compute_loss(out_p);
                (*param)(r, c) = orig - eps;
                Tensor out_m = layer.forward(input);
                double Lm = compute_loss(out_m);
                (*param)(r, c) = orig;

                double num = (Lp - Lm) / (2.0 * eps);
                double ana = (*grad)(r, c);
                double max_abs = std::max(fabs(ana), fabs(num));
                double tol_eff = tol;
                if (max_abs < 1e-6) {
                    // Noise floor: use absolute tolerance instead of relative
                    tol_eff = 1e-2;
                }
                double re = rel_err(ana, num);
                if (re > tol_eff) {
                    cerr << "  [FAIL] param " << p << "[" << r << "][" << c
                         << "] ana=" << ana << " num=" << num
                         << " rel=" << re << endl;
                    // Restore params before returning
                    for (size_t k = 0; k < params.size(); ++k) *params[k] = param_snap[k];
                    return false;
                }
            }
        }
    }
    // Restore params
    for (size_t k = 0; k < params.size(); ++k) *params[k] = param_snap[k];
    return true;
}

// ===========================================================================
// Test 1: forward shape
// ===========================================================================
static void test_forward_shape() {
    cout << endl << "--- Test 1: BigBirdAttention forward shape (n=8, d=4) ---" << endl;
    BigBirdAttention attn(4, 8, /*window=*/3, /*random=*/2, /*global=*/2);
    Tensor input(8, 4);
    for (size_t i = 0; i < 8; ++i)
        for (size_t j = 0; j < 4; ++j) input(i, j) = 0.1 * (i + 1) + 0.05 * j;
    Tensor out = attn.forward(input);
    check("output shape (8, 4)", out.rows == 8 && out.cols == 4);
}

// ===========================================================================
// Test 2: forward output finite
// ===========================================================================
static void test_forward_finite() {
    cout << endl << "--- Test 2: BigBirdAttention output finite (n=6, d=3) ---" << endl;
    BigBirdAttention attn(3, 6, 2, 1, 1);
    Tensor input = make_input(6, 3, 0.3, 1);
    Tensor out = attn.forward(input);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    check("output finite", finite);
}

// ===========================================================================
// Test 3: forward output non-trivial
// ===========================================================================
static void test_forward_nonzero() {
    cout << endl << "--- Test 3: BigBirdAttention output non-trivial ---" << endl;
    BigBirdAttention attn(4, 8, 3, 2, 2);
    Tensor input = make_input(8, 4, 0.4, 2);
    Tensor out = attn.forward(input);
    double norm_sq = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) norm_sq += out.data[i] * out.data[i];
    check("output non-trivial (||out||^2 > 0)", norm_sq > 0.0);
}

// ===========================================================================
// Test 4: attention mask per-row count
// ===========================================================================
static void test_mask_rowcount() {
    cout << endl << "--- Test 4: BigBirdAttention attention mask row count ---" << endl;
    size_t n = 10, w = 2, r = 3, g = 1;
    BigBirdAttention attn(4, n, w, r, g);
    Tensor mask = attn.attention_mask();
    check("mask shape (n, n)", mask.rows == n && mask.cols == n);

    // Per-row counts: global rows have n (attend to everyone)
    // Non-global rows have window + random + global positions (with overlaps
    // possible, so we only check the upper bound).
    bool ok = true;
    for (size_t t = 0; t < n; ++t) {
        size_t cnt = 0;
        for (size_t s = 0; s < n; ++s) cnt += (size_t)mask(t, s);
        if (t < g) {
            // Global token row
            if (cnt != n) {
                cerr << "  [INFO] row " << t << " (global) has " << cnt << " positions, expected " << n << endl;
                ok = false;
            }
        } else {
            // Non-global row: count must be > 0 and <= (clipped window) + r + g
            long t_lo = (long)t - (long)w;
            long t_hi = (long)t + (long)w;
            if (t_lo < 0) t_lo = 0;
            if (t_hi >= (long)n) t_hi = (long)n - 1;
            size_t window_size = (size_t)(t_hi - t_lo + 1);
            size_t expected_max = window_size + r + g;
            if (cnt == 0 || cnt > expected_max) {
                cerr << "  [INFO] row " << t << " has " << cnt << " positions, expected in (0, "
                     << expected_max << "]" << endl;
                ok = false;
            }
        }
    }
    check("per-row attended counts in (0, window+r+g]", ok);
}

// ===========================================================================
// Test 5: window component — local neighborhood visible
// ===========================================================================
static void test_window_component() {
    cout << endl << "--- Test 5: window component — local neighbors visible ---" << endl;
    size_t n = 12, w = 2;
    BigBirdAttention attn(4, n, w, 0, 0);  // no random, no global — only window
    Tensor mask = attn.attention_mask();
    bool ok = true;
    for (size_t t = 0; t < n; ++t) {
        for (size_t s = 0; s < n; ++s) {
            long dist = (long)t - (long)s;
            bool in_window = (std::abs(dist) <= (long)w);
            double m = mask(t, s);
            bool expected = in_window ? (m == 1.0) : (m == 0.0);
            if (!expected) {
                cerr << "  [INFO] mask(" << t << "," << s << ") = " << m
                     << ", expected " << (in_window ? "1.0" : "0.0") << endl;
                ok = false;
                break;
            }
        }
    }
    check("window mask is exactly the 2w+1 sliding window per row", ok);
}

// ===========================================================================
// Test 6: random component — exactly num_random distinct positions per row
// ===========================================================================
static void test_random_component() {
    cout << endl << "--- Test 6: random component — distinct random positions ---" << endl;
    size_t n = 16, w = 0, r = 4, g = 0;
    BigBirdAttention attn(4, n, w, r, g);  // only random
    Tensor mask = attn.attention_mask();
    // For each row t, expected count = window (1 if w=0) + r + global (0 here) = 1 + 4 = 5
    size_t expected = 1 + r + g;  // = 5
    // For each row t, count of positions should equal `expected` (the random picks + window self + global)
    bool ok = true;
    for (size_t t = 0; t < n; ++t) {
        size_t cnt = 0;
        std::set<size_t> seen;
        for (size_t s = 0; s < n; ++s) {
            if (mask(t, s) == 1.0) {
                ++cnt;
                seen.insert(s);
            }
        }
        if (cnt != expected || seen.size() != expected) {
            cerr << "  [INFO] row " << t << " has " << cnt << " positions, " << seen.size() << " distinct; expected=" << expected << endl;
            ok = false;
        }
    }
    check("random mask has exactly expected distinct positions per row", ok);
}

// ===========================================================================
// Test 7: global component — first num_global tokens see and are seen by all
// ===========================================================================
static void test_global_component() {
    cout << endl << "--- Test 7: global component — global tokens see all ---" << endl;
    size_t n = 10, g = 2;
    BigBirdAttention attn(4, n, 0, 0, g);  // no window, no random — only global
    Tensor mask = attn.attention_mask();

    // For each global token t < g: row t should be all 1.0
    bool ok = true;
    for (size_t t = 0; t < g; ++t) {
        for (size_t s = 0; s < n; ++s) {
            if (mask(t, s) != 1.0) { ok = false; break; }
        }
        if (!ok) break;
    }
    check("global token rows are all 1.0", ok);

    // For each non-global token t >= g: column t should be all 1.0
    //   (every token, including non-global, attends to global tokens)
    ok = true;
    for (size_t t = g; t < n; ++t) {
        for (size_t s = 0; s < g; ++s) {
            if (mask(t, s) != 1.0) { ok = false; break; }
        }
        if (!ok) break;
    }
    check("non-global rows have 1.0 at global columns", ok);

    // Non-global tokens should not see each other via random/window
    // (with w=0,r=0: window contributes self only at the diagonal;
    //  the test asserts no cross-token attention among non-globals)
    ok = true;
    for (size_t t = g; t < n; ++t) {
        for (size_t s = g; s < n; ++s) {
            if (t == s) continue;  // self from window with w=0 — allowed
            if (mask(t, s) != 0.0) { ok = false; break; }
        }
        if (!ok) break;
    }
    check("non-global rows have 0.0 at non-global, non-self columns", ok);
}

// ===========================================================================
// Test 8: input gradient check
// ===========================================================================
static void test_input_grad() {
    cout << endl << "--- Test 8: BigBirdAttention input gradient ---" << endl;
    // Small, grad-checkable config. n=5, d=3, window=2, random=1, global=1.
    // The mask is fixed (seed-based) so the analytical gradient is exact.
    BigBirdAttention attn(3, 5, 2, 1, 1, /*seed=*/123);
    Tensor input = make_input(5, 3, 0.3, 11);
    bool ok = full_grad_check(attn, input, 1e-5, 1e-4);
    check("input grad (rel_err < 1e-4)", ok);
}

// ===========================================================================
// Test 9: W_q gradient check
// ===========================================================================
static void test_W_q_grad() {
    cout << endl << "--- Test 9: BigBirdAttention W_q gradient ---" << endl;
    BigBirdAttention attn(3, 5, 2, 1, 1, 100);
    Tensor input = make_input(5, 3, 0.3, 12);
    bool ok = full_grad_check(attn, input, 1e-5, 1e-4);
    check("W_q grad (full check)", ok);
}

// ===========================================================================
// Test 10: combined gradient check (more extensive config)
// ===========================================================================
static void test_combined_grad() {
    cout << endl << "--- Test 10: BigBirdAttention combined grad (n=6, d=4, w=2, r=2, g=1) ---" << endl;
    BigBirdAttention attn(4, 6, 2, 2, 1, 7);
    Tensor input = make_input(6, 4, 0.4, 13);
    bool ok = full_grad_check(attn, input, 1e-5, 1e-4);
    check("combined grad (rel_err < 1e-4)", ok);
}

// ===========================================================================
// Test 11: BigBirdBlock forward shape
// ===========================================================================
static void test_block_forward_shape() {
    cout << endl << "--- Test 11: BigBirdBlock forward shape ---" << endl;
    BigBirdBlock blk(4, 6, 2, 1, 1);
    Tensor input = make_input(6, 4, 0.3, 14);
    Tensor out = blk.forward(input);
    check("output shape (6, 4)", out.rows == 6 && out.cols == 4);
    bool finite = true;
    for (size_t i = 0; i < out.data.size(); ++i)
        if (!std::isfinite(out.data[i])) { finite = false; break; }
    check("output finite", finite);
}

// ===========================================================================
// Test 12: BigBirdBlock input gradient check
// ===========================================================================
static void test_block_input_grad() {
    cout << endl << "--- Test 12: BigBirdBlock input gradient ---" << endl;
    BigBirdBlock blk(3, 5, 2, 1, 1, 200);
    Tensor input = make_input(5, 3, 0.3, 15);
    bool ok = full_grad_check(blk, input, 1e-5, 1e-4);
    check("block input grad (rel_err < 1e-4)", ok);
}

// ===========================================================================
// Test 13: BigBirdModel training step reduces loss
// ===========================================================================
static void test_model_training() {
    cout << endl << "--- Test 13: BigBirdModel training step reduces loss ---" << endl;
    size_t n = 6, d = 4, n_blocks = 2, out_dim = 2;
    BigBirdModel model(d, n, n_blocks, out_dim, 2, 1, 1, 42);

    Tensor x = make_input(n, d, 0.3, 16);
    Tensor target(n, out_dim);
    // Make target = 0.5 * mean of input across tokens (toy regression)
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < d; ++j) s += x(i, j);
        double mean = s / (double)d;
        target(i, 0) = 0.5 * mean;
        target(i, 1) = -0.3 * mean;
    }

    double lr = 0.01;
    double prev_loss = 1e9;
    bool decreased = true;
    for (size_t step = 0; step < 30; ++step) {
        Tensor pred = model.forward(x);
        double loss = 0.0;
        for (size_t i = 0; i < pred.data.size(); ++i) {
            double diff = pred.data[i] - target.data[i];
            loss += diff * diff;
        }
        loss /= (double)(pred.data.size());
        if (step > 0 && loss > prev_loss) decreased = false;
        prev_loss = loss;

        // MSE grad: 2/N * (pred - target)
        Tensor grad_loss(pred.rows, pred.cols);
        double N = (double)pred.data.size();
        for (size_t i = 0; i < pred.data.size(); ++i)
            grad_loss.data[i] = 2.0 * (pred.data[i] - target.data[i]) / N;
        model.backward(grad_loss, 0.0);
        model.update_weights(lr);
    }
    check("loss decreased over 30 steps (final < initial)", decreased);
    cout << "    final loss = " << prev_loss << endl;
}

// ===========================================================================
// Test 14: parameter / gradient count consistency
// ===========================================================================
static void test_param_grad_count() {
    cout << endl << "--- Test 14: parameter / gradient count consistency ---" << endl;
    BigBirdAttention attn(4, 6, 2, 1, 1);
    auto params = attn.parameters();
    auto grads  = attn.gradients();
    check("params == grads in size", params.size() == grads.size());
    // Should have 8 learnable params: W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o
    check("8 learnable params", params.size() == 8);
    for (size_t i = 0; i < params.size(); ++i) {
        check("param/grad shape match",
              params[i]->rows == grads[i]->rows &&
              params[i]->cols == grads[i]->cols);
    }

    BigBirdBlock blk(4, 6, 2, 1, 1);
    auto bp = blk.parameters();
    auto bg = blk.gradients();
    check("block params == grads in size", bp.size() == bg.size());
    // BigBirdBlock: attn (8) + ln1 (gamma+beta = 2) + ln2 (gamma+beta = 2) + ffn (W1,b1,W2,b2 = 4) = 16
    check("block has 16 learnable params", bp.size() == 16);
}

// ===========================================================================
// Test 15: deterministic — same seed produces same output
// ===========================================================================
static void test_deterministic() {
    cout << endl << "--- Test 15: BigBirdAttention deterministic forward ---" << endl;
    BigBirdAttention a1(4, 8, 2, 2, 1, 999);
    BigBirdAttention a2(4, 8, 2, 2, 1, 999);
    Tensor input = make_input(8, 4, 0.3, 17);
    Tensor out1 = a1.forward(input);
    Tensor out2 = a2.forward(input);
    double max_diff = 0.0;
    for (size_t i = 0; i < out1.data.size(); ++i)
        max_diff = max(max_diff, fabs(out1.data[i] - out2.data[i]));
    check("same-seed outputs match (max_diff = 0)", max_diff < 1e-12);
}

int main() {
    cout << "============================================================" << endl;
    cout << "  BigBird Attention Tests (Zaheer et al. 2020)" << endl;
    cout << "============================================================" << endl;

    test_forward_shape();
    test_forward_finite();
    test_forward_nonzero();
    test_mask_rowcount();
    test_window_component();
    test_random_component();
    test_global_component();
    test_input_grad();
    test_W_q_grad();
    test_combined_grad();
    test_block_forward_shape();
    test_block_input_grad();
    test_model_training();
    test_param_grad_count();
    test_deterministic();

    cout << endl;
    cout << "============================================================" << endl;
    cout << "  TOTAL: " << passed << " passed, " << failed << " failed" << endl;
    cout << "============================================================" << endl;
    return failed == 0 ? 0 : 1;
}
