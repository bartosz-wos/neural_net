// test_three_d_axial_attention.cpp — Gradient correctness tests for 3D Axial
//   Attention (Ho et al. 2019 §3 — video).
//
// Tests:
//   1.  ThreeDAxialAttention constructor validation (5 invalid + 1 valid + accessors)
//   2.  ThreeDAxialAttentionBlock constructor validation (3 invalid + 1 valid)
//   3.  ThreeDAxialAttentionModel constructor validation (5 invalid + 1 valid)
//   4.  Forward shape + finiteness (H=2, W=3, T=2, d=8, num_heads=2)
//   5.  H=1, W=1, T=1 degenerate forward
//   6.  W=1, T=1 degenerate forward (H-only axis)
//   7.  H-axis mask: only same (j, t) keys contribute (with random input)
//   8.  W-axis mask: only same (i, t) keys contribute
//   9.  T-axis mask: only same (i, j) keys contribute
//  10.  Causal mask: per-axis causal blocks future-direction
//  11.  Per-axis causal flag toggle works independently
//  12.  FD input gradient (random non-uniform init, rel_err < 1e-4)
//  13.  FD W_q/k/v/o_h (boosted Q/K init for FD-precision)
//  14.  FD W_q/k/v/o_w
//  15.  FD W_q/k/v/o_t
//  16.  FD b_h, b_w, b_t (clean, no softmax chain)
//  17.  ThreeDAxialAttentionBlock forward shape (with and without FFN)
//  18.  ThreeDAxialAttentionModel forward shape (H=2, W=3, T=2, num_blocks=2)
//  19.  ThreeDAxialAttentionBlock FD input gradient (rel_err < 1e-3 — LN+FFN chain)
//  20.  End-to-end ThreeDAxialAttentionModel training reduces MSE loss > 30%
//  21.  update_weights moves all 15 parameter groups

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <stdexcept>
#include <cstdlib>
#include "nn/layers/attention/three_d_axial_attention.h"

using namespace std;

static int passed = 0;
static int failed = 0;
static void check(const string& name, bool cond, double err = 0.0) {
    if (cond) { ++passed; cout << "  [PASS] " << name << "\n"; }
    else      { ++failed; cout << "  [FAIL] " << name << " (err=" << err << ")\n"; }
}

static double tensor_max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e9;
    double m = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i)
        m = max(m, fabs(a.data[i] - b.data[i]));
    return m;
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

static double l2_loss_value(const Tensor& out, const Tensor& tgt) {
    double s = 0.0;
    for (size_t i = 0; i < out.data.size(); ++i) {
        double d = out.data[i] - tgt.data[i];
        s += 0.5 * d * d;
    }
    return s;
}
static Tensor l2_loss_grad(const Tensor& out, const Tensor& tgt) {
    Tensor g(out.rows, out.cols);
    for (size_t i = 0; i < out.data.size(); ++i)
        g.data[i] = out.data[i] - tgt.data[i];
    return g;
}

// Flatten helper matching the impl: idx = i*(W*T) + j*T + t.
static inline size_t flat_idx(size_t i, size_t j, size_t t, size_t W, size_t T) {
    return i * (W * T) + j * T + t;
}

// FD: input gradient.
static Tensor fd_input_grad(ThreeDAxialAttention& attn, Tensor& input,
                             const Tensor& target, double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            double lp = l2_loss_value(attn.forward(input), target);
            input[i][j] = orig - eps;
            double lm = l2_loss_value(attn.forward(input), target);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}
static Tensor fd_input_grad_block(ThreeDAxialAttentionBlock& blk, Tensor& input,
                                   const Tensor& target, double eps = 1e-5) {
    Tensor grad(input.rows, input.cols);
    for (size_t i = 0; i < input.rows; ++i) {
        for (size_t j = 0; j < input.cols; ++j) {
            double orig = input[i][j];
            input[i][j] = orig + eps;
            double lp = l2_loss_value(blk.forward(input), target);
            input[i][j] = orig - eps;
            double lm = l2_loss_value(blk.forward(input), target);
            input[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

// FD: param gradient.
static Tensor fd_param_grad(ThreeDAxialAttention& attn, size_t pidx,
                             Tensor& input, const Tensor& target,
                             double eps = 1e-5) {
    Tensor* p = attn.parameters()[pidx];
    Tensor grad(p->rows, p->cols);
    for (size_t i = 0; i < p->rows; ++i) {
        for (size_t j = 0; j < p->cols; ++j) {
            double orig = (*p)[i][j];
            (*p)[i][j] = orig + eps;
            double lp = l2_loss_value(attn.forward(input), target);
            (*p)[i][j] = orig - eps;
            double lm = l2_loss_value(attn.forward(input), target);
            (*p)[i][j] = orig;
            grad[i][j] = (lp - lm) / (2.0 * eps);
        }
    }
    return grad;
}

static Tensor random_tensor(size_t r, size_t c, double scale, mt19937& rng) {
    normal_distribution<double> dist(0.0, scale);
    Tensor t(r, c);
    for (size_t i = 0; i < t.data.size(); ++i)
        t.data[i] = dist(rng);
    return t;
}

static Tensor random_input(size_t r, size_t c, double scale = 0.5,
                            mt19937* rng_p = nullptr) {
    static thread_local mt19937* tl_rng = nullptr;
    if (rng_p) tl_rng = rng_p;
    if (!tl_rng) {
        static thread_local mt19937 fallback(42);
        tl_rng = &fallback;
    }
    normal_distribution<double> dist(0.0, scale);
    Tensor t(r, c);
    for (size_t i = 0; i < t.data.size(); ++i)
        t.data[i] = dist(*tl_rng);
    return t;
}

int main() {
    cout << "=== 3D Axial Attention Tests ===" << endl;
    cout.setf(std::ios::unitbuf);

    // ============================================================
    // Test 1: ThreeDAxialAttention constructor validation
    // ============================================================
    cout << "\n--- Test 1: ThreeDAxialAttention constructor validation ---\n";
    {
        int sub = 0, total_sub = 0;
        // d_model=0
        ++total_sub;
        try { ThreeDAxialAttention a(0, 2, 3, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        // H=0
        ++total_sub;
        try { ThreeDAxialAttention a(8, 0, 3, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        // W=0
        ++total_sub;
        try { ThreeDAxialAttention a(8, 2, 0, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        // T=0
        ++total_sub;
        try { ThreeDAxialAttention a(8, 2, 3, 0); }
        catch (const std::invalid_argument&) { ++sub; }
        // d_model % num_heads != 0
        ++total_sub;
        try { ThreeDAxialAttention a(8, 2, 3, 2, 3); }
        catch (const std::invalid_argument&) { ++sub; }
        // valid
        ++total_sub;
        try {
            ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
            if (a.d_model() == 8 && a.H() == 2 && a.W() == 3 && a.T() == 2 &&
                a.num_heads() == 2 &&
                a.causal_h() == true && a.causal_w() == true && a.causal_t() == true &&
                a.name() == "ThreeDAxialAttention") ++sub;
        } catch (...) {}
        // parameters() returns 15
        ++total_sub;
        try {
            ThreeDAxialAttention a(8, 2, 3, 2);
            if (a.parameters().size() == 15 &&
                a.gradients().size() == 15) ++sub;
        } catch (...) {}
        check("T1 constructor (7 sub-checks)", sub == total_sub);
    }

    // ============================================================
    // Test 2: ThreeDAxialAttentionBlock constructor validation
    // ============================================================
    cout << "\n--- Test 2: ThreeDAxialAttentionBlock constructor ---\n";
    {
        int sub = 0, total_sub = 0;
        ++total_sub;
        try { ThreeDAxialAttentionBlock b(0, 2, 2, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try { ThreeDAxialAttentionBlock b(8, 0, 2, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try { ThreeDAxialAttentionBlock b(8, 2, 2, 0); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try {
            ThreeDAxialAttentionBlock b(8, 2, 3, 2, 1, 16);
            // 3D axial (15) + LN (2) + LN (2) + Dense (2) + Dense (2) = 23
            if (b.d_model() == 8 && b.ffn_dim() == 16 &&
                b.name() == "ThreeDAxialAttentionBlock" &&
                b.parameters().size() == 23) ++sub;
        } catch (...) {}
        check("T2 block constructor (4 sub-checks)", sub == total_sub);
    }

    // ============================================================
    // Test 3: ThreeDAxialAttentionModel constructor validation
    // ============================================================
    cout << "\n--- Test 3: ThreeDAxialAttentionModel constructor ---\n";
    {
        int sub = 0, total_sub = 0;
        ++total_sub;
        try { ThreeDAxialAttentionModel m(0, 8, 3, 2, 3, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try { ThreeDAxialAttentionModel m(4, 0, 3, 2, 3, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try { ThreeDAxialAttentionModel m(4, 8, 0, 2, 3, 2); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try { ThreeDAxialAttentionModel m(4, 8, 3, 2, 3, 0); }
        catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try {
            ThreeDAxialAttentionModel m(4, 8, 3, 2, 3, 2, 0);
            // num_blocks = 0 throws
        } catch (const std::invalid_argument&) { ++sub; }
        ++total_sub;
        try {
            ThreeDAxialAttentionModel m(4, 8, 3, 2, 3, 2);
            if (m.name() == "ThreeDAxialAttentionModel") ++sub;
        } catch (...) {}
        check("T3 model constructor (6 sub-checks)", sub == total_sub);
    }

    // ============================================================
    // Test 4: Forward shape + finiteness (H=2, W=3, T=2, d=8, 2 heads)
    // ============================================================
    cout << "\n--- Test 4: forward shape (H=2, W=3, T=2, d=8, 2 heads) ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        Tensor input = random_input(12, 8, 0.3);  // H*W*T = 12
        Tensor output = a.forward(input);
        bool ok_shape = (output.rows == 12 && output.cols == 8);
        bool ok_finite = true;
        double s = 0.0;
        for (size_t i = 0; i < output.data.size(); ++i) {
            if (!isfinite(output.data[i])) ok_finite = false;
            s += fabs(output.data[i]);
        }
        bool ok_nonzero = (s > 1e-6);
        check("T4 output shape (12, 8)", ok_shape);
        check("T4 output finite",         ok_finite);
        check("T4 output nonzero",        ok_nonzero);
    }

    // ============================================================
    // Test 5: H=1, W=1, T=1 degenerate (single token)
    // ============================================================
    cout << "\n--- Test 5: H=1 W=1 T=1 degenerate ---\n";
    {
        ThreeDAxialAttention a(4, 1, 1, 1, 1, false, false, false);
        Tensor input = random_input(1, 4, 0.3);
        Tensor output = a.forward(input);
        bool ok_shape = (output.rows == 1 && output.cols == 4);
        bool ok_finite = true;
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!isfinite(output.data[i])) ok_finite = false;
        check("T5 H=1 W=1 T=1 shape",   ok_shape);
        check("T5 H=1 W=1 T=1 finite",  ok_finite);
    }

    // ============================================================
    // Test 6: W=1, T=1 degenerate (H-only axis)
    // ============================================================
    cout << "\n--- Test 6: W=1 T=1 degenerate ---\n";
    {
        ThreeDAxialAttention a(8, 4, 1, 1, 2, true, true, true);
        Tensor input = random_input(4, 8, 0.3);
        Tensor output = a.forward(input);
        bool ok = (output.rows == 4 && output.cols == 8);
        for (size_t i = 0; i < output.data.size(); ++i)
            if (!isfinite(output.data[i])) ok = false;
        check("T6 W=1 T=1 (H-only axis) finite", ok);
    }

    // ============================================================
    // Test 7: H-axis mask — only same (j, t) keys contribute
    // ============================================================
    cout << "\n--- Test 7: H-axis mask (only same (j,t) keys) ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        Tensor input = random_input(12, 8, 0.3);
        a.forward(input);
        const Tensor& A = a.last_attn_h();
        bool ok = true;
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t ki = 0; ki < 2; ++ki) {
                        for (size_t kj = 0; kj < 3; ++kj) {
                            for (size_t kt = 0; kt < 2; ++kt) {
                                size_t k = flat_idx(ki, kj, kt, 3, 2);
                                if (kj != qj || kt != qt) {
                                    if (fabs(A[q][k]) > 1e-9) ok = false;
                                }
                            }
                        }
                    }
                }
            }
        }
        check("T7 H-axis only same (j,t)", ok);
    }

    // ============================================================
    // Test 8: W-axis mask — only same (i, t) keys contribute
    // ============================================================
    cout << "\n--- Test 8: W-axis mask (only same (i,t) keys) ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        Tensor input = random_input(12, 8, 0.3);
        a.forward(input);
        const Tensor& A = a.last_attn_w();
        bool ok = true;
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t ki = 0; ki < 2; ++ki) {
                        for (size_t kj = 0; kj < 3; ++kj) {
                            for (size_t kt = 0; kt < 2; ++kt) {
                                size_t k = flat_idx(ki, kj, kt, 3, 2);
                                if (ki != qi || kt != qt) {
                                    if (fabs(A[q][k]) > 1e-9) ok = false;
                                }
                            }
                        }
                    }
                }
            }
        }
        check("T8 W-axis only same (i,t)", ok);
    }

    // ============================================================
    // Test 9: T-axis mask — only same (i, j) keys contribute
    // ============================================================
    cout << "\n--- Test 9: T-axis mask (only same (i,j) keys) ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        Tensor input = random_input(12, 8, 0.3);
        a.forward(input);
        const Tensor& A = a.last_attn_t();
        bool ok = true;
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t ki = 0; ki < 2; ++ki) {
                        for (size_t kj = 0; kj < 3; ++kj) {
                            for (size_t kt = 0; kt < 2; ++kt) {
                                size_t k = flat_idx(ki, kj, kt, 3, 2);
                                if (ki != qi || kj != qj) {
                                    if (fabs(A[q][k]) > 1e-9) ok = false;
                                }
                            }
                        }
                    }
                }
            }
        }
        check("T9 T-axis only same (i,j)", ok);
    }

    // ============================================================
    // Test 10: Causal mask — per-axis causal blocks future-direction
    // ============================================================
    cout << "\n--- Test 10: causal mask (3 axes) ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        Tensor input = random_input(12, 8, 0.3);
        a.forward(input);
        const Tensor& Ah = a.last_attn_h();
        const Tensor& Aw = a.last_attn_w();
        const Tensor& At = a.last_attn_t();
        bool ok_h = true, ok_w = true, ok_t = true;
        // H-axis causal: for each (qj, qt) only i' ≤ i attend; i' > i must be 0.
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t ki = 0; ki < 2; ++ki) {
                        size_t k = flat_idx(ki, qj, qt, 3, 2);
                        if (ki > qi && fabs(Ah[q][k]) > 1e-9) ok_h = false;
                    }
                }
            }
        }
        // W-axis causal: j' > j blocked.
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t kj = 0; kj < 3; ++kj) {
                        size_t k = flat_idx(qi, kj, qt, 3, 2);
                        if (kj > qj && fabs(Aw[q][k]) > 1e-9) ok_w = false;
                    }
                }
            }
        }
        // T-axis causal: t' > t blocked.
        for (size_t qi = 0; qi < 2; ++qi) {
            for (size_t qj = 0; qj < 3; ++qj) {
                for (size_t qt = 0; qt < 2; ++qt) {
                    size_t q = flat_idx(qi, qj, qt, 3, 2);
                    for (size_t kt = 0; kt < 2; ++kt) {
                        size_t k = flat_idx(qi, qj, kt, 3, 2);
                        if (kt > qt && fabs(At[q][k]) > 1e-9) ok_t = false;
                    }
                }
            }
        }
        check("T10 H causal (i'>i dropped)", ok_h);
        check("T10 W causal (j'>j dropped)", ok_w);
        check("T10 T causal (t'>t dropped)", ok_t);
    }

    // ============================================================
    // Test 11: Per-axis causal flag independence
    // ============================================================
    cout << "\n--- Test 11: causal_h/w/t independence ---\n";
    {
        // causal_h=false, causal_w=true, causal_t=true
        ThreeDAxialAttention a(8, 2, 3, 2, 2, false, true, true);
        Tensor input = random_input(12, 8, 0.5);
        a.forward(input);
        const Tensor& Ah = a.last_attn_h();
        const Tensor& Aw = a.last_attn_w();
        const Tensor& At = a.last_attn_t();
        // With causal_h=false: a "future" H coordinate (ki > qi) at the same (qj, qt)
        // should still have nonzero attention.
        size_t q = flat_idx(0, 0, 0, 3, 2);
        size_t k_past_h = flat_idx(0, 0, 0, 3, 2);
        size_t k_fut_h  = flat_idx(1, 0, 0, 3, 2);
        // ki = 0, so k_past_h is qi=0 / ki=0 (allowed); k_fut_h is qi=0 / ki=1.
        double past_h = fabs(Ah[q][k_past_h]);
        double fut_h  = fabs(Ah[q][k_fut_h]);
        check("T11 causal_h=false: future H nonzero", fut_h > 1e-9, fut_h);
        // sanity: past still works
        check("T11 causal_h=false: past H nonzero",  past_h > 1e-9, past_h);
        // W-axis with causal_w=true: ki=qj, kj=1 (future) at (qi, qj=0, qt=0) blocked.
        size_t k_fut_w = flat_idx(0, 1, 0, 3, 2);
        double fut_w = fabs(Aw[q][k_fut_w]);
        check("T11 causal_w=true: future W zero",    fut_w < 1e-9, fut_w);
        // T-axis with causal_t=true: t'=1 blocked at t=0.
        size_t k_fut_t = flat_idx(0, 0, 1, 3, 2);
        double fut_t = fabs(At[q][k_fut_t]);
        check("T11 causal_t=true: future T zero",    fut_t < 1e-9, fut_t);
    }

    // ============================================================
    // Test 12: FD input gradient
    // ============================================================
    cout << "\n--- Test 12: FD input gradient ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        mt19937 rng(123);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out_ana = a.forward(input);
        Tensor d_out   = l2_loss_grad(out_ana, target);
        a.backward(d_out, 0.0);
        Tensor ana = a.grad_input();
        Tensor num = fd_input_grad(a, input, target, 1e-5);
        double rel = tensor_rel_err(ana, num);
        check("T12 FD input grad rel_err < 1e-4", rel < 1e-4, rel);
    }

    // ============================================================
    // Test 13: FD W_q/k/v/o_h (boosted Q/K init scale)
    // ============================================================
    cout << "\n--- Test 13: FD W_*_h ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        // Boost Q/K init scale so the gradient through softmax chain is
        // well above the FD noise floor.
        mt19937 rng_boost(789);
        normal_distribution<double> boost_dist(0.0, 0.3);
        for (auto& v : a.W_q_h.data) v = boost_dist(rng_boost);
        for (auto& v : a.W_k_h.data) v = boost_dist(rng_boost);
        mt19937 rng(456);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out = a.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        a.zero_grad();
        a.backward(d_out, 0.0);
        std::vector<Tensor> ana;
        for (size_t k = 0; k < a.gradients().size(); ++k)
            ana.push_back(a.gradients()[k]->clone());
        const char* names[4] = {"W_q_h", "W_k_h", "W_v_h", "W_o_h"};
        for (size_t k = 0; k < 4; ++k) {
            a.zero_grad();
            Tensor num = fd_param_grad(a, k, input, target, 1e-5);
            double rel = tensor_rel_err(ana[k], num);
            double absd = tensor_max_abs_diff(ana[k], num);
            bool ok = (rel < 1e-3) || (absd < 5e-3);
            check(string("T13 FD ") + names[k] + " (rel=" + to_string(rel) +
                  " abs=" + to_string(absd) + ")", ok, rel);
        }
    }

    // ============================================================
    // Test 14: FD W_q/k/v/o_w
    // ============================================================
    cout << "\n--- Test 14: FD W_*_w ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        mt19937 rng_boost(789);
        normal_distribution<double> boost_dist(0.0, 0.3);
        for (auto& v : a.W_q_w.data) v = boost_dist(rng_boost);
        for (auto& v : a.W_k_w.data) v = boost_dist(rng_boost);
        mt19937 rng(457);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out = a.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        a.zero_grad();
        a.backward(d_out, 0.0);
        std::vector<Tensor> ana;
        for (size_t k = 0; k < a.gradients().size(); ++k)
            ana.push_back(a.gradients()[k]->clone());
        const char* names[4] = {"W_q_w", "W_k_w", "W_v_w", "W_o_w"};
        for (size_t k = 0; k < 4; ++k) {
            a.zero_grad();
            Tensor num = fd_param_grad(a, 4 + k, input, target, 1e-5);
            double rel = tensor_rel_err(ana[4 + k], num);
            double absd = tensor_max_abs_diff(ana[4 + k], num);
            bool ok = (rel < 1e-3) || (absd < 5e-3);
            check(string("T14 FD ") + names[k] + " (rel=" + to_string(rel) +
                  " abs=" + to_string(absd) + ")", ok, rel);
        }
    }

    // ============================================================
    // Test 15: FD W_q/k/v/o_t
    // ============================================================
    cout << "\n--- Test 15: FD W_*_t ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        mt19937 rng_boost(789);
        normal_distribution<double> boost_dist(0.0, 0.3);
        for (auto& v : a.W_q_t.data) v = boost_dist(rng_boost);
        for (auto& v : a.W_k_t.data) v = boost_dist(rng_boost);
        mt19937 rng(458);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out = a.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        a.zero_grad();
        a.backward(d_out, 0.0);
        std::vector<Tensor> ana;
        for (size_t k = 0; k < a.gradients().size(); ++k)
            ana.push_back(a.gradients()[k]->clone());
        const char* names[4] = {"W_q_t", "W_k_t", "W_v_t", "W_o_t"};
        for (size_t k = 0; k < 4; ++k) {
            a.zero_grad();
            Tensor num = fd_param_grad(a, 8 + k, input, target, 1e-5);
            double rel = tensor_rel_err(ana[8 + k], num);
            double absd = tensor_max_abs_diff(ana[8 + k], num);
            bool ok = (rel < 1e-3) || (absd < 5e-3);
            check(string("T15 FD ") + names[k] + " (rel=" + to_string(rel) +
                  " abs=" + to_string(absd) + ")", ok, rel);
        }
    }

    // ============================================================
    // Test 16: FD b_h, b_w, b_t (clean, no softmax chain)
    // ============================================================
    cout << "\n--- Test 16: FD b_h, b_w, b_t ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        mt19937 rng(2024);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out = a.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        a.zero_grad();
        a.backward(d_out, 0.0);
        std::vector<Tensor> ana;
        for (size_t k = 0; k < a.gradients().size(); ++k)
            ana.push_back(a.gradients()[k]->clone());
        const char* names[3] = {"b_h", "b_w", "b_t"};
        for (size_t k = 0; k < 3; ++k) {
            a.zero_grad();
            Tensor num = fd_param_grad(a, 12 + k, input, target, 1e-5);
            double rel = tensor_rel_err(ana[12 + k], num);
            check(string("T16 FD ") + names[k] + " rel_err < 1e-5",
                  rel < 1e-5, rel);
        }
    }

    // ============================================================
    // Test 17: ThreeDAxialAttentionBlock forward shape (with/without FFN)
    // ============================================================
    cout << "\n--- Test 17: block forward shape ---\n";
    {
        // no FFN
        ThreeDAxialAttentionBlock b1(8, 2, 3, 2, 2, 0);
        Tensor input = random_input(12, 8, 0.3);
        Tensor out1 = b1.forward(input);
        bool ok1 = (out1.rows == 12 && out1.cols == 8);
        for (size_t i = 0; i < out1.data.size(); ++i)
            if (!isfinite(out1.data[i])) ok1 = false;
        check("T17 block (no FFN) shape + finite", ok1);

        // with FFN
        ThreeDAxialAttentionBlock b2(8, 2, 3, 2, 2, 16);
        Tensor out2 = b2.forward(input);
        bool ok2 = (out2.rows == 12 && out2.cols == 8);
        for (size_t i = 0; i < out2.data.size(); ++i)
            if (!isfinite(out2.data[i])) ok2 = false;
        check("T17 block (with FFN) shape + finite", ok2);
    }

    // ============================================================
    // Test 18: ThreeDAxialAttentionModel forward shape
    // ============================================================
    cout << "\n--- Test 18: model forward shape ---\n";
    {
        ThreeDAxialAttentionModel m(4, 8, 3, 2, 3, 2, 2);
        Tensor input = random_input(12, 4, 0.3);  // H*W*T=12, d_input=4
        Tensor out = m.forward(input);
        bool ok = (out.rows == 12 && out.cols == 3);
        for (size_t i = 0; i < out.data.size(); ++i)
            if (!isfinite(out.data[i])) ok = false;
        check("T18 model forward (12,4)->(12,3)", ok);
    }

    // ============================================================
    // Test 19: ThreeDAxialAttentionBlock FD input gradient
    // ============================================================
    cout << "\n--- Test 19: block FD input gradient ---\n";
    {
        ThreeDAxialAttentionBlock b(8, 2, 3, 2, 2, 16);
        mt19937 rng(99);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        Tensor out = b.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        b.backward(d_out, 0.0);
        Tensor ana = b.grad_input();
        Tensor num = fd_input_grad_block(b, input, target, 1e-5);
        double rel = tensor_rel_err(ana, num);
        check("T19 block FD input grad rel_err < 1e-3", rel < 1e-3, rel);
    }

    // ============================================================
    // Test 20: End-to-end training
    // ============================================================
    cout << "\n--- Test 20: end-to-end training ---\n";
    {
        ThreeDAxialAttentionModel m(4, 8, 3, 2, 3, 2, 2);
        mt19937 rng(7);
        Tensor X = random_tensor(12, 4, 0.5, rng);
        Tensor Y = random_tensor(12, 3, 0.5, rng);
        double lr = 0.01;
        double init = -1.0, last = -1.0;
        for (int step = 0; step < 60; ++step) {
            Tensor out = m.forward(X);
            double L = l2_loss_value(out, Y);
            if (step == 0) init = L;
            last = L;
            Tensor d = l2_loss_grad(out, Y);
            m.zero_grad();
            m.backward(d, lr);
            m.update_weights(lr);
        }
        double reduction = (init - last) / max(init, 1e-12);
        check("T20 training reduces loss > 30%", reduction > 0.30, reduction);
    }

    // ============================================================
    // Test 21: update_weights moves all 15 parameter groups
    // ============================================================
    cout << "\n--- Test 21: update_weights moves all params ---\n";
    {
        ThreeDAxialAttention a(8, 2, 3, 2, 2, true, true, true);
        mt19937 rng(1234);
        Tensor input  = random_tensor(12, 8, 0.3, rng);
        Tensor target = random_tensor(12, 8, 0.3, rng);
        std::vector<Tensor> snap;
        for (auto* p : a.parameters()) snap.push_back(p->clone());
        Tensor out = a.forward(input);
        Tensor d_out = l2_loss_grad(out, target);
        a.backward(d_out, 0.0);
        a.update_weights(0.01);
        bool all_moved = true;
        for (size_t k = 0; k < snap.size(); ++k) {
            double d = tensor_max_abs_diff(snap[k], *a.parameters()[k]);
            if (d < 1e-10) all_moved = false;
        }
        check("T21 all 15 params moved by update", all_moved);
    }

    cout << "\n=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
