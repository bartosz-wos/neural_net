// test_moba.cpp — MoBA: Mixture of Block Attention (Lu et al., arXiv:2502.13189)
//
// Tests:
//   1.  Constructor validation + accessors + 8-tensor param/grad contract
//   2.  Forward shape / finite + gate is exactly 0/1 + causal block routing
//   3.  Gate score is UNSCALED mean-pooled affinity (hand-derived, Eq. 6)
//   4.  Forward rejects bad shapes (cols != d_model, empty, N % block_size)
//   5.  Forward determinism (two consecutive calls bit-identical)
//   6.  Input gradient FD (N=8, num_heads=2 — exercises head row indexing)
//   7.  All 8 parameter gradients FD
//   8.  Unselected keys get EXACTLY zero K gradient (+ non-vacuity guard)
//   9.  zero_grad clears all 8 gradients
//  10.  update_weights moves all 8 parameters
//  11.  MoBABlock forward + input gradient FD
//  12.  MoBAModel forward + training reduces loss

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include "nn/layers/attention/moba.h"

using namespace std;

static int passed = 0, failed = 0;
static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else      { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}
static double rel_err(double a, double b) {
    double m = max(fabs(a), fabs(b));
    if (m < 1e-9) return fabs(a - b) / 1e-9;
    return fabs(a - b) / m;
}
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
    for (size_t i = 0; i < out.data.size(); ++i)
        g.data[i] = out.data[i] - tgt.data[i];
    return g;
}
// Deterministic pseudo-random fill (independent of global RNG state, so tests
// are reproducible regardless of what other layers consumed from the RNG).
static void fill_det(Tensor& t, unsigned seed, double scale) {
    unsigned s = seed;
    for (size_t i = 0; i < t.data.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        double u = ((s >> 8) & 0xFFFFFF) / double(0xFFFFFF);  // [0,1)
        t.data[i] = (2.0 * u - 1.0) * scale;
    }
}
static Tensor rand_tensor(size_t r, size_t c, unsigned seed, double scale) {
    Tensor t(r, c);
    fill_det(t, seed, scale);
    return t;
}
// Non-degenerate, ASYMMETRIC parameter set. Uniform or symmetric init hides
// row-vs-column transposition bugs (TDD skill).
static void randomize(MoBAAttention& a, unsigned base) {
    fill_det(a.W_q.weights, base + 1, 0.5);
    fill_det(a.W_k.weights, base + 2, 0.5);
    fill_det(a.W_v.weights, base + 3, 0.5);
    fill_det(a.W_o.weights, base + 4, 0.5);
    fill_det(a.W_q.bias,    base + 5, 0.2);
    fill_det(a.W_k.bias,    base + 6, 0.2);
    fill_det(a.W_v.bias,    base + 7, 0.2);
    fill_det(a.W_o.bias,    base + 8, 0.2);
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation + accessors + parameter contract
// ---------------------------------------------------------------------------
static void test_constructor() {
    cout << endl << "--- Test 1: constructor validation ---" << endl;
    bool t_dm=false, t_nh=false, t_div=false, t_bs=false, t_k=false, ok=true;
    try { MoBAAttention bad(0, 1);       } catch (const exception&) { t_dm  = true; }
    try { MoBAAttention bad(4, 0);       } catch (const exception&) { t_nh  = true; }
    try { MoBAAttention bad(4, 3);       } catch (const exception&) { t_div = true; }
    try { MoBAAttention bad(4, 1, 0);    } catch (const exception&) { t_bs  = true; }
    try { MoBAAttention bad(4, 1, 2, 0); } catch (const exception&) { t_k   = true; }
    try { MoBAAttention good(8, 2, 4, 1); } catch (const exception&) { ok = false; }
    check("d_model=0 throws", t_dm);
    check("num_heads=0 throws", t_nh);
    check("d_model%num_heads!=0 throws", t_div);
    check("block_size=0 throws", t_bs);
    check("top_k=0 throws", t_k);
    check("valid (8,2,4,1) constructs", ok);

    MoBAAttention a(8, 2, 4, 1);
    check("head_dim == d_model/num_heads", a.head_dim() == 4);
    check("inv_temp == 1/sqrt(head_dim)",
          rel_err(a.inv_temp(), 1.0 / sqrt(4.0)) < 1e-15);
    check("block_size accessor", a.block_size() == 4);
    check("top_k accessor", a.top_k() == 1);
    check("causal accessor default true", a.causal() == true);
    check("name()", a.name() == "MoBAAttention");

    check("parameters() contract: 8 tensors", a.parameters().size() == 8);
    check("gradients() contract: 8 tensors", a.gradients().size() == 8);
    bool shapes_ok = true;
    for (size_t p = 0; p < a.parameters().size(); ++p)
        if (a.parameters()[p]->rows != a.gradients()[p]->rows ||
            a.parameters()[p]->cols != a.gradients()[p]->cols) shapes_ok = false;
    check("every param/grad pair shape-matched", shapes_ok);
}

// ---------------------------------------------------------------------------
// Test 3 helper: gate score hand-computed straight from Eq. 6, WITHOUT the layer.
// Dense convention: y = x*W^T + b  =>  K[t,c] = sum_j x[t,j]*W_k[c,j] + b_k[c]
// ---------------------------------------------------------------------------
static double hand_gate_score(MoBAAttention& a, const Tensor& x,
                              size_t t, size_t b, size_t block_size) {
    double dot = 0.0;
    for (size_t c = 0; c < a.head_dim(); ++c) {
        // `a` is non-const here: Tensor's one-arg operator[] is non-const only,
        // so a const MoBAAttention& would force us to the 2-D subscript.
        double q = a.W_q.bias(0, c);
        for (size_t j = 0; j < x.cols; ++j) q += x(t, j) * a.W_q.weights(c, j);
        double ksum = 0.0;
        for (size_t s = b * block_size; s < (b + 1) * block_size; ++s) {
            double kv = a.W_k.bias(0, c);
            for (size_t j = 0; j < x.cols; ++j) kv += x(s, j) * a.W_k.weights(c, j);
            ksum += kv;
        }
        dot += q * (ksum / static_cast<double>(block_size));   // MEAN pool, not sum
    }
    return dot;                    // NO inv_temp -- the assertion's whole point
}

// Tensor has no (r, c, fill) constructor -- build and fill explicitly.
static Tensor filled(size_t r, size_t c, double v) {
    Tensor t(r, c);
    t.fill(v);
    return t;
}

// Access a single cached gate entry. last_gate() returns a const Tensor&, so we
// take a local copy-free reference and index through operator()(r, c).
static double gate_at(const Tensor& g, size_t row, size_t col) {
    return g(row, col);
}

// ---------------------------------------------------------------------------
// Test 2: forward shape / finite + gate invariants
// ---------------------------------------------------------------------------
static void test_forward_and_gate() {
    cout << endl << "--- Test 2: forward + gate invariants ---" << endl;
    MoBAAttention a(8, 2, 4, 1);            // N=8, block=4 -> 2 blocks
    randomize(a, 700);
    Tensor x = rand_tensor(8, 8, 31, 1.0);
    Tensor out = a.forward(x);
    check("forward shape (N, d_model)", out.rows == 8 && out.cols == 8);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) finite = false;
    check("forward finite", finite);

    check("gate shape (num_heads*N, n_blk)",
          a.last_gate().rows == 2 * 8 && a.last_gate().cols == 2);
    bool b01 = true;
    for (double v : a.last_gate().data) if (v != 0.0 && v != 1.0) b01 = false;
    check("gate is exactly 0/1", b01);

    // Query 0 lives in block 0: no historical blocks exist, so it can select
    // ONLY block 0. Any selection of block 1 would be a causality violation.
    check("query 0 selects only block 0 (no future blocks)",
          gate_at(a.last_gate(), 0, 0) == 1.0 && gate_at(a.last_gate(), 0, 1) == 0.0);
    // Query 5 lives in block 1: current block always forced on.
    check("query 5 (block 1) always selects current block 1",
          gate_at(a.last_gate(), 1 * 8 + 5, 1) == 1.0);
    // With top_k=1 and 1 historical block visible, |I| = 2 for t >= 4.
    size_t sel5 = 0;
    for (size_t b = 0; b < 2; ++b)
        if (gate_at(a.last_gate(), 1 * 8 + 5, b) != 0.0) ++sel5;
    check("query 5 selects exactly top_k + 1 = 2 blocks", sel5 == 2);
}

// ---------------------------------------------------------------------------
// Test 3: gate score is UNSCALED mean-pooled affinity
// ---------------------------------------------------------------------------
static void test_gate_is_unscaled() {
    cout << endl << "--- Test 3: gate score unscaled + mean-pooled ---" << endl;
    MoBAAttention a(4, 1, 2, 1);            // inv_temp = 1/sqrt(4) = 0.5 exactly
    fill_det(a.W_q.weights, 1, 1.0); fill_det(a.W_k.weights, 2, 1.0);
    fill_det(a.W_q.bias, 3, 0.0);    fill_det(a.W_k.bias, 4, 0.0);
    fill_det(a.W_v.weights, 5, 1.0); fill_det(a.W_o.weights, 6, 1.0);
    fill_det(a.W_v.bias, 7, 0.0);    fill_det(a.W_o.bias, 8, 0.0);
    Tensor x = rand_tensor(4, 4, 41, 1.0);  // N=4, block=2 -> 2 blocks
    a.forward(x);

    check("scores shape (num_heads*N, n_blk)",
          a.last_scores().rows == 4 && a.last_scores().cols == 2);
    double worst = 0.0;
    for (size_t b = 0; b < 2; ++b)
        worst = max(worst, rel_err(a.last_scores()(0, b),
                                   hand_gate_score(a, x, 0, b, 2)));
    cout << "  hand-derived vs impl gate, worst rel_err = " << scientific << worst << endl;
    check("gate score is UNSCALED mean-pooled affinity", worst < 1e-12);
}

// ---------------------------------------------------------------------------
// Test 4: forward input validation
// ---------------------------------------------------------------------------
static void test_forward_rejects_bad_shapes() {
    cout << endl << "--- Test 4: forward input validation ---" << endl;
    MoBAAttention a(8, 2, 4, 1);
    bool c1=false, c2=false, c3=false;
    try { a.forward(filled(4, 6, 1.0)); } catch (const exception&) { c1 = true; }
    try { a.forward(Tensor(0, 8)); }     catch (const exception&) { c2 = true; }
    try { a.forward(filled(6, 8, 1.0)); } catch (const exception&) { c3 = true; }
    check("forward rejects cols != d_model", c1);
    check("forward rejects empty input", c2);
    check("forward rejects N % block_size != 0", c3);
}

// ---------------------------------------------------------------------------
// Test 5: forward determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    cout << endl << "--- Test 5: forward determinism ---" << endl;
    MoBAAttention a(8, 2, 4, 1);
    randomize(a, 705);
    Tensor x = rand_tensor(8, 8, 43, 1.0);
    Tensor o1 = a.forward(x).clone();
    Tensor o2 = a.forward(x).clone();
    double worst = 0.0;
    for (size_t i = 0; i < o1.data.size(); ++i)
        worst = max(worst, fabs(o1.data[i] - o2.data[i]));
    cout << "  max |out1 - out2| = " << scientific << worst << endl;
    check("two consecutive forwards bit-identical", worst == 0.0);
}

// ---------------------------------------------------------------------------
// FD helpers (adapted from tests/test_stick_breaking.cpp lines 318-355)
//
// NOTE on the error criterion. These use a SCALE-NORMALIZED absolute error,
// not a per-entry relative error:
//
//   err = max |ana - num| / max |ana|   over the whole gradient tensor
//
// A per-entry rel_err is the wrong metric for an input gradient here because
// some entries are legitimately ~1e-5 while the largest is ~1.6. A central
// difference has an ABSOLUTE error floor (O(eps^2) truncation + O(eps_machine)
// roundoff), so on a 1e-5 entry that floor shows up as a ~1e-4 "relative error"
// even when the implementation is exact to 4e-9 absolute.
//
// Verified with an eps sweep (systematic-debugging 5c): worst per-entry rel_err
// was 6.2e-5 / 5.2e-7 / 6.7e-6 / 1.1e-4 / 7.2e-4 at eps = 1e-3/1e-4/1e-5/
// 1e-6/1e-7 — the U-shape is the FD noise signature, not a bug. The max
// ABSOLUTE difference across all entries was 4.4e-9 (machine precision).
//
// The normalized-absolute form is NOT blind to constant-factor bugs: a
// half-scale implementation still scores err ~0.5 because `num` is an
// independent finite difference while the scale comes from `ana`.
// ---------------------------------------------------------------------------
static double fd_input_check(MoBAAttention& a, Tensor x, const Tensor& tgt,
                             double eps = 1e-6) {
    a.zero_grad();
    Tensor out = a.forward(x);
    Tensor gi = a.backward(l2_grad(out, tgt), 0.0);
    double scale = 0.0;
    for (size_t i = 0; i < gi.data.size(); ++i) scale = max(scale, fabs(gi.data[i]));
    if (scale < 1e-12) return 0.0;               // gradient is ~0 everywhere
    double worst = 0.0;
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c) {
            Tensor xp = x.clone(); xp(r, c) += eps;
            Tensor xm = x.clone(); xm(r, c) -= eps;
            double num = (l2_loss(a.forward(xp), tgt) -
                          l2_loss(a.forward(xm), tgt)) / (2 * eps);
            worst = max(worst, fabs(gi(r, c) - num));
        }
    return worst / scale;
}
// `param` and `grad` are matched by POINTER, never by shape.
static double fd_param_check(MoBAAttention& a, Tensor* param, Tensor* grad,
                             const Tensor& x, const Tensor& tgt,
                             double eps = 1e-6) {
    a.zero_grad();
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, tgt), 0.0);
    Tensor ana = grad->clone();
    double scale = 0.0;
    for (size_t i = 0; i < ana.data.size(); ++i) scale = max(scale, fabs(ana.data[i]));
    if (scale < 1e-12) return 0.0;
    double worst = 0.0;
    for (size_t i = 0; i < param->data.size(); ++i) {
        double orig = param->data[i];
        param->data[i] = orig + eps;
        double lp = l2_loss(a.forward(x), tgt);
        param->data[i] = orig - eps;
        double lm = l2_loss(a.forward(x), tgt);
        param->data[i] = orig;
        worst = max(worst, fabs(ana.data[i] - (lp - lm) / (2 * eps)));
    }
    return worst / scale;
}

// ---------------------------------------------------------------------------
// Test 6 / 7: gradient FD checks
// ---------------------------------------------------------------------------
static void test_gradients() {
    cout << endl << "--- Test 6-7: gradient FD checks ---" << endl;
    MoBAAttention a(8, 2, 4, 1);            // N=8, 2 heads -> exercises head indexing
    randomize(a, 600);
    Tensor x = rand_tensor(8, 8, 51, 1.0);
    Tensor tgt = rand_tensor(8, 8, 53, 1.0);

    double e_in = fd_input_check(a, x, tgt);
    cout << "  input grad err (N=8, H=2, norm-abs) = " << scientific << e_in << endl;
    check("input gradient FD", e_in < 1e-6);

    double worst = 0.0;
    for (size_t p = 0; p < a.parameters().size(); ++p) {
        double e = fd_param_check(a, a.parameters()[p], a.gradients()[p], x, tgt);
        cout << "  param " << p << " grad err = " << e << endl;
        worst = max(worst, e);
    }
    check("all 8 parameter gradients FD", worst < 1e-6);
}

// ---------------------------------------------------------------------------
// Test 8: EXACT zero K-gradient on keys no query can see.
//
// N=8, d_model=4, block_size=4, top_k=1, num_heads=1 -> blocks {0:[0,4), 1:[4,8)}.
// Queries 0..3 live in block 0, so they have NO historical blocks visible:
// I_t = {0} for all t in 0..3, i.e. they see ONLY keys 0..3. Queries 4..7 do
// see block 1, so key rows 4..7 of dK would be non-zero from THEM alone.
//
// The isolation trick: drive grad_output to zero on rows 4..7 so only queries
// 0..3 contribute. Their key set is {0..3}, hence dK rows 4..7 must be
// EXACTLY 0.0 while rows 0..3 are non-zero.
//
// Note WHY this reads dK rather than grad_W_k: grad_W_k[c,j] = sum_t dk[t,c] *
// x[t,j], so its second index j is an INPUT FEATURE, not a key. Reading key
// visibility off grad_W_k would be a category error that happens to give a
// plausible-looking number.
// ---------------------------------------------------------------------------
static void test_unselected_keys_zero_k_grad() {
    cout << endl << "--- Test 8: unselected keys get exactly zero K grad ---" << endl;
    MoBAAttention a(4, 1, 4, 1);
    randomize(a, 640);
    Tensor x = rand_tensor(8, 4, 61, 1.0);
    Tensor out = a.forward(x);
    Tensor g(out.rows, out.cols);
    for (size_t t = 0; t < 8; ++t)
        for (size_t c = 0; c < 4; ++c)
            g(t, c) = (t < 4) ? (0.3 + 0.1 * c) : 0.0;    // rows 4..7 silent
    a.zero_grad();
    a.backward(g, 0.0);

    const Tensor& dK = a.last_dK();
    check("last_dK shape (N, d_model)", dK.rows == 8 && dK.cols == 4);
    double max_invisible = 0.0, max_visible = 0.0;
    for (size_t i = 0; i < 8; ++i)
        for (size_t c = 0; c < 4; ++c) {
            double m = std::fabs(dK(i, c));
            if (i < 4) max_visible  = max(max_visible,  m);
            else       max_invisible = max(max_invisible, m);
        }
    cout << "  max |dK| invisible rows = " << scientific << max_invisible
         << ", visible rows = " << max_visible << endl;
    check("unselected keys get EXACTLY zero K gradient", max_invisible == 0.0);
    // Non-vacuity: visible rows MUST be non-zero, else the assertion above
    // would pass for a backward that computes nothing at all.
    check("visible keys have NON-zero K gradient (test is non-vacuous)",
          max_visible > 1e-12);
}

// ---------------------------------------------------------------------------
// Test 9: zero_grad
// ---------------------------------------------------------------------------
static void test_zero_grad() {
    cout << endl << "--- Test 9: zero_grad ---" << endl;
    MoBAAttention a(8, 2, 4, 1);
    randomize(a, 650);
    Tensor x = rand_tensor(8, 8, 63, 1.0);
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, Tensor::zeros(out.rows, out.cols)), 0.0);
    a.zero_grad();
    bool clean = true;
    for (Tensor* g : a.gradients())
        for (double v : g->data) if (v != 0.0) clean = false;
    check("zero_grad clears all 8 gradients", clean);
}

// ---------------------------------------------------------------------------
// Test 10: update_weights moves every parameter
// ---------------------------------------------------------------------------
static void test_update_weights() {
    cout << endl << "--- Test 10: update_weights ---" << endl;
    MoBAAttention a(8, 2, 4, 1);
    randomize(a, 660);
    Tensor x = rand_tensor(8, 8, 65, 1.0);
    Tensor out = a.forward(x);
    a.backward(l2_grad(out, Tensor::zeros(out.rows, out.cols)), 0.0);

    const double lr = 0.1;
    vector<Tensor> before;
    for (Tensor* p : a.parameters()) before.push_back(p->clone());
    a.update_weights(lr);
    bool all_moved = true;
    size_t i = 0;
    for (Tensor* p : a.parameters()) {
        bool moved = false;
        for (size_t j = 0; j < p->data.size(); ++j)
            if (p->data[j] != before[i].data[j]) moved = true;
        if (!moved) all_moved = false;
        ++i;
    }
    check("update_weights moves all 8 parameters", all_moved);
}

// ---------------------------------------------------------------------------
// Test 11: MoBABlock
// ---------------------------------------------------------------------------
static double fd_input_check_block(MoBABlock& b, Tensor x, const Tensor& tgt,
                                   double eps = 1e-6) {
    b.zero_grad();
    Tensor out = b.forward(x);
    Tensor gi = b.backward(l2_grad(out, tgt), 0.0);
    double scale = 0.0;
    for (size_t i = 0; i < gi.data.size(); ++i) scale = max(scale, fabs(gi.data[i]));
    if (scale < 1e-12) return 0.0;
    double worst = 0.0;
    for (size_t r = 0; r < x.rows; ++r)
        for (size_t c = 0; c < x.cols; ++c) {
            Tensor xp = x.clone(); xp(r, c) += eps;
            Tensor xm = x.clone(); xm(r, c) -= eps;
            double num = (l2_loss(b.forward(xp), tgt) -
                          l2_loss(b.forward(xm), tgt)) / (2 * eps);
            worst = max(worst, fabs(gi(r, c) - num));
        }
    return worst / scale;
}

static void test_block() {
    cout << endl << "--- Test 11: MoBABlock ---" << endl;
    bool t_dm=false, t_nh=false, t_div=false, ok=true;
    try { MoBABlock bad(0, 2, 4, 1); } catch (const exception&) { t_dm  = true; }
    try { MoBABlock bad(8, 0, 4, 1); } catch (const exception&) { t_nh  = true; }
    try { MoBABlock bad(8, 3, 4, 1); } catch (const exception&) { t_div = true; }
    try { MoBABlock good(8, 2, 4, 1); } catch (const exception&) { ok = false; }
    check("MoBABlock d_model=0 throws", t_dm);
    check("MoBABlock num_heads=0 throws", t_nh);
    check("MoBABlock non-divisible throws", t_div);
    check("MoBABlock valid constructs", ok);

    MoBABlock b(8, 2, 4, 1, 16);
    Tensor x = rand_tensor(8, 8, 67, 1.0);
    Tensor out = b.forward(x);
    check("MoBABlock forward shape", out.rows == 8 && out.cols == 8);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) finite = false;
    check("MoBABlock forward finite", finite);

    Tensor tgt = rand_tensor(8, 8, 71, 0.3);
    double e = fd_input_check_block(b, x, tgt);
    cout << "  MoBABlock input grad err (norm-abs) = " << scientific << e << endl;
    check("MoBABlock input gradient FD", e < 1e-5);
}

// ---------------------------------------------------------------------------
// Test 12: MoBAModel + end-to-end training
// ---------------------------------------------------------------------------
static void test_model() {
    cout << endl << "--- Test 12: MoBAModel + training ---" << endl;
    bool t_in=false, t_dm=false, t_out=false, t_nb=false, t_div=false, ok=true;
        try { MoBAModel bad(0, 8, 2, 1, 4, 1, 2); } catch (const exception&) { t_in  = true; }
    try { MoBAModel bad(3, 0, 2, 1, 4, 1, 2); } catch (const exception&) { t_dm  = true; }
    try { MoBAModel bad(3, 8, 0, 1, 4, 1, 2); } catch (const exception&) { t_out = true; }
    try { MoBAModel bad(3, 8, 2, 0, 4, 1, 2); } catch (const exception&) { t_nb  = true; }
    try { MoBAModel bad(3, 8, 2, 1, 4, 1, 3); } catch (const exception&) { t_div = true; }
    try { MoBAModel good(3, 8, 2, 2, 4, 1, 2); } catch (const exception&) { ok = false; }
    check("MoBAModel input_dim=0 throws", t_in);
    check("MoBAModel d_model=0 throws", t_dm);
    check("MoBAModel output_dim=0 throws", t_out);
    check("MoBAModel num_blocks=0 throws", t_nb);
    check("MoBAModel non-divisible d_model/num_heads throws", t_div);
    check("MoBAModel valid constructs", ok);

    // Fixture: d_model=8, 2 blocks of 4, top_k=1, num_heads=2 (head indexing
    // exercised end-to-end through two blocks).
    MoBAModel m(3, 8, 2, 2, 4, 1, 2);
    Tensor x = rand_tensor(8, 3, 73, 1.0);
    Tensor out = m.forward(x);
    check("MoBAModel forward shape (N, output_dim)", out.rows == 8 && out.cols == 2);
    bool finite = true;
    for (double v : out.data) if (!std::isfinite(v)) finite = false;
    check("MoBAModel forward finite", finite);

    bool shapes_ok = true;
    for (size_t p = 0; p < m.parameters().size(); ++p)
        if (m.parameters()[p]->rows != m.gradients()[p]->rows ||
            m.parameters()[p]->cols != m.gradients()[p]->cols) shapes_ok = false;
    check("MoBAModel param/grad shapes match", shapes_ok);
    cout << "  MoBAModel parameter tensors = " << m.parameters().size() << endl;

    // Deterministic parameter fixture for the training test.
    //
    // MoBAModel's constructor uses Dense::init_weights, which draws from the
    // GLOBAL RNG — so a model built inside the test binary starts from
    // different weights than the same model built in a standalone probe, and
    // the loss curve differs (measured 70% vs 44% reduction for the SAME
    // lr/steps). A fixed threshold on that is fragile. Overwriting every
    // parameter with a deterministic LCG fill makes the run reproducible no
    // matter what ran before it.
    {
        unsigned s = 99991u;
        for (Tensor* q : m.parameters()) {
            for (size_t i = 0; i < q->data.size(); ++i) {
                s = s * 1664525u + 1013904223u;
                double u = ((s >> 8) & 0xFFFFFF) / double(0xFFFFFF);
                (*q)(i / q->cols, i % q->cols) = (2.0 * u - 1.0) * 0.3;
            }
        }
    }

    // Training: fit a fixed target so the loss must come down.
    //
    // 300 steps at lr=0.01 on the deterministic fixture above reduces the loss
    // by well over 50%. The step count is not arbitrary: the loss is
    // NON-MONOTONIC (LayerNorm + two blocks makes the surface rough at N=8),
    // so a short run samples one point on that wiggle rather than the trend.
    Tensor tgt = rand_tensor(8, 2, 79, 0.2);
    double l0 = l2_loss(m.forward(x), tgt);
    for (int step = 0; step < 300; ++step) {
        Tensor o = m.forward(x);
        m.zero_grad();
        m.backward(l2_grad(o, tgt), 0.0);
        m.update_weights(0.01);
    }
    double l1 = l2_loss(m.forward(x), tgt);
    cout << "  loss " << l0 << " -> " << l1 << " ("
         << (100.0 * (1 - l1 / l0)) << "% down)" << endl;
    check("MoBAModel training reduces loss > 50%", l1 < l0 * 0.5);
}

int main() {
    cout << "========================================" << endl;
    cout << "  MoBA Test Suite (arXiv:2502.13189)" << endl;
    cout << "========================================" << endl;

    test_constructor();
    test_forward_and_gate();
    test_gate_is_unscaled();
    test_forward_rejects_bad_shapes();
    test_determinism();
    test_gradients();
    test_unselected_keys_zero_k_grad();
    test_zero_grad();
    test_update_weights();
    test_block();
    test_model();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}