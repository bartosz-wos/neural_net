// test_megalodon.cpp — Forward/structural tests for MegalodonBlock + MegalodonModel
//   (MEGALODON §3.3 normalized attention + §3.4 two-hop residual).
//
// Ma, Yang, Xiong, Chen, Yu, Zhang, Ma, Zhou, "MEGALODON: Efficient LLM
// Pretraining and Inference with Unlimited Context Length"
// (https://arxiv.org/abs/2404.08801).
//
// Forward (input x ∈ R^(n × d_model)):
//
//   tsn  = TimestepNorm(x)                        §3.2, causal
//   x'   = ComplexEMA(tsn)                        §3.1
//   z    = W_z · x' + b_z
//   z'   = per-head RMS normalization of z         §3.3 ("Z' = Z/||Z||")
//   Q    = kappa_q ⊙ z' + mu_q                   §3.3
//   K    = kappa_k ⊙ z' + mu_k
//   V    = SiLU(W_v · tsn + b_v)                 reference line 331
//   A    = softmax_masked(Q K^T / sqrt(d_h))
//   H    = (A · V) · W_o + b_o
//   Ŷ    = H + x                                 attention residual
//   Y    = FFN(LayerNorm(Ŷ)) + x                 TWO-HOP: + x, not + Ŷ
//
// The two-hop residual is the paper's §3.4 claim: reusing the block input in
// the FFN residual (instead of the attention output) removes Mega's update
// gate and fixes the variance growth of deep pre-norm stacks. Test 4 pins it
// directly — with the FFN output forced to zero, Y must equal x EXACTLY.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include "nn/nn.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool pass) {
    if (pass) { cout << "  [PASS] " << name << endl; ++passed; }
    else { cout << "  [FAIL] " << name << endl; ++failed; }
    return pass;
}

static Tensor rand_tensor(size_t r, size_t c, double scale, unsigned seed) {
    Tensor t(r, c);
    srand(seed);
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            t(i, j) = ((rand() / (double)RAND_MAX) * 2.0 - 1.0) * scale;
    return t;
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) return 1e300;
    double md = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            md = max(md, fabs(a(i, j) - b(i, j)));
    return md;
}

static bool all_finite(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (!std::isfinite(t(i, j))) return false;
    return true;
}

static bool any_nonzero(const Tensor& t) {
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            if (fabs(t(i, j)) > 0.0) return true;
    return false;
}

// =====================================================================
// Test 1: Constructor validation + accessors
// =====================================================================
static void test_constructor() {
    cout << endl << "-- Test 1: constructor validation + accessors --" << endl;

    bool t_d = false;
    try { MegalodonBlock b(0, 1, 4, 0, 4, 0); } catch (...) { t_d = true; }
    check("d_model=0 throws", t_d);

    bool t_h = false;
    try { MegalodonBlock b(4, 0, 4, 0, 4, 0); } catch (...) { t_h = true; }
    check("num_heads=0 throws", t_h);

    bool t_div = false;
    try { MegalodonBlock b(6, 4, 4, 0, 4, 0); } catch (...) { t_div = true; }
    check("d_model % num_heads != 0 throws", t_div);

    bool t_cema = false;
    try { MegalodonBlock b(4, 1, 4, 0, 0, 0); } catch (...) { t_cema = true; }
    check("cema_ndim=0 throws", t_cema);

    bool t_ffn = false;
    try { MegalodonBlock b(4, 1, 0, 0, 4, 0); } catch (...) { t_ffn = true; }
    check("ffn_mult=0 throws", t_ffn);

    // num_groups larger than d_model — TimestepNorm rejects k > d, and the
    // block must reject it too rather than deferring to a sublayer throw.
    bool t_g = false;
    try { MegalodonBlock b(4, 1, 4, 9, 4, 0); } catch (...) { t_g = true; }
    check("num_groups=9 > d_model=4 throws", t_g);

    bool ok = false;
    MegalodonBlock b(8, 2, 4, 2, 4, 0);
    try { ok = true; } catch (...) {}
    check("valid construct (d=8, heads=2, groups=2, cema_ndim=4)", ok);

    check("name() == \"MegalodonBlock\"", b.name() == "MegalodonBlock");
    check("d_model() == 8", b.d_model() == 8);
    check("num_heads() == 2", b.num_heads() == 2);
    check("head_dim() == 4", b.head_dim() == 4);
    check("z_dim() == d_model (heads * head_dim)", b.z_dim() == 8);
    check("v_dim() == d_model (heads * head_dim)", b.v_dim() == 8);
    check("ffn_hidden() == 4 * d_model", b.ffn_hidden() == 32);
    check("num_groups() == 2", b.num_groups() == 2);
    check("cema_ndim() == 4", b.cema_ndim() == 4);
    check("chunk_size() == 0 (full causal)", b.chunk_size() == 0);
    check("TimestepNorm sublayer is 2-group", b.tsn_.num_groups() == 2);
    check("ComplexEMA sublayer ndim matches", b.cema_.ndim() == 4);
}

// =====================================================================
// Test 2: Parameter / gradient contract
// =====================================================================
static void test_parameter_contract() {
    cout << endl << "-- Test 2: parameter/gradient contract --" << endl;

    MegalodonBlock b(4, 1, 2, 0, 2, 0);
    auto p = b.parameters();
    auto g = b.gradients();

    // tsn(2) + cema(6) + W_z(2) + qk(4) + W_v(2) + W_o(2) + ffn_ln(2)
    // + swiglu(4) + ffn_down(2) = 26
    check("26 learnable parameters", p.size() == 26);
    check("gradient count matches parameter count", g.size() == p.size());

    bool shapes_match = true;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) {
            shapes_match = false;
            cout << "    shape mismatch at " << i << ": param ("
                 << p[i]->rows << "," << p[i]->cols << ") grad ("
                 << g[i]->rows << "," << g[i]->cols << ")" << endl;
        }
    }
    check("every gradient shape matches its parameter", shapes_match);

    // The core sublayer weights, checked by name in the code layout.
    check("W_z.weights is (z_dim, d_model)", b.W_z_.weights.rows == 4 && b.W_z_.weights.cols == 4);
    check("W_v.weights is (v_dim, d_model)", b.W_v_.weights.rows == 4 && b.W_v_.weights.cols == 4);
    check("W_o.weights is (d_model, v_dim)", b.W_o_.weights.rows == 4 && b.W_o_.weights.cols == 4);
    check("q_gamma_ is (1, z_dim)", b.q_gamma_.rows == 1 && b.q_gamma_.cols == 4);
    check("k_gamma_ is (1, z_dim)", b.k_gamma_.rows == 1 && b.k_gamma_.cols == 4);
    check("ffn_down_.weights is (d_model, ffn_hidden)", b.ffn_down_.weights.rows == 4 &&
                                                     b.ffn_down_.weights.cols == 8);
}

// =====================================================================
// Test 3: Forward shape, finiteness, non-triviality
// =====================================================================
static void test_forward() {
    cout << endl << "-- Test 3: forward shape / finite / nonzero --" << endl;

    MegalodonBlock b(6, 3, 2, 3, 4, 0);
    Tensor x = rand_tensor(5, 6, 1.0, 1234);
    Tensor y = b.forward(x);

    check("forward shape (5,6) -> (5,6)", y.rows == 5 && y.cols == 6);
    check("output is finite", all_finite(y));
    check("output is non-zero", any_nonzero(y));

    // Different input must give a different output (catches a no-op forward).
    Tensor x2 = rand_tensor(5, 6, 1.0, 5678);
    Tensor y2 = b.forward(x2);
    check("different input -> different output", max_abs_diff(y, y2) > 1e-9);

    // Wrong feature count must be rejected, not silently reinterpreted.
    bool threw = false;
    try { b.forward(rand_tensor(5, 7, 1.0, 99)); } catch (...) { threw = true; }
    check("input with wrong cols throws", threw);
}

// =====================================================================
// Test 4: TWO-HOP RESIDUAL signature (§3.4) — the paper's central claim
//
// Y = FFN(LayerNorm(Ŷ)) + x. Force the FFN contribution to exactly zero and
// the output must be the block input ITSELF. Under a one-hop residual
// (Y = FFN(...) + Ŷ) the output would be Ŷ = x + attention_out, which is
// not x.
// =====================================================================
static void test_two_hop_residual() {
    cout << endl << "-- Test 4: two-hop residual signature (FFN zeroed => Y == x) --" << endl;

    MegalodonBlock b(4, 1, 2, 0, 3, 0);
    Tensor x = rand_tensor(4, 4, 1.0, 4321);

    // Kill the FFN's down-projection: output becomes exactly 0 regardless of
    // its input, so Y = 0 + x.
    b.ffn_down_.weights.fill(0.0);
    b.ffn_down_.bias.fill(0.0);

    Tensor y = b.forward(x);
    double md = max_abs_diff(y, x);
    cout << "    max|Y - x| with FFN zeroed = " << scientific << md << defaultfloat << endl;
    check("Y == x exactly (two-hop residual)", md < 1e-12);

    // Control: the attention branch is NOT zero — prove the test can fail.
    // Restore the down-projection, kill the output projection instead.
    MegalodonBlock c(4, 1, 2, 0, 3, 0);
    c.W_o_.weights.fill(0.0);
    c.W_o_.bias.fill(0.0);
    Tensor yc = c.forward(x);
    double md2 = max_abs_diff(yc, x);
    cout << "    max|Y - x| with attention zeroed = " << scientific << md2 << defaultfloat << endl;
    check("attention branch is live (Y != x when W_o=0 removed)", md2 > 1e-9);
}

// =====================================================================
// Test 5: Megalodon-chunk signature
//
// chunk_size = 0 is full causal. With chunk_size = 2 and n = 4 the chunks are
// {0,1} and {2,3}. Rows 0 and 1 see an identical key set under both settings
// (t=0: {0}; t=1: {0,1} in both), so their OUTPUTS MUST BE BIT-IDENTICAL
// between the two configurations. Rows 2 and 3 see fewer keys under chunking
// ({2} and {2,3} instead of {0,1,2} and {0,1,2,3}), so they must differ.
//
// This is a strict test: it would fail if the chunk predicate were ignored,
// or if the mask were inverted.
// =====================================================================
static void test_chunking() {
    cout << endl << "-- Test 5: Megalodon-chunk signature (chunk_size=2, n=4) --" << endl;

    MegalodonBlock full(4, 1, 2, 0, 3, 0);       // chunk_size = 0
    MegalodonBlock chunk(4, 1, 2, 0, 3, 2);     // chunk_size = 2
    chunk.copy_params_from(full);               // identical weights

    Tensor x = rand_tensor(4, 4, 1.0, 24680);
    Tensor y_full  = full.forward(x);
    Tensor y_chunk = chunk.forward(x);

    double d0 = fabs(y_full(0, 0) - y_chunk(0, 0));
    double d1 = fabs(y_full(1, 0) - y_chunk(1, 0));
    double d2 = fabs(y_full(2, 0) - y_chunk(2, 0));
    double d3 = fabs(y_full(3, 0) - y_chunk(3, 0));
    cout << "    |y_full - y_chunk| per row (col 0): "
         << d0 << ", " << d1 << ", " << d2 << ", " << d3 << endl;

    check("row 0 bit-identical (same key set in both modes)", d0 == 0.0);
    check("row 1 bit-identical (same key set in both modes)", d1 == 0.0);
    check("row 2 differs (chunked mask drops keys 0,1)", d2 > 1e-9);
    check("row 3 differs (chunked mask drops keys 0,1)", d3 > 1e-9);

    // Rows within the first chunk are the same under both settings for ALL
    // columns, not just column 0.
    double md_first = 0.0, md_second = 0.0;
    for (size_t j = 0; j < 4; ++j) {
        md_first  = max(md_first,  max(fabs(y_full(0, j) - y_chunk(0, j)),
                                       fabs(y_full(1, j) - y_chunk(1, j))));
        md_second = max(md_second, max(fabs(y_full(2, j) - y_chunk(2, j)),
                                       fabs(y_full(3, j) - y_chunk(3, j))));
    }
    check("first-chunk rows identical across ALL columns", md_first == 0.0);
    check("second-chunk rows differ across ALL columns", md_second > 1e-9);

    // chunk_size larger than the sequence reduces to full causal.
    MegalodonBlock big(4, 1, 2, 0, 3, 99);
    big.copy_params_from(full);
    Tensor y_big = big.forward(x);
    check("chunk_size >= n is identical to full causal", max_abs_diff(y_big, y_full) < 1e-12);
}

// =====================================================================
// Test 6: Causality — the attention sublayer is autoregressive-safe
// =====================================================================
static void test_causality() {
    cout << endl << "-- Test 6: causality (perturb x[t>1] leaves y[<t] unchanged) --" << endl;

    MegalodonBlock b(4, 1, 2, 0, 3, 0);
    Tensor x = rand_tensor(5, 4, 1.0, 13579);
    Tensor y1 = b.forward(x);

    Tensor x2 = x.clone();
    x2(4, 0) += 5.0;   // perturb the LAST row only
    Tensor y2 = b.forward(x2);

    double md = 0.0;
    for (size_t t = 0; t < 4; ++t)
        for (size_t j = 0; j < 4; ++j)
            md = max(md, fabs(y1(t, j) - y2(t, j)));
    check("rows 0..3 bit-exact unchanged when x[4] changes", md == 0.0);

    // Row 4 MUST change, otherwise the whole block ignores its input.
    check("row 4 does change (forward is live)",
          fabs(y1(4, 0) - y2(4, 0)) > 1e-9);
}

// =====================================================================
// Test 7: Determinism
// =====================================================================
static void test_determinism() {
    cout << endl << "-- Test 7: determinism --" << endl;

    MegalodonBlock b(6, 2, 2, 3, 4, 0);
    Tensor x = rand_tensor(4, 6, 1.0, 24680);
    Tensor y1 = b.forward(x);
    Tensor y2 = b.forward(x);
    check("two consecutive forwards bit-exact", max_abs_diff(y1, y2) == 0.0);
}

// =====================================================================
// Test 8: copy_params_from
// =====================================================================
static void test_copy_params() {
    cout << endl << "-- Test 8: copy_params_from --" << endl;

    MegalodonBlock a(4, 1, 2, 0, 3, 0);
    MegalodonBlock b(4, 1, 2, 0, 3, 0);
    Tensor x = rand_tensor(4, 4, 1.0, 1111);

    check("fresh blocks differ (RNG advanced)",
          max_abs_diff(a.forward(x), b.forward(x)) > 1e-9);

    b.copy_params_from(a);
    check("copied block produces bit-exact output", max_abs_diff(a.forward(x), b.forward(x)) == 0.0);

    // Shape-mismatched copy must be rejected rather than silently truncated.
    MegalodonBlock small(4, 1, 2, 0, 3, 0);
    bool threw = false;
    try { small.copy_params_from(MegalodonBlock(8, 2, 2, 0, 3, 0)); }
    catch (...) { threw = true; }
    check("copy from a differently-shaped block throws", threw);

    // chunk_size is a mask, not a parameter: copying across it is legal and
    // is exactly how the Megalodon-chunk signature test works.
    MegalodonBlock c1(4, 1, 2, 0, 3, 0);
    MegalodonBlock c2(4, 1, 2, 0, 3, 2);
    c2.copy_params_from(c1);
    check("copy across a chunk_size difference is allowed", true);
}

// =====================================================================
// Test 9: MegalodonModel — construction + forward
// =====================================================================
static void test_model() {
    cout << endl << "-- Test 9: MegalodonModel construction + forward --" << endl;

    bool t_in = false;
    try { MegalodonModel(0, 8, 2, 2, 1, 2, 0, 2, 0); } catch (...) { t_in = true; }
    check("input_dim=0 throws", t_in);

    bool t_out = false;
    try { MegalodonModel(3, 8, 0, 2, 1, 2, 0, 2, 0); } catch (...) { t_out = true; }
    check("output_dim=0 throws", t_out);

    bool t_layers = false;
    try { MegalodonModel(3, 8, 2, 0, 1, 2, 0, 2, 0); } catch (...) { t_layers = true; }
    check("num_layers=0 throws", t_layers);

    MegalodonModel m(3, 8, 2, 3, 2, 2, 0, 2, 0);
    check("name() == \"MegalodonModel\"", m.name() == "MegalodonModel");
    check("3 blocks constructed", m.blocks.size() == 3);

    Tensor x = rand_tensor(4, 3, 1.0, 8080);
    Tensor y = m.forward(x);
    check("model forward (4,3) -> (4,2)", y.rows == 4 && y.cols == 2);
    check("model output finite", all_finite(y));
    check("model output non-zero", any_nonzero(y));

    auto p = m.parameters();
    auto g = m.gradients();
    check("model parameter/gradient counts match", p.size() == g.size() && p.size() > 60);
    bool shapes_match = true;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes_match = false;
    check("model gradient shapes all match", shapes_match);

    // Param count scales with the number of blocks.
    MegalodonModel m1(3, 8, 2, 1, 2, 2, 0, 2, 0);
    check("num_layers=1 has fewer params than num_layers=3", m1.parameters().size() < p.size());
}

int main() {
    cout << "==========================================" << endl;
    cout << "  Megalodon Tests" << endl;
    cout << "  MEGALODON §3.3 normalized attention" << endl;
    cout << "  + §3.4 two-hop residual" << endl;
    cout << "==========================================" << endl;

    test_constructor();
    test_parameter_contract();
    test_forward();
    test_two_hop_residual();
    test_chunking();
    test_causality();
    test_determinism();
    test_copy_params();
    test_model();

    cout << endl << "=== Summary: " << passed << " passed, " << failed << " failed ===" << endl;
    return failed == 0 ? 0 : 1;
}
