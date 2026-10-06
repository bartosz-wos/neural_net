// Tests for MultiHeadHyena (MultiHyena) — the multi-head long-convolution
// operator.
//
// Massaroli, Poli et al., "Laughing Hyena Distillery: Extracting Compact
// Recurrences From Convolutions", arXiv:2310.18780 §4:
//
//   1. "Given the projections q,k,v ∈ R^{L×D}, we split them into M chunks of
//       size N = D/M, q^m,k^m,v^m ∈ R^{L×N}."
//   2. "Each chunk is processed by a modified Hyena operator: first, we
//       perform the outer product of k^m and v^m along the spatial dimension,
//       z^m ≜ k^m ⊗ v^m ∈ R^{L×N×N}, apply a long convolution with filter h^m
//       to all N×N elements independently, then compute
//       y^m_t = (h^m * z^m)_t q^m_t, y^m ∈ R^{L×N}."
//   3. "Finally, we compose y^1,…,y^m into a single output y ∈ R^{L×D} via
//       concatenation."
//
// This is the layer MAD (arXiv:2403.17844 App. B.3.2) evaluates as its
// "Multi-Head Hyena" baseline (its ref [24]): heads=16, state dim=2, filter
// order=2, short filter order=3.
//
// THE TWO FACTS THE TESTS PIN, because each is a natural mistake:
//   (1) `h^m` is ONE filter applied to all N×N outer-product channels
//       ("to all N×N elements independently"). This is the layer's thesis —
//       §4 is motivated by filters being under-utilized. A per-(i,j) filter is
//       the natural wrong implementation.
//   (2) `·q^m_t` is the Hadamard product, not a matmul over N. The paper's
//       y^m ∈ R^{L×N} requires the element-wise reading.
//
// Conventions: sequences are (L, d_model) row-major Tensors. Inside the
// operator the L×N×N outer-product plane is flattened to (L, N*N) with index
// c = i*N + j.

#include "nn/nn.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const std::string& name) {
    tests_run++;
    if (cond) {
        tests_passed++;
        std::printf("  [PASS] %s\n", name.c_str());
    } else {
        tests_failed++;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
}

static double max_abs_diff(const Tensor& a, const Tensor& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            double d = std::fabs(a[i][j] - b[i][j]);
            if (d > m) m = d;
        }
    return m;
}

static double max_abs(const Tensor& a) {
    double m = 0.0;
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j) {
            double v = std::fabs(a[i][j]);
            if (v > m) m = v;
        }
    return m;
}

static bool all_finite(const Tensor& a) {
    for (size_t i = 0; i < a.rows; ++i)
        for (size_t j = 0; j < a.cols; ++j)
            if (!std::isfinite(a[i][j])) return false;
    return true;
}

static Tensor random_tensor(size_t r, size_t c, double scale) {
    return Tensor::random(r, c, scale);
}

static void zero_params(Layer* l) {
    for (auto* p : l->parameters())
        for (size_t i = 0; i < p->rows; ++i)
            for (size_t j = 0; j < p->cols; ++j) (*p)[i][j] = 0.0;
}

static double total_grad_norm(Layer* l) {
    double s = 0.0;
    for (auto* g : l->gradients())
        for (size_t i = 0; i < g->rows; ++i)
            for (size_t j = 0; j < g->cols; ++j) s += std::fabs((*g)[i][j]);
    return s;
}

// ---------------------------------------------------------------------------
// Independent reference: §4 written out by hand, NOT calling the impl.
// Given the head chunks q,k,v (each L×N) and the filter h (L×N*N, one shared
// across the whole outer-product plane), produce y (L×N).
// ---------------------------------------------------------------------------
static Tensor ref_head(const Tensor& q, const Tensor& k, const Tensor& v,
                       const Tensor& h) {
    size_t L = q.rows, N = q.cols;
    Tensor y(L, N);
    y.fill(0.0);
    for (size_t t = 0; t < L; ++t) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t c = i * N + j;
                // Causal conv, repo convention (hyena.cpp causal_conv_1d):
                //   y[t] = Σ_{s=0..t} h[s] * z[t-s]
                // h[s] is the filter at LAG s, so it multiplies z at t-s —
                // NOT z at s. Indexing h by absolute position instead of lag
                // is the classic convolution transpose.
                double w = 0.0;
                for (size_t s = 0; s <= t; ++s)
                    w += h[s][c] * k[t - s][i] * v[t - s][j];
                y[t][i] += w * q[t][i];
            }
        }
    }
    return y;
}

// ---------------------------------------------------------------------------
// FD harness. `loss(layer, x)` calls forward then returns 0.5*sum(grad^T out)
// so the analytical backward has a well-defined scalar to differentiate.
// ---------------------------------------------------------------------------
static Tensor make_grad(const Tensor& out) {
    Tensor g(out.rows, out.cols);
    g.fill(0.0);
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j) g[i][j] = 1.0 + 0.1 * ((i + j) % 3);
    return g;
}

// FD harness. `loss(layer, x)` calls forward then returns sum(grad^T out).
//
// NO 0.5 FACTOR. The repo's Layer::backward contract is the gradient of the
// loss L = Σ grad_output ⊙ forward(x) — that is exactly what Dense::backward
// computes (`grad_weights = grad_output^T * last_input`, no ½ anywhere). A
// conventional ½·Σ out² would halve every FD value while leaving the
// analytical full, giving a clean but entirely spurious factor-of-2 on every
// cell. Match the contract, not the textbook.
static double fd_loss(Layer* l, const Tensor& x, const Tensor& gy) {
    Tensor out = l->forward(x);
    double s = 0.0;
    for (size_t i = 0; i < out.rows; ++i)
        for (size_t j = 0; j < out.cols; ++j) s += gy[i][j] * out[i][j];
    return s;
}

// The §4 math (outer product → conv → Hadamard → concat) happens BEFORE
// out_proj_. Tests that check that math against the reference must therefore
// neutralize out_proj_, or they compare a pre-projection value against a
// post-projection one. Setting its weights to the identity and its bias to
// zero makes forward_from_qkv return exactly the concatenated head outputs.
static void make_out_proj_identity(MultiHeadHyenaOperator& op) {
    Dense& p = op.out_proj_;
    for (size_t i = 0; i < p.weights.rows; ++i)
        for (size_t j = 0; j < p.weights.cols; ++j)
            p.weights[i][j] = (i == j) ? 1.0 : 0.0;
    for (size_t j = 0; j < p.bias.cols; ++j) p.bias[0][j] = 0.0;
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor_validation() {
    std::printf("Test 1: constructor validation\n");

    bool threw = false;
    try { MultiHeadHyenaOperator op(0, 4, 2, 4); } catch (...) { threw = true; }
    check(threw, "operator throws on d_model=0");

    threw = false;
    try { MultiHeadHyenaOperator op(8, 0, 2, 4); } catch (...) { threw = true; }
    check(threw, "operator throws on seq_len=0");

    threw = false;
    try { MultiHeadHyenaOperator op(8, 4, 0, 4); } catch (...) { threw = true; }
    check(threw, "operator throws on num_heads=0");

    threw = false;
    try { MultiHeadHyenaOperator op(9, 4, 2, 4); } catch (...) { threw = true; }
    check(threw, "operator throws on d_model % num_heads != 0");

    // 9 IS divisible by 3, so it must NOT throw. Guard against a divisibility
    // check that rejects everything not divisible by 2.
    bool no_throw = true;
    try { MultiHeadHyenaOperator ok(9, 4, 3, 4); } catch (...) { no_throw = false; }
    check(no_throw, "d_model divisible by num_heads is accepted (9 % 3 == 0)");

    threw = false;
    try { MultiHeadHyenaOperator op(8, 4, 2, 0); } catch (...) { threw = true; }
    check(threw, "operator throws on filter_order=0");

    MultiHeadHyenaOperator op(8, 4, 2, 4);
    check(op.d_model() == 8 && op.seq_len() == 4, "operator accessors: d_model, seq_len");
    check(op.num_heads() == 2 && op.head_dim() == 4, "operator accessors: num_heads, head_dim");
    check(op.filter_order() == 4, "operator accessor: filter_order");
    check(op.name() == "MultiHeadHyenaOperator", "operator name()");

    threw = false;
    try { MultiHeadHyenaBlock b(8, 4, 0); } catch (...) { threw = true; }
    check(threw, "block throws on num_heads=0");

    MultiHeadHyenaBlock b(8, 4, 2);
    check(b.name() == "MultiHeadHyenaBlock", "block name()");
    check(b.d_model() == 8 && b.num_heads() == 2, "block accessors");

    threw = false;
    try { MultiHeadHyenaModel m(0, 8, 2, 2, 4, 2); } catch (...) { threw = true; }
    check(threw, "model throws on input_dim=0");

    threw = false;
    try { MultiHeadHyenaModel m(4, 8, 0, 2, 4, 2); } catch (...) { threw = true; }
    check(threw, "model throws on num_layers=0");

    threw = false;
    try { MultiHeadHyenaModel m(4, 8, 2, 0, 4, 2); } catch (...) { threw = true; }
    check(threw, "model throws on output_dim=0");

    MultiHeadHyenaModel m(4, 8, 3, 2, 4, 2);
    check(m.name() == "MultiHeadHyenaModel", "model name()");
    check(m.num_layers() == 3 && m.seq_len() == 4, "model accessors");
}

// ---------------------------------------------------------------------------
// Test 2: forward shape, finiteness
// ---------------------------------------------------------------------------
static void test_forward_shape() {
    std::printf("Test 2: forward shape / finite / nonzero\n");

    MultiHeadHyenaOperator op(8, 4, 2, 4);
    Tensor x = random_tensor(4, 8, 0.5);
    Tensor y = op.forward(x);
    check(y.rows == 4 && y.cols == 8, "operator forward preserves (L, d_model)");
    check(all_finite(y), "operator output finite");
    check(max_abs(y) > 1e-9, "operator output non-zero");

    // 4 heads, L=6
    MultiHeadHyenaOperator op4(8, 6, 4, 2);
    Tensor x4 = random_tensor(6, 8, 0.5);
    Tensor y4 = op4.forward(x4);
    check(y4.rows == 6 && y4.cols == 8 && all_finite(y4), "operator forward at 4 heads, L=6");

    // L=1 degenerate
    MultiHeadHyenaOperator op1(8, 1, 2, 4);
    Tensor x1 = random_tensor(1, 8, 0.5);
    Tensor y1 = op1.forward(x1);
    check(y1.rows == 1 && y1.cols == 8 && all_finite(y1), "L=1 degenerate forward");

    // wrong input width rejected
    bool threw = false;
    try { op.forward(random_tensor(4, 5, 0.5)); } catch (...) { threw = true; }
    check(threw, "operator throws on wrong input width");
}

// ---------------------------------------------------------------------------
// Test 3: the §4 three-step definition, against an INDEPENDENT reference.
//
// This is the test that pins the whole operator. It feeds known q,k,v and a
// known filter h through ref_head() and compares to the impl's cached head
// outputs. To do that the impl must expose its per-head caches.
// ---------------------------------------------------------------------------
static void test_matches_independent_reference() {
    std::printf("Test 3: matches independent hand-rolled reference\n");

    const size_t L = 4, D = 8, M = 2, N = 4;
    MultiHeadHyenaOperator op(D, L, M, 4);
    make_out_proj_identity(op);   // isolate the §4 math from out_proj_
    Tensor q = random_tensor(L, D, 0.5);
    Tensor k = random_tensor(L, D, 0.5);
    Tensor v = random_tensor(L, D, 0.5);

    Tensor y = op.forward_from_qkv(q, k, v);
    check(y.rows == L && y.cols == D, "forward_from_qkv shape");

    // For each head, the impl's cached filter must reproduce ref_head when we
    // use that same filter. This separates "outer product + conv + Hadamard"
    // (the §4 math) from "filter generation" (HyenaFilter's job, already
    // tested there) — a failure localizes to the right one of the two.
    double worst = 0.0;
    for (size_t m = 0; m < M; ++m) {
        Tensor h = op.last_filter(m);   // (L, N*N)
        check(h.rows == L && h.cols == N * N, "cached head filter shape (L, N*N)");
        Tensor qm(L, N), km(L, N), vm(L, N);
        for (size_t t = 0; t < L; ++t)
            for (size_t i = 0; i < N; ++i) {
                qm[t][i] = q[t][m * N + i];
                km[t][i] = k[t][m * N + i];
                vm[t][i] = v[t][m * N + i];
            }
        Tensor yref = ref_head(qm, km, vm, h);
        // impl output for this head
        Tensor ym(L, N);
        for (size_t t = 0; t < L; ++t)
            for (size_t i = 0; i < N; ++i) ym[t][i] = y[t][m * N + i];
        double d = max_abs_diff(yref, ym);
        if (d > worst) worst = d;
    }
    check(worst < 1e-12, "each head matches ref_head to <1e-12");

    // Cached outer product must equal k ⊗ v exactly (index c = i*N + j).
    double worst_z = 0.0;
    for (size_t m = 0; m < M; ++m) {
        Tensor z = op.last_outer(m);   // (L, N*N)
        for (size_t t = 0; t < L; ++t)
            for (size_t i = 0; i < N; ++i)
                for (size_t j = 0; j < N; ++j) {
                    double want = k[t][m * N + i] * v[t][m * N + j];
                    double got = z[t][i * N + j];
                    double d = std::fabs(want - got);
                    if (d > worst_z) worst_z = d;
                }
    }
    check(worst_z < 1e-15, "cached outer product equals k⊗v with index i*N+j");
}

// ---------------------------------------------------------------------------
// Test 4: the SHARED-FILTER contract (the layer's whole point)
//
// If each of the N×N outer-product channels got its OWN filter, the operator
// would not be MultiHyena — that is just plain Hyena on an outer-product
// featurization. Two checks:
//   (1) the cached filter really is one row-vector shared across the plane;
//   (2) a head whose filter we overwrite MANUALLY produces exactly the
//       reference computed with that same manual filter, proving every one of
//       the N² channels is driven by the same h.
// ---------------------------------------------------------------------------
static void test_shared_filter_contract() {
    std::printf("Test 4: shared-filter across the N×N outer-product plane\n");

    const size_t L = 4, D = 8, M = 2, N = 4;
    MultiHeadHyenaOperator op(D, L, M, 4);
    make_out_proj_identity(op);
    Tensor q = random_tensor(L, D, 0.5);
    Tensor k = random_tensor(L, D, 0.5);
    Tensor v = random_tensor(L, D, 0.5);
    op.forward_from_qkv(q, k, v);

    // Set a distinctive, NON-degenerate manual filter for head 0 and re-run.
    Tensor h(L, N * N);
    for (size_t t = 0; t < L; ++t)
        for (size_t c = 0; c < N * N; ++c)
            h[t][c] = std::sin(0.7 * (double)(t * N * N + c)) + 0.25;
    op.set_filter_for_test(0, h);
    Tensor y = op.forward_from_qkv(q, k, v);

    Tensor got = op.last_filter(0);
    check(max_abs_diff(got, h) == 0.0, "set_filter_for_test installs the filter bit-exactly");

    // Every channel (i,j) of head 0 must now follow the SAME h. We verify by
    // computing ref_head with h and comparing all of head 0's output.
    Tensor qm(L, N), km(L, N), vm(L, N);
    for (size_t t = 0; t < L; ++t)
        for (size_t i = 0; i < N; ++i) {
            qm[t][i] = q[t][i];          // head 0 → columns [0, N)
            km[t][i] = k[t][i];
            vm[t][i] = v[t][i];
        }
    Tensor yref = ref_head(qm, km, vm, h);
    Tensor ym(L, N);
    for (size_t t = 0; t < L; ++t)
        for (size_t i = 0; i < N; ++i) ym[t][i] = y[t][i];
    check(max_abs_diff(yref, ym) < 1e-12,
          "all N×N channels of head 0 driven by the single shared filter");

    // Discriminator: a per-channel filter (channel c uses only h[:, c] but
    // with a channel-dependent gain) must NOT match. We simulate the wrong
    // implementation by scaling the output per-channel and confirming the
    // shared-filter reference rejects it — i.e. the test above is not vacuous.
    Tensor h_wrong(L, N * N);
    for (size_t t = 0; t < L; ++t)
        for (size_t c = 0; c < N * N; ++c) h_wrong[t][c] = h[t][c] * (1.0 + 0.1 * (double)c);
    Tensor yref_wrong = ref_head(qm, km, vm, h_wrong);
    check(max_abs_diff(yref_wrong, ym) > 1e-9,
          "a per-channel-varying filter would NOT match (test is non-vacuous)");

    // THE GENERATED FILTER MUST ALSO BE SHARED. The pinned-filter checks above
    // bypass HyenaFilter entirely, so on their own they would not catch an
    // implementation that generates N² independent filter columns and then
    // broadcasts only when a test pins one. Assert directly on the filter the
    // operator generates for itself: h[l][c] must be identical for every c.
    {
        MultiHeadHyenaOperator op2(D, L, M, 4);   // no pin: real generated filter
        Tensor q2 = random_tensor(L, D, 0.5);
        Tensor k2 = random_tensor(L, D, 0.5);
        Tensor v2 = random_tensor(L, D, 0.5);
        op2.forward_from_qkv(q2, k2, v2);
        double spread = 0.0;
        bool all_fin = true;
        for (size_t m = 0; m < M; ++m) {
            Tensor hg = op2.last_filter(m);
            for (size_t t = 0; t < L; ++t) {
                double mn = hg[t][0], mx = hg[t][0];
                for (size_t c = 1; c < N * N; ++c) {
                    mn = std::min(mn, hg[t][c]);
                    mx = std::max(mx, hg[t][c]);
                    if (!std::isfinite(hg[t][c])) all_fin = false;
                }
                spread = std::max(spread, mx - mn);
            }
        }
        check(spread == 0.0,
              "GENERATED filter is identical across all N×N channels (spread == 0)");
        check(all_fin, "generated filter is finite");
    }
}

// ---------------------------------------------------------------------------
// Test 5: Hadamard, not matmul
//
// y_t = (h*z)_t q_t element-wise. If the impl instead did a matmul over N it
// would emit a scalar per head. Check: the Hadamard reading and the matmul
// reading give DIFFERENT outputs for the same input, and the impl matches the
// Hadamard one.
// ---------------------------------------------------------------------------
static void test_hadamard_not_matmul() {
    std::printf("Test 5: q enters as Hadamard, not matmul\n");

    const size_t L = 4, D = 4, M = 1, N = 4;
    MultiHeadHyenaOperator op(D, L, M, 4);
    make_out_proj_identity(op);
    Tensor q = random_tensor(L, D, 0.5);
    Tensor k = random_tensor(L, D, 0.5);
    Tensor v = random_tensor(L, D, 0.5);
    Tensor y = op.forward_from_qkv(q, k, v);

    Tensor h = op.last_filter(0);
    Tensor w(L, N * N);
    for (size_t t = 0; t < L; ++t)
        for (size_t c = 0; c < N * N; ++c)
            w[t][c] = std::sin(0.5 * (double)(t * N * N + c)) + 0.3;
    op.set_filter_for_test(0, w);
    y = op.forward_from_qkv(q, k, v);

    // Hadamard reference (what we expect).
    Tensor yref = ref_head(q, k, v, w);

    // Matmul reference: y[t] = q[t] · (sum_j (h*z)[t][i*N+j] j) — i.e. collapse
    // the N×N plane with the SAME q twice. Distinct value. Same lag convention
    // as ref_head: h[s] multiplies z at t-s.
    Tensor ymat(L, N);
    for (size_t t = 0; t < L; ++t) {
        for (size_t i = 0; i < N; ++i) {
            double acc = 0.0;
            for (size_t j = 0; j < N; ++j) {
                double s = 0.0;
                for (size_t ss = 0; ss <= t; ++ss)
                    s += w[ss][i * N + j] * k[t - ss][i] * v[t - ss][j];
                acc += s * q[t][j];
            }
            ymat[t][i] = acc;
        }
    }
    check(max_abs_diff(yref, ymat) > 1e-9, "Hadamard and matmul readings differ");
    check(max_abs_diff(yref, y) < 1e-12, "impl matches the Hadamard reading");
    check(max_abs_diff(ymat, y) > 1e-6, "impl does NOT match the matmul reading");
}

// ---------------------------------------------------------------------------
// Test 6: causality
//
// (h*z)_t depends only on z_s for s <= t, so perturbing k or v at position t
// must not change the output at any position < t.
// ---------------------------------------------------------------------------
static void test_causality() {
    std::printf("Test 6: causality\n");

    const size_t L = 5, D = 4, M = 1;
    MultiHeadHyenaOperator op(D, L, M, 4);
    make_out_proj_identity(op);
    Tensor q = random_tensor(L, D, 0.5);
    Tensor k = random_tensor(L, D, 0.5);
    Tensor v = random_tensor(L, D, 0.5);
    Tensor y0 = op.forward_from_qkv(q, k, v);

    // Perturb k at position 3 only.
    Tensor k2 = k.clone();
    k2[3][1] += 3.0;
    Tensor y1 = op.forward_from_qkv(q, k2, v);

    double leak = 0.0;
    for (size_t t = 0; t < 3; ++t)
        for (size_t i = 0; i < D; ++i)
            leak = std::max(leak, std::fabs(y1[t][i] - y0[t][i]));
    check(leak < 1e-14, "perturbing k at t=3 does not affect outputs at t<3");
    double moved = std::fabs(y1[3][1] - y0[3][1]) + std::fabs(y1[4][0] - y0[4][0]);
    check(moved > 1e-9, "perturbing k at t=3 DOES affect outputs at t>=3");
}

// ---------------------------------------------------------------------------
// Test 7: M=1 recovery and head independence
// ---------------------------------------------------------------------------
static void test_head_structure() {
    std::printf("Test 7: head structure\n");

    const size_t L = 4, D = 8;
    // Perturbing head 1's q must not change head 0's output columns.
    MultiHeadHyenaOperator op(D, L, 2, 4);
    make_out_proj_identity(op);
    Tensor q = random_tensor(L, D, 0.5);
    Tensor k = random_tensor(L, D, 0.5);
    Tensor v = random_tensor(L, D, 0.5);
    Tensor y0 = op.forward_from_qkv(q, k, v);

    Tensor q2 = q.clone();
    q2[1][4] += 5.0;   // column 4 is in head 1 (N=4, head1 = cols [4,8))
    Tensor y1 = op.forward_from_qkv(q2, k, v);

    double d0 = 0.0, d1 = 0.0;
    for (size_t t = 0; t < L; ++t) {
        for (size_t i = 0; i < 4; ++i) d0 = std::max(d0, std::fabs(y1[t][i] - y0[t][i]));
        for (size_t i = 4; i < 8; ++i) d1 = std::max(d1, std::fabs(y1[t][i] - y0[t][i]));
    }
    check(d0 < 1e-14, "head 0 output unchanged by a head-1 q perturbation");
    check(d1 > 1e-9, "head 1 output changed by a head-1 q perturbation");

    // M=1 (one head spanning D) is the degenerate valid config.
    MultiHeadHyenaOperator op1(D, L, 1, 4);
    Tensor ym = op1.forward_from_qkv(q, k, v);
    check(ym.rows == L && ym.cols == D && all_finite(ym), "M=1 single-head forward works");
}

// ---------------------------------------------------------------------------
// Test 8: determinism
// ---------------------------------------------------------------------------
static void test_determinism() {
    std::printf("Test 8: determinism\n");

    MultiHeadHyenaOperator op(8, 4, 2, 4);
    Tensor x = random_tensor(4, 8, 0.5);
    check(max_abs_diff(op.forward(x), op.forward(x)) == 0.0,
          "operator forward bit-exact across two calls");

    MultiHeadHyenaBlock b(8, 4, 2);
    Tensor z = random_tensor(4, 8, 0.5);
    check(max_abs_diff(b.forward(z), b.forward(z)) == 0.0,
          "block forward bit-exact across two calls");

    MultiHeadHyenaModel m(4, 8, 3, 2, 4, 2);
    Tensor w = random_tensor(4, 4, 0.5);
    check(max_abs_diff(m.forward(w), m.forward(w)) == 0.0,
          "model forward bit-exact across two calls");
}

// ---------------------------------------------------------------------------
// Test 9: parameter / gradient contract
// ---------------------------------------------------------------------------
static void test_parameter_contract() {
    std::printf("Test 9: parameter and gradient contract\n");

    MultiHeadHyenaOperator op(8, 4, 2, 4);
    auto p = op.parameters();
    auto g = op.gradients();
    check(p.size() == g.size(), "operator params().size() == grads().size()");
    check(p.size() > 0, "operator exposes parameters");
    bool shapes = true;
    for (size_t i = 0; i < p.size(); ++i)
        if (p[i]->rows != g[i]->rows || p[i]->cols != g[i]->cols) shapes = false;
    check(shapes, "every operator param has a shape-matched grad");

    MultiHeadHyenaBlock b(8, 4, 2);
    check(b.parameters().size() == b.gradients().size(), "block param/grad counts agree");

    MultiHeadHyenaModel m(4, 8, 3, 2, 4, 2);
    check(m.parameters().size() == m.gradients().size(), "model param/grad counts agree");

    Tensor x = random_tensor(4, 8, 0.5);
    Tensor y = op.forward(x);
    Tensor gy = make_grad(y);
    op.zero_grad();
    check(total_grad_norm(&op) == 0.0, "zero_grad clears every operator gradient");
    op.backward(gy, 0.0);
    check(total_grad_norm(&op) > 0.0, "operator backward produces non-zero gradients");
}

// ---------------------------------------------------------------------------
// Test 10: FD gradient checks — input and every parameter group
//
// Uses random non-uniform init (mandatory: uniform init would make the
// row-vs-column sums in the outer-product backward identical and let a
// transposed implementation pass vacuously).
//
// Criterion: |ana - num| / max(1, |ana|, |num|) < tol. The max(1,·) floor
// keeps the relative error meaningful when both sides sit near zero.
// ---------------------------------------------------------------------------

static void test_fd_gradients() {
    std::printf("Test 10: FD gradient checks\n");

    const size_t L = 5, D = 8, M = 2, N = 4;
    const double TOL = 1e-4;

    // --- FD on INPUT of the operator
    {
        MultiHeadHyenaOperator op(D, L, M, 4);
        Tensor x = random_tensor(L, D, 0.5);
        Tensor out = op.forward(x);
        Tensor gy = make_grad(out);

        // analytical
        op.zero_grad();
        op.backward(gy, 0.0);
        Tensor dgrad = op.last_input_grad();   // (L, D)

        // FD on a couple of input cells
        double worst = 0.0;
        size_t cells[4][2] = {{0,0},{1,3},{2,5},{4,7}};
        for (auto& cell : cells) {
            size_t r = cell[0], c = cell[1];
            double eps = 1e-6;
            double orig = x[r][c];
            x[r][c] = orig + eps; double lp = fd_loss(&op, x, gy);
            x[r][c] = orig - eps; double lm = fd_loss(&op, x, gy);
            x[r][c] = orig;
            double num = (lp - lm) / (2.0 * eps);
            double ana = dgrad[r][c];
            double scale = std::max(1.0, std::max(std::fabs(ana), std::fabs(num)));
            worst = std::max(worst, std::fabs(ana - num) / scale);
        }
        check(worst < TOL, "operator input gradient matches FD");
    }

    // --- FD on in_proj / out_proj weights (find them by shape)
    {
        MultiHeadHyenaOperator op(D, L, M, 4);
        Tensor x = random_tensor(L, D, 0.5);
        Tensor out = op.forward(x);
        Tensor gy = make_grad(out);
        op.zero_grad();
        op.backward(gy, 0.0);

        auto ps = op.parameters();
        auto gs = op.gradients();
        // in_proj.weights is (3D, D); out_proj.weights is (D, D)
        double worst_in = 0.0, worst_out = 0.0;
        for (size_t pi = 0; pi < ps.size(); ++pi) {
            bool is_in = (ps[pi]->rows == 3 * D && ps[pi]->cols == D);
            bool is_out = (ps[pi]->rows == D && ps[pi]->cols == D);
            if (!is_in && !is_out) continue;
            for (size_t k = 0; k < 2; ++k) {
                size_t r = (k == 0) ? 0 : 1;
                size_t c = (k == 0) ? 0 : 2;
                double eps = 1e-6;
                double orig = (*ps[pi])[r][c];
                (*ps[pi])[r][c] = orig + eps; double lp = fd_loss(&op, x, gy);
                (*ps[pi])[r][c] = orig - eps; double lm = fd_loss(&op, x, gy);
                (*ps[pi])[r][c] = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = (*gs[pi])[r][c];
                double scale = std::max(1.0, std::max(std::fabs(ana), std::fabs(num)));
                double err = std::fabs(ana - num) / scale;
                if (is_in) worst_in = std::max(worst_in, err);
                else worst_out = std::max(worst_out, err);
            }
        }
        check(worst_in < TOL, "operator in_proj weight gradient matches FD");
        check(worst_out < TOL, "operator out_proj weight gradient matches FD");
    }

    // --- FD on a per-head filter parameter (mlp_out_W of head 0)
    {
        MultiHeadHyenaOperator op(D, L, M, 4);
        Tensor x = random_tensor(L, D, 0.5);
        Tensor out = op.forward(x);
        Tensor gy = make_grad(out);
        op.zero_grad();
        op.backward(gy, 0.0);

        // HyenaFilter params: mlp_out_W is (N, filter_order) per head
        auto ps = op.parameters();
        auto gs = op.gradients();
        double worst = 0.0;
        for (size_t pi = 0; pi < ps.size(); ++pi) {
            if (ps[pi]->rows != (size_t)N || ps[pi]->cols != 4) continue;
            for (size_t k = 0; k < 2; ++k) {
                size_t r = k, c = (k == 0) ? 0 : 1;
                double eps = 1e-6;
                double orig = (*ps[pi])[r][c];
                (*ps[pi])[r][c] = orig + eps; double lp = fd_loss(&op, x, gy);
                (*ps[pi])[r][c] = orig - eps; double lm = fd_loss(&op, x, gy);
                (*ps[pi])[r][c] = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = (*gs[pi])[r][c];
                double scale = std::max(1.0, std::max(std::fabs(ana), std::fabs(num)));
                worst = std::max(worst, std::fabs(ana - num) / scale);
            }
            break;  // only head 0's copy needs checking; identical code path
        }
        check(worst < TOL, "per-head filter (mlp_out_W) gradient matches FD");
    }
}

// ---------------------------------------------------------------------------
// Test 11: block forward + block input FD + identity invariant
// ---------------------------------------------------------------------------
static void test_block() {
    std::printf("Test 11: block\n");

    MultiHeadHyenaBlock b(8, 4, 2);
    Tensor x = random_tensor(4, 8, 0.5);
    Tensor y = b.forward(x);
    check(y.rows == 4 && y.cols == 8, "block forward shape");
    check(all_finite(y) && max_abs(y) > 1e-9, "block output finite and non-zero");

    // Zeroed params ⇒ exact identity (residual only).
    MultiHeadHyenaBlock b2(8, 4, 2);
    zero_params(&b2);
    Tensor x2 = random_tensor(4, 8, 0.5);
    check(max_abs_diff(b2.forward(x2), x2) == 0.0, "zeroed block is exactly the identity");

    // Block input gradient FD.
    {
        MultiHeadHyenaBlock b3(8, 4, 2);
        Tensor z = random_tensor(4, 8, 0.5);
        Tensor out = b3.forward(z);
        Tensor gy = make_grad(out);
        b3.zero_grad();
        b3.backward(gy, 0.0);
        Tensor dg = b3.last_input_grad();
        double worst = 0.0;
        size_t cells[3][2] = {{0,0},{2,4},{3,7}};
        for (auto& cell : cells) {
            size_t r = cell[0], c = cell[1];
            double eps = 1e-6;
            double orig = z[r][c];
            z[r][c] = orig + eps; double lp = fd_loss(&b3, z, gy);
            z[r][c] = orig - eps; double lm = fd_loss(&b3, z, gy);
            z[r][c] = orig;
            double num = (lp - lm) / (2.0 * eps);
            double ana = dg[r][c];
            double scale = std::max(1.0, std::max(std::fabs(ana), std::fabs(num)));
            worst = std::max(worst, std::fabs(ana - num) / scale);
        }
        check(worst < 1e-4, "block input gradient matches FD");
    }
}

// ---------------------------------------------------------------------------
// Test 12: model forward + training reduces loss
// ---------------------------------------------------------------------------
static void test_model() {
    std::printf("Test 12: model\n");

    MultiHeadHyenaModel m(4, 8, 2, 3, 4, 2);
    Tensor x = random_tensor(4, 4, 0.5);
    Tensor y = m.forward(x);
    check(y.rows == 1 && y.cols == 3, "model forward shape (mean-pool → 1×output_dim)");
    check(all_finite(y) && max_abs(y) > 1e-9, "model output finite and non-zero");

    bool threw = false;
    try { m.forward(random_tensor(4, 5, 0.5)); } catch (...) { threw = true; }
    check(threw, "model throws on wrong input width");

    // Training reduces loss on a tiny regression task.
    MultiHeadHyenaModel tr(3, 8, 2, 1, 5, 2);
    Tensor xin = random_tensor(5, 3, 0.5);
    Tensor target = random_tensor(1, 1, 0.5);
    auto loss = [&]() {
        Tensor o = tr.forward(xin);
        double d = o[0][0] - target[0][0];
        return d * d;
    };
    double l0 = loss();
    Tensor gy(1, 1);
    gy.fill(0.0);
    for (int step = 0; step < 60; ++step) {
        Tensor o = tr.forward(xin);
        double d = o[0][0] - target[0][0];
        gy[0][0] = 2.0 * d;
        tr.zero_grad();
        tr.backward(gy, 0.0);
        tr.update_weights(0.05);
    }
    double l1 = loss();
    check(l1 < l0 * 0.9, "model training reduces loss >10%");
}

int main() {
    std::printf("=== MultiHeadHyena Tests ===\n\n");
    test_constructor_validation();
    test_forward_shape();
    test_matches_independent_reference();
    test_shared_filter_contract();
    test_hadamard_not_matmul();
    test_causality();
    test_head_structure();
    test_determinism();
    test_parameter_contract();
    test_fd_gradients();
    test_block();
    test_model();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}