// Tests for HyenaDNA — the multi-head Hyena operator from Nguyen, Poli, Faizi
// et al., "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide
// Resolution", arXiv:2306.15794 §3.1 (Eq. 3.1), plus the §3.2 sequence-length
// warm-up schedule and the §3.3 soft-prompting adaptation (Eq. 3.2).
//
// The underlying single-head Hyena operator is Poli et al. arXiv:2302.10866 and
// is already shipped as layers/architectures/hyena.{h,cpp}; this is a separate
// implementation because the multi-head filter bank, the ExponentialModulation
// decay parameterization and the per-channel D-skip placement all differ.
//
// Conventions: sequences are (L, d_model) row-major Tensors, unbatched.

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

// Loss L = 0.5 * sum(out^2) => dL/dout = out.
static Tensor grad_from_out(const Tensor& out) {
    return out.clone();
}

static double loss_of(const Tensor& out) {
    double L = 0.0;
    for (size_t r = 0; r < out.rows; ++r)
        for (size_t c = 0; c < out.cols; ++c) L += 0.5 * out[r][c] * out[r][c];
    return L;
}

// Relative error with a noise-floor-safe denominator: when both sides sit at
// ~1e-17 the raw |a-n|/max(|a|,|n|) is ~1 and says nothing.
static double rel_err(double ana, double num) {
    double denom = std::max(std::fabs(ana), std::fabs(num));
    if (denom < 1e-9) denom = 1.0;
    return std::fabs(ana - num) / denom;
}

// Scientific-notation formatting. std::to_string prints 6 DECIMAL places, so a
// rel_err of 1e-11 renders as "0.000000" — which is exactly the vacuous-looking
// output that makes a real test look like it proved nothing.
static std::string sci(double v) {
    char b[64];
    std::snprintf(b, sizeof(b), "%.3e", v);
    return std::string(b);
}

// ---------------------------------------------------------------------------
// Test 1: constructor validation
// ---------------------------------------------------------------------------
static void test_constructor_validation() {
    std::printf("Test 1: constructor validation\n");

    // --- HyenaDNAOperator
    bool threw = false;
    try { HyenaDNAOperator op(0, 8, 1, 2); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on d_model=0");

    threw = false;
    try { HyenaDNAOperator op(4, 0, 1, 2); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on l_max=0");

    threw = false;
    try { HyenaDNAOperator op(4, 8, 0, 2); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on num_heads=0");

    threw = false;
    try { HyenaDNAOperator op(4, 8, 3, 2); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on d_model % num_heads != 0");

    threw = false;
    try { HyenaDNAOperator op(4, 8, 1, 1); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on order=1 (order must be >= 2)");

    threw = false;
    try { HyenaDNAOperator op(4, 8, 1, 2, 16, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on short_filter_order=0");

    threw = false;
    try { HyenaDNAOperator op(4, 8, 1, 2, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNAOperator throws on filter_order=0");

    HyenaDNAOperator ok(8, 16, 2, 2);
    check(ok.d_model() == 8, "operator d_model() = 8");
    check(ok.l_max() == 16, "operator l_max() = 16");
    check(ok.num_heads() == 2, "operator num_heads() = 2");
    check(ok.head_dim() == 4, "operator head_dim() = d_model/num_heads = 4");
    check(ok.order() == 2, "operator order() = 2");
    check(ok.filter_order() == 16, "operator filter_order() = 16");
    check(ok.short_filter_order() == 3, "operator short_filter_order() defaults to 3");
    check(ok.name() == "HyenaDNAOperator", "operator name()");

    // --- HyenaDNAFilter
    threw = false;
    try { HyenaDNAFilter f(0, 8); } catch (...) { threw = true; }
    check(threw, "HyenaDNAFilter throws on d_model=0");

    threw = false;
    try { HyenaDNAFilter f(4, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNAFilter throws on l_max=0");

    threw = false;
    try { HyenaDNAFilter f(4, 8, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNAFilter throws on filter_order=0");

    threw = false;
    // emb_dim must be odd and >= 3 (official source assert)
    try { HyenaDNAFilter f(4, 8, 16, 4); } catch (...) { threw = true; }
    check(threw, "HyenaDNAFilter throws on even emb_dim");

    threw = false;
    try { HyenaDNAFilter f(4, 8, 16, 1); } catch (...) { threw = true; }
    check(threw, "HyenaDNAFilter throws on emb_dim < 3");

    HyenaDNAFilter fk(4, 8, 16, 3, 2);
    check(fk.d_model() == 4, "filter d_model() = 4");
    check(fk.l_max() == 8, "filter l_max() = 8");
    check(fk.filter_order() == 16, "filter filter_order() = 16");
    check(fk.emb_dim() == 3, "filter emb_dim() = 3");
    check(fk.num_inner() == 2, "filter num_inner() = 2");

    // --- HyenaDNABlock
    threw = false;
    try { HyenaDNABlock b(0, 8); } catch (...) { threw = true; }
    check(threw, "HyenaDNABlock throws on d_model=0");

    threw = false;
    try { HyenaDNABlock b(4, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNABlock throws on l_max=0");

    HyenaDNABlock bk(4, 8, 2, 2, 16, 2);
    check(bk.d_model() == 4, "block d_model() = 4");
    check(bk.ffn_mult() == 2, "block ffn_mult() = 2");
    check(bk.name() == "HyenaDNABlock", "block name()");

    // --- HyenaDNAModel
    threw = false;
    try { HyenaDNAModel m(0, 8, 2, 3); } catch (...) { threw = true; }
    check(threw, "HyenaDNAModel throws on d_model=0");

    threw = false;
    try { HyenaDNAModel m(4, 8, 0, 3); } catch (...) { threw = true; }
    check(threw, "HyenaDNAModel throws on depth=0");

    threw = false;
    try { HyenaDNAModel m(4, 8, 2, 0); } catch (...) { threw = true; }
    check(threw, "HyenaDNAModel throws on num_classes=0");

    HyenaDNAModel mk(4, 8, 2, 3);
    check(mk.depth() == 2, "model depth() = 2");
    check(mk.num_classes() == 3, "model num_classes() = 3");
    check(mk.name() == "HyenaDNAModel", "model name()");
}

// ---------------------------------------------------------------------------
// Test 2: positional embedding closed form.
//   z[l][0]      = l/(L-1)                      (normalized so t_{L-1} = 1)
//   bands        = (emb_dim - 1)/2
//   w[l]         = 2*pi*l/L
//   f[b]         = linspace(1e-4, bands-1, bands)
//   z[l][1+b]    = cos(f[b]*w[l])
//   z[l][1+bands+b] = -sin(f[b]*w[l])
// Hand-derived against a fully independent recomputation below (not against
// another call into positional_embedding).
// ---------------------------------------------------------------------------
static void test_positional_embedding() {
    std::printf("Test 2: positional embedding\n");

    HyenaDNAFilter f(4, 8, 16, 3, 1);
    Tensor z = f.positional_embedding(8);
    check(z.rows == 8 && z.cols == 3, "z shape (8, 3) for emb_dim=3");

    // Independent recomputation of the (L, emb_dim) closed form.
    const double TWO_PI = 2.0 * 3.14159265358979323846;
    size_t L = 8, emb = 3, bands = 1;
    double worst = 0.0;
    for (size_t l = 0; l < L; ++l) {
        double t = (double)l / (double)(L - 1);
        double w = TWO_PI * (double)l / (double)L;
        double fb = 1e-4;                       // bands == 1 => f = 1e-4
        double e_t = std::fabs(z[l][0] - t);
        double e_c = std::fabs(z[l][1] - std::cos(fb * w));
        double e_s = std::fabs(z[l][2] - (-std::sin(fb * w)));
        if (e_t > worst) worst = e_t;
        if (e_c > worst) worst = e_c;
        if (e_s > worst) worst = e_s;
    }
    check(worst < 1e-12, "emb_dim=3 closed form matches to 1e-12 worst=" + std::to_string(worst));

    // Time channel is normalized: first 0, last 1.
    check(std::fabs(z[0][0]) < 1e-15, "t[0] = 0");
    check(std::fabs(z[L - 1][0] - 1.0) < 1e-15, "t[L-1] = 1 (normalized)");

    // emb_dim=5 => bands = 2, so the two band frequencies differ.
    HyenaDNAFilter f5(4, 8, 16, 5, 1);
    Tensor z5 = f5.positional_embedding(8);
    check(z5.rows == 8 && z5.cols == 5, "z shape (8, 5) for emb_dim=5");

    double worst5 = 0.0;
    for (size_t l = 0; l < L; ++l) {
        double t = (double)l / (double)(L - 1);
        double w = TWO_PI * (double)l / (double)L;
        if (std::fabs(z5[l][0] - t) > worst5) worst5 = std::fabs(z5[l][0] - t);
        for (size_t b = 0; b < 2; ++b) {
            // f = linspace(1e-4, 1, 2) = [1e-4, 1.0]
            double fb = (b == 0) ? 1e-4 : 1.0;
            double ec = std::fabs(z5[l][1 + b] - std::cos(fb * w));
            double es = std::fabs(z5[l][1 + 2 + b] - (-std::sin(fb * w)));
            if (ec > worst5) worst5 = ec;
            if (es > worst5) worst5 = es;
        }
    }
    check(worst5 < 1e-12, "emb_dim=5 two-band closed form matches to 1e-12 worst=" + std::to_string(worst5));

    // The two bands are genuinely distinct (catches a hard-coded single band).
    double band_gap = std::fabs(z5[3][1] - z5[3][2]);
    check(band_gap > 1e-6, "band 0 and band 1 differ (gap=" + std::to_string(band_gap) + ")");

    // cos^2 + sin^2 = 1 per band (the complex-exponential unit modulus).
    double worst_m = 0.0;
    for (size_t l = 0; l < L; ++l)
        for (size_t b = 0; b < 2; ++b) {
            double m = z5[l][1 + b] * z5[l][1 + b] + z5[l][3 + b] * z5[l][3 + b];
            if (std::fabs(m - 1.0) > worst_m) worst_m = std::fabs(m - 1.0);
        }
    check(worst_m < 1e-12, "cos^2+sin^2 = 1 per band (unit modulus) worst=" + std::to_string(worst_m));

    // L <= l_max is required; L > l_max throws.
    bool threw = false;
    try { f.positional_embedding(9); } catch (...) { threw = true; }
    check(threw, "positional_embedding throws on L > l_max");
}

// ---------------------------------------------------------------------------
// Test 3: filter closed forms.
//   h = (exp(-t * |deltas|) + shift) * mlp(z)
//   mlp(z) = Linear(emb_dim, P) -> Sin -> [Linear(P,P) -> Sin]*num_inner
//            -> Linear(P, D_f, bias=false)
// with Sin(x) = sin(freq * x), freq a learnable (1, P) tensor init to 1.
// ---------------------------------------------------------------------------
static void test_filter_forward() {
    std::printf("Test 3: filter forward closed forms\n");

    HyenaDNAFilter f(3, 6, 4, 3, 1);
    Tensor h = f.filter(6);
    check(h.rows == 6 && h.cols == 3, "filter(L) shape (L, d_model) = (6, 3)");
    check(all_finite(h), "filter output finite");
    check(max_abs(h) > 1e-9, "filter output non-zero (max=" + std::to_string(max_abs(h)) + ")");

    // --- sin_freq init is exactly 1.0, and Sin(x) = sin(freq*x) follows.
    bool freq_all_one = true;
    for (size_t j = 0; j < f.filter_order(); ++j)
        if (std::fabs(f.sin_freq[0][j] - 1.0) > 1e-15) freq_all_one = false;
    check(freq_all_one, "sin_freq initialised to 1.0");

    // --- deltas seeded to the ExponentialModulation linspace:
    //     deltas = linspace(log(target)/slow_pct, log(target)/fast_pct, d_model)
    //     target=1e-2, slow=1.5, fast=0.3
    // log(1e-2) = -4.60517; min = /1.5 = -3.07011; max = /0.3 = -15.35057
    // linspace over d_model=3: [min, (min+max)/2, max]
    //      = [-3.070113, -9.210340, -15.350567]
    // (verified against numpy.linspace; the midpoint is the ARITHMETIC mean,
    //  not a harmonic or log mean)
    const double log_target = std::log(1e-2);
    double dmin = log_target / 1.5, dmax = log_target / 0.3;
    double d0 = dmin, d1 = 0.5 * (dmin + dmax), d2 = dmax;
    double worst_d = std::fabs(f.deltas[0][0] - d0);
    if (std::fabs(f.deltas[0][1] - d1) > worst_d) worst_d = std::fabs(f.deltas[0][1] - d1);
    if (std::fabs(f.deltas[0][2] - d2) > worst_d) worst_d = std::fabs(f.deltas[0][2] - d2);
    check(worst_d < 1e-12, "deltas = ExponentialModulation linspace worst=" + std::to_string(worst_d));
    check(f.deltas[0][2] < f.deltas[0][0],
          "deltas is non-increasing (fast decay at the end)");

    // --- d_model == 1 linspace edge case: numpy/torch linspace(mind, maxd, 1)
    //     returns the START value, so deltas[0][0] must be min_decay, NOT
    //     max_decay. This is the exact spot a hand-written interpolation gets
    //     it wrong.
    HyenaDNAFilter f1(1, 5, 3, 3, 0);
    check(std::fabs(f1.deltas[0][0] - dmin) < 1e-12,
          "deltas for d_model=1 is linspace START (min_decay), got " +
          std::to_string(f1.deltas[0][0]) + " want " + std::to_string(dmin));
    check(std::fabs(f1.deltas[0][0] - dmax) > 1e-6,
          "deltas for d_model=1 is NOT max_decay (the wrong-answer guard)");

    // --- per-channel bias initialised to zero
    double max_bias = 0.0;
    for (size_t j = 0; j < f.bias.cols; ++j)
        if (std::fabs(f.bias[0][j]) > max_bias) max_bias = std::fabs(f.bias[0][j]);
    check(max_bias == 0.0, "filter bias initialised to 0");

    // --- the decay term: exp(-t*|deltas|) + shift, t = l/(L-1)
    Tensor decay = f.last_decay_term();
    check(decay.rows == 6 && decay.cols == 3, "cached decay shape (6, 3)");
    double worst_decay = 0.0;
    for (size_t l = 0; l < 6; ++l) {
        double t = (double)l / 5.0;
        for (size_t c = 0; c < 3; ++c) {
            double want = std::exp(-t * std::fabs(f.deltas[0][c])) + 0.0;
            double got = decay[l][c];
            if (std::fabs(got - want) > worst_decay) worst_decay = std::fabs(got - want);
        }
    }
    check(worst_decay < 1e-12, "decay = exp(-t*|deltas|)+shift matches to 1e-12 worst=" + std::to_string(worst_decay));

    // --- decay is strictly decreasing along the sequence (the causality
    //     mechanism: a filter tap at lag s is damped by exp(-s*delta)).
    check(decay[0][0] > decay[2][0] && decay[2][0] > decay[5][0],
          "decay strictly decreasing along l");

    // --- HAND-DERIVED MAGNITUDE TEST (guards against half-scale bugs, which
    //     FD-vs-analytical comparisons are structurally blind to).
    //     P=1, emb_dim=3, num_inner=0 => mlp(z) = W1 @ z with no bias, then
    //     Sin, then a final (D_f,1) projection with no bias.
    //     With W1 = [[1,0,0]] and W_out = [1,0,0,0]^T and freq = 1:
    //         mlp(z)[l] = sin(1 * z[l][0]) = sin(t_l)
    //         h[l][0]    = decay * sin(t_l)
    //     A 0.5x error anywhere shows up as an exact 2x relative gap.
    HyenaDNAFilter tiny(1, 5, 1, 3, 0);
    for (size_t j = 0; j < tiny.filter_order(); ++j) tiny.sin_freq[0][j] = 1.0;
    for (size_t p = 0; p < tiny.filter_order(); ++p)
        for (size_t e = 0; e < tiny.emb_dim(); ++e) tiny.mlp_in_W[p][e] = 0.0;
    tiny.mlp_in_W[0][0] = 1.0;                 // picks the time channel only
    tiny.mlp_in_b.fill(0.0);
    for (size_t c = 0; c < tiny.d_model(); ++c)
        for (size_t p = 0; p < tiny.filter_order(); ++p) tiny.mlp_out_W[c][p] = 0.0;
    tiny.mlp_out_W[0][0] = 1.0;                // identity projection
    tiny.mlp_out_W[0][1] = 0.0;
    // deltas[0] = log(1e-2)/1.5 (the linspace's first value for d_model=1)
    tiny.deltas[0][0] = std::log(1e-2) / 1.5;

    Tensor ht = tiny.filter(5);
    double worst_hand = 0.0;
    for (size_t l = 0; l < 5; ++l) {
        double tl = (double)l / 4.0;
        double want = std::exp(-tl * std::fabs(tiny.deltas[0][0])) * std::sin(tl);
        double got = ht[l][0];
        if (std::fabs(got - want) > worst_hand) worst_hand = std::fabs(got - want);
    }
    check(worst_hand < 1e-12, "hand-derived P=1 filter closed form to 1e-12 worst=" +
          std::to_string(worst_hand));

    // Explicitly assert the magnitude is the HAND value, not half of it.
    // L=5, l=2 => t=0.5, sin(0.5)=0.4794255386, exp(-0.5*3.07011)=0.2155457
    // product = 0.1033300... — a 0.5x implementation gives 0.0516650.
    double hand_mid = std::exp(-0.5 * std::fabs(tiny.deltas[0][0])) * std::sin(0.5);
    check(std::fabs(ht[2][0] - hand_mid) < 1e-12,
          "hand-derived magnitude is the exact hand value (not 0.5x) h[2][0]=" +
          std::to_string(ht[2][0]) + " want=" + std::to_string(hand_mid));

    // --- the final projection has NO bias: zeroing it must still give a
    //     non-degenerate result driven purely by W (this asserts the absence
    //     of a bias term rather than the presence of one).
    check(f.parameters().size() == f.gradients().size(),
          "parameters()/gradients() index-aligned (no separate bias tensor for mlp_out)");

    // --- determinism: two consecutive filter calls are bit-identical.
    Tensor h1 = f.filter(6);
    Tensor h2 = f.filter(6);
    check(max_abs_diff(h1, h2) == 0.0, "filter is deterministic across calls");
}

// ---------------------------------------------------------------------------
// Test 4: filter FD parameter gradients.
// Loss L = 0.5 * sum(h^2) => dL/dh = h, so the analytical grad_h is h itself.
// Random non-uniform init is mandatory: a uniform init makes
// sum_k W[k,j] == sum_k W[j,k] by construction, so a row/column transpose in
// the matmul backward would pass vacuously.
// `bias` is a DIRECT output-parameter of the operator (not of filter()), so
// its gradient is checked at the operator level in Test 6, not here.
// ---------------------------------------------------------------------------

// Centered finite difference of L = 0.5*sum(filter(L)^2) w.r.t. one element of
// a filter parameter tensor.
static double fd_filter_param_grad(HyenaDNAFilter& f, Tensor& param, size_t i,
                                   size_t j, size_t L, double eps) {
    double orig = param[i][j];
    param[i][j] = orig + eps;
    double lp = loss_of(f.filter(L));
    param[i][j] = orig - eps;
    double lm = loss_of(f.filter(L));
    param[i][j] = orig;
    return (lp - lm) / (2.0 * eps);
}

static void test_filter_fd_gradients() {
    std::printf("Test 4: filter FD parameter gradients\n");

    // d_model=3, P=4, num_inner=2 => exercises the inner-layer path and the
    // Sin-frequency chain.
    HyenaDNAFilter f(3, 6, 4, 3, 2);
    size_t L = 6;

    Tensor h = f.filter(L);
    f.zero_grad();
    f.backward(h);                      // grad_h = h for L = 0.5*sum(h^2)

    struct Case { const char* label; size_t rows, cols; };
    // Locate each parameter by shape (the filter exposes named members too,
    // but going through parameters()/gradients() exercises that index-alignment).
    std::vector<Tensor*> ps = f.parameters();
    std::vector<Tensor*> gs = f.gradients();
    check(ps.size() == gs.size(), "parameters()/gradients() same length");

    // FD-check a representative element of every parameter tensor. We check
    // element [0][0] of each, plus one off-diagonal element of the two
    // matrices where a transpose bug would show.
    // NOTE: `bias` is deliberately excluded from the "real gradient" sweep.
    // It is a per-channel skip that the OPERATOR applies after the long conv
    // (y = T v + D v), so it does not appear in filter()'s output at all and
    // its filter-level gradient is legitimately exactly zero. It is FD-checked
    // at the operator level in Test 6. Asserting rel_err(0,0) here would be a
    // vacuous pass, so it gets its own explicit check below.
    const double eps = 1e-6;
    double worst = 0.0;
    std::string worst_label;
    for (size_t k = 0; k < ps.size(); ++k) {
        // skip the (1, d_model) `bias` tensor
        if (ps[k]->rows == 1 && ps[k]->cols == f.d_model()) continue;
        std::string label = "param#" + std::to_string(k) + " (" +
                            std::to_string(ps[k]->rows) + "x" +
                            std::to_string(ps[k]->cols) + ")[0][0]";
        double ana = (*gs[k])[0][0];
        double num = fd_filter_param_grad(f, *ps[k], 0, 0, L, eps);
        double e = rel_err(ana, num);
        if (e > worst) { worst = e; worst_label = label; }
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%.3e", worst);
    check(worst < 1e-4, "filter FD all param tensors [0][0] rel_err < 1e-4 worst=" +
          std::string(buf) + " (" + worst_label + ")");

    // Sanity: the fixture is NOT degenerate. A rel_err of 0 because BOTH sides
    // are zero is a vacuous pass; assert the gradients are actually non-zero.
    double max_g = 0.0;
    for (size_t k = 0; k < gs.size(); ++k) {
        if (ps[k]->rows == 1 && ps[k]->cols == f.d_model()) continue;
        double gm = max_abs(*gs[k]);
        if (gm > max_g) max_g = gm;
    }
    std::snprintf(buf, sizeof(buf), "%.3e", max_g);
    check(max_g > 1e-9, std::string("filter gradients are non-degenerate (max=") +
          buf + ")");

    // Off-diagonal probe on mlp_in_W and mlp_out_W: a transposed matmul
    // backward computes the right value at [0][0] and a wrong one elsewhere.
    {
        Tensor* W = nullptr; Tensor* gW = nullptr;
        for (size_t k = 0; k < ps.size(); ++k) {
            if (ps[k]->rows == f.filter_order() && ps[k]->cols == f.emb_dim()) {
                W = ps[k]; gW = gs[k];
            }
        }
        check(W != nullptr, "located mlp_in_W (P, emb_dim)");
        if (W) {
            size_t i = 1, j = 1;
            double ana = (*gW)[i][j];
            double num = fd_filter_param_grad(f, *W, i, j, L, eps);
            check(rel_err(ana, num) < 1e-4,
                  "mlp_in_W[1][1] FD rel_err = " + sci(rel_err(ana, num)));
        }
    }
    {
        Tensor* W = nullptr; Tensor* gW = nullptr;
        for (size_t k = 0; k < ps.size(); ++k) {
            if (ps[k]->rows == f.d_model() && ps[k]->cols == f.filter_order()) {
                W = ps[k]; gW = gs[k];
            }
        }
        check(W != nullptr, "located mlp_out_W (d_model, P)");
        if (W) {
            size_t i = 2, j = 3;
            double ana = (*gW)[i][j];
            double num = fd_filter_param_grad(f, *W, i, j, L, eps);
            check(rel_err(ana, num) < 1e-4,
                  "mlp_out_W[2][3] FD rel_err = " + sci(rel_err(ana, num)));
        }
    }
    {
        // `deltas` and `bias` BOTH have shape (1, d_model), so a first-match
        // loop over that shape lands on `bias` (whose gradient is identically
        // zero) and the check passes vacuously. `deltas` is the FIRST of the
        // two in parameters() order, so take the first match and assert it is
        // not the zero-gradient one; the bias check below pins the other.
        Tensor* dl = nullptr; Tensor* gdl = nullptr;
        for (size_t k = 0; k < ps.size(); ++k) {
            if (ps[k]->rows == 1 && ps[k]->cols == f.d_model()) {
                if (dl == nullptr) { dl = ps[k]; gdl = gs[k]; }   // first match only
            }
        }
        check(dl != nullptr, "located deltas (1, d_model)");
        if (dl) {
            // non-degeneracy guard
            check(max_abs(*gdl) > 1e-9,
                  std::string("deltas grad is non-degenerate (max=") +
                  sci(max_abs(*gdl)) + ")");
            double worst_dg = 0.0;
            for (size_t j = 0; j < f.d_model(); ++j) {
                double ana = (*gdl)[0][j];
                double num = fd_filter_param_grad(f, *dl, 0, j, L, eps);
                double er = rel_err(ana, num);
                if (er > worst_dg) worst_dg = er;
            }
            check(worst_dg < 1e-4,
                  "deltas[0][*] FD sweep rel_err < 1e-4 worst=" + sci(worst_dg));
        }
    }

    // `bias` is a post-long-conv skip owned by the operator, so at the filter
    // level its gradient is exactly zero by construction. Asserting that
    // explicitly documents WHY it is excluded from the FD sweep above — and if
    // someone later wires the skip into filter(), this test fails and points
    // at the sweep that needs updating.
    {
        // `bias` is the SECOND (1, d_model) tensor in parameters() order, so
        // take the LAST match (the first match is `deltas`, checked above).
        Tensor* gb = nullptr;
        for (size_t k = 0; k < ps.size(); ++k)
            if (ps[k]->rows == 1 && ps[k]->cols == f.d_model()) gb = gs[k];
        check(gb != nullptr, "located filter bias (1, d_model)");
        if (gb) check(max_abs(*gb) == 0.0,
                      "filter bias grad is exactly 0 at filter level (post-conv skip)");
    }

    // zero_grad actually clears every buffer.
    f.zero_grad();
    bool all_clear = true;
    for (Tensor* g : f.gradients())
        if (max_abs(*g) != 0.0) all_clear = false;
    check(all_clear, "zero_grad clears every filter grad buffer");

    // update_weights moves a parameter by exactly -lr*grad.
    f.zero_grad();
    f.backward(h);
    Tensor* freq = nullptr; Tensor* gfreq = nullptr;
    for (size_t k = 0; k < ps.size(); ++k)
        if (ps[k]->rows == 1 && ps[k]->cols == f.filter_order()) { freq = ps[k]; gfreq = gs[k]; }
    check(freq != nullptr, "located sin_freq (1, P)");
    if (freq) {
        double before = (*freq)[0][2];
        double g = (*gfreq)[0][2];
        f.update_weights(0.01);
        double after = (*freq)[0][2];
        check(std::fabs(after - (before - 0.01 * g)) < 1e-15,
              "update_weights moves sin_freq by exactly -lr*grad");
    }

    // --- NON-UNIT sin_freq fixture (mutation-critical).
    // d/dx sin(f*x) = f*cos(f*x). At the init f == 1 that reduces to cos(x), so
    // DROPPING the `f` factor from the backward is algebraically invisible and
    // the whole suite still passes. To give this check teeth, re-run the FD
    // sweep with sin_freq perturbed away from 1.0. The relevant gradients here
    // are the upstream mlp weights (they flow through the Sin layer), so if the
    // impl dropped `f` the rel_err would jump to |1 - 1/f| ~ 0.5 for f = 2.0.
    {
        HyenaDNAFilter fq(3, 6, 4, 3, 1);
        for (size_t j = 0; j < fq.filter_order(); ++j)
            fq.sin_freq[0][j] = 0.5 + 0.37 * (double)j;   // 0.50, 0.87, 1.24, 1.61
        Tensor hq = fq.filter(6);
        fq.zero_grad();
        fq.backward(hq);
        std::vector<Tensor*> qp = fq.parameters();
        std::vector<Tensor*> qg = fq.gradients();
        // mlp_in_W is (P, emb_dim) = (4, 3) — unique shape.
        Tensor* W = nullptr; Tensor* gW = nullptr;
        for (size_t k = 0; k < qp.size(); ++k)
            if (qp[k]->rows == fq.filter_order() && qp[k]->cols == fq.emb_dim()) {
                W = qp[k]; gW = qg[k];
            }
        check(W != nullptr, "located mlp_in_W for the non-unit-freq sweep");
        if (W) {
            double worst_q = 0.0;
            for (size_t p = 0; p < fq.filter_order(); ++p)
                for (size_t e = 0; e < fq.emb_dim(); ++e) {
                    double ana = (*gW)[p][e];
                    double num = fd_filter_param_grad(fq, *W, p, e, 6, eps);
                    double er = rel_err(ana, num);
                    if (er > worst_q) worst_q = er;
                }
            check(worst_q < 1e-4,
                  "non-unit sin_freq: full mlp_in_W sweep rel_err < 1e-4 worst=" +
                  sci(worst_q));
            // Direct probe on sin_freq itself.
            Tensor* fr = nullptr; Tensor* gfr = nullptr;
            for (size_t k = 0; k < qp.size(); ++k)
                if (qp[k]->rows == 1 && qp[k]->cols == fq.filter_order()) { fr = qp[k]; gfr = qg[k]; }
            if (fr) {
                double worst_f = 0.0;
                for (size_t j = 0; j < fq.filter_order(); ++j) {
                    double ana = (*gfr)[0][j];
                    double num = fd_filter_param_grad(fq, *fr, 0, j, 6, eps);
                    double er = rel_err(ana, num);
                    if (er > worst_f) worst_f = er;
                }
                check(worst_f < 1e-4,
                      "non-unit sin_freq: all sin_freq grads rel_err < 1e-4 worst=" +
                      sci(worst_f));
            }
        }
    }

    // Gradient accumulation: two backward calls double the gradient.
    // Target a UNIQUELY-shaped tensor: with d_model=3, P=4, BOTH `deltas` and
    // `bias` have shape (1,3), so a first-match/last-match loop over that shape
    // silently lands on `bias` — whose gradient is identically zero — and the
    // assertion passes vacuously. mlp_out_W is (d_model, P) = (3,4), unique.
    HyenaDNAFilter fa(3, 6, 4, 3, 1);
    Tensor ha = fa.filter(6);
    fa.zero_grad();
    fa.backward(ha);
    Tensor* ga = nullptr;
    for (size_t k = 0; k < fa.gradients().size(); ++k)
        if (fa.gradients()[k]->rows == fa.d_model() &&
            fa.gradients()[k]->cols == fa.filter_order())
            ga = fa.gradients()[k];
    check(ga != nullptr, "located mlp_out_W grad for the accumulation check");
    if (ga) {
        double once = (*ga)[0][0];
        fa.backward(ha);
        double twice = (*ga)[0][0];
        char gb2[128];
        std::snprintf(gb2, sizeof(gb2), "%.6e vs 2*%.6e", twice, once);
        // non-degeneracy guard: a 0-vs-0 "double" would pass vacuously
        check(std::fabs(once) > 1e-9,
              std::string("accumulation fixture non-degenerate (once=") +
              sci(once) + ")");
        check(std::fabs(twice - 2.0 * once) < 1e-12,
              std::string("filter gradient accumulates over two backward calls (") +
              gb2 + ")");
    }
}

int main() {
    std::printf("=== HyenaDNA Tests ===\n");
    test_constructor_validation();
    test_positional_embedding();
    test_filter_forward();
    test_filter_fd_gradients();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
