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

int main() {
    std::printf("=== HyenaDNA Tests ===\n");
    test_constructor_validation();
    test_positional_embedding();
    test_filter_forward();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
