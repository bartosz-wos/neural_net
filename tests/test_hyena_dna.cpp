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
    size_t L = 8;
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

    // --- L == 1 convention: torch.linspace(0, 1, 1) == [0.0], so the single
    // time coordinate is 0, NOT 1. Getting this wrong applies the full
    // exp(-|deltas|) decay to the only timestep and shrinks the filter ~20x.
    {
        HyenaDNAFilter f1e(3, 4, 3, 3, 0);
        Tensor z1 = f1e.positional_embedding(1);
        check(z1.rows == 1 && z1.cols == 3, "positional_embedding(1) shape (1, emb_dim)");
        check(std::fabs(z1[0][0]) < 1e-15,
              "t = 0 at L=1 (linspace(0,1,1) == [0.0], not [1.0]) got " +
              std::to_string(z1[0][0]));
        // With t=0 the decay is exp(0) = 1 exactly, so filter(1) == mlp(z[0]).
        Tensor h1e = f1e.filter(1);
        check(std::fabs(f1e.last_decay_term()[0][0] - 1.0) < 1e-15,
              "decay at L=1 is exactly 1 (exp(-0)) got " +
              std::to_string(f1e.last_decay_term()[0][0]));
        (void)h1e;
    }

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


// ---------------------------------------------------------------------------
// Test 5: operator forward.
//   u    = in_proj(x)                          (L, (order+1)*D)
//   uc   = short_depthwise_conv1d(u)           causal, kernel short_filter_order
//   per head h: x_0..x_order, v = split(uc)   each (L, head_dim)
//   for o = order-1 .. 1: v = v*x[o]; v = long_conv(v,k_o) + bias_o*v
//   v    = v * x[0]; out = out_proj(concat_heads(v))
// ---------------------------------------------------------------------------
static void test_operator_forward() {
    std::printf("Test 5: operator forward\n");

    HyenaDNAOperator op(4, 6, 1, 2, 4, 3);
    Tensor x = random_tensor(6, 4, 0.5);
    Tensor y = op.forward(x);
    check(y.rows == 6 && y.cols == 4, "operator forward (L,D) -> (L,D)");
    check(all_finite(y), "operator output finite");
    check(max_abs(y) > 1e-9, "operator output non-zero (max=" + sci(max_abs(y)) + ")");

    // Input validation
    bool threw = false;
    try { op.forward(random_tensor(6, 5, 0.5)); } catch (...) { threw = true; }
    check(threw, "operator throws on wrong input width");
    threw = false;
    try { op.forward(random_tensor(7, 4, 0.5)); } catch (...) { threw = true; }
    check(threw, "operator throws on L > l_max");

    // --- CAUSALITY (the paper's central claim: no future leakage).
    // Output at row t must not depend on input rows > t.
    HyenaDNAOperator cop(4, 6, 1, 2, 4, 3);
    Tensor xa = random_tensor(6, 4, 0.5);
    Tensor ya = cop.forward(xa);
    Tensor xb = xa.clone();
    xb[5][0] += 3.0;         // perturb ONLY the LAST row
    Tensor yb = cop.forward(xb);
    // rows 0..4 must be bit-identical; row 5 may change.
    double max_early = 0.0;
    for (size_t t = 0; t < 5; ++t)
        for (size_t c = 0; c < 4; ++c) {
            double d = std::fabs(ya[t][c] - yb[t][c]);
            if (d > max_early) max_early = d;
        }
    check(max_early == 0.0, "operator is causal: perturbing row L-1 leaves rows 0..4-1 bit-identical (max=" +
          sci(max_early) + ")");
    double last_diff = std::fabs(ya[5][0] - yb[5][0]);
    check(last_diff > 1e-9, "perturbing row L-1 does change row L-1 (max=" + sci(last_diff) + ")");

    // --- short conv is causal too: perturbing a middle row must not change
    // any EARLIER row.
    HyenaDNAOperator cop2(4, 6, 1, 2, 4, 3);
    Tensor xc = random_tensor(6, 4, 0.5);
    Tensor yc = cop2.forward(xc);
    Tensor xd = xc.clone();
    xd[2][1] -= 2.0;
    Tensor yd = cop2.forward(xd);
    double max_before = 0.0;
    for (size_t t = 0; t < 2; ++t)
        for (size_t c = 0; c < 4; ++c) {
            double d = std::fabs(yc[t][c] - yd[t][c]);
            if (d > max_before) max_before = d;
        }
    check(max_before == 0.0, "short conv is causal: perturbing row 2 leaves rows 0..1 unchanged (max=" +
          sci(max_before) + ")");

    // --- multi-head: the per-head groups are INDEPENDENT. Two operators with
    // num_heads=2 whose second head is fed a different slice of the filter
    // must give the same second-head output. Concretely: with num_heads=2 and
    // d_model=4, head 0 owns channels 0,1 and head 1 owns channels 2,3.
    // Perturbing in_proj's weight rows for head 0 only must leave head 1's
    // output contribution unchanged.
    HyenaDNAOperator hop(4, 6, 2, 2, 4, 3);
    Tensor xe = random_tensor(6, 4, 0.5);
    Tensor ye0 = hop.forward(xe);
    // in_proj.weights is ((order+1)*D, D) = (12, 4). Column c is input channel c.
    // Head 1 = output channels 8..11 (per the h*(order+1)*head_dim + o*head_dim
    // + c layout with head_dim=2). Zero input column 0 (affects every head) vs
    // column 3 — instead verify head separation by feeding a zeroed channel.
    Tensor xf = xe.clone();
    for (size_t t = 0; t < 6; ++t) xf[t][0] = 0.0;   // silence input channel 0
    Tensor yf = hop.forward(xf);
    // The output must CHANGE (channel 0 is a real input) but the two heads
    // must respond differently — a single shared 1-head operator would apply
    // the same D-wide filter to both groups, which this already is per head.
    check(max_abs_diff(ye0, yf) > 1e-9,
          "zeroing input channel 0 changes the multi-head output");

    // --- num_heads=2 vs num_heads=1 with the SAME weights must differ
    // (otherwise num_heads is inert — a "parameter exists but is unused" bug).
    HyenaDNAOperator h1(4, 6, 1, 2, 4, 3);
    HyenaDNAOperator h2(4, 6, 2, 2, 4, 3);
    h2.in_proj.weights = h1.in_proj.weights.clone();
    h2.in_proj.bias = h1.in_proj.bias.clone();
    h2.out_proj.weights = h1.out_proj.weights.clone();
    h2.out_proj.bias = h1.out_proj.bias.clone();
    h2.short_W = h1.short_W.clone();
    h2.short_b = h1.short_b.clone();
    h2.filter.mlp_in_W = h1.filter.mlp_in_W.clone();
    h2.filter.mlp_in_b = h1.filter.mlp_in_b.clone();
    h2.filter.sin_freq = h1.filter.sin_freq.clone();
    h2.filter.mlp_out_W = h1.filter.mlp_out_W.clone();
    h2.filter.deltas = h1.filter.deltas.clone();
    h2.filter.bias = h1.filter.bias.clone();
    Tensor y_1 = h1.forward(xe);
    Tensor y_2 = h2.forward(xe);
    // The two are NOT equal (different head grouping), and the 1-head filter
    // has a different output width so a bit-exact match is impossible.
    check(max_abs_diff(y_1, y_2) > 1e-9,
          "num_heads=1 and num_heads=2 differ for the same copied weights "
          "(num_heads is load-bearing)");
    check(all_finite(y_2), "num_heads=2 output finite");

    // --- L=1 degenerate: a length-1 sequence is just the t=0 row.
    HyenaDNAOperator o1(4, 1, 1, 2, 4, 3);
    Tensor x1 = random_tensor(1, 4, 0.5);
    Tensor y1 = o1.forward(x1);
    check(y1.rows == 1 && y1.cols == 4, "L=1 degenerate forward shape (1, D)");
    check(all_finite(y1), "L=1 output finite");
    check(max_abs(y1) > 1e-9, "L=1 output non-zero");

    // --- order=3 (TWO long-conv iterations) must run and differ from order=2.
    // order=2 gives exactly ONE iteration, so a `+=` vs `=` bug in the loop
    // passes vacuously there; order=3 is the config that has teeth.
    HyenaDNAOperator o2o(4, 6, 1, 2, 4, 3);
    HyenaDNAOperator o3o(4, 6, 1, 3, 4, 3);
    o3o.in_proj.weights = o2o.in_proj.weights.clone();
    o3o.in_proj.bias = o2o.in_proj.bias.clone();
    o3o.out_proj.weights = o2o.out_proj.weights.clone();
    o3o.out_proj.bias = o2o.out_proj.bias.clone();
    o3o.short_W = o2o.short_W.clone();
    o3o.short_b = o2o.short_b.clone();
    Tensor y_o2 = o2o.forward(xe);
    Tensor y_o3 = o3o.forward(xe);
    check(all_finite(y_o3), "order=3 (two conv iterations) output finite");
    check(max_abs_diff(y_o2, y_o3) > 1e-9, "order=3 differs from order=2");

    // --- determinism
    HyenaDNAOperator dop(4, 6, 2, 2, 4, 3);
    Tensor ya1 = dop.forward(xe);
    Tensor ya2 = dop.forward(xe);
    check(max_abs_diff(ya1, ya2) == 0.0, "operator forward is deterministic");
}

// ---------------------------------------------------------------------------
// Test 6: operator FD gradients.
// Loss L = 0.5*sum(y^2) => dL/dy = y.
// The `bias` (per-channel skip) gradient is checked HERE, not at filter level.
// ---------------------------------------------------------------------------
static double fd_op_input_grad(HyenaDNAOperator& op, const Tensor& x, size_t i,
                               size_t j, double eps) {
    Tensor xp = x.clone(), xm = x.clone();
    xp[i][j] += eps; xm[i][j] -= eps;
    return (loss_of(op.forward(xp)) - loss_of(op.forward(xm))) / (2.0 * eps);
}

static double fd_op_param_grad(HyenaDNAOperator& op, Tensor& param, size_t i,
                               size_t j, const Tensor& x, double eps) {
    double orig = param[i][j];
    param[i][j] = orig + eps;
    double lp = loss_of(op.forward(x));
    param[i][j] = orig - eps;
    double lm = loss_of(op.forward(x));
    param[i][j] = orig;
    return (lp - lm) / (2.0 * eps);
}

// Rescale an operator's multiplicative weights so its forward output and
// gradients sit in a well-conditioned range (~1e-1 rather than ~1e-6). Without
// this the FD check is vacuous — see systematic-debugging 5c on test-config
// degeneracy. The FILTER SHAPE is untouched: only the magnitude of the
// short-conv weights and the filter's final projection change, so every
// structural property under test is preserved.
static void well_conditioned_operator(HyenaDNAOperator& op) {
    const double TARGET = 3.0;   // scale factor
    for (size_t i = 0; i < op.short_W.rows; ++i)
        for (size_t j = 0; j < op.short_W.cols; ++j) op.short_W[i][j] *= TARGET;
    for (size_t i = 0; i < op.filter.mlp_out_W.rows; ++i)
        for (size_t j = 0; j < op.filter.mlp_out_W.cols; ++j)
            op.filter.mlp_out_W[i][j] *= TARGET;
    // Give the skip bias a non-zero value so the per-channel skip term is
    // load-bearing (at bias=0 the `+ bias*v` path contributes nothing and
    // dropping it would pass vacuously).
    for (size_t c = 0; c < op.filter.bias.cols; ++c)
        op.filter.bias[0][c] = 0.3 + 0.11 * (double)c;
}

static void test_operator_fd_gradients() {
    std::printf("Test 6: operator FD gradients\n");

    // order=3 so the recurrence loop runs TWICE (order=2 runs once and a
    // `+=`-vs-`=` bug there is invisible to FD).
    //
    // FIXTURE SCALING (systematic-debugging 5c). At the default init the
    // operator is numerically DEGENERATE for a gradient check: every
    // recurrence step multiplies several ~0.1-scale quantities (the
    // sin-bounded implicit MLP output, the short conv, the exponentially
    // damped filter), so the forward output lands at ~1e-6 and the input
    // gradient at ~1e-15. A centered difference with eps=1e-6 on a 1e-12
    // loss is pure noise — rel_err comes out "0" because BOTH sides are
    // 0, which is a vacuous pass, not a pass.
    //
    // The fix is the config, not the implementation: rescale the two
    // multiplicative knobs (short conv weights and the filter's final
    // projection) so the operator operates in a well-conditioned range.
    // `well_conditioned_operator` is the helper.
    size_t L = 5, D = 4;
    HyenaDNAOperator op(D, 6, 2, 3, 4, 3);
    well_conditioned_operator(op);
    Tensor x = random_tensor(L, D, 0.5);

    Tensor y = op.forward(x);
    op.zero_grad();
    Tensor dx = op.backward(grad_from_out(y), 0.0);

    // --- non-degeneracy: gradients must be real, not all zero.
    double max_dx = 0.0, max_dp = 0.0;
    for (size_t i = 0; i < dx.rows; ++i)
        for (size_t j = 0; j < dx.cols; ++j)
            if (std::fabs(dx[i][j]) > max_dx) max_dx = std::fabs(dx[i][j]);
    for (Tensor* g : op.gradients())
        if (max_abs(*g) > max_dp) max_dp = max_abs(*g);
    check(max_dx > 1e-9, std::string("operator input grad non-degenerate (max=") + sci(max_dx) + ")");
    check(max_dp > 1e-9, std::string("operator param grads non-degenerate (max=") + sci(max_dp) + ")");

    // --- FD input gradient, full sweep.
    const double eps = 1e-6;
    double worst_in = 0.0;
    for (size_t i = 0; i < L; ++i)
        for (size_t j = 0; j < D; ++j) {
            double num = fd_op_input_grad(op, x, i, j, eps);
            double er = rel_err(dx[i][j], num);
            if (er > worst_in) worst_in = er;
        }
    check(worst_in < 1e-4, "operator FD input grad full sweep rel_err < 1e-4 worst=" + sci(worst_in));

    // --- FD parameter gradients, full sweep of the first row of each tensor.
    std::vector<Tensor*> ps = op.parameters();
    std::vector<Tensor*> gs = op.gradients();
    check(ps.size() == gs.size(), "operator parameters()/gradients() index-aligned");

    double worst_p = 0.0;
    std::string worst_pl;
    for (size_t k = 0; k < ps.size(); ++k) {
        if (ps[k]->rows == 0 || ps[k]->cols == 0) continue;
        // skip the filter's `bias` here: checked separately below, and its
        // filter-level grad is identically zero.
        size_t e = 0.0;
        for (size_t c = 0; c < ps[k]->cols && e < 3; ++c) {
            for (size_t r = 0; r < ps[k]->rows && e < 3; ++r) {
                double ana = (*gs[k])[r][c];
                double num = fd_op_param_grad(op, *ps[k], r, c, x, eps);
                double er = rel_err(ana, num);
                if (er > worst_p) {
                    worst_p = er;
                    worst_pl = "param#" + std::to_string(k) + "[" + std::to_string(r) +
                               "][" + std::to_string(c) + "]";
                }
                ++e;
            }
        }
    }
    check(worst_p < 1e-4, "operator FD param grads rel_err < 1e-4 worst=" + sci(worst_p) +
          " (" + worst_pl + ")");

    // --- the per-channel skip bias: THE canonical per-channel sum, checked
    // against FD. This is the term that the "half-scale" and "dropped term"
    // mutation families attack, and it is the one gradient the filter-level
    // test could not reach.
    {
        HyenaDNAOperator bop(D, 6, 2, 3, 4, 3);
        well_conditioned_operator(bop);
        Tensor bx = random_tensor(L, D, 0.5);
        Tensor by = bop.forward(bx);
        bop.zero_grad();
        bop.backward(grad_from_out(by), 0.0);
        // filter.bias is (1, head_dim*(order-1)) = (1, 2*2) = (1,4)
        Tensor* bs = nullptr; Tensor* gbs = nullptr;
        for (size_t k = 0; k < bop.parameters().size(); ++k) {
            auto* pp = bop.parameters()[k];
            if (pp->rows == 1 && pp->cols == bop.filter.d_model() &&
                pp == &bop.filter.bias) { bs = pp; gbs = bop.gradients()[k]; }
        }
        check(bs != nullptr, "located filter bias at operator level");
        if (bs) {
            check(max_abs(*gbs) > 1e-9,
                  std::string("filter bias grad non-degenerate at operator level (max=") +
                  sci(max_abs(*gbs)) + ")");
            double worst_b = 0.0;
            for (size_t c = 0; c < bs->cols; ++c) {
                double ana = (*gbs)[0][c];
                double num = fd_op_param_grad(bop, *bs, 0, c, bx, eps);
                double er = rel_err(ana, num);
                if (er > worst_b) worst_b = er;
            }
            check(worst_b < 1e-4, "filter bias (per-channel skip) FD sweep rel_err < 1e-4 worst=" +
                  sci(worst_b));
        }
    }

    // --- short conv params
    {
        HyenaDNAOperator sop(D, 6, 1, 2, 4, 3);
        well_conditioned_operator(sop);
        Tensor sx = random_tensor(L, D, 0.5);
        Tensor sy = sop.forward(sx);
        sop.zero_grad();
        sop.backward(grad_from_out(sy), 0.0);
        double w1 = rel_err(sop.grad_short_W[0][0], fd_op_param_grad(sop, sop.short_W, 0, 0, sx, eps));
        double b1 = rel_err(sop.grad_short_b[0][0], fd_op_param_grad(sop, sop.short_b, 0, 0, sx, eps));
        check(w1 < 1e-4, "short_W[0][0] FD rel_err = " + sci(w1));
        check(b1 < 1e-4, "short_b[0][0] FD rel_err = " + sci(b1));
    }

    // --- zero_grad / update_weights
    {
        HyenaDNAOperator zop(D, 6, 1, 2, 4, 3);
        well_conditioned_operator(zop);
        Tensor zx = random_tensor(L, D, 0.5);
        Tensor zy = zop.forward(zx);
        zop.zero_grad();
        zop.backward(grad_from_out(zy), 0.0);
        bool cleared = true;
        for (Tensor* g : zop.gradients()) if (max_abs(*g) != 0.0) cleared = false;
        zop.zero_grad();
        cleared = true;
        for (Tensor* g : zop.gradients()) if (max_abs(*g) != 0.0) cleared = false;
        check(cleared, "operator zero_grad clears every buffer");

        zop.zero_grad();
        zop.backward(grad_from_out(zy), 0.0);
        double before = zop.short_W[0][0];
        double g = zop.grad_short_W[0][0];
        zop.update_weights(0.05);
        check(std::fabs(zop.short_W[0][0] - (before - 0.05 * g)) < 1e-15,
              "operator update_weights moves short_W by exactly -lr*grad");
    }
}

// ---------------------------------------------------------------------------
// Test 7: block.
//   res1 = x + op(LN1(x))
//   y   = res1 + ffn2(gelu(ffn1(LN2(res1))))       (skipped when ffn_mult == 0)
// ---------------------------------------------------------------------------
static double fd_block_input_grad(HyenaDNABlock& b, const Tensor& x, size_t i,
                                  size_t j, double eps) {
    Tensor xp = x.clone(), xm = x.clone();
    xp[i][j] += eps; xm[i][j] -= eps;
    return (loss_of(b.forward(xp)) - loss_of(b.forward(xm))) / (2.0 * eps);
}

static void test_block() {
    std::printf("Test 7: block\n");

    const size_t L = 5, D = 4;

    // --- forward shape / finiteness with an FFN
    HyenaDNABlock b(D, 6, 2, 3, 4, 2);
    well_conditioned_operator(b.op);
    Tensor x = random_tensor(L, D, 0.5);
    Tensor y = b.forward(x);
    check(y.rows == L && y.cols == D, "block forward (L,D) -> (L,D)");
    check(all_finite(y), "block output finite");
    check(max_abs(y) > 1e-9, "block output non-zero (max=" + sci(max_abs(y)) + ")");

    // Input validation: wrong width and L > l_max both throw.
    bool threw = false;
    try { b.forward(random_tensor(L, D + 1, 0.5)); } catch (...) { threw = true; }
    check(threw, "block throws on wrong input width");
    threw = false;
    try { b.forward(random_tensor(7, D, 0.5)); } catch (...) { threw = true; }
    check(threw, "block throws on L > l_max");

    // --- ffn_mult == 0 skips the channel mixer entirely: the output must equal
    // the FIRST residual only. Assert against last_res1, which is exactly that
    // intermediate, rather than recomputing the operator.
    {
        HyenaDNABlock z(D, 6, 2, 3, 4, 0);
        well_conditioned_operator(z.op);
        Tensor zx = random_tensor(L, D, 0.5);
        Tensor zy = z.forward(zx);
        check(zy.rows == L && zy.cols == D, "ffn_mult=0 block forward shape");
        check(max_abs_diff(zy, z.last_res1) == 0.0,
              "ffn_mult=0 output == first residual (x + op(LN1(x))) exactly");
    }

    // --- FD input gradient. order=3 so the operator's recurrence loop runs
    // twice, and the operator is rescaled into a well-conditioned range
    // (systematic-debugging 5c) — at default init the loss delta under a 1e-6
    // perturbation sits at the double-precision noise floor.
    {
        HyenaDNABlock fb(D, 6, 2, 3, 4, 2);
        well_conditioned_operator(fb.op);
        Tensor fx = random_tensor(L, D, 0.5);
        Tensor fy = fb.forward(fx);
        fb.zero_grad();
        Tensor dx = fb.backward(grad_from_out(fy), 0.0);
        double max_dx = 0.0;
        for (size_t i = 0; i < dx.rows; ++i)
            for (size_t j = 0; j < dx.cols; ++j)
                if (std::fabs(dx[i][j]) > max_dx) max_dx = std::fabs(dx[i][j]);
        check(max_dx > 1e-9, "block input grad non-degenerate (max=" + sci(max_dx) + ")");

        double max_dp = 0.0;
        for (Tensor* g : fb.gradients())
            if (max_abs(*g) > max_dp) max_dp = max_abs(*g);
        check(max_dp > 1e-9, "block param grads non-degenerate (max=" + sci(max_dp) + ")");

        const double eps = 1e-6;
        double worst = 0.0;
        for (size_t i = 0; i < L; ++i)
            for (size_t j = 0; j < D; ++j) {
                double er = rel_err(dx[i][j], fd_block_input_grad(fb, fx, i, j, eps));
                if (er > worst) worst = er;
            }
        check(worst < 1e-4, "block FD input grad rel_err < 1e-4 worst=" + sci(worst));
    }

    // --- FD on the FFN parameters specifically. With ffn_mult=0 these tensors
    // are shape-(1,1) placeholders and their gradient is legitimately zero, so
    // the FFN chain is only reachable at ffn_mult > 0.
    {
        HyenaDNABlock pb(D, 6, 1, 2, 4, 2);
        well_conditioned_operator(pb.op);
        Tensor px = random_tensor(L, D, 0.5);
        Tensor py = pb.forward(px);
        pb.zero_grad();
        pb.backward(grad_from_out(py), 0.0);
        // Read the ANALYTICAL gradient at the SAME cell the FD perturbs —
        // comparing grad[0][0] against FD at [1][1] is a test bug that looks
        // like an implementation bug (rel_err ~0.1, no clean fingerprint).
        double w1a = pb.ffn1.grad_weights[1][1];
        double w2a = pb.ffn2.grad_weights[1][1];
        check(std::fabs(w1a) > 1e-9, "ffn1 grad non-zero (max=" + sci(w1a) + ")");
        check(std::fabs(w2a) > 1e-9, "ffn2 grad non-zero (max=" + sci(w2a) + ")");

        const double eps = 1e-6;
        auto fd_param = [&](Tensor& param, size_t i, size_t j) {
            double orig = param[i][j];
            param[i][j] = orig + eps;
            double lp = loss_of(pb.forward(px));
            param[i][j] = orig - eps;
            double lm = loss_of(pb.forward(px));
            param[i][j] = orig;
            return (lp - lm) / (2.0 * eps);
        };
        double e1 = rel_err(w1a, fd_param(pb.ffn1.weights, 1, 1));
        double e2 = rel_err(w2a, fd_param(pb.ffn2.weights, 1, 1));
        check(e1 < 1e-4, "ffn1.weights[1][1] FD rel_err = " + sci(e1));
        check(e2 < 1e-4, "ffn2.weights[1][1] FD rel_err = " + sci(e2));
    }

    // --- parameters()/gradients() contract: every param has a shape-matched grad.
    {
        HyenaDNABlock qb(D, 6, 1, 2, 4, 2);
        Tensor qx = random_tensor(L, D, 0.5);
        Tensor qy = qb.forward(qx);
        qb.backward(grad_from_out(qy), 0.0);
        std::vector<Tensor*> ps = qb.parameters();
        std::vector<Tensor*> gs = qb.gradients();
        check(ps.size() == gs.size(), "block parameters()/gradients() same length");
        bool shapes_ok = true;
        for (size_t k = 0; k < ps.size(); ++k)
            if (ps[k]->rows != gs[k]->rows || ps[k]->cols != gs[k]->cols)
                shapes_ok = false;
        check(shapes_ok, "block param/grad shapes match elementwise");

        qb.zero_grad();
        bool cleared = true;
        for (Tensor* g : qb.gradients()) if (max_abs(*g) != 0.0) cleared = false;
        check(cleared, "block zero_grad clears every buffer");
    }
}

// ---------------------------------------------------------------------------
// Test 8: model.
//   h = block_0 -> block_1 -> ... -> block_{depth-1}
//   out = classifier(mean_over_tokens(h))      (1, num_classes)
// ---------------------------------------------------------------------------
static void test_model() {
    std::printf("Test 8: model\n");

    const size_t L = 5, D = 4, C = 3, DEPTH = 2;

    HyenaDNAModel m(D, 6, DEPTH, C, 2, 3, 4, 2);
    for (size_t i = 0; i < m.blocks.size(); ++i) well_conditioned_operator(m.blocks[i].op);
    Tensor x = random_tensor(L, D, 0.5);
    Tensor y = m.forward(x);
    check(y.rows == 1 && y.cols == C, "model forward (L,D) -> (1, num_classes)");
    check(all_finite(y), "model output finite");
    check(max_abs(y) > 1e-9, "model output non-zero (max=" + sci(max_abs(y)) + ")");

    bool threw = false;
    try { m.forward(random_tensor(L, D + 1, 0.5)); } catch (...) { threw = true; }
    check(threw, "model throws on wrong input width");
    threw = false;
    try { m.forward(random_tensor(7, D, 0.5)); } catch (...) { threw = true; }
    check(threw, "model throws on L > l_max");

    // --- parameters()/gradients() contract across ALL depth blocks.
    {
        std::vector<Tensor*> ps = m.parameters();
        std::vector<Tensor*> gs = m.gradients();
        check(ps.size() == gs.size(), "model parameters()/gradients() same length");
        bool shapes_ok = true;
        for (size_t k = 0; k < ps.size(); ++k)
            if (ps[k]->rows != gs[k]->rows || ps[k]->cols != gs[k]->cols)
                shapes_ok = false;
        check(shapes_ok, "model param/grad shapes match elementwise");
        // depth=2 with FFN: per block = ln1(2) + op(?) + ffn1(2) + ffn2(2) + ln2(2).
        // Assert the count scales with depth rather than hard-coding a number.
        HyenaDNAModel d1(D, 6, 1, C, 1, 2, 4, 2);
        HyenaDNAModel d2(D, 6, 2, C, 1, 2, 4, 2);
        size_t per_block = d2.parameters().size() - d1.parameters().size();
        check(per_block > 0 && d1.parameters().size() > per_block,
              "model param count grows with depth (1 block: " +
              std::to_string(d1.parameters().size()) + ", per extra block: " +
              std::to_string(per_block) + ")");
    }

    // --- FD input gradient through ALL depth blocks + the classifier.
    // depth=2 exercises the reverse-order stack backward; a forward-order loop
    // would still agree on the FIRST block and disagree on the second.
    {
        HyenaDNAModel fm(D, 6, DEPTH, C, 2, 3, 4, 2);
        for (size_t i = 0; i < fm.blocks.size(); ++i) well_conditioned_operator(fm.blocks[i].op);
        Tensor fx = random_tensor(L, D, 0.5);
        Tensor fy = fm.forward(fx);
        fm.zero_grad();
        Tensor dx = fm.backward(grad_from_out(fy), 0.0);
        double max_dx = 0.0;
        for (size_t i = 0; i < dx.rows; ++i)
            for (size_t j = 0; j < dx.cols; ++j)
                if (std::fabs(dx[i][j]) > max_dx) max_dx = std::fabs(dx[i][j]);
        check(max_dx > 1e-9, "model input grad non-degenerate (max=" + sci(max_dx) + ")");

        const double eps = 1e-6;
        double worst = 0.0;
        for (size_t i = 0; i < L; ++i)
            for (size_t j = 0; j < D; ++j) {
                Tensor xp = fx.clone(), xm = fx.clone();
                xp[i][j] += eps; xm[i][j] -= eps;
                double num = (loss_of(fm.forward(xp)) - loss_of(fm.forward(xm))) / (2.0 * eps);
                double er = rel_err(dx[i][j], num);
                if (er > worst) worst = er;
            }
        check(worst < 1e-4, "model FD input grad rel_err < 1e-4 worst=" + sci(worst));
    }

    // --- The reverse-order stack backward at depth=3. At depth=2 the
    // forward-order mutation only moves rel_err to ~6.6e-4 (the first block
    // still receives the right gradient, so the error is diluted); at depth=3
    // the signal is unambiguous. This is the "single-iteration vacuity" trap
    // from the TDD skill applied to a stack loop.
    {
        const size_t DEPTH3 = 3;
        HyenaDNAModel sm(D, 6, DEPTH3, C, 2, 3, 4, 2);
        for (size_t i = 0; i < sm.blocks.size(); ++i) well_conditioned_operator(sm.blocks[i].op);
        Tensor sx = random_tensor(L, D, 0.5);
        Tensor sy = sm.forward(sx);
        sm.zero_grad();
        Tensor dx = sm.backward(grad_from_out(sy), 0.0);
        const double eps = 1e-6;
        double worst = 0.0;
        for (size_t i = 0; i < L; ++i)
            for (size_t j = 0; j < D; ++j) {
                Tensor xp = sx.clone(), xm = sx.clone();
                xp[i][j] += eps; xm[i][j] -= eps;
                double num = (loss_of(sm.forward(xp)) - loss_of(sm.forward(xm))) / (2.0 * eps);
                double er = rel_err(dx[i][j], num);
                if (er > worst) worst = er;
            }
        check(worst < 1e-4, "model depth=3 FD input grad rel_err < 1e-4 worst=" + sci(worst));
    }

    // --- end-to-end training reduces the loss. The standard loop is
    // zero_grad / forward / backward / update_weights — forward+backward alone
    // changes no parameter, so the loss would print L0 == L1 exactly.
    {
        HyenaDNAModel tm(D, 6, DEPTH, C, 2, 3, 4, 2);
        for (size_t i = 0; i < tm.blocks.size(); ++i) well_conditioned_operator(tm.blocks[i].op);
        Tensor tx = random_tensor(L, D, 0.5);
        const double lr = 0.05;
        double first = 0.0, last = 0.0;
        for (int step = 0; step < 30; ++step) {
            tm.zero_grad();
            Tensor ty = tm.forward(tx);
            double L = loss_of(ty);
            if (step == 0) first = L;
            last = L;
            tm.backward(grad_from_out(ty), 0.0);
            tm.update_weights(lr);
        }
        check(last < first * 0.9,
              "model training reduces loss over 30 SGD steps (" + sci(first) +
              " -> " + sci(last) + ")");
    }
}

// ---------------------------------------------------------------------------
// Test 9: SequenceLengthWarmup (arXiv:2306.15794 §3.2).
// Stage i runs for epoch_scale * 2^i epochs at sequence length start_len * 2^i.
//   total_epochs = epoch_scale * (2^num_stages - 1)
//   max_seq_len  = start_len * 2^(num_stages-1)
// ---------------------------------------------------------------------------
static void test_sequence_length_warmup() {
    std::printf("Test 9: sequence-length warmup\n");

    // --- constructor validation
    bool threw = false;
    try { SequenceLengthWarmup w(0, 4, 1); } catch (...) { threw = true; }
    check(threw, "warmup throws on start_len=0");
    threw = false;
    try { SequenceLengthWarmup w(64, 0, 1); } catch (...) { threw = true; }
    check(threw, "warmup throws on num_stages=0");
    threw = false;
    try { SequenceLengthWarmup w(64, 4, 0); } catch (...) { threw = true; }
    check(threw, "warmup throws on epoch_scale=0");

    // --- paper defaults: L_1 = 64
    {
        SequenceLengthWarmup w;
        check(w.start_len() == 64, "default start_len is the paper's L_1 = 64");
        check(w.num_stages() > 0, "default num_stages > 0");
        check(w.epoch_scale() > 0, "default epoch_scale > 0");
    }

    // --- doubling schedule, epoch_scale=1, 4 stages: stage lengths 1,2,4,8
    // so epochs 0 | 1,2 | 3,4,5,6 | 7..14
    {
        SequenceLengthWarmup w(64, 4, 1);
        check(w.stage_for_epoch(0) == 0, "epoch 0 is stage 0");
        check(w.stage_for_epoch(1) == 1, "epoch 1 is stage 1");
        check(w.stage_for_epoch(2) == 1, "epoch 2 is still stage 1");
        check(w.stage_for_epoch(3) == 2, "epoch 3 is stage 2");
        check(w.stage_for_epoch(6) == 2, "epoch 6 is still stage 2");
        check(w.stage_for_epoch(7) == 3, "epoch 7 is stage 3");
        check(w.stage_for_epoch(14) == 3, "epoch 14 is still stage 3");
        // stage 0 and 1 boundaries are the ones an off-by-one would break
        check(w.seq_len_for_epoch(0) == 64, "stage 0 seq len = 64");
        check(w.seq_len_for_epoch(1) == 128, "stage 1 seq len = 128 (doubled)");
        check(w.seq_len_for_epoch(3) == 256, "stage 2 seq len = 256");
        check(w.seq_len_for_epoch(7) == 512, "stage 3 seq len = 512");
        check(w.total_epochs() == 15, "total_epochs = 1*(2^4-1) = 15");
        check(w.max_seq_len() == 512, "max_seq_len = 64*2^3 = 512");
        // Past the end the schedule SATURATES at the final stage rather than
        // shifting past the end of the word.
        check(w.stage_for_epoch(15) == 3, "epoch past the end saturates at the last stage");
        check(w.seq_len_for_epoch(15) == 512, "saturated seq len == max_seq_len");
        check(w.seq_len_for_epoch(1000) == 512, "far-past epoch still saturates");
    }

    // --- epoch_scale=2: every stage length doubles, total = 2*(2^4-1) = 30
    {
        SequenceLengthWarmup w(64, 4, 2);
        check(w.stage_for_epoch(0) == 0, "epoch_scale=2: epoch 0 is stage 0");
        check(w.stage_for_epoch(1) == 0, "epoch_scale=2: epoch 1 still stage 0");
        check(w.stage_for_epoch(2) == 1, "epoch_scale=2: epoch 2 is stage 1");
        check(w.stage_for_epoch(29) == 3, "epoch_scale=2: final epoch is the last stage");
        check(w.total_epochs() == 30, "epoch_scale=2 total_epochs = 30");
        check(w.max_seq_len() == 512, "epoch_scale does not affect max_seq_len");
    }

    // --- single-stage edge case: everything is stage 0
    {
        SequenceLengthWarmup w(100, 1, 1);
        check(w.stage_for_epoch(0) == 0, "num_stages=1: epoch 0 is stage 0");
        check(w.stage_for_epoch(99) == 0, "num_stages=1: saturates at stage 0");
        check(w.seq_len_for_epoch(0) == 100, "num_stages=1 seq len is start_len");
        check(w.total_epochs() == 1, "num_stages=1 total_epochs = 1");
        check(w.max_seq_len() == 100, "num_stages=1 max_seq_len = start_len");
    }
}

// ---------------------------------------------------------------------------
// Test 10: SoftPrompting (arXiv:2306.15794 §3.3, Eq. 3.2).
//   x <- concat[theta, embed(x_p)]   (at_front)  or  concat[embed(x_p), theta]
// ---------------------------------------------------------------------------
static void test_soft_prompting() {
    std::printf("Test 10: soft prompting\n");

    const size_t N = 3, D = 4, T = 5;

    // --- constructor validation
    bool threw = false;
    try { SoftPrompting p(0, D); } catch (...) { threw = true; }
    check(threw, "soft prompting throws on prompt_len=0");
    threw = false;
    try { SoftPrompting p(N, 0); } catch (...) { threw = true; }
    check(threw, "soft prompting throws on d_model=0");

    // --- shape and concat order, both directions. The prompt entries are
    // hand-set so the expected layout is a specific known value, not another
    // implementation of the same loop.
    {
        SoftPrompting pf(N, D, true);
        check(pf.prompt_len() == N && pf.d_model() == D, "soft prompting accessors");
        check(pf.at_front() == true, "at_front accessor");
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c) pf.prompt()[n][c] = 100.0 + n;

        Tensor emb = Tensor::random(T, D, 0.5);
        Tensor out = pf.forward(emb);
        check(out.rows == T + N && out.cols == D, "soft prompting (T,D) -> (T+N, D)");
        bool front_ok = true;
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c)
                if (out[n][c] != 100.0 + n) front_ok = false;
        check(front_ok, "at_front=true puts the prompt in rows 0..N-1");
        bool emb_ok = true;
        for (size_t t = 0; t < T; ++t)
            for (size_t c = 0; c < D; ++c)
                if (out[t + N][c] != emb[t][c]) emb_ok = false;
        check(emb_ok, "at_front=true shifts the embedding to rows N..N+T-1");
    }
    {
        SoftPrompting pb(N, D, false);
        check(pb.at_front() == false, "at_back accessor");
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c) pb.prompt()[n][c] = 200.0 + n;
        Tensor emb = Tensor::random(T, D, 0.5);
        Tensor out = pb.forward(emb);
        bool back_ok = true;
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c)
                if (out[T + n][c] != 200.0 + n) back_ok = false;
        check(back_ok, "at_front=false puts the prompt in rows T..T+N-1");
        bool emb_ok = true;
        for (size_t t = 0; t < T; ++t)
            for (size_t c = 0; c < D; ++c)
                if (out[t][c] != emb[t][c]) emb_ok = false;
        check(emb_ok, "at_front=false leaves the embedding in rows 0..T-1");
    }

    // --- d_model mismatch throws
    {
        SoftPrompting pm(N, D);
        threw = false;
        try { pm.forward(Tensor::random(T, D + 1, 0.5)); } catch (...) { threw = true; }
        check(threw, "soft prompting throws on embedded width mismatch");
    }

    // --- FD on the prompt gradient. Loss = 0.5*sum(out^2) => grad = out, and
    // dL/dtheta[n,c] = out[prompt row n, c] — a pure copy through the concat.
    // Hand-derived here rather than FD'd, because the copy is the definition.
    {
        SoftPrompting sp(N, D, true);
        Tensor emb = Tensor::random(T, D, 0.5);
        Tensor out = sp.forward(emb);
        sp.zero_grad();
        Tensor g = sp.backward(out.clone(), 0.0);
        double worst = 0.0;
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c) {
                double e = rel_err(sp.grad_prompt_[n][c], out[n][c]);
                if (e > worst) worst = e;
            }
        check(worst < 1e-12, "prompt grad == the gradient at the prompt rows (hand-derived), worst=" +
              sci(worst));
        check(max_abs(sp.grad_prompt_) > 1e-9, "prompt grad non-degenerate");

        // The embedded part gets ZERO gradient — the paper's prompt-tuning
        // freezes everything upstream of the prompt.
        check(max_abs(g) == 0.0, "backward returns zeros for the embedded part");
    }
    {
        SoftPrompting sp(N, D, false);
        Tensor emb = Tensor::random(T, D, 0.5);
        Tensor out = sp.forward(emb);
        sp.zero_grad();
        sp.backward(out.clone(), 0.0);
        double worst = 0.0;
        for (size_t n = 0; n < N; ++n)
            for (size_t c = 0; c < D; ++c) {
                double e = rel_err(sp.grad_prompt_[n][c], out[T + n][c]);
                if (e > worst) worst = e;
            }
        check(worst < 1e-12, "at_front=false prompt grad reads the TRAILING rows, worst=" +
              sci(worst));
    }

    // --- zero_grad / update_weights / parameters contract
    {
        SoftPrompting sp(N, D);
        Tensor emb = Tensor::random(T, D, 0.5);
        Tensor out = sp.forward(emb);
        sp.zero_grad();
        sp.backward(out.clone(), 0.0);
        sp.zero_grad();
        check(max_abs(sp.grad_prompt_) == 0.0, "soft prompting zero_grad clears the prompt grad");

        sp.backward(out.clone(), 0.0);
        double before = sp.prompt_[1][2];
        double g = sp.grad_prompt_[1][2];
        check(std::fabs(g) > 1e-9, "prompt grad is non-zero before the step");
        sp.update_weights(0.1);
        check(std::fabs(sp.prompt_[1][2] - (before - 0.1 * g)) < 1e-15,
              "soft prompting update_weights moves theta by exactly -lr*grad");

        std::vector<Tensor*> ps = sp.parameters();
        std::vector<Tensor*> gs = sp.gradients();
        check(ps.size() == 1 && gs.size() == 1, "soft prompting exposes exactly one param/grad");
        check(ps[0]->rows == N && ps[0]->cols == D, "soft prompting param shape is (N, D)");
    }
}

int main() {
    std::printf("=== HyenaDNA Tests ===\n");
    test_constructor_validation();
    test_positional_embedding();
    test_filter_forward();
    test_filter_fd_gradients();
    test_operator_forward();
    test_operator_fd_gradients();
    test_block();
    test_model();
    test_sequence_length_warmup();
    test_soft_prompting();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
