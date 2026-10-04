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

int main() {
    std::printf("=== HyenaDNA Tests ===\n");
    test_constructor_validation();

    std::printf("\n=== Summary: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed == 0 ? 0 : 1;
}
