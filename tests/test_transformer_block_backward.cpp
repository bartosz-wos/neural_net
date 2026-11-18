// =============================================================================
// tests/test_transformer_block_backward.cpp -- the TRANSFORMER BLOCK's
// analytical backward, checked against FINITE DIFFERENCES.
//
// WHY THIS FILE EXISTS
//
// `tests/test_mha_backward.cpp` already pins MultiHeadAttention's backward,
// and `tests/test_block_cached_decode.cpp` pins the block's cached FORWARD
// and the block-level cached decode. NEITHER checks the block's BACKWARD
// against finite differences -- the block's backward is exercised only for
// "it runs and the gradients are finite", which a wrong gradient satisfies
// exactly as well as a right one.
//
// A probe found the block's backward disagrees with centered finite
// differences by rel_err ~ 1.0 (i.e. the analytical and numerical gradients
// are of OPPOSITE SIGN and different magnitude). Two defects, both in
// `TransformerBlock::backward`:
//
//   D1  THE GELU DERIVATIVE IS THE `pdf` TERM ONLY.
//
//        The block's forward GELU is
//
//            GELU(x) = 0.5 * x * (1 + tanh(u)),
//            u       = sqrt(2/pi) * (x + 0.044715 * x^3)
//
//        whose derivative is
//
//            GELU'(x) = 0.5 * (1 + tanh(u))
//                     + 0.5 * x * (1 - tanh(u)^2) * u'
//            u'       = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
//
//        i.e. `cdf + x * pdf`. The shipped code computes only
//
//            0.5 * (1 - tanh(u)^2) * sqrt(2/pi) * (1 + 3*0.044715*x^2)
//
//        which is `pdf` -- the `cdf` term AND the factor `x` are both
//        missing. Measured against finite differences of the block's own
//        GELU, the shipped formula is off by up to 1.03 in absolute value.
//
//        NOTE the repo's `GELU::derivative` (activations.cpp) IS correct
//        (`cdf + x_clamped * pdf`), so this is a LOCAL divergence inside
//        transformer.cpp, not a repo-wide convention.
//
//   D2  THE RESIDUAL IS WIRED TO dL/dnorm1 INSTEAD OF dL/dresidual.
//
//        The forward is
//
//            residual = x + attn_tokens
//            norm1    = LayerNorm(residual)
//            ...
//
//        so BOTH `x` and `attn_tokens` reach the loss only THROUGH the
//        residual, and the gradient that reaches them is the LayerNorm's
//        BACKWARD OUTPUT (dL/dresidual) -- not dL/dnorm1, which is the
//        gradient w.r.t. the LayerNorm's OUTPUT.
//
//        The shipped code computes
//
//            grad_attn_tokens = grad_norm1;                 // should be grad_residual1
//            grad_residual1   = ln1.backward(grad_norm1);
//            grad_x           = grad_residual1 + grad_attn_tokens;
//
//        so `grad_attn_tokens` skips the LayerNorm's Jacobian entirely, and
//        `grad_x` has the pre-LayerNorm gradient ADDED to it.
//
//   D3  THE BLOCK'S LAYERNORMS ARE NOT IN ITS PARAMETER LIST.
//
//        `parameters()` / `gradients()` / `zero_grad()` cover the attention
//        and the FFN but NOT ln1.gamma, ln1.beta, ln2.gamma, ln2.beta.
//        So the optimizer never updates the block's LayerNorm scale and
//        shift, and their gradient accumulators (which LayerNorm's backward
//        grows with `+=`) are never cleared.
//
// WHY FINITE DIFFERENCES ARE THE ORACLE HERE
//
// The repo has a float64 numpy oracle for the cached decode, but the
// BACKWARD has no independent specification -- and the shipped C++ backward
// is what a numpy transcription would be compared against, so an oracle
// written from the same reading would share the bug. Finite differences of
// the FORWARD are the ground truth: they need no specification of the
// backward at all, and they are exactly what the backward must reproduce.
//
// The fixture is deliberately ASYMMETRIC (the input varies with both the
// feature and the token index, and the weights are random rather than
// uniform) because a symmetric fixture can hide a transposed gradient.
//
// The non-degenerate init scale matters too: the block ships with 0.01,
// which puts the loss change under a 1e-5 perturbation at the
// double-precision noise floor and makes the FD estimate meaningless.
//
// AND A CLEAN SIGN INVERSION IS THE FINGERPRINT OF A TEST BUG, NOT AN
// IMPLEMENTATION BUG: a real implementation error is large and
// shape-dependent, not exact-and-negated. So the tolerances here are
// absolute-and-relative against the FD value, and the eps-sweep check
// distinguishes a real bug (a CONSTANT rel_err) from FD noise (an error
// that shrinks with eps).
// =============================================================================

#include "nn/layers/attention/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        ++g_pass;
        std::printf("  [PASS] %s\n", name.c_str());
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", name.c_str());
    }
}

// The repo's `Layer::backward` contract: the returned tensor is the
// gradient of L = sum(grad_output * forward(x)) -- there is no 1/2 factor
// anywhere (see Dense::backward, which computes grad_weights =
// grad_output^T * last_input).
static double block_loss(TransformerBlock& b, const Tensor& x, const Tensor& g) {
    Tensor y = b.forward(x);
    double L = 0.0;
    for (size_t i = 0; i < y.rows; ++i)
        for (size_t j = 0; j < y.cols; ++j) L += g(i, j) * y(i, j);
    return L;
}

// A block with a NON-DEGENERATE init. The shipped 0.01 makes the loss
// change under a 1e-5 perturbation land at the double-precision noise
// floor.
static void init_block(TransformerBlock& b, size_t d_model, size_t /*heads*/,
                       double scale) {
    b.W1 = Tensor::random(d_model, d_model, scale);
    b.W2 = Tensor::random(d_model, d_model, scale);
    b.b1 = Tensor::zeros(1, d_model);
    b.b2 = Tensor::zeros(1, d_model);
    b.attn.W_q = Tensor::random(d_model, d_model, scale);
    b.attn.W_k = Tensor::random(d_model, d_model, scale);
    b.attn.W_v = Tensor::random(d_model, d_model, scale);
    b.attn.W_o = Tensor::random(d_model, d_model, scale);
}

// An ASYMMETRIC fixture: it varies with the feature index AND the token
// index, so a transposed gradient cannot agree with it by accident.
//
// IT MUST ALSO BE SMALL ENOUGH THAT THE CAUSAL SOFTMAX DOES NOT SATURATE.
// The first version of this fixture was a large monotone ramp
// (`1 + 0.37f + 1.13s + 0.11fs`), which made the score gaps reach 50.9 --
// the softmax became exactly one-hot, so dL/d(scores) = P*(dP - rowsum) was
// exactly zero, and the FD on W_q and W_k was EXACTLY 0.0.
//
// **A CHECK ON A DEGENERATE FIXTURE MEASURES NOTHING, AND IT PASSES
// VACUOUSLY.** The tell was a reported rel_err of 1.47e-28 for W_q in one
// configuration and 0.993 in another: a rel_err that tiny is not a good
// agreement, it is two zeros. The fixture is now O(1) and was MEASURED to
// give a usable FD signal (|FD_W_q| ~ 1e-5, six orders of magnitude above
// the double-precision noise floor) in every configuration below.
static Tensor make_input(size_t d_model, size_t seq_len) {
    Tensor x(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x(f, s) = 0.30 * std::sin(1.7 * f + 0.9 * s)
                    + 0.20 * std::cos(0.6 * s + 0.3 * f);
    return x;
}

// ---------------------------------------------------------------------------
// The input gradient.
// ---------------------------------------------------------------------------
static void fd_input_grad(size_t d_model, size_t seq_len, size_t heads,
                           double scale) {
    TransformerBlock b(d_model, heads);
    init_block(b, d_model, heads, scale);

    Tensor x = make_input(d_model, seq_len);
    Tensor y = b.forward(x);
    Tensor g(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s) g(f, s) = 0.7 * y(f, s) - 0.3;

    b.zero_grad();
    Tensor dx = b.backward(g, 0.0);

    const double eps = 1e-5;
    double worst = 0.0;
    for (size_t f = 0; f < d_model; ++f) {
        for (size_t s = 0; s < seq_len; ++s) {
            Tensor xp = x.clone(), xm = x.clone();
            xp(f, s) += eps;
            xm(f, s) -= eps;
            double num = (block_loss(b, xp, g) - block_loss(b, xm, g)) / (2.0 * eps);
            double ana = dx(f, s);
            double den = std::max(std::fabs(num), std::fabs(ana));
            if (den < 1e-9) den = 1.0;
            double rel = std::fabs(num - ana) / den;
            if (rel > worst) worst = rel;
        }
    }
    char buf[220];
    std::snprintf(buf, sizeof(buf),
                  "FD input grad d_model=%zu T=%zu H=%zu scale=%.2f: "
                  "worst rel_err=%.3g < 1e-4",
                  d_model, seq_len, heads, scale, worst);
    check(worst < 1e-4, buf);
}

// ---------------------------------------------------------------------------
// The parameter gradients.
// ---------------------------------------------------------------------------
static void fd_param_grad(size_t d_model, size_t seq_len, size_t heads,
                           double scale) {
    TransformerBlock b(d_model, heads);
    init_block(b, d_model, heads, scale);

    Tensor x = make_input(d_model, seq_len);
    Tensor y = b.forward(x);
    Tensor g(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s) g(f, s) = 0.7 * y(f, s) - 0.3;

    b.zero_grad();
    b.backward(g, 0.0);

    struct Target { Tensor* p; Tensor* grad; const char* name; };
    Target targets[] = {
        {&b.W1,         &b.grad_W1,     "W1"},
        {&b.b1,         &b.grad_b1,     "b1"},
        {&b.W2,         &b.grad_W2,     "W2"},
        {&b.b2,         &b.grad_b2,     "b2"},
        {&b.attn.W_q,   &b.attn.grad_W_q, "W_q"},
        {&b.attn.W_k,   &b.attn.grad_W_k, "W_k"},
        {&b.attn.W_v,   &b.attn.grad_W_v, "W_v"},
        {&b.attn.W_o,   &b.attn.grad_W_o, "W_o"},
    };

    for (auto& t : targets) {
        double worst = 0.0;
        for (size_t i = 0; i < t.p->rows; ++i) {
            for (size_t j = 0; j < t.p->cols; ++j) {
                const double eps = 1e-5;
                const double orig = (*t.p)(i, j);
                (*t.p)(i, j) = orig + eps;
                const double lp = block_loss(b, x, g);
                (*t.p)(i, j) = orig - eps;
                const double lm = block_loss(b, x, g);
                (*t.p)(i, j) = orig;
                const double num = (lp - lm) / (2.0 * eps);
                const double ana = (*t.grad)(i, j);
                double den = std::max(std::fabs(num), std::fabs(ana));
                if (den < 1e-9) den = 1.0;
                const double rel = std::fabs(num - ana) / den;
                if (rel > worst) worst = rel;
            }
        }
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "FD %s gradient: worst rel_err=%.3g < 1e-3", t.name, worst);
        check(worst < 1e-3, buf);
    }
}

// ---------------------------------------------------------------------------
// THE EPS SWEEP: a real bug keeps a CONSTANT rel_err, FD noise shrinks.
//
// This is the check that distinguishes "the implementation is wrong" from
// "the test's fixture is too small to measure".
// ---------------------------------------------------------------------------
static void eps_sweep_is_stable() {
    TransformerBlock b(4, 1);
    init_block(b, 4, 1, 0.6);
    Tensor x = make_input(4, 3);
    Tensor y = b.forward(x);
    Tensor g(4, 3);
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 3; ++s) g(f, s) = y(f, s);

    b.zero_grad();
    Tensor dx = b.backward(g, 0.0);

    double worst_1e5 = 0.0, worst_1e3 = 0.0;
    for (size_t f = 0; f < 4; ++f) {
        for (size_t s = 0; s < 3; ++s) {
            for (int k = 0; k < 2; ++k) {
                const double eps = (k == 0) ? 1e-5 : 1e-3;
                Tensor xp = x.clone(), xm = x.clone();
                xp(f, s) += eps;
                xm(f, s) -= eps;
                const double num =
                    (block_loss(b, xp, g) - block_loss(b, xm, g)) / (2.0 * eps);
                const double ana = dx(f, s);
                double den = std::max(std::fabs(num), std::fabs(ana));
                if (den < 1e-9) den = 1.0;
                const double rel = std::fabs(num - ana) / den;
                if (k == 0) { if (rel > worst_1e5) worst_1e5 = rel; }
                else       { if (rel > worst_1e3) worst_1e3 = rel; }
            }
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "rel_err stable across eps (1e-5:%.3g vs 1e-3:%.3g), both < 1e-3",
                  worst_1e5, worst_1e3);
    check(worst_1e5 < 1e-3 && worst_1e3 < 1e-3, buf);
}

// ---------------------------------------------------------------------------
// D1 IN ISOLATION: the block's GELU derivative, checked against finite
// differences of the block's OWN GELU.
//
// This is a DIRECT check of the formula, independent of the whole backward,
// so it pins the defect where it lives rather than through a long chain.
// ---------------------------------------------------------------------------
static double block_gelu(double x) {
    const double A = std::sqrt(2.0 / M_PI);
    return 0.5 * x * (1.0 + std::tanh(A * (x + 0.044715 * x * x * x)));
}

static void gelu_derivative_matches_fd() {
    // The formula as this session found it in TransformerBlock::backward: the
    // `pdf` term only, with both the `cdf` term and the factor `x` missing.
    auto pdf_only = [](double x) {
        const double A = std::sqrt(2.0 / M_PI);
        const double z = A * (x + 0.044715 * x * x * x);
        const double t = std::tanh(z);
        return 0.5 * (1.0 - t * t) * A * (1.0 + 3.0 * 0.044715 * x * x);
    };
    // The correct derivative of the block's own GELU: cdf + x * pdf.
    auto correct = [](double x) {
        const double A = std::sqrt(2.0 / M_PI);
        const double z = A * (x + 0.044715 * x * x * x);
        const double t = std::tanh(z);
        return 0.5 * (1.0 + t)
             + 0.5 * x * (1.0 - t * t) * A * (1.0 + 3.0 * 0.044715 * x * x);
    };

    const double eps = 1e-6;
    double worst_correct = 0.0;
    for (double x = -3.0; x <= 3.0; x += 0.25) {
        const double fd = (block_gelu(x + eps) - block_gelu(x - eps)) / (2.0 * eps);
        worst_correct = std::max(worst_correct, std::fabs(correct(x) - fd));
    }
    char buf[180];
    std::snprintf(buf, sizeof(buf),
                  "the block's GELU derivative (cdf + x*pdf) matches FD: "
                  "worst abs diff = %.3g < 1e-6", worst_correct);
    check(worst_correct < 1e-6, buf);

    // THE COUNTERFACTUAL: the `pdf`-only formula that shipped before this
    // fix must NOT match the FD. Without this check, the check above could
    // be satisfied by a formula that is wrong everywhere -- and the two
    // formulas differ by up to 1.03 in absolute value, so the
    // counterfactual is a genuinely discriminating witness.
    //
    // (If this ever fails, the fix has been reverted, not the test broken.)
    double worst_old = 0.0;
    for (double x = -3.0; x <= 3.0; x += 0.25) {
        const double fd = (block_gelu(x + eps) - block_gelu(x - eps)) / (2.0 * eps);
        worst_old = std::max(worst_old, std::fabs(pdf_only(x) - fd));
    }
    std::snprintf(buf, sizeof(buf),
                  "COUNTERFACTUAL: the `pdf`-only formula does NOT match FD "
                  "(worst abs diff = %.3g > 1e-3)", worst_old);
    check(worst_old > 1e-3, buf);
}

// ---------------------------------------------------------------------------
// D3 IN ISOLATION: the block's LayerNorms must be trainable.
//
// A LayerNorm whose gamma/beta are not in the block's parameter list is
// never updated by the optimizer AND never has its gradient accumulators
// cleared -- and LayerNorm::backward grows them with `+=`.
// ---------------------------------------------------------------------------
static void layer_norms_are_trainable() {
    TransformerBlock b(4, 2);
    init_block(b, 4, 2, 0.4);
    Tensor x = make_input(4, 3);
    Tensor y = b.forward(x);
    Tensor g(4, 3);
    for (size_t f = 0; f < 4; ++f)
        for (size_t s = 0; s < 3; ++s) g(f, s) = y(f, s);

    // Every trainable tensor must be reachable through parameters() and
    // gradients(), with the SAME LENGTH (the optimizers index them in
    // parallel).
    auto params = b.parameters();
    auto grads = b.gradients();
    char buf[200];
    std::snprintf(buf, sizeof(buf),
                  "parameters() and gradients() have the same length (%zu vs %zu)",
                  params.size(), grads.size());
    check(params.size() == grads.size(), buf);

    auto has = [&](const std::vector<Tensor*>& v, const Tensor* t) {
        for (auto* p : v) if (p == t) return true;
        return false;
    };
    std::snprintf(buf, sizeof(buf),
                  "ln1.gamma is in parameters() (the block's LayerNorms are "
                  "trainable)");
    check(has(params, &b.ln1.gamma), buf);
    std::snprintf(buf, sizeof(buf), "ln1.beta is in parameters()");
    check(has(params, &b.ln1.beta), buf);
    std::snprintf(buf, sizeof(buf), "ln2.gamma is in parameters()");
    check(has(params, &b.ln2.gamma), buf);
    std::snprintf(buf, sizeof(buf), "ln2.beta is in parameters()");
    check(has(params, &b.ln2.beta), buf);

    std::snprintf(buf, sizeof(buf), "ln1.grad_gamma_ is in gradients()");
    check(has(grads, &b.ln1.grad_gamma_), buf);
    std::snprintf(buf, sizeof(buf), "ln2.grad_beta_ is in gradients()");
    check(has(grads, &b.ln2.grad_beta_), buf);

    // The gradients must be NON-EMPTY (the LayerNorm's gradient buffers
    // are only allocated on the first backward, and a 0x0 buffer is the
    // shape of the memory-corruption bug this repo has already hit once).
    b.zero_grad();
    b.backward(g, 0.0);
    std::snprintf(buf, sizeof(buf),
                  "ln1.grad_gamma_ has the right shape after a backward pass "
                  "(%zux%zu)", b.ln1.grad_gamma_.rows, b.ln1.grad_gamma_.cols);
    check(b.ln1.grad_gamma_.rows == 1 && b.ln1.grad_gamma_.cols == 4, buf);

    // A gradient accumulator that zero_grad() does not clear grows without
    // bound across steps. Two backward passes with a zero_grad() in between
    // must leave the accumulator at ONE step's worth.
    const double after_first = b.ln1.grad_gamma_.data[0];
    b.zero_grad();
    b.backward(g, 0.0);
    const double after_second = b.ln1.grad_gamma_.data[0];
    std::snprintf(buf, sizeof(buf),
                  "zero_grad() clears the LayerNorm's gradient accumulator "
                  "(%.6g then %.6g)", after_first, after_second);
    check(std::fabs(after_first - after_second) <=
              1e-12 * std::max(1.0, std::fabs(after_first)),
          buf);
}

int main() {
    std::printf("=== TransformerBlock backward vs finite differences ===\n");

    // The input gradient.
    fd_input_grad(4, 3, 1, 0.6);
    fd_input_grad(4, 4, 1, 0.6);
    fd_input_grad(8, 4, 2, 0.6);
    fd_input_grad(8, 6, 2, 0.5);
    fd_input_grad(6, 5, 3, 0.4);

    // The parameter gradients.
    fd_param_grad(4, 3, 1, 0.6);
    fd_param_grad(8, 4, 2, 0.6);

    // The eps sweep.
    eps_sweep_is_stable();

    // The two defects in isolation.
    gelu_derivative_matches_fd();
    layer_norms_are_trainable();

    std::printf("=== Summary: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
