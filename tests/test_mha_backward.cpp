// Focused regression test for MultiHeadAttention's analytical backward.
//
// The backward shipped in transformer.cpp computes dK and dQ as if the
// attention map were LINEAR in the scores:
//
//   grad_K = P^T @ (grad_attn_h @ Q) / sqrt(d_k)
//   grad_Q = P^T @ (grad_attn_h @ K) / sqrt(d_k)
//
// That omits the softmax Jacobian entirely. The correct chain requires
// dScores = P .* (dP - rowsum(P .* dP)) BEFORE projecting back to dQ/dK,
// where dP[i][j] = sum_dk grad_attn_h[i][dk] * V[j][dk] is the gradient
// w.r.t. the attention probabilities.
//
// Symptom this pins: the analytical input gradient disagrees with centered
// finite differences by a factor of ~1 (rel_err ~0.5-1.0), and the
// disagreement does NOT shrink as eps shrinks — a real bug, not FD noise.
//
// The layer's public layout is TRANSPOSED: (d_model, seq_len), unlike most of
// the repo which is (seq_len, d_model). See transformer.cpp:31-35.

#include "nn/nn.h"
#include <cstdio>
#include <cmath>
#include <string>

static int passed = 0, failed = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { ++passed; std::printf("  [PASS] %s\n", name.c_str()); }
    else      { ++failed; std::printf("  [FAIL] %s\n", name.c_str()); }
}

// Loss must be a function of the FORWARD OUTPUT only, so dL/dout = out for
// L = 0.5 * sum(out^2).
static double mha_loss(MultiHeadAttention& a, const Tensor& x) {
    Tensor o = a.forward(x);
    double L = 0.0;
    for (size_t i = 0; i < o.rows; ++i)
        for (size_t j = 0; j < o.cols; ++j) L += 0.5 * o[i][j] * o[i][j];
    return L;
}

// Loss with an EXPLICIT upstream gradient: L = sum(g * out), so dL/dout = g.
// The gradient handed to backward() must be the gradient of THIS loss, or the
// finite difference and the analytical gradient describe different functions.
static double mha_loss_with(MultiHeadAttention& a, const Tensor& x, const Tensor& g) {
    Tensor o = a.forward(x);
    double L = 0.0;
    for (size_t i = 0; i < o.rows; ++i)
        for (size_t j = 0; j < o.cols; ++j) L += g(i, j) * o(i, j);
    return L;
}

static void fd_input_grad(size_t d_model, size_t seq_len, size_t heads, double init_scale) {
    MultiHeadAttention a(d_model, heads);
    // Re-init at a non-degenerate scale: MHA ships with 0.01, which puts the
    // loss change under a 1e-5 perturbation at the double-precision noise
    // floor and makes the FD estimate meaningless (see the degenerate-config
    // case below, which pins that failure mode deliberately).
    a.W_q = Tensor::random(d_model, d_model, init_scale);
    a.W_k = Tensor::random(d_model, d_model, init_scale);
    a.W_v = Tensor::random(d_model, d_model, init_scale);
    a.W_o = Tensor::random(d_model, d_model, init_scale);

    // Asymmetric fixture: a symmetric one can hide a transposed grad.
    Tensor x(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x(f, s) = 1.0 + 0.37 * f + 1.13 * s + 0.11 * f * s;

    Tensor y = a.forward(x);
    Tensor gy = Tensor::zeros(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s) gy(f, s) = y(f, s);   // dL/dy = y

    a.zero_grad();
    Tensor dx = a.backward(gy, 0.0);

    const double eps = 1e-5;
    double worst = 0.0;
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s) {
            Tensor xp = x.clone(), xm = x.clone();
            xp(f, s) += eps; xm(f, s) -= eps;
            double num = (mha_loss(a, xp) - mha_loss(a, xm)) / (2.0 * eps);
            double ana = dx(f, s);
            double den = std::max(std::fabs(ana), std::fabs(num));
            if (den < 1e-12) den = 1.0;
            double rel = std::fabs(ana - num) / den;
            if (rel > worst) worst = rel;
        }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "FD input grad d_model=%zu L=%zu H=%zu scale=%.2f: worst rel_err=%.3g < 1e-4",
                  d_model, seq_len, heads, init_scale, worst);
    check(worst < 1e-4, buf);
}

// ---------------------------------------------------------------------------
// MULTI-HEAD parameter gradients.
//
// The parameter-gradient check above uses MultiHeadAttention a(4, 1) -- ONE
// head -- and the input gradient check uses a saturating ramp that drives the
// causal softmax to a one-hot distribution. Neither can see a defect that
// only appears with two or more heads.
//
// The defect this pins: MultiHeadAttention::backward accumulates
//     grad_W_q += last_x^T @ grad_q
// INSIDE the per-head loop, while grad_q is progressively filled with EVERY
// head's contribution (each head writes its own column slice). Head 0's
// contribution was therefore counted twice -- once on its own, then again
// together with head 1 -- so every gradient column belonging to head 0 was
// exactly doubled. With one head the error is invisible; with two or more it
// is a clean factor of two.
//
// The fixture must not saturate the softmax, or the FD signal for W_q/W_k/W_v
// collapses to zero and the check becomes vacuous. The bounded sinusoidal
// input below keeps the attention scores O(1).
static void fd_param_grad_multihead(size_t d_model, size_t seq_len, size_t heads) {
    MultiHeadAttention a(d_model, heads);
    a.W_q = Tensor::random(d_model, d_model, 0.6);
    a.W_k = Tensor::random(d_model, d_model, 0.6);
    a.W_v = Tensor::random(d_model, d_model, 0.6);
    a.W_o = Tensor::random(d_model, d_model, 0.6);

    // Bounded, O(1) input: the scores stay in a range where the softmax is
    // genuinely sensitive to W_q/W_k.
    Tensor x(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s)
            x(f, s) = 0.30 * std::sin(1.7 * f + 0.9 * s)
                    + 0.20 * std::cos(0.6 * s + 0.3 * f);

    Tensor y = a.forward(x);
    Tensor gy = Tensor::zeros(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f)
        for (size_t s = 0; s < seq_len; ++s) gy(f, s) = 0.7 * y(f, s) - 0.3;
    a.zero_grad();
    a.backward(gy, 0.0);

    struct { Tensor* p; Tensor* g; const char* nm; } targets[] = {
        {&a.W_q, &a.grad_W_q, "W_q"},
        {&a.W_k, &a.grad_W_k, "W_k"},
        {&a.W_v, &a.grad_W_v, "W_v"},
        {&a.W_o, &a.grad_W_o, "W_o"},
    };
    for (auto& t : targets) {
        const double eps = 1e-5;
        double worst = 0.0;
        double fd_max = 0.0;
        for (size_t i = 0; i < t.p->rows; ++i)
            for (size_t j = 0; j < t.p->cols; ++j) {
                double orig = (*t.p)(i, j);
                (*t.p)(i, j) = orig + eps; double lp = mha_loss_with(a, x, gy);
                (*t.p)(i, j) = orig - eps; double lm = mha_loss_with(a, x, gy);
                (*t.p)(i, j) = orig;
                double num = (lp - lm) / (2.0 * eps);
                double ana = (*t.g)(i, j);
                fd_max = std::max(fd_max, std::fabs(num));
                double den = std::max(std::fabs(ana), std::fabs(num));
                if (den < 1e-12) den = 1.0;
                double rel = std::fabs(ana - num) / den;
                if (rel > worst) worst = rel;
            }
        char buf[200];
        // Guard against a vacuous pass: the FD signal must be far above the
        // double-precision noise floor, otherwise "they agree" means nothing.
        std::snprintf(buf, sizeof(buf),
                      "FD %s gradient (H=%zu): worst rel_err=%.3g < 1e-3 (|FD|max=%.3g > 1e-9)",
                      t.nm, heads, worst, fd_max);
        check(worst < 1e-3 && fd_max > 1e-9, buf);
    }
}

int main() {
    std::printf("=== MultiHeadAttention backward regression ===\n");

    // These are the shapes that matter and that the striped-hyena block uses.
    fd_input_grad(4, 3, 1, 0.3);
    fd_input_grad(4, 4, 1, 0.3);
    fd_input_grad(8, 4, 2, 0.3);   // multi-head
    fd_input_grad(8, 6, 2, 0.5);
    fd_input_grad(6, 5, 3, 0.4);   // 3 heads

    // eps-independence: a real bug keeps a CONSTANT rel_err as eps changes;
    // FD noise shrinks. Pin that the error does not blow up with a coarser step.
    {
        MultiHeadAttention a(4, 1);
        a.W_q = Tensor::random(4, 4, 0.3); a.W_k = Tensor::random(4, 4, 0.3);
        a.W_v = Tensor::random(4, 4, 0.3); a.W_o = Tensor::random(4, 4, 0.3);
        Tensor x(4, 3);
        for (size_t f = 0; f < 4; ++f) for (size_t s = 0; s < 3; ++s) x(f, s) = 1.0 + 0.4 * f + 0.9 * s;
        Tensor y = a.forward(x);
        Tensor gy = Tensor::zeros(4, 3);
        for (size_t f = 0; f < 4; ++f) for (size_t s = 0; s < 3; ++s) gy(f, s) = y(f, s);
        a.zero_grad();
        Tensor dx = a.backward(gy, 0.0);

        double worst_1e5 = 0.0, worst_1e3 = 0.0;
        for (size_t f = 0; f < 4; ++f)
            for (size_t s = 0; s < 3; ++s) {
                for (int k = 0; k < 2; ++k) {
                    double eps = (k == 0) ? 1e-5 : 1e-3;
                    Tensor xp = x.clone(), xm = x.clone();
                    xp(f, s) += eps; xm(f, s) -= eps;
                    double num = (mha_loss(a, xp) - mha_loss(a, xm)) / (2.0 * eps);
                    double ana = dx(f, s);
                    double den = std::max(std::fabs(ana), std::fabs(num));
                    if (den < 1e-12) den = 1.0;
                    double rel = std::fabs(ana - num) / den;
                    if (k == 0) { if (rel > worst_1e5) worst_1e5 = rel; }
                    else       { if (rel > worst_1e3) worst_1e3 = rel; }
                }
            }
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "rel_err stable across eps (1e-5:%.3g vs 1e-3:%.3g), both < 1e-3",
                      worst_1e5, worst_1e3);
        check(worst_1e5 < 1e-3 && worst_1e3 < 1e-3, buf);
    }

    // Parameter gradients must also match FD (catches the missing softmax
    // Jacobian even if the input path happened to be right).
    {
        MultiHeadAttention a(4, 1);
        a.W_q = Tensor::random(4, 4, 0.3); a.W_k = Tensor::random(4, 4, 0.3);
        a.W_v = Tensor::random(4, 4, 0.3); a.W_o = Tensor::random(4, 4, 0.3);
        Tensor x(4, 3);
        for (size_t f = 0; f < 4; ++f) for (size_t s = 0; s < 3; ++s) x(f, s) = 1.0 + 0.4 * f + 0.9 * s;
        Tensor y = a.forward(x);
        Tensor gy = Tensor::zeros(4, 3);
        for (size_t f = 0; f < 4; ++f) for (size_t s = 0; s < 3; ++s) gy(f, s) = y(f, s);
        a.zero_grad();
        a.backward(gy, 0.0);

        struct { Tensor* p; Tensor* g; const char* nm; } targets[] = {
            {&a.W_q, &a.grad_W_q, "W_q"},
            {&a.W_k, &a.grad_W_k, "W_k"},
            {&a.W_v, &a.grad_W_v, "W_v"},
            {&a.W_o, &a.grad_W_o, "W_o"},
        };
        for (auto& t : targets) {
            double worst = 0.0;
            for (size_t i = 0; i < t.p->rows; ++i)
                for (size_t j = 0; j < t.p->cols; ++j) {
                    const double eps = 1e-5;
                    double orig = (*t.p)(i, j);
                    (*t.p)(i, j) = orig + eps; double lp = mha_loss(a, x);
                    (*t.p)(i, j) = orig - eps; double lm = mha_loss(a, x);
                    (*t.p)(i, j) = orig;
                    double num = (lp - lm) / (2.0 * eps);
                    double ana = (*t.g)(i, j);
                    double den = std::max(std::fabs(ana), std::fabs(num));
                    if (den < 1e-12) den = 1.0;
                    double rel = std::fabs(ana - num) / den;
                    if (rel > worst) worst = rel;
                }
            char buf[128];
            std::snprintf(buf, sizeof(buf), "FD %s gradient: worst rel_err=%.3g < 1e-3", t.nm, worst);
            check(worst < 1e-3, buf);
        }
    }

    // Multi-head parameter gradients: the defect they pin is invisible with
    // one head, so H=1 is included only as a control.
    fd_param_grad_multihead(4, 3, 1);
    fd_param_grad_multihead(8, 4, 2);
    fd_param_grad_multihead(8, 4, 3);
    fd_param_grad_multihead(6, 5, 3);

    std::printf("=== Summary: %d passed, %d failed ===\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
