#ifndef MAMBA_H
#define MAMBA_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Mamba (S6) — Gu & Dao 2023
//   "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
// ============================================================================
//
// The Mamba layer is a sequence mixer that replaces attention with a
// *selective* state-space model (SSM). "Selective" means the SSM parameters
// (Δ, B, C) are FUNCTIONS OF THE INPUT, allowing the model to content-awarely
// filter what to remember and what to forget. This is the central
// contribution relative to S4, where (A, B, C) are fixed (HIPPO-initialized).
//
// Mathematical formulation (per sequence position t, with N=batch tokens):
//
//   Input:        x_t in R^{d_model}                  (t = 0..T-1)
//   Block input:  u = [u_0; ...; u_{T-1}],  u in R^{T x d_model}
//
// Step 1 — Input projection (combine ssm and gate paths):
//   p_t = in_proj(u_t)                              in R^{2 * d_inner}
//   x_pre_t = p_t[:d_inner]                         (raw ssm path input)
//   g_t     = p_t[d_inner:]                         (gate path input)
//   x_tilde_t = silu(x_pre_t)                       (silu before SSM — canonical Mamba pattern without conv)
//
// Step 2 — Input-dependent SSM parameters:
//   Δ_t  = softplus(dt_proj(u_t))                   in R^{d_inner}        (d_inner learnable dt projection: d_model -> d_inner)
//   B_t  = B_proj(u_t)                              in R^{d_state}        (d_state learnable B projection: d_model -> d_state)
//   C_t  = C_proj(u_t)                              in R^{d_state}        (d_state learnable C projection: d_model -> d_state)
//   Note: Δ_t > 0 elementwise; softplus ensures positivity.
//
// Step 3 — Discretization of the continuous-time SSM (A is fixed in Mamba-1
//          at the layer level, Δ_t makes it selective):
//   A is a learnable matrix in R^{d_inner x d_state}, constrained to be
//   negative: A = -exp(A_log). (We store A_log, the unconstrained parameter.)
//   Ā_t = exp(Δ_t ⊗ A)                             in R^{d_inner x d_state}   (Zero-Order Hold discretization)
//   B̄_t = Δ_t ⊗ B_t                                in R^{d_inner x d_state}
//
// Step 4 — Selective state-space recurrence (run for t = 0..T-1):
//   h_0 = 0
//   h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊗ x_tilde_t         in R^{d_inner x d_state}
//   y_t = C_t · h_t                                 in R^{d_inner}              (contraction over d_state)
//
// Step 5 — Skip / gating + output projection:
//   gated_t  = silu(g_t) ⊙ y_t                     in R^{d_inner}
//   out_t    = out_proj(gated_t)                    in R^{d_model}
//
// All 6 parameter sources are learnable:
//   in_proj, out_proj, dt_proj, B_proj, C_proj, A_log, D
// where D is a learnable diagonal skip-connection vector in R^{d_inner}.
//
// ----------------------------------------------------------------------------
// Implementation notes:
//
//  * We follow the Mamba paper's convention of "exp(A_log)" reparameterization
//    (A = -exp(A_log)) so A stays negative without constraint violation.
//  * The state h has shape (T, d_inner, d_state) per token (we treat each
//    row of the input as a "token" of dimension d_model). For numerical
//    gradient checks we use small d_inner, d_state (e.g. 2, 2) and T = 3-4.
//  * The recurrence is a pure Python-style loop; we cache all intermediate
//    states (Ā_t, B̄_t, h_t, Δ_t, B_t, C_t) for analytical backward.
//  * We do NOT include the depthwise 1D conv that the canonical Mamba has
//    before the SSM — it's a regularizer/performance add-on, not part of
//    the selective scan math. Keeping it out makes the gradient check
//    tractable and the code clean.
// ============================================================================

class MambaBlock : public Layer {
public:
    // d_model:    input/output feature dim
    // d_state:    SSM state dimension (N in the paper, number of "modes")
    // d_inner:    inner feature dim (default = 2 * d_model, matches paper convention)
    MambaBlock(size_t d_model, size_t d_state, size_t d_inner = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "MambaBlock"; }

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }
    size_t d_inner() const { return d_inner_; }

    // Parameter tensors (public for gradient tests)
    Dense in_proj;          // (d_model -> 2*d_inner)
    Dense out_proj;         // (d_inner -> d_model)
    Dense dt_proj;          // (d_model -> d_inner)  — pre-softplus Δ
    Dense B_proj;           // (d_model -> d_state)
    Dense C_proj;           // (d_model -> d_state)
    Tensor A_log;           // (d_inner, d_state)  — unconstrained A; A = -exp(A_log)
    Tensor D_skip;          // (1, d_inner)         — learnable skip connection

    // Hidden gradient buffers (set in backward, used in update_weights)
    Tensor grad_A_log_;     // (d_inner, d_state)
    Tensor grad_D_skip_;    // (1, d_inner)

private:
    size_t d_model_;
    size_t d_state_;
    size_t d_inner_;

    // Caches for forward (all reshaped to (T, d_inner) or (T, d_inner, d_state))
    Tensor last_input_;         // (T, d_model)
    Tensor last_p_;             // (T, 2*d_inner)   in_proj output
    Tensor last_ssm_in_;        // (T, d_inner)     ssm path input (= p[:, :d_inner])
    Tensor last_gate_;          // (T, d_inner)     gate path (= p[:, d_inner:])
    Tensor last_Delta_;         // (T, d_inner)     post-softplus Δ
    Tensor last_Delta_pre_;     // (T, d_inner)     pre-softplus Δ (dt_proj output)
    Tensor last_B_t_;           // (T, d_state)     per-token B
    Tensor last_C_t_;           // (T, d_state)     per-token C
    Tensor last_A_bar_;         // (T, d_inner, d_state)  exp(Δ_t ⊗ A)
    Tensor last_B_bar_;         // (T, d_inner, d_state)  Δ_t ⊗ B_t
    Tensor last_h_;             // (T+1, d_inner, d_state)  state sequence (h_0 = 0)
    Tensor last_y_;             // (T, d_inner)     SSM output per token
    Tensor last_gated_;         // (T, d_inner)     silu(gate) ⊙ y

    // dt_proj has a special form: y = softplus(dt_proj(x)), so the "weights"
    // we want to backprop are dt_proj.weights, dt_proj.bias. We don't use
    // the Dense's own update — we manage it directly. This is so we can
    // backprop through the softplus element-wise.

    // Helpers
    static double softplus(double x) {
        // Numerically stable softplus: log(1 + exp(x))
        // For large x, log(1 + exp(x)) ~ x.
        if (x > 30.0) return x;
        if (x < -30.0) return std::exp(x);
        return std::log(1.0 + std::exp(x));
    }
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double silu(double x) {
        return x * sigmoid(x);
    }
};

#endif // MAMBA_H
