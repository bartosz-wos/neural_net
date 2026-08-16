#ifndef MAMBA3_H
#define MAMBA3_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Mamba-3 — Dao & Gu 2025
//   "Mamba-3: State Space Models as General Purpose Backbone"
//
// Mamba-3 is the third generation of the Mamba SSM family, building on
// Mamba-1 (selective SSM with input-dependent Δ, B, C) and Mamba-2
// (scalar-times-identity structured state space with structured mask).
//
// The three central innovations of Mamba-3 vs Mamba-1/2:
//
//   1. **Per-channel complex eigenvalues**. Instead of real-valued scalar
//      decay, Mamba-3 stores `A[c] = a[c] · exp(i · θ[c])` per channel, where
//      `a[c] ∈ (0, 1)` is the decay magnitude and `θ[c] ∈ (-π, π)` is the
//      rotation phase. This is a damped oscillator per channel: the
//      recurrence becomes 2-D in (real, imag) space.
//
//   2. **Trapezoidal-rule discretization** instead of Zero-Order Hold
//      (Euler). For per-step `Δ_t`:
//          Ā_t = (1 + 0.5 · Δ_t · A) / (1 - 0.5 · Δ_t · A)
//          B̄_t = Δ_t · B / (1 - 0.5 · Δ_t · A)
//      This is the second-order accurate variant and gives Mamba-3 its
//      better gradient flow.
//
//   3. **Rotational positional encoding** (RoPE-style) on the input path:
//      `x_ssm_t[c] *= exp(i · m · pos_t · θ_base[c])` per channel. This
//      replaces convolutions from Mamba-1 and gives the rotational SSM
//      property — the model can attend to relative positions without an
//      explicit position embedding.
//
// ----------------------------------------------------------------------------
// Mathematical formulation (single channel c, time t):
//
//   Input: x_t ∈ R^{d_model}, t = 0..T-1.
//
//   Per-channel complex eigenvalue: A[c] = exp(A_log[c]) · exp(i · theta[c])
//   where exp(A_log[c]) ∈ (0, ∞) gives the decay magnitude (after the
//   `exp(-exp(A_log))` shape — matching Mamba convention; here we use
//   `exp(A_log) ∈ (0, ∞)` and the trapezoidal form handles stability).
//   For numerical safety we clamp A_log[c] ≤ 0 so exp(A_log) ≤ 1.
//
//   Rotational input encoding: angle[c, t] = t · theta_base[c]
//     where theta_base[c] is a learnable per-channel base rotation rate
//     (we parameterize via a separate `theta_base` tensor).
//     x_ssm_t[c] (rotated) = x_ssm_t[c] · (cos(angle) + i · sin(angle))
//
//   Discretization (trapezoidal rule):
//     denom[c, t]   = 1 - 0.5 · Δ_t · A[c]   (complex)
//     A_bar[c, t]  = (1 + 0.5 · Δ_t · A[c]) / denom[c, t]   (complex)
//     B_bar[c, t]  = Δ_t · B[c] / denom[c, t]              (complex)
//     where Δ_t = softplus(dt_proj(x_t) + dt_bias)  (per-head scalar, but
//     used as a scalar multiplier on the per-channel complex A[c]).
//
//   SSD recurrence (per channel):
//     h_0[c] = 0   (complex)
//     h_t[c] = A_bar[c, t] · h_{t-1}[c] + B_bar[c, t] · x_ssm_t_rotated[c]
//     o_t[c] = Re(h_t[c]) + D_skip[c] · x_ssm_t_raw[c]
//
//   Output:
//     gated_t = silu(gate_t) ⊙ o_t
//     out_t   = out_proj(gated_t)   ∈ R^{d_model}
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   x:               (T, d_model)
//   in_proj output:  (T, 2*d_inner)         weights: (2*d_inner, d_model)
//   x_ssm_t:         (T, d_inner)
//   gate_t:          (T, d_inner)
//   Δ_t (per head):  (T, n_heads)
//   b_proj output:   (T, d_inner)           weights: (d_inner, d_model)
//   A_log:           (1, d_inner)
//   theta:           (1, d_inner)           (rotation phase)
//   theta_base:      (1, d_inner)           (per-channel RoPE base)
//   D_skip:          (1, d_inner)
//   dt_bias:         (1, n_heads)
//   h state:         ((T+1)*d_inner, 2)     (real, imag)
//   o_t:             (T, d_inner)
//   out_proj output: (T, d_model)
// ----------------------------------------------------------------------------
// Initialization convention (Mamba-3 paper):
//   * A_log initialized to 0.0 (decay magnitude = 1.0 — fully oscillatory
//     at init; allows gradient to learn the decay).
//   * theta initialized to small values (0.0 → no rotation at init; can be
//     perturbed by training).
//   * theta_base initialized to small values (paper uses ~0.01).
//   * dt_bias initialized to 1.0 (sigmoid(1) ≈ 0.731, softplus(sigmoid^-1(0.5)) ≈ 0.5).
//   * D_skip initialized to 1.0 (matching Mamba-1 convention).
//   * All Dense projection biases initialized to 0 (dt_proj bias is folded
//     into dt_bias explicitly).
// ============================================================================

class Mamba3Block : public Layer {
public:
    // d_model:  input/output feature dim
    // n_heads:  number of attention heads (= number of channel groups)
    // d_inner:  inner feature dim (default = 2 * d_model, matching Mamba-1)
    //           must be divisible by n_heads
    Mamba3Block(size_t d_model, size_t n_heads, size_t d_inner = 0);

    // Input: (T, d_model). Output: (T, d_model).
    Tensor forward(const Tensor& input) override;

    // Standard backward contract: receives grad_output (T, d_model) and
    // returns grad_input (T, d_model). Internal Dense projections
    // accumulate their own gradients via the standard Layer convention.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "Mamba3Block"; }

    // Test introspection
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t d_inner() const { return d_inner_; }
    size_t head_dim() const { return head_dim_; }

    // Parameter tensors (public for gradient tests)
    Dense in_proj;          // (d_model -> 2*d_inner)
    Dense out_proj;         // (d_inner -> d_model)
    Dense dt_proj;          // (d_model -> n_heads)
    Dense b_proj;           // (d_model -> d_inner)

    Tensor A_log;           // (1, d_inner) — log of decay magnitude (≤ 0)
    Tensor theta;           // (1, d_inner) — rotation phase (init small)
    Tensor theta_base;      // (1, d_inner) — RoPE-style per-channel base rotation
    Tensor D_skip;          // (1, d_inner) — skip connection
    Tensor dt_bias;         // (1, n_heads)

    // Hidden gradient buffers (for the non-Dense parameters)
    Tensor grad_A_log_;
    Tensor grad_theta_;
    Tensor grad_theta_base_;
    Tensor grad_D_skip_;
    Tensor grad_dt_bias_;

private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_inner_;
    size_t head_dim_;       // = d_inner / n_heads

public:  // (test introspection — these caches need to be readable from tests for debugging)
    // Caches for forward (filled in forward, used in backward)
    Tensor last_input_;             // (T, d_model)
    Tensor last_p_;                 // (T, 2*d_inner)
    Tensor last_x_ssm_;             // (T, d_inner) — raw ssm input (real)
    Tensor last_gate_;              // (T, d_inner)
    Tensor last_dt_pre_;            // (T, n_heads)
    Tensor last_dt_;                // (T, n_heads) — softplus(dt_pre + dt_bias)
    Tensor last_b_;                 // (T, d_inner)
    Tensor last_x_ssm_rotated_;     // (T * d_inner, 2) — (real, imag)
    Tensor last_h_;                 // ((T+1)*d_inner, 2) — complex state
    Tensor last_A_bar_;             // (T * d_inner, 2) — discretized A_bar (complex)
    Tensor last_B_bar_;             // (T * d_inner, 2) — discretized B_bar (complex)
    Tensor last_o_;                 // (T, d_inner) — output per token (real)
    Tensor last_gated_;             // (T, d_inner) — silu(gate) � o

private:

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double silu(double x) {
        return x * sigmoid(x);
    }
    static double silu_deriv(double x) {
        // d/dx [x·sigmoid(x)] = sigmoid(x) + x · sigmoid(x) · (1 - sigmoid(x))
        double s = sigmoid(x);
        return s + x * s * (1.0 - s);
    }
};

#endif // MAMBA3_H
