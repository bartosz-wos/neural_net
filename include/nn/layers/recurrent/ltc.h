#ifndef LTC_H
#define LTC_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <string>
#include <cstddef>

// ============================================================================
// Liquid Time-Constant (LTC) Networks — Hasani, Lechner, Amini, Rusch, Grosu
//   "Liquid Time-Constant Networks" (AAAI 2021)
//   https://arxiv.org/abs/2006.04439
//
// ============================================================================
//
// LTC is an ODE-inspired recurrent layer where each neuron has a learned,
// INPUT- AND STATE-DEPENDENT time constant τ_i(x, h, I) > 0. The dynamics come
// from the "synaptic" ODE:
//
//   dh_i/dt = -h_i/τ_i(x, h, I) + (1/τ_i(x, h, I)) · A_i(x, h)
//
// where A_i(x, h) = W_ih[i]·x + W_hh[i]·h + b[i] is the standard RNN affine
// map and τ_i is a positive scalar that depends on the current input and
// state (the "Liquid" part). For piecewise-constant τ between observations,
// the ODE has an exact closed-form solution between time steps Δt apart:
//
//   h_i(t+Δt) = h_i(t)·exp(-Δt/τ_i) + A_i · (1 - exp(-Δt/τ_i))
//
// We use Δt = 1 (one step per input token) and saturate via tanh to give the
// bi-stable dynamics the LTC paper highlights:
//
//   h_i(t+1) = tanh( g_i · h_i(t) + α_i · A_i(t) )
//
// with g_i = exp(-1/τ_i) and α_i = 1 - g_i. Note that this can be rewritten
// as h_t_input = A_t + g_t·(h_{t-1} - A_t), so dh_t_input/dg = h_{t-1} - A_t
// (the form we use in backward).
//
// ----------------------------------------------------------------------------
// Per-neuron, per-time step (single-step BPTT):
//
// Input:  X in R^{T x input_dim}
//
// Step 1 — Affine map A_t = W_ih @ x_t + W_hh @ h_{t-1} + b ∈ R^{hidden}
// Step 2 — Time-constant logit z_τ_t = W_tx @ x_t + W_th @ h_{t-1} + b_τ ∈ R^{hidden}
// Step 3 — Per-neuron τ_t = exp(log_τ_base) · softplus(z_τ_t) > 0
// Step 4 — Decay coefficient g_t = exp(-1/τ_t) ∈ (0, 1), drive α_t = 1 - g_t
// Step 5 — h_t = tanh(g_t · h_{t-1} + α_t · A_t)
//
// Output: H in R^{T x hidden_size}
//
// ----------------------------------------------------------------------------
// Learnable parameters (7 tensors total):
//   W_ih        (hidden x input)   — input-to-hidden affine
//   W_hh        (hidden x hidden)  — hidden-to-hidden affine
//   b           (hidden x 1)       — hidden bias
//   W_tx        (hidden x input)   — input contribution to τ
//   W_th        (hidden x hidden)  — state contribution to τ
//   b_τ         (hidden x 1)       — τ bias
//   log_τ_base  (hidden x 1)       — log of per-neuron baseline τ (τ_base = exp(log_τ_base))
//
// We store log_τ_base as the unconstrained parameter (positive without
// constraint violation via exp()), with default init τ_base = 2.0 so
// log_τ_base = ln(2.0) ≈ 0.693. softplus is used inside τ = τ_base · sp(z_τ)
// to keep τ positive. τ ≈ 2.0·ln(2) ≈ 1.39 at init → g ≈ exp(-0.72) ≈ 0.49
// (even split between "carry" and "drive").
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   X:              (T, input_dim)
//   h_t, A_t, z_τ_t, τ_t, g_t, α_t: (hidden,)
//   W_ih, W_tx:     (hidden, input_dim)
//   W_hh, W_th:     (hidden, hidden)
//   b, b_τ, log_τ_base: (hidden, 1)
//
// ----------------------------------------------------------------------------
// Implementation choices:
//   * Single layer (user stacks for multi-layer).
//   * Per-neuron τ depends on BOTH input x_t AND previous state h_{t-1}
//     (the canonical paper formulation; the constant-τ variant is a strict
//     subset).
//   * Saturation is tanh (canonical paper choice; sigmoid would also work).
//   * softplus is computed in the stable form max(z,0) + log1p(exp(-|z|)).
//   * We do NOT track the full continuous-time trajectory — one closed-form
//     step per input is the canonical LTC used in practice (cf. ncps library).
//
// ============================================================================

class LTC : public Layer {
public:
    LTC(size_t input_dim, size_t hidden_size, double tau_base_init = 2.0);

    // Input: (T, input_dim). Output: (T, hidden_size).
    Tensor forward(const Tensor& input) override;

    // Standard backward: receives grad_output (T, hidden_size) and returns
    // grad_input (T, input_dim). Internal parameter gradients accumulate
    // via the standard Layer convention.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override { return W_ih_; }
    Tensor get_gradients() const override { return grad_W_ih_; }
    std::string name() const override { return "LTC"; }

    // Test introspection: cached values from the most recent forward call.
    // All are shape (T, hidden_size); empty Tensor when forward has not run.
    Tensor last_tau() const { return cache_tau_; }
    Tensor last_g() const { return cache_g_; }
    Tensor last_alpha() const { return cache_alpha_; }
    Tensor last_h() const { return cache_h_; }
    Tensor last_A() const { return cache_A_; }
    Tensor last_pre_tanh() const { return cache_pre_tanh_; }
    Tensor last_z_tau() const { return cache_z_t_; }

    // Public Dense-style parameter accessors for testing/inspection.
    size_t input_dim() const { return input_dim_; }
    size_t hidden_size() const { return hidden_size_; }

public:
    // Public parameter tensors (for tests that need to set them directly).
    Tensor W_ih_, W_hh_, b_;
    Tensor W_tx_, W_th_, b_t_;
    Tensor log_tau_base_;

    // Public gradient tensors.
    Tensor grad_W_ih_, grad_W_hh_, grad_b_;
    Tensor grad_W_tx_, grad_W_th_, grad_b_t_;
    Tensor grad_log_tau_base_;

private:
    size_t input_dim_;
    size_t hidden_size_;

    // Caches from forward (all per-token, per-neuron; shape (T, hidden_size)).
    Tensor cache_x_;            // (T, input_dim)  — input snapshot
    Tensor cache_A_;            // (T, hidden_size) — affine pre-activation
    Tensor cache_z_t_;          // (T, hidden_size) — τ logit
    Tensor cache_tau_;          // (T, hidden_size) — τ values (>0)
    Tensor cache_g_;            // (T, hidden_size) — decay coefficients (0,1)
    Tensor cache_alpha_;        // (T, hidden_size) — drive coefficients (0,1)
    Tensor cache_pre_tanh_;     // (T, hidden_size) — pre-tanh input
    Tensor cache_h_;            // (T, hidden_size) — output (post-tanh)
};

#endif // LTC_H