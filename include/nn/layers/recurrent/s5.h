#ifndef S5_H
#define S5_H

#include "../../core/layer.h"
#include "../../core/tensor.h"
#include <vector>
#include <complex>
#include <cmath>

// ============================================================================
// S5 — Simplified State Space Layers for Sequence Modeling
//   Smith, Warrington, Linderman, Gu  (ICLR 2022)
//   https://arxiv.org/abs/2208.04933
//
// ============================================================================
//
// The S5 layer is a single-input multi-output (SIMO) state space layer with a
// DIAGONAL state matrix A ∈ ℂ^{N×N}, parameterised by HiPPO-LegS eigenvalues.
// Per the paper §2.2, diagonalising A reduces the SSM to a bank of N
// independent scalar SSMs, all driven by the same scalar input. Each channel
// is then bilinear-discretised (Tustin) with a fixed step Δ:
//
//   Ā_n = (1 + Δ/2 · A_n) / (1 − Δ/2 · A_n)
//   B̄_n = B_n / (1 − Δ/2 · A_n)
//
// Discrete recurrence (paper eq. 5):
//   h_t[n] = Ā_n · h_{t-1}[n] + B̄_n · x_t
//   y_t    = Re( Σ_n C[n] · h_t[n] ) + D · x_t
//
// This is the SINGLE-CHANNEL S5 primitive. For multi-channel inputs
// (d_model > 1), we apply an input Dense projection, run d_state independent
// channels (each driven by the same scalar x_t within a channel), and apply
// an output Dense projection. The Dense projections handle cross-channel
// mixing; the SSM itself is depthwise (one channel per A/B/C entry).
//
// We do NOT diagonalise a non-diagonal A — instead we use the closed-form
// HiPPO-LegS eigenvalues A_n = −sqrt((2n+1)(2n+2)), which are already real
// and negative. B_n = sqrt(2n+1) (paper §2.2.1). C is randomly initialised
// (both real and imaginary parts).
//
// Bidirectional extension (paper §3.1): a backward SSM processes the input
// in reverse, and the two outputs are mixed via a per-channel learnable gate
// Λ_gate ∈ (0, 1) parameterised as sigmoid(Λ_gate_log).
//
// Backward derivation (hand-derived; see plan §"Backward derivation"):
//   1. dh_t = conj(C) · d_y_t (per channel) — the SSM is linear, so the
//      adjoint of "y = Re(C^H h)" w.r.t. h is the complex conjugate of C.
//   2. Reverse sweep: propagate dh_t into dh_{t-1} via Ā.
//   3. dĀ = Σ_t dh_t · conj(h_{t-1}); dB̄ = Σ_t dh_t · conj(x_t).
//   4. Tustin inverse chain rule (per channel).
//   5. dC_n = Σ_t conj(h_t) · d_y_t.
//   6. d_x (SSM-level) = Re(conj(B̄) · dh_t).
//
// All parameter tensors (A_re, A_im, B_re, B_im, C_re, C_im, D_skip, and
// Λ_gate_log) are exposed as public members for test inspection.
// ============================================================================

class S5Layer : public Layer {
public:
    // d_model:    input/output feature dim
    // d_state:    SSM state dim N (one complex diagonal entry per dimension)
    // bidirectional: if true, run a forward + a backward SSM and mix with a
    //                per-channel learnable gate Λ_gate.
    // dt:         fixed Tustin step size (paper convention: small constant)
    S5Layer(size_t d_model, size_t d_state, bool bidirectional = true, double dt = 0.1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return A_re_; }
    Tensor get_gradients() const override { return grad_A_re_; }
    std::string name() const override { return "S5Layer"; }

    size_t d_model() const { return d_model_; }
    size_t d_state() const { return d_state_; }
    double dt() const { return dt_; }

    // Accessors for tests — return complex A_n as (re, im) pair.
    // Stored as 2 Tensors of shape (d_state,).
    const Tensor& A_re() const { return A_re_; }
    const Tensor& A_im() const { return A_im_; }
    const Tensor& B_re() const { return B_re_; }
    const Tensor& B_im() const { return B_im_; }
    const Tensor& C_re() const { return C_re_; }
    const Tensor& C_im() const { return C_im_; }

    // For the S5Block wrapper, expose the post-discretisation parameters too:
    const Tensor& Abar_re() const { return Abar_re_; }
    const Tensor& Abar_im() const { return Abar_im_; }
    const Tensor& Bbar_re() const { return Bbar_re_; }
    const Tensor& Bbar_im() const { return Bbar_im_; }

    // The forward caches (for testing / debugging).
    const Tensor& last_input() const { return last_input_; }
    const Tensor& last_h_re() const { return last_h_re_; }
    const Tensor& last_h_im() const { return last_h_im_; }
    const Tensor& last_h_bwd_re() const { return last_h_bwd_re_; }
    const Tensor& last_h_bwd_im() const { return last_h_bwd_im_; }

    // Parameters — public for test access
    Tensor A_re_;            // (d_state,) real part of A diagonal
    Tensor A_im_;            // (d_state,) imag part of A diagonal
    Tensor B_re_;            // (d_state,)
    Tensor B_im_;            // (d_state,)
    Tensor C_re_;            // (d_state,)
    Tensor C_im_;            // (d_state,)
    Tensor D_skip_;          // (1, 1) — skip connection scalar (per-channel is overkill)
    Tensor gate_log_;        // (d_state,) — bidirectional gate, sigmoid(gate_log_) ∈ (0,1)

    // Gradient accumulators
    Tensor grad_A_re_;
    Tensor grad_A_im_;
    Tensor grad_B_re_;
    Tensor grad_B_im_;
    Tensor grad_C_re_;
    Tensor grad_C_im_;
    Tensor grad_D_skip_;
    Tensor grad_gate_log_;

    // Internal accumulators used inside backward (zeroed at start).
    Tensor grad_Abar_re_acc_;
    Tensor grad_Abar_im_acc_;
    Tensor grad_Bbar_re_acc_;
    Tensor grad_Bbar_im_acc_;

private:
    // (initialised first — scalars)
    size_t d_model_;
    size_t d_state_;
    double dt_;
    bool bidirectional_;

    // Discretised parameters (computed at forward, cached for backward).
    Tensor Abar_re_;
    Tensor Abar_im_;
    Tensor Bbar_re_;
    Tensor Bbar_im_;

    // Forward caches
    Tensor last_input_;
    Tensor last_h_re_;
    Tensor last_h_im_;
    size_t last_T_ = 0;

    // Backward-direction caches (only used when bidirectional_ = true).
    Tensor last_h_bwd_re_;
    Tensor last_h_bwd_im_;
    size_t last_T_bwd_ = 0;
};

// ============================================================================
// S5Block — wraps an S5Layer with input/output Dense projections and residual
// gating. Forward: y = Dense_out(S5Layer(Dense_in(x))) + x. The residual
// gives a well-conditioned optimisation target and is the standard S4D/S5
// pattern.
// ============================================================================

class S5Block : public Layer {
public:
    // d_model:     input/output feature dim
    // d_state:     S5 internal state dim
    // bidirectional: passed to S5Layer
    S5Block(size_t d_model, size_t d_state, bool bidirectional = true, double dt = 0.1);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return dense_in_.get_weights(); }
    Tensor get_gradients() const override { return dense_in_.get_gradients(); }
    std::string name() const override { return "S5Block"; }

    Dense dense_in_;
    S5Layer s5_;
    Dense dense_out_;

    // Caches
    Tensor last_input_;
    Tensor last_proj_in_;
};

#endif // S5_H