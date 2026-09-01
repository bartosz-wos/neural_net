#ifndef TTT_MLP_H
#define TTT_MLP_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <string>

// ============================================================================
// TTT-MLP (Test-Time Training, MLP Variant) — Sun et al. NeurIPS 2024
//   "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
//   https://arxiv.org/abs/2407.04620  (TTTMLP in https://github.com/test-time-training/ttt-lm-pytorch/blob/main/ttt.py)
//
// ============================================================================
//
// TTT-MLP generalises TTT-Linear: the persistent recurrent STATE is the two
// weight matrices (W1, W2) of a 2-layer GELU MLP. At each time step both
// matrices are updated via a single closed-form GD step on the same
// self-supervised reconstruction loss:
//
//   z_t   = W_in · input_t + b_in                ∈ R^{d_inner}      (pre-projection)
//   Z1_t  = W1_{t-1} · z_t + b1_{t-1}            ∈ R^{d_hidden}     (mlp_ratio=4 expansion)
//   X2_t  = GELU(Z1_t)                            ∈ R^{d_hidden}     (post-gelu hidden)
//   Z2_t  = W2_{t-1} · X2_t + b2_{t-1}            ∈ R^{d_inner}      (output of inner MLP)
//   err_t = Z2_t - z_t                           ∈ R^{d_inner}      (the "innovation")
//
//   # Single GD step on 0.5 · ||W2·gelu(W1·z) - z||² + 0.5 · λ · ||W - W_{t-1}||²
//   delta_t  = GELU'(Z1_t) ⊙ (W2_t^T · err_t)    ∈ R^{d_hidden}     (gradient w.r.t. Z1)
//   W2_t  = W2_{t-1} - η · err_t ⊗ X2_t / (||X2_t||² + λ)
//   b2_t  = b2_{t-1} - η · err_t / (||X2_t||² + λ)
//   W1_t  = W1_{t-1} - η · delta_t ⊗ z_t / (||z_t||² + λ)
//   b1_t  = b1_{t-1} - η · delta_t / (||z_t||² + λ)
//   o_t   = W2_t · GELU(W1_t · z_t + b1_t) + b2_t + b_in             ∈ R^{d_inner}
//   y_t   = W_out · o_t + b_out                    ∈ R^{d_model}
//
// The state IS (W1, W2, b1, b2). The "memory" of past tokens is encoded in
// how the matrices have been updated. As with TTTLinear, we adopt the
// single-step GD simplification (the paper's closed-form dual form is
// intractable for our standalone FD-vs-analytical gradient check).
//
// ----------------------------------------------------------------------------
// SHAPE CONVENTIONS:
//   x ∈ R^{T × d_model}
//   W_in  : (d_inner, d_model)        — pre-projection
//   W_out : (d_model, d_inner)        — post-projection
//   W1 ∈ R^{d_hidden × d_inner}       — persistent hidden-layer weights (fast)
//   W2 ∈ R^{d_inner × d_hidden}       — persistent output-layer weights (fast)
//   b1 ∈ R^{d_hidden}                 — persistent hidden-layer bias   (fast)
//   b2 ∈ R^{d_inner}                  — persistent output-layer bias   (fast)
//   b_in_  ∈ R^{d_inner}              — slow bias added after inner MLP
//   λ > 0                             — regularisation scalar (constant)
//   η > 0                             — per-step learning rate (constant)
//
// PER-TOKEN FORWARD CACHE (for backward):
//   last_input_   : (T, d_model)
//   last_z_       : (T, d_inner)
//   last_W1_t_    : (T+1, d_hidden, d_inner) flat
//   last_W2_t_    : (T+1, d_inner, d_hidden) flat
//   last_b1_t_    : (T+1, d_hidden)
//   last_b2_t_    : (T+1, d_inner)
//   last_Z1_      : (T, d_hidden)        — pre-activation, hidden
//   last_X2_      : (T, d_hidden)        — post-GELU(Z1)
//   last_Z2_      : (T, d_inner)         — W2_prev · X2 + b2_prev (pre-update)
//   last_o_pre_   : (T, d_inner)         — W2_t · gelu(W1_t · z + b1_t) + b2_t  (post-update)
//   last_o_       : (T, d_inner)         — last_o_pre_ + b_in_                  (input to W_out)
//
// PER-TOKEN BACKWARD (full BPTT through both fast-weight update rules):
//   Direct gradients (one per step, accumulated into W1/W2/b1/b2 grad buffers):
//     dW2_step[t] = grad_o[t] ⊗ X2_t / d_X
//     db2_step[t] = grad_o[t] / d_X
//     delta_t     = GELU'(Z1_t) ⊙ (W2_t^T · grad_o[t])    (note: W2 POST-update)
//     dW1_step[t] = delta_t ⊗ z_t / d_Z
//     db1_step[t] = delta_t / d_Z
//   where d_X = ||X2_t||² + λ and d_Z = ||z_t||² + λ.
//
//   Recurrence Jacobians (same structure as TTTLinear for each matrix):
//     A2_t = I - (η/d_X) · X2_t X2_t^T    — ∂W2_t/∂W2_{t-1}
//     A1_t = I - (η/d_Z) · z_t z_t^T       — ∂W1_t/∂W1_{t-1}
//
//   Reverse-time chain:
//     dL/dW2_t = dW2_step[t] + A2_{t+1}^T · dL/dW2_{t+1}
//     dL/dW1_t = dW1_step[t] + A1_{t+1}^T · dL/dW1_{t+1}
//     dL/dW2_{t-1} = A2_t^T · dL/dW2_t     (becomes dW2_total for next iter)
//     dL/dW1_{t-1} = A1_t^T · dL/dW1_t     (becomes dW1_total for next iter)
//
//   dz_t (for backprop through W_in) has TWO contributions:
//     (1) direct:   ∂o_pre_t/∂z_t  →  sum_j GELU'(Z1_t[j]) · W1_t[j,k] · (W2_t^T · grad_o[t])[j]
//     (2) recurrence: dL/dW1_t → ∂W1_t/∂z_t  (formula identical to TTTLinear's,
//         using delta_t as the "err" and z_t as the "z", with the (W1_prev - I) term
//         and the η · delta · z · 2z / d² denominator derivative)
//
// INVARIANTS:
//   - d_model > 0, d_inner > 0, mlp_ratio >= 1
//   - eta_ > 0 (learning rate)
//   - lambda_reg_ >= 0 (regularisation)
// ============================================================================

class TTTMLP : public Layer {
public:
    // d_model: input/output feature dim
    // d_inner: inner MLP dim (= state dim of W2, also the "embedding" of z)
    // mlp_ratio: W1 expands by this factor → d_hidden = mlp_ratio * d_inner
    TTTMLP(size_t d_model, size_t d_inner = 0,
           double eta = 0.1, double lambda_reg = 1e-4,
           size_t mlp_ratio = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "TTTMLP"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_inner() const { return d_inner_; }
    size_t d_hidden() const { return d_hidden_; }
    size_t mlp_ratio() const { return mlp_ratio_; }
    double eta() const { return eta_; }
    double lambda_reg() const { return lambda_reg_; }
    const Tensor& W1_state() const { return W1_state_; }
    const Tensor& W2_state() const { return W2_state_; }
    const Tensor& b1_state() const { return b1_state_; }
    const Tensor& b2_state() const { return b2_state_; }
    // Non-const accessors for in-place mutation (used by tests, custom init, etc.)
    Tensor& W1_state_ref() { return W1_state_; }
    Tensor& W2_state_ref() { return W2_state_; }
    Tensor& b1_state_ref() { return b1_state_; }
    Tensor& b2_state_ref() { return b2_state_; }
    const Tensor& W_in() const { return W_in_.weights; }
    Tensor& W_in_ref() { return W_in_.weights; }
    const Tensor& W_out() const { return W_out_.weights; }
    Tensor& W_out_ref() { return W_out_.weights; }
    const Tensor& b_in() const { return b_in_; }
    Tensor& b_in_ref() { return b_in_; }

    // Reset the persistent state (call between independent sequences)
    void reset_state();

private:
    // Dimensions
    size_t d_model_;
    size_t d_inner_;
    size_t d_hidden_;
    size_t mlp_ratio_;

    // Hyperparameters
    double eta_;
    double lambda_reg_;

    // Slow (learnable) projections + the inner-output slow bias b_in_
    Dense W_in_;          // (d_inner, d_model)
    Dense W_out_;         // (d_model, d_inner)
    Tensor b_in_;         // (1, d_inner)

    // Fast (persistent) state
    Tensor W1_state_;     // (d_hidden, d_inner)
    Tensor W2_state_;     // (d_inner, d_hidden)
    Tensor b1_state_;     // (1, d_hidden)
    Tensor b2_state_;     // (1, d_inner)

    // Gradient buffers for the slow parameters
    Tensor grad_W_in_w_;
    Tensor grad_W_in_b_;
    Tensor grad_W_out_w_;
    Tensor grad_W_out_b_;
    Tensor grad_b_in_;

    // Forward cache
    Tensor last_input_;
    Tensor last_z_;
    Tensor last_W1_t_;       // flat (T+1) * d_hidden * d_inner
    Tensor last_W2_t_;       // flat (T+1) * d_inner * d_hidden
    Tensor last_b1_t_;       // flat (T+1) * d_hidden
    Tensor last_b2_t_;       // flat (T+1) * d_inner
    Tensor last_Z1_;         // (T, d_hidden)
    Tensor last_X2_;         // (T, d_hidden)
    Tensor last_Z2_;         // (T, d_inner)
    Tensor last_o_pre_;      // (T, d_inner)
    Tensor last_o_;          // (T, d_inner)
    size_t last_T_;

    // Initialize the persistent state (called at construction and on reset)
    void initialize_state();
};

// Convenience wrapper: input projection -> TTTMLP -> TTTMLP -> output projection.
// Stacks two TTTMLP layers for a complete forward pass.
class TTTMLPModel : public Layer {
public:
    TTTMLPModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
                double eta = 0.1, double lambda_reg = 1e-4,
                size_t mlp_ratio = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "TTTMLPModel"; }

private:
    TTTMLP layer1_;
    TTTMLP layer2_;
    Dense proj_in_;
    Dense proj_out_;
};

#endif // TTT_MLP_H