#ifndef TTT_LINEAR_H
#define TTT_LINEAR_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <string>

// ============================================================================
// TTT-Linear (Test-Time Training, Linear Variant) — Sun et al. NeurIPS 2024
//   "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
//   https://arxiv.org/abs/2407.04620
//
// ============================================================================
//
// TTT-Linear is a sequence mixer where the recurrent STATE is the weight
// matrix W of a small linear "MLP". At each time step the layer updates W
// via a single closed-form gradient-descent step on a self-supervised
// reconstruction loss applied to the current input:
//
//   z_t   = LN(x_t) · W_in + b_in        ∈ R^{d_inner}    (pre-projection + LN)
//   target = z_t                                            (self-supervised)
//   y_hat_t = W_{t-1} · z_t                                 (linear-MLP output)
//   # Single GD step on quadratic loss 0.5 ||W z - z||² + 0.5 λ ||W - W_{t-1}||²
//   W_t   = W_{t-1} - η · (W_{t-1} z_t - target) ⊗ z_t / (||z_t||² + λ)
//   o_t   = W_t · z_t + b
//
// The state IS the matrix W; there is no separate hidden vector. The "memory"
// of past tokens is encoded in how W has been updated.
//
// ----------------------------------------------------------------------------
// SHAPE CONVENTIONS:
//   x ∈ R^{T × d_model}
//   W_in : (d_inner, d_model)        — pre-projection
//   W_out: (d_model, d_inner)        — post-projection
//   W ∈ R^{d_inner × d_inner}        — persistent state (the "fast weights")
//   b ∈ R^{d_inner}                  — slow bias
//   λ > 0                            — regularization scalar (constant)
//   η > 0                            — per-step learning rate (constant for v1)
//
// PER-TOKEN FORWARD CACHE (for backward):
//   last_z_         : (T, d_inner)             pre-projected inputs
//   last_W_t_       : (T+1, d_inner, d_inner)  W at each time step (W_{-1} is initial, W_T is final)
//   last_o_pre_     : (T, d_inner)             W_t · z_t (pre-bias, post-W update)
//   last_input_     : (T, d_model)             original input
//
// PER-TOKEN BACKWARD:
//   Full BPTT through the T updates. The gradient on W_0 (initial state) is
//   the sum of indirect gradients from each later token via the recurrence:
//     dW_0 = sum_{t=0..T-1} (Π_{s=t+1..T-1} dW_s/dW_{s-1})^T · dW_t_direct
//   We compute this by reverse-time accumulation: starting from dW_T (zero),
//   at each t we add the direct contribution dW_t_direct and then propagate
//   backward through the update rule.
//
// KEY SIMPLIFICATION (paper §3.2):
//   The full closed-form "TTT-Linear with ridge regression" has the closed
//   form W_t = (1-α_t) W_{t-1} + α_t (regression fit) where α_t depends on
//   the inner product z_t^T W_{t-1} z_t / (||z_t||² + λ). The PAPER's
//   simplification (and the one that makes the gradient check tractable) is
//   the single-step GD rule:
//
//     W_t = W_{t-1} - η_t · (W_{t-1} z_t - z_t) ⊗ z_t / (||z_t||² + λ)
//
//   This is what we implement. The gradient through this is well-defined and
//   the FD check works out cleanly.
//
// INVARIANTS:
//   - d_model > 0, d_inner > 0
//   - eta_ > 0 (learning rate)
//   - lambda_ >= 0 (regularization)
// ============================================================================

class TTTLinear : public Layer {
public:
    // d_model: input/output feature dim
    // d_inner: hidden dim of the linear MLP (== state size)
    TTTLinear(size_t d_model, size_t d_inner = 0,
              double eta = 0.1, double lambda_reg = 1e-4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "TTTLinear"; }

    // Public accessors
    size_t d_model() const { return d_model_; }
    size_t d_inner() const { return d_inner_; }
    double eta() const { return eta_; }
    double lambda_reg() const { return lambda_reg_; }
    const Tensor& W_in() const { return W_in_.weights; }
    const Tensor& W_out() const { return W_out_.weights; }
    const Tensor& bias() const { return bias_; }
    const Tensor& W_state() const { return W_state_; }

    // Reset the persistent state (call between independent sequences)
    void reset_state();

private:
    // Dimensions
    size_t d_model_;
    size_t d_inner_;

    // Hyperparameters
    double eta_;
    double lambda_reg_;

    // Learnable parameters (slow weights)
    Dense W_in_;        // (d_inner, d_model) — pre-projection
    Dense W_out_;       // (d_model, d_inner) — post-projection
    Tensor bias_;       // (1, d_inner) — per-channel bias

    // Persistent state (fast weights)
    Tensor W_state_;    // (d_inner, d_inner)

    // Gradient buffers (slow weights only — state gradient is accumulated via last_W_t_)
    Tensor grad_W_in_w_;
    Tensor grad_W_in_b_;
    Tensor grad_W_out_w_;
    Tensor grad_W_out_b_;
    Tensor grad_bias_;

    // Cached forward state for backward
    Tensor last_input_;      // (T, d_model)
    Tensor last_z_;          // (T, d_inner) — post-projection pre-LN
    Tensor last_W_t_;        // (T+1, d_inner, d_inner) — flat, indexed as (t * d_inner + i) * d_inner + j
    Tensor last_o_pre_;      // (T, d_inner) — W_t · z_t (pre-bias)
    Tensor last_resid_;      // (T, d_inner) — last_W_t_ · z_t - z_t (the "innovation" for backward)
    size_t last_T_;

    // Initialize the persistent state (called at construction and on reset)
    void initialize_state();

    // Per-token single GD step: W_{t} = W_{t-1} - η · (W_{t-1} z - z) ⊗ z / (||z||² + λ)
    // Returns the new W_t.
    static Tensor apply_step(const Tensor& W_prev, const Tensor& z_row,
                             double eta, double lambda_reg);
};

// Convenience wrapper: input projection -> TTTLinear -> output projection.
// Stacks two TTTLinear layers for a complete forward pass.
class TTTLinearModel : public Layer {
public:
    TTTLinearModel(size_t input_dim, size_t hidden_dim, size_t output_dim,
                   double eta = 0.1, double lambda_reg = 1e-4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "TTTLinearModel"; }

private:
    TTTLinear layer1_;
    TTTLinear layer2_;
    Dense proj_in_;
    Dense proj_out_;
};

#endif // TTT_LINEAR_H