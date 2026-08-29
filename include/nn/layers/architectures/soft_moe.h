#ifndef SOFT_MOE_H
#define SOFT_MOE_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>

// ============================================================================
// Soft MoE (Puigcerver et al. ICLR 2024, https://arxiv.org/abs/2308.00951,
// "From Sparse to Soft Mixtures of Experts")
//
// A fully-differentiable alternative to top-k sparse MoE. Instead of routing
// each token to ONE expert (hard selection), each expert owns a fixed number of
// "slots" and tokens are SOFTLY distributed across slots via a learned
// softmax dispatch matrix.
//
// Math (paper §2.2):
//   X  ∈ R^(T × d_model)
//   W_disp ∈ R^(E·S × d_model)            per-slot linear
//   D  = softmax(X · W_disp^T, axis=1)     (T, E·S) — dispatch weights
//   X' = D^T @ X                           (E·S, d_model) — slot inputs
//   W_comb ∈ R^(E·S × d_model)            per-slot linear
//   C  = softmax(X' · W_comb^T, axis=1)    (E·S, T) — combine weights
//   Y  = expert_1(X'_1) ; ... ; expert_E(X'_E)    (E·S, d_model)
//   output = C^T @ Y                       (T, d_model)
//
// Each expert is a 2-layer FFN: slot → hidden → slot, with ReLU between.
// W1_e : (d_expert, d_model), b1_e : (1, d_expert)
// W2_e : (d_model, d_expert), b2_e : (1, d_model)
//
// Backward chain (forward caches are bolded):
//   dC        = d_output^T @ Y                              (E·S, T)
//   dY        = C @ d_output                                (E·S, d_model)
//   dX'       = chain through (per-expert) experts and combine:
//               dX' += dY_via_expert  (from expert FFN backward)
//               dX' += W_comb^T @ (dC ⊙ row-jacobian of softmax)
//   dW_comb   = X'^T @ (dC ⊙ row-jacobian of softmax)
//   dD        = chain through slot inputs:
//               dD_via_Y  = (per-expert output gradient back to X')
//               dD_via_C  = X @ W_comb^T @ softmax-jacobian (∂C/∂X')
//               dD[t, i] = sum_k dD_via_X'[i] · X[t, k]   + dD_via_C chain
//   dW_disp   = X^T @ (dD_logits ⊙ row-jacobian of softmax)
//   d_input   = chain through dispatch: D chain back to X via W_disp^T
//
// The implementation keeps all cached tensors public-facing for FD validation.
// ============================================================================

class SoftMoELayer : public Layer {
public:
    size_t d_model_;
    size_t num_experts_;
    size_t slots_per_expert_;
    size_t d_expert_;
    size_t num_slots_;     // = E * S

    // Per-slot linear projections for dispatch (shared across slots — one matmul).
    Dense W_disp_;                                            // (E*S, d_model)

    // Per-slot combine: each slot has its own Dense mapping (1, d_model) -> (1, T)
    // with weights (T, d_model). We instantiate E*S Dense instances on first forward,
    // once we know T (the seq length of the input).
    std::vector<Dense> W_comb_;          // length num_slots_
    bool W_comb_initialized_ = false;

    // Per-slot FFNs (one Dense pair per slot, so each has its own last_input cache).
    // W1_e_s: (d_expert, d_model), W2_e_s: (d_model, d_expert).
    std::vector<Dense> W1_;          // length num_slots_, W1_[slot_idx]
    std::vector<Dense> W2_;          // length num_slots_, W2_[slot_idx]

    // Forward caches (for backward) — exposed for tests.
    Tensor last_input;             // (T, d_model)
    Tensor last_D_logits;          // (T, E*S)         pre-softmax dispatch logits
    Tensor last_D_;                // (T, E*S)         dispatch weights (rows sum to 1)
    Tensor last_Xp_;               // (E*S, d_model)   slot inputs (X' = D^T @ X)
    Tensor last_C_logits;          // (E*S, T)         pre-softmax combine logits
    Tensor last_C_;                // (E*S, T)         combine weights (cols sum to 1)
    Tensor last_Y_;                // (E*S, d_model)   slot outputs (post-experts)
    // Per-slot intermediates (one Tensor per slot)
    std::vector<Tensor> expert_h_pre_;   // (num_slots_, 1, d_expert)  pre-ReLU
    std::vector<Tensor> expert_h_act_;   // (num_slots_, 1, d_expert)  post-ReLU

    SoftMoELayer(size_t d_model, size_t num_experts, size_t slots_per_expert,
                 size_t d_expert);
    ~SoftMoELayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_disp_.weights; }
    Tensor get_gradients() const override { return W_disp_.grad_weights; }
    std::string name() const override { return "SoftMoELayer"; }

    // Copy all learnable parameters (W_disp, W_comb, per-expert W1/W2) from
    // `other` into this layer. Both layers must have identical config.
    void copy_params_from(const SoftMoELayer& other);

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t num_experts() const { return num_experts_; }
    size_t slots_per_expert() const { return slots_per_expert_; }
    size_t d_expert() const { return d_expert_; }
    size_t num_slots() const { return num_slots_; }
    size_t count_parameters() const;

private:
    void ensure_W_comb_(size_t T);
};

// Stack of `num_layers` SoftMoELayer blocks + final LayerNorm + classifier.
class SoftMoEModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t num_experts_;
    size_t slots_per_expert_;
    size_t d_expert_;

    Dense embed_;
    std::vector<std::unique_ptr<SoftMoELayer>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;

    SoftMoEModel(size_t input_dim, size_t d_model, size_t output_dim,
                 size_t num_layers, size_t num_experts, size_t slots_per_expert,
                 size_t d_expert);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed_.weights; }
    Tensor get_gradients() const override { return embed_.grad_weights; }
    std::string name() const override { return "SoftMoEModel"; }
};

#endif // SOFT_MOE_H