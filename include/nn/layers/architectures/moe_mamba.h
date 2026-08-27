#ifndef MOE_MAMBA_H
#define MOE_MAMBA_H

#include "../../core/layer.h"
#include "../recurrent/mamba2.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>

// ============================================================================
// MoE-Mamba (Pióro et al. 2024, https://arxiv.org/abs/2402.03262,
// "MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts")
//
// Combines Mamba-2's SSD recurrence with Switch-Transformer-style top-1
// sparse expert routing. Each "expert" is a complete Mamba2Block operating on
// the full (T, d_model) input.
//
// Forward per token t:
//   gate_logits[t] = W_g · x_t                (T, num_experts)
//   gate_scores[t] = sigmoid(gate_logits[t])
//   route_t        = argmax_i gate_scores[t, i]   (Switch — top-1, discrete)
//   y_t            = Expert_{route_t}(x_t)    (sparse — only the chosen expert fires)
//
// Capacity factor: each expert can serve at most ceil(T * cap_factor / N) tokens.
// Tokens that would overflow capacity are dropped (treated as zero contribution)
// — Switch Transformer trick to keep a runaway expert from exploding the gradient.
//
// Auxiliary load-balance loss (Fedus et al. 2022):
//   f_e = (# tokens routed to e) / T
//   p_e = (1/T) · sum_t gate_scores[t, e]
//   L_aux = alpha · N · sum_e f_e · p_e
// Gradient on gate_scores (t, e): alpha · N · f_e / T (constant per row, e).
//
// Layer interface contract: forward / backward / update_weights / zero_grad /
// parameters / gradients / get_weights / get_gradients / name (per core/layer.h).
// ============================================================================

class MoEMambaBlock : public Layer {
public:
    // Architecture.
    size_t d_model_;
    size_t n_heads_;
    size_t expert_d_inner_;
    size_t num_experts_;
    double capacity_factor_;
    double aux_loss_alpha_;

    // Router: gate_scores = sigmoid(x · W_g^T + b_g).  Shape (T, num_experts).
    Dense W_g_;                                          // (num_experts, d_model)

    // Forward cache (last_input_): used for backward.
    Tensor last_input_;                                  // (T, d_model)
    Tensor last_gate_logits_;                            // (T, num_experts)
    Tensor last_gate_scores_;                            // (T, num_experts) — sigmoid
    Tensor last_route_indices_;                          // (T, 1)         — int-as-double (argmax)
    Tensor last_route_mask_;                             // (T, num_experts) — 1.0 if this token routes to expert i, else 0.0
    Tensor last_capacity_mask_;                          // (T, num_experts) — capacity-mask ∧ route_mask (dropped tokens → 0)
    Tensor last_token_expert_count_;                     // (num_experts,)   — how many tokens per expert (pre-capacity-drop)
    double last_load_balance_loss_;                       // scalar

    // Experts — each is a complete Mamba2Block.
    std::vector<std::unique_ptr<Mamba2Block>> experts_;

    MoEMambaBlock(size_t d_model, size_t n_heads, size_t num_experts,
                  size_t expert_d_inner = 0, double capacity_factor = 1.0,
                  double aux_loss_alpha = 0.01);
    ~MoEMambaBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_g_.weights; }
    Tensor get_gradients() const override { return W_g_.grad_weights; }
    std::string name() const override { return "MoEMambaBlock"; }

    // Copy all learnable params (router + every expert) from another layer.
    void copy_params_from(const MoEMambaBlock& other);

    // Accessors.
    size_t d_model() const { return d_model_; }
    size_t num_experts() const { return num_experts_; }
    size_t n_heads() const { return n_heads_; }
    double load_balance_loss() const { return last_load_balance_loss_; }
    size_t count_parameters() const;
};

// Stack of `num_layers` MoEMambaBlocks + input projection + final LayerNorm + classifier.
class MoEMambaModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t n_heads_;
    size_t num_experts_;
    size_t expert_d_inner_;

    Dense embed_;
    std::vector<std::unique_ptr<MoEMambaBlock>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;

    MoEMambaModel(size_t input_dim, size_t d_model, size_t output_dim,
                  size_t num_layers, size_t n_heads, size_t num_experts,
                  size_t expert_d_inner = 0, double capacity_factor = 1.0,
                  double aux_loss_alpha = 0.01);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed_.weights; }
    Tensor get_gradients() const override { return embed_.grad_weights; }
    std::string name() const override { return "MoEMambaModel"; }
};

#endif // MOE_MAMBA_H