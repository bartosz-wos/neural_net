#ifndef DEEPSEEK_MOE_H
#define DEEPSEEK_MOE_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>

// ============================================================================
// DeepSeekMoE (DeepSeek-AI 2024, https://arxiv.org/abs/2401.06066,
// "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts
// Language Models")
//
// Two innovations over standard top-k MoE:
//   1. Fine-grained expert segmentation: each of `num_routed` routed experts
//      handles only a SEGMENT of size (d_model / num_routed) of the input —
//      not the full input. This gives combinatorial flexibility without
//      changing the parameter count.
//   2. Shared expert isolation: `num_shared` experts that ALWAYS fire (no
//      routing), capturing common knowledge and freeing the routed experts
//      to specialize.
//
// Per token:
//   shared_path = sum_j SharedFFN_j(x)                    // no gating
//   gate_scores = sigmoid(W_g · x)                         // (T, num_routed)
//   top_k, top_w = top_k_with_renormalize(gate_scores, k)
//   routed_path = sum_{i in top_k} top_w_i · RoutedFFN_i(x_seg_i)
//   y = shared_path + routed_path
//
// where x_seg_i = x[:, i*seg : (i+1)*seg] is the per-expert segment of
// the input (each routed FFN sees only its own segment, both in and out).
//
// Layout: (T, d_model) end-to-end. The gated and shared paths are computed
// independently and summed. Output shape matches input.
//
// Invariants enforced in the constructor:
//   - d_model > 0, d_expert > 0
//   - when num_routed > 0:  d_model % num_routed == 0, top_k_routed >= 1,
//                           top_k_routed <= num_routed
//   - num_shared >= 0 (zero allowed → pure routed MoE)
//   - num_routed = 0 → pure shared experts (no gating path)
//
// Gate gradient: the renormalization `w_i = s_i / Σ_{j∈top-k} s_j` introduces
// a coupling Jacobian `∂w_i/∂s_j = (δ_{ij}·Σ - s_i) / Σ²` for `j ∈ top-k`,
// which we include in backward().
// ============================================================================

class DeepSeekMoELayer : public Layer {
public:
    size_t d_model_;
    size_t d_expert_;
    size_t num_routed_;
    size_t num_shared_;
    size_t top_k_routed_;
    size_t seg_size_;     // = d_model / num_routed (only meaningful when num_routed > 0)

    // Gate linear projection: W_g · x produces (T, num_routed) gate scores.
    Dense W_g_;                                                          // (num_routed, d_model)

    // Routed experts — each is a self-contained FFN over its input segment.
    // W1: (d_expert, seg_size), W2: (seg_size, d_expert).
    struct RoutedExpert {
        Dense W1, W2;
        RoutedExpert(size_t in_dim, size_t out_dim) : W1(in_dim, out_dim), W2(out_dim, in_dim) {}
    };
    std::vector<RoutedExpert> experts_;

    // Shared experts — each is a self-contained FFN over the FULL input.
    // W1: (d_expert, d_model), W2: (d_model, d_expert).
    struct SharedExpert {
        Dense W1, W2;
        SharedExpert(size_t in_dim, size_t out_dim) : W1(in_dim, out_dim), W2(out_dim, in_dim) {}
    };
    std::vector<SharedExpert> shared_experts_;

    // Forward caches (for backward) — exposed for tests.
    Tensor last_input;             // (T, d_model)
    Tensor last_gate_logits;       // (T, num_routed) — W_g · x (pre-sigmoid)
    Tensor last_gate_scores;       // (T, num_routed) — sigmoid(W_g · x)
    Tensor last_top_k_indices;     // (T, top_k_routed) — int-as-double
    Tensor last_top_k_weights;     // (T, top_k_routed) — renormalized s_i
    Tensor last_shared_out;        // (T, d_model)
    Tensor last_routed_out;        // (T, d_model)

    // Load-balance auxiliary loss: α · E · Σ_e (f_e · p_e).
    //   f_e = fraction of (token, top_k_routed) selections landing on e
    //   p_e = mean gate probability to e across tokens
    double last_load_balance_loss_;

    DeepSeekMoELayer(size_t d_model, size_t d_expert,
                     size_t num_routed, size_t num_shared,
                     size_t top_k_routed);
    ~DeepSeekMoELayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_g_.weights; }
    Tensor get_gradients() const override { return W_g_.grad_weights; }
    std::string name() const override { return "DeepSeekMoELayer"; }

    // Copy all learnable parameters (gate + per-expert + shared FFNs) from
    // `other` into this layer. Both layers must have identical config.
    void copy_params_from(const DeepSeekMoELayer& other);

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_expert() const { return d_expert_; }
    size_t num_routed() const { return num_routed_; }
    size_t num_shared() const { return num_shared_; }
    size_t top_k_routed() const { return top_k_routed_; }
    size_t seg_size() const { return seg_size_; }
    double load_balance_loss() const { return last_load_balance_loss_; }
    size_t count_parameters() const;
};

// Stack of `num_layers` DeepSeekMoELayer blocks + final LayerNorm + classifier.
class DeepSeekMoEModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t d_expert_;
    size_t num_routed_;
    size_t num_shared_;
    size_t top_k_routed_;

    Dense embed_;
    std::vector<std::unique_ptr<DeepSeekMoELayer>> blocks_;
    LayerNorm final_ln_;
    Dense classifier_;

    DeepSeekMoEModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t num_layers, size_t d_expert,
                     size_t num_routed, size_t num_shared,
                     size_t top_k_routed);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed_.weights; }
    Tensor get_gradients() const override { return embed_.grad_weights; }
    std::string name() const override { return "DeepSeekMoEModel"; }
};

#endif // DEEPSEEK_MOE_H