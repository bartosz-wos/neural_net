#ifndef SPARSE_MOE_H
#define SPARSE_MOE_H

#include "../../core/layer.h"
#include <vector>

// ============================================================================
// Sparse Mixture of Experts (Sparse MoE) — Shazeer et al. 2017 / Fedus et al. 2022
// ============================================================================
//
// Reference: Shazeer et al. 2017 "Outrageously Large Neural Networks: The
//   Sparsely-Gated Mixture-of-Experts Layer" (https://arxiv.org/abs/1701.06538),
//   and Fedus, Zoph, Shazeer 2022 "Switch Transformers: Scaling to Trillion
//   Parameter Models with Simple and Efficient Sparsity"
//   (https://arxiv.org/abs/2101.03961).
//
// This layer routes each input token to the TOP-K experts (k=1 or k=2
// typical; configurable). The output is a weighted combination of the
// k selected expert outputs, weighted by a normalized gate value (the
// softmax probability of the selected expert).
//
// Architecture
// ------------
//   input (batch, d_model)  --+-->  router Dense(d_model -> num_experts)
//                              |     logits_e = sum_d x[d] * W_router[d, e] + b_router[e]
//                              |     gate_logits  (batch, num_experts)
//                              |     softmax_topk  ->  g  (batch, num_experts)  (sparse)
//                              |
//                              +-->  for each token b:
//                                       for each (e, w_e) in top-k of g[b, :]:
//                                           out[b, :] += w_e * expert_e(x[b, :])
//
// Auxiliary load-balancing loss
// -----------------------------
//   L_aux = alpha * num_experts * sum_e ( f_e * p_e )
// where
//   f_e = (# tokens routed to expert e) / batch                (fraction of tokens)
//   p_e = (1/batch) * sum_b g[b, e]                            (mean gate value)
// We use the unweighted variant (Shazeer) and expose `load_balance_loss()`.
// The aux loss is NOT added to the layer's forward — the user can fetch it
// via get_load_balance_loss() and add it to the model loss manually (matches
// the standard "aux loss" pattern in Switch/GShard MoE implementations).
//
// BPTT
// ----
//   Forward caches:
//     - input_                  (batch, d_model)
//     - gate_logits_            (batch, num_experts)  raw logits
//     - gate_probs_             (batch, num_experts)  softmax-topk probs
//     - topk_idx_               (batch, k)            indices of selected experts
//     - expert_input_cache_     (batch, d_model)      input for the dispatcher
//     - expert_outputs_         list of (batch, d_model) per-expert outputs
//
//   Backward flow:
//     1. grad_out (batch, d_model) -> for each (b, k) pair: d_expert_out_e += g[b,e] * grad_out[b, :]
//     2. each expert's backward (batched: per-expert, with its own grad_in) -> d_expert_w, d_expert_b
//     3. gate probs: d g[b,e] = sum_j grad_out[b, j] * expert_e_out[b, j] (for selected e only)
//     4. softmax-topk backward with topk-mask: d_logits[b, e] = mask_topk(b,e) * g[b,e] * (d g[b,e] - sum_e' g[b,e'] * d g[b,e'])
//     5. gate Dense backward: d_input += d_logits @ W_router, d_W_router += d_logits^T @ input, d_b_router += sum d_logits
//
//   We use the "softmax over the masked logits" form of top-k routing (GShard
//   / Switch style) so the gate probabilities sum to 1 across the k selected
//   experts. This is differentiable and standard in modern MoE papers.
// ============================================================================

class SparseMoELayer : public Layer {
public:
    // d_model:       input and output feature dim
    // num_experts:   number of parallel FFN experts
    // k:             top-k experts to route to per token (typically 1 or 2)
    // expert_hidden: hidden dim of each expert's FFN (defaults to 4 * d_model)
    // aux_loss_coef: coefficient alpha for the load-balancing auxiliary loss
    SparseMoELayer(size_t d_model, size_t num_experts, size_t k = 2,
                   size_t expert_hidden = 0, double aux_loss_coef = 0.01);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;   // returns router weights (for inspection)
    Tensor get_gradients() const override; // returns router grad weights

    // Get the load-balancing aux loss computed during the most recent forward.
    // Convention (Shazeer): multiplied by alpha and num_experts.
    double get_load_balance_loss() const { return load_balance_loss_; }
    // Get the most recent per-expert dispatch fractions f_e (size num_experts).
    std::vector<double> get_dispatch_fractions() const { return dispatch_frac_; }
    // Get the most recent per-expert mean gate probs p_e (size num_experts).
    std::vector<double> get_mean_gate_probs() const { return mean_gate_prob_; }

    std::string name() const override { return "SparseMoELayer"; }

private:
    size_t d_model_;
    size_t num_experts_;
    size_t k_;
    size_t expert_hidden_;
    double aux_loss_coef_;

    // Router: gate logits = x @ W_router^T + b_router
    // Dense convention: W_router is (num_experts, d_model) so that
    //   logits[b, e] = sum_d input[b, d] * W_router[e, d] + b_router[0, e].
    Tensor W_router_;        // (num_experts, d_model)
    Tensor b_router_;        // (1, num_experts)
    Tensor grad_W_router_;   // (num_experts, d_model)
    Tensor grad_b_router_;   // (1, num_experts)

    // Experts — each is a 2-layer FFN: input -> hidden -> output (with ReLU).
    // Using raw Tensors for full control over gradient layout.
    // Dense convention: W1_ (expert_hidden, d_model), W2_ (d_model, expert_hidden).
    std::vector<Tensor> W1_, W2_;     // expert weights
    std::vector<Tensor> b1_, b2_;     // expert biases
    std::vector<Tensor> gW1_, gW2_;   // weight grads
    std::vector<Tensor> gb1_, gb2_;   // bias grads

    // Forward cache
    Tensor input_;             // (batch, d_model)
    Tensor gate_logits_;       // (batch, num_experts)
    Tensor gate_probs_;        // (batch, num_experts)  (sparse — non-selected entries are 0)
    std::vector<std::vector<size_t>> topk_idx_;   // (batch, k)

    // Per-expert intermediate caches (for backward)
    std::vector<Tensor> expert_h_pre_;      // (batch, expert_hidden) per expert  (pre-ReLU)
    std::vector<Tensor> expert_h_act_;      // (batch, expert_hidden) per expert  (post-ReLU)
    std::vector<Tensor> expert_outputs_;    // (batch, d_model) per expert

    // Aux loss info
    double load_balance_loss_;
    std::vector<double> dispatch_frac_;     // (num_experts,)
    std::vector<double> mean_gate_prob_;    // (num_experts,)

    // Helpers
    void softmax_topk(const Tensor& logits, Tensor& probs, std::vector<std::vector<size_t>>& topk);
};

#endif
