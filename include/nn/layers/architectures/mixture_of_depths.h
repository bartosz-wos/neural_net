#ifndef MIXTURE_OF_DEPTHS_H
#define MIXTURE_OF_DEPTHS_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <random>

// ============================================================================
// Mixture of Depths (MoD) — Raposo et al. 2024
//   "Mixture-of-Depths: Dynamically allocating compute in transformer-based
//    models" (https://arxiv.org/abs/2404.02258)
//
// Core idea: instead of routing tokens between EXPERTS (MoE), route tokens
// between LAYER FANOUTS (process vs skip). At every transformer block each
// token independently decides whether to go through the heavy sub-layer
// (process) or be passed through as identity (skip). This *reduces FLOPs*
// per forward pass without changing the parameter count.
//
// Math (per MoDLayer call, where x has shape (n, d_model)):
//
//   For every token t in [0, n):
//     router_logits[t, 0] = x[t, :] @ W_router[:] + b_router[0]   # (n, 1) per-token
//   router_probs = sigmoid(router_logits)                          # (n, 1)
//
//   capacity  = capacity_factor * n                               # scalar
//   tokens routed through sub-layer = top-capacity tokens by router_logits
//                                          (ties broken by smaller token index)
//   mask[t] = 1.0 if t in selected set, else 0.0                  # (n, 1)
//
//   For the SELECTED tokens, run the inner sub-layer:
//     sub_out[selected, :] = inner_block(x[selected, :])
//   Sub_out for unselected tokens is filled with zeros.
//
//   Final output:
//     y = x + mask * sub_out                                       # residual with mask
//
// We use a SOFT router (sigmoid + top-k mask), which is the standard
// "MoD paper" convention: gradients flow to BOTH the router and the inner
// block via the mask. The router learns to assign higher logits to tokens
// that benefit from more compute.
//
// Aux loss (load balancing)
// -------------------------
//   Following the MoD paper's "expert-choice"-style load balance:
//     L_aux = alpha * n * var(per_token_selected)
//            = alpha * n * (mean(mask^2) - mean(mask)^2)
//   We use the SIMPLER form from the paper appendix:
//     L_aux = alpha * n * (mean(mask) - capacity_factor)^2
//   which penalizes deviation from the target mean (the fraction of tokens
//   routed through the sub-layer).
//
// BPTT
// ----
//   We use the same "run inner on the full batch, mask unselected outputs to
//   zero in forward and zero their grads in backward" convention as
//   SparseMoELayer. This avoids scatter/gather bookkeeping in the inner
//   sub-layer and keeps BPTT straightforward.
//
//   Forward caches:
//     - input_                  (n, d_model)
//     - router_logits_          (n, 1)
//     - router_probs_           (n, 1)
//     - mask_                   (n, 1)  — hard top-k mask {0, 1}
//     - selected_indices_       (capacity,) — flat indices of selected tokens
//     - sub_out_                (n, d_model)  (zeros at unselected rows)
//
//   Backward flow:
//     1) grad_out (n, d_model) is masked: grad_sub_out = grad_out * mask
//        (unselected rows become 0; gradient does NOT flow through unselected
//        tokens — they were skipped in forward, so they get nothing here).
//     2) inner_block.backward(grad_sub_out) → d_input_inner, dW_block
//     3) Add the residual gradient: d_input = d_input_inner + grad_out
//        (since y = x + mask * sub_out, gradient w.r.t. x is grad_out from the
//        residual path PLUS d_input_inner from the gated-sub-layer path).
//     4) Router backward (sigmoid + aux-loss):
//        d_router_probs = 2 * (mean(mask) - capacity_factor) * alpha
//        (the aux loss is the only path back into the router, same as
//        SparseMoE's aux-loss gradient treatment).
//        d_router_logits = d_router_probs * router_probs * (1 - router_probs)
//     5) Router Dense backward: dW_router += d_router_logits^T @ input,
//                                d_input += d_router_logits @ W_router
// ============================================================================

class MoDLayer : public Layer {
public:
    // d_model:         input and output feature dim
    // capacity_factor: fraction of tokens to route through the inner block
    //                  (e.g. 0.5 = route 50% of tokens through, skip 50%)
    // aux_loss_coef:   coefficient alpha for the load-balancing auxiliary loss
    // inner:           the inner sub-layer (any Layer; will be owned by us)
    MoDLayer(size_t d_model, double capacity_factor = 0.5,
             double aux_loss_coef = 0.01, Layer* inner = nullptr);

    // Move-only (the inner_ unique_ptr is move-only).
    MoDLayer(const MoDLayer&) = delete;
    MoDLayer& operator=(const MoDLayer&) = delete;
    MoDLayer(MoDLayer&&) = default;
    MoDLayer& operator=(MoDLayer&&) = default;
    ~MoDLayer() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;

    Tensor get_weights() const override;   // returns router weights
    Tensor get_gradients() const override; // returns router grad weights

    // Inspectors
    double get_load_balance_loss() const { return load_balance_loss_; }
    size_t num_selected() const { return selected_indices_.size(); }
    size_t capacity() const { return capacity_; }
    double capacity_factor() const { return capacity_factor_; }
    const std::vector<size_t>& selected_indices() const { return selected_indices_; }
    std::string name() const override { return "MoDLayer"; }

private:
    size_t d_model_;
    double capacity_factor_;
    size_t capacity_;       // capacity_factor * n, rounded, at least 1, at most n
    double aux_loss_coef_;

    // Inner block (sub-layer that processes the selected tokens)
    std::unique_ptr<Layer> inner_;

    double load_balance_loss_;
    std::mt19937 rng_;

    // Forward cache
    Tensor input_;            // (n, d_model)
    Tensor router_logits_;    // (n, 1)
    Tensor router_probs_;     // (n, 1)  sigmoid(router_logits)
    std::vector<size_t> selected_indices_;   // indices of selected tokens
    Tensor sub_out_;          // (n, d_model) — zeros at unselected rows

    // Helper: top-k selection on the (n,) vector. Returns indices sorted
    // descending by score; we take the first `capacity` of them.
    void select_top_k(const std::vector<double>& scores,
                      std::vector<size_t>& indices_out);

public:
    // Public access for tests / debugging (mirrors MLA style)
    Tensor W_router_;        // (1, d_model)
    Tensor b_router_;        // (1, 1)
    Tensor grad_W_router_;   // (1, d_model)
    Tensor grad_b_router_;   // (1, 1)
    Tensor mask_;             // (n, 1)  hard top-k mask {0, 1}
};

// ============================================================================
// MoDBlock — pre-LN → MoDLayer(attn+ffn) → residual
// ============================================================================

class MoDBlock : public Layer {
public:
    MoDBlock(size_t d_model, double capacity_factor = 0.5,
             size_t ffn_dim = 0, double aux_loss_coef = 0.01);

    // Move-only (mod1_ contains a unique_ptr).
    MoDBlock(const MoDBlock&) = delete;
    MoDBlock& operator=(const MoDBlock&) = delete;
    MoDBlock(MoDBlock&&) = default;
    MoDBlock& operator=(MoDBlock&&) = default;
    ~MoDBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "MoDBlock"; }

    double get_load_balance_loss() const { return mod1_.get_load_balance_loss(); }

private:
    size_t d_model_;
    size_t ffn_dim_;
    double capacity_factor_;

    // The two pre-LN layers
    LayerNorm ln1_;             // (d_model)
    LayerNorm ln2_;             // (d_model)

    // The two MoD-wrapped sub-layers (FFN-style is simplest; attention is
    // optional). We use simple Dense-based FFN sub-layers inside MoD, which
    // gives us a clean shape and tractable gradient check.
    Dense ffn_fc1_;             // (ffn_dim, d_model)
    Dense ffn_fc2_;             // (d_model, ffn_dim)
    MoDLayer mod1_;             // MoD wrap of ffn_fc1+ffn_fc2 (an "FFN MoD layer")

    Tensor last_input_;
    Tensor last_h_pre_;         // pre-ReLU (for ReLU backward)
};

// ============================================================================
// MoDModel — stack of MoDBlocks + classifier head
// ============================================================================

class MoDModel : public Layer {
public:
    MoDModel(size_t d_model, size_t out_features, size_t num_blocks = 2,
             double capacity_factor = 0.5, size_t ffn_dim = 0,
             double aux_loss_coef = 0.01);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }

    // Total aux loss across all blocks (for the trainer to add to the loss)
    double get_total_aux_loss() const { return total_aux_loss_; }

    std::string name() const override { return "MoDModel"; }

private:
    size_t d_model_;
    size_t out_features_;
    double total_aux_loss_;
    std::vector<MoDBlock> blocks_;
    Dense classifier_;
    Tensor last_input_;
};

#endif
