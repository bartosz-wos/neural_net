#ifndef SWITCH_TRANSFORMER_H
#define SWITCH_TRANSFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>

// ============================================================================
// Switch Transformer — Fedus, Zoph, Shazeer 2022
// ============================================================================
//
// Reference: Fedus, Zoph, Shazeer 2022 "Switch Transformers: Scaling to
//   Trillion Parameter Models with Simple and Efficient Sparsity"
//   (https://arxiv.org/abs/2101.03961).
//
// This file implements three classes:
//
//   1. SwitchMoELayer       — the top-1 routed mixture-of-experts FFN with
//                             capacity-factor dropping and the paper's z-loss.
//
//   2. SwitchTransformerBlock — pre-LN → SwitchMoE → residual (a "Switch FFN
//                               block", paper §3.2).
//
//   3. SwitchTransformerModel — input projection → LayerNorm → N blocks →
//                                classifier, exposing the SUM of per-block
//                                aux + z losses via `get_aux_loss()` /
//                                `get_z_loss()` for the user to add to the
//                                total loss.
//
// Distinctive ingredients vs the existing `SparseMoELayer` (Shazeer 2017):
//
//   - Hard top-1 routing only (k=1). The paper's "simplification" of
//     Shazeer's top-k routing — `top1[b] = argmax_e gate_logits[b, e]`.
//
//   - Expert capacity factor (paper §3.1, Eq. 2). Each expert can process
//     at most `capacity = ceil(B / num_experts) · capacity_factor` tokens
//     per forward pass. Overflow tokens get `output = 0` and zero
//     gradient. This is the *defining* Switch Transformer trick that lets
//     them shard each expert to a separate device without imbalance
//     stalling the run.
//
//   - Router z-loss (paper Eq. 5). `L_z = z_coef · (1/B) · Σ_b
//     log(Σ_e exp(z_e)²)`. Penalizes large router logits — the paper's
//     stability fix that prevents the logits from growing without bound
//     during long training runs.
//
//   - Differentiable load-balancing loss (paper Eq. 4). `L_aux = α · N ·
//     Σ_e f_e · P_e`, where `f_e` is the fraction of tokens routed to
//     expert e and `P_e` is the mean router probability for e.
//     Identical form to `SparseMoELayer::get_load_balance_loss()` but
//     exposed separately here for the canonical Switch Transformer
//     convention `aux_loss_coef · num_experts · Σ f_e · P_e`.
//
// Implementation notes:
//
//   - Each expert is a 2-layer FFN: `h = ReLU(W1 · x + b1)`, `y = W2 · h +
//     b2`. Weights stored as raw Tensors (Dense convention), He-init.
//
//   - Router is a Dense layer with one output per expert; we cache the
//     FULL softmax `gate_probs[b, e] = softmax_e(gate_logits[b, :])` so
//     the router backward can be the standard softmax chain even though
//     only the top-1 was selected for forward dispatch.
//
//   - Per-expert forward caches the pre-acts / post-acts only for the
//     dispatched (non-dropped) tokens. The expert's local gradient
//     computation operates on a `capacity_e`-sized batch dimension.
//
//   - Capacity-limited dispatch: per-expert counter increments as tokens
//     route to it; once a counter reaches `capacity`, additional tokens
//     targeted at that expert are DROPPED (their output is 0, they
//     contribute zero gradient).
// ============================================================================


class SwitchMoELayer : public Layer {
public:
    // d_model:        input and output feature dim.
    // num_experts:    number of parallel FFN experts.
    // expert_hidden:  hidden dim of each expert's FFN (defaults to 4 * d_model).
    // capacity_factor: per-expert token cap as a multiple of the uniform
    //                  quota (paper default 1.25). Must be > 0.
    // aux_loss_coef:  coefficient α on the differentiable load-balancing
    //                  loss (paper Eq. 4). Must be ≥ 0.
    // z_loss_coef:    coefficient on the router z-loss (paper Eq. 5).
    //                  Must be ≥ 0.
    SwitchMoELayer(size_t d_model, size_t num_experts,
                   size_t expert_hidden = 0,
                   double capacity_factor = 1.25,
                   double aux_loss_coef = 0.01,
                   double z_loss_coef = 0.001);

    size_t d_model() const { return d_model_; }
    size_t num_experts() const { return num_experts_; }
    size_t expert_hidden() const { return expert_hidden_; }
    double capacity_factor() const { return capacity_factor_; }
    double aux_loss_coef() const { return aux_loss_coef_; }
    double z_loss_coef() const { return z_loss_coef_; }

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;     // router weights
    Tensor get_gradients() const override;   // router grad weights

    // Aux + z losses from the most recent forward. Caller adds these
    // to the total loss; the layer does NOT mix them into `forward`.
    double get_aux_loss() const { return load_balance_loss_; }
    double get_z_loss() const { return z_loss_; }

    // Accessors for the most recent forward (handy for tests / debugging).
    const Tensor& last_gate_logits() const { return gate_logits_; }
    const Tensor& last_gate_probs()  const { return gate_probs_; }
    const Tensor& last_top1_indices() const { return top1_idx_; }
    const Tensor& last_dispatch_mask() const { return dispatch_mask_; }
    std::vector<double> get_dispatch_fractions() const { return dispatch_frac_; }
    std::vector<double> get_mean_router_probs() const { return mean_router_prob_; }

    // Testing helper: directly set router weights/biases (used by unit
    // tests to force a particular routing pattern). NOT for production.
    void set_router_for_test(const Tensor& W, const Tensor& b) {
        W_router_ = W;
        b_router_ = b;
    }
    // Testing helper: directly set an expert's weights (W1, b1, W2, b2).
    void set_expert_for_test(size_t e, const Tensor& W1, const Tensor& b1,
                             const Tensor& W2, const Tensor& b2) {
        W1_[e] = W1; b1_[e] = b1;
        W2_[e] = W2; b2_[e] = b2;
    }

    std::string name() const override { return "SwitchMoELayer"; }

private:
    size_t d_model_, num_experts_, expert_hidden_;
    double capacity_factor_, aux_loss_coef_, z_loss_coef_;

    // Router: logits[b, e] = sum_d input[b, d] * W_router[e, d] + b_router[0, e].
    Tensor W_router_;          // (num_experts, d_model)
    Tensor b_router_;          // (1, num_experts)
    Tensor grad_W_router_;     // (num_experts, d_model)
    Tensor grad_b_router_;     // (1, num_experts)

    // Experts — 2-layer FFN (ReLU hidden).
    // W1_[e]: (expert_hidden, d_model)  W2_[e]: (d_model, expert_hidden)
    std::vector<Tensor> W1_, W2_, b1_, b2_, gW1_, gW2_, gb1_, gb2_;

    // Forward caches.
    Tensor input_;             // (B, d_model)
    Tensor gate_logits_;       // (B, num_experts)
    Tensor gate_probs_;        // (B, num_experts) full softmax (not sparse)
    Tensor top1_idx_;          // (B, 1)
    Tensor dispatch_mask_;     // (B, num_experts) 1 iff token b routed to e AND not dropped
    Tensor token_dropped_;     // (B, 1) 1 iff the token's expert was over capacity

    // Per-expert caches of size (capacity_per_expert, ...) for the
    // dispatched (non-dropped) tokens routed to that expert.
    std::vector<Tensor> expert_h_pre_;     // (cap_e, expert_hidden)
    std::vector<Tensor> expert_h_act_;     // (cap_e, expert_hidden)
    std::vector<Tensor> expert_x_in_;      // (cap_e, d_model) — the routed token inputs
    std::vector<size_t> expert_count_;     // how many tokens routed (and not dropped) per expert

    // Loss info from the most recent forward.
    double load_balance_loss_;
    double z_loss_;
    std::vector<double> dispatch_frac_;
    std::vector<double> mean_router_prob_;

    // Helper.
    void compute_losses_();   // populates load_balance_loss_ and z_loss_
};


// ============================================================================
// SwitchTransformerBlock — pre-LN → SwitchMoE → residual
// ============================================================================
//
// Paper §3.2: the Switch encoder block is identical in shape to a standard
// transformer FFN block, except the dense FFN is replaced by a routed
// SwitchMoE. We skip the attention sublayer here because (a) attention is
// orthogonal to the routing trick, and (b) every other block in this repo
// (HiLoBlock, PowerBlock, StickBreakingBlock, TokenformerBlock, etc.)
// provides its own pre-LN-attn-FFN pattern that callers can compose with
// this block via `Model::add_layer`.

class SwitchTransformerBlock : public Layer {
public:
    SwitchTransformerBlock(size_t d_model, size_t num_experts,
                           size_t expert_hidden = 0,
                           double capacity_factor = 1.25,
                           double aux_loss_coef = 0.01,
                           double z_loss_coef = 0.001);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;

    double get_aux_loss() const { return moe_.get_aux_loss(); }
    double get_z_loss()   const { return moe_.get_z_loss(); }

    std::string name() const override { return "SwitchTransformerBlock"; }

private:
    LayerNorm ln_;     // pre-LN
    SwitchMoELayer moe_;
    Tensor input_;     // for the residual
};


// ============================================================================
// SwitchTransformerModel — input projection → LN → N blocks → classifier
// ============================================================================

class SwitchTransformerModel : public Layer {
public:
    SwitchTransformerModel(size_t input_dim, size_t d_model,
                           size_t num_experts, size_t output_dim,
                           size_t num_blocks = 2,
                           size_t expert_hidden = 0,
                           double capacity_factor = 1.25,
                           double aux_loss_coef = 0.01,
                           double z_loss_coef = 0.001);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;

    // SUM of per-block aux + z losses — the canonical Switch Transformer
    // convention (paper §3.2). Caller adds these to the total loss.
    double get_aux_loss() const { return sum_aux_; }
    double get_z_loss() const   { return sum_z_; }

    std::string name() const override { return "SwitchTransformerModel"; }

private:
    Dense input_proj_;
    LayerNorm pre_ln_;
    std::vector<SwitchTransformerBlock> blocks_;
    Dense classifier_;

    Tensor input_;
    double sum_aux_;
    double sum_z_;
};

#endif
