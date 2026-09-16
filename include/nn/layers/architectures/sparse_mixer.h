#ifndef SPARSE_MIXER_H
#define SPARSE_MIXER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>
#include <stdexcept>

// ============================================================================
// Sparse Mixer — Lee-Thorp & Ainslie, EMNLP Findings 2022
// ============================================================================
//
// Reference: "Sparse Mixers: Combining MoE and Mixing to build a more
//   efficient BERT" (https://arxiv.org/abs/2205.12399).
//
// Combines sparsely-gated Mixture-of-Experts (MoE) capacity with the speed
// and stability of linear token-mixing to design an efficient BERT-like
// encoder. The default recipe (paper §4.4): 14 layers, d_m=512, d_ff=2056;
// 4 MIDDLE MoE layers with 16 experts each, Experts-Choice router, capacity
// factor 1.0; Linear mixing with 4 TOP self-attention layers.
//
// This file implements four classes:
//
//   (1) LinearMixingSublayer(d_model, seq_len) — drop-in for self-attention.
//       Math: y = T_h · T_seq · x where T_h : (d_model, d_model) and
//       T_seq : (seq_len, seq_len) are two dense token-INDEPENDENT linear
//       projections. NO attention — just two matmuls (paper §3.3 "Linear").
//
//   (2) ExpertsChoiceMoE(d_model, num_experts, expert_hidden=0, cf=1.0) —
//       Zhou et al. 2022 router. Each expert picks its top-k tokens
//       (k = ceil(B/E)·cf), so an expert ALWAYS fills its buffer and a
//       token may be routed to 0, 1, or multiple experts. Output is mean-
//       pool over experts that selected the token (so any token gets at
//       least the residual through, even if no expert picked it). No load-
//       balancing loss needed because every expert fills capacity.
//
//   (3) SparseMixerBlock(d_model, seq_len, mode, num_experts=0,
//       expert_hidden=0, cf=1.0, ffn_mult=4) — pre-norm + mixing/attention
//       sublayer + residual + pre-norm + MoE/dense FFN sublayer + residual.
//       mode ∈ {LINEAR_MIXING, SELF_ATTENTION} selects the first sublayer.
//       Second sublayer is ExpertsChoiceMoE if num_experts > 0, else a
//       dense 2-layer GELU FFN with ffn_mult hidden width.
//
//   (4) SparseMixerModel(input_dim, d_model, output_dim, seq_len,
//       num_layers, num_attention_layers, num_experts, expert_hidden=0,
//       cf=1.0, ffn_mult=4) — Dense input projection → stack of
//       num_layers SparseMixerBlocks where the LAST num_attention_layers
//       use SELF_ATTENTION and the first (num_layers - num_attention_layers)
//       use LINEAR_MIXING (matching the paper's "Linear, 4 TOP Attention
//       layers" arrangement) → mean-pool over seq_len → final Dense
//       classifier.
//
// Conventions:
//   * Input to a layer/block/model: (B, S*d_model) — flat (B, S, d_model)
//     with the last two dims flattened to (S*d_model,). Same convention
//     used by MlpMixerBlock in this repo.
//   * Dense convention: weights is (out, in), y = x @ W^T + b.
// ============================================================================

// ============================================================================
// LinearMixingSublayer — token-INDEPENDENT linear mixing
// ============================================================================

class LinearMixingSublayer : public Layer {
public:
    // d_model: feature dim
    // seq_len: token count (S) — must be known statically because T_seq
    //          has shape (S, S)
    LinearMixingSublayer(size_t d_model, size_t seq_len);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_h; }
    Tensor get_gradients() const override { return grad_W_h; }
    std::string name() const override { return "LinearMixingSublayer"; }

    // Public accessors (tests).
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    Tensor& mix_W_h() { return W_h; }
    Tensor& mix_W_seq() { return W_seq; }
    Tensor& mix_b_h() { return b_h; }
    Tensor& grad_mix_W_h() { return grad_W_h; }
    Tensor& grad_mix_W_seq() { return grad_W_seq; }
    Tensor& grad_mix_b_h() { return grad_b_h; }
    Tensor& grad_input() { return last_d_input_; }

private:
    size_t d_model_;
    size_t seq_len_;

    // T_h: (d_model, d_model), T_seq: (seq_len, seq_len), b_h: (1, d_model)
    Tensor W_h, W_seq, b_h;
    Tensor grad_W_h, grad_W_seq, grad_b_h;

    // Forward cache
    Tensor input_;             // (B, S*D) flat
    Tensor last_z_seq_;         // (B, S*D) after T_seq: each "row" (token) is the linear mix over S
    Tensor last_output_;       // (B, S*D) after T_h + b
    Tensor last_d_input_;      // (B, S*D) input gradient
};

// ============================================================================
// ExpertsChoiceMoE — Zhou et al. 2022 Experts-Choice router
// ============================================================================
//
// Math: router logits = x @ W_router^T + b_router  (B, E).
// For each expert e, sort logits[e, :] descending; pick top-k where
// k = ceil(B/E)·cf — every expert processes EXACTLY k tokens. A token may
// end up routed to 0, 1, or multiple experts.
//
// Output: mean over all experts that selected the token of the expert's
// FFN output. If no expert selected the token, output is zero (the
// residual in the block will carry the input through).
//
// Backward:
//   - Each expert's FFN computes its own gradient on its (k, d_model)
//     batch of dispatched tokens (independent across experts).
//   - The router gradient is the standard softmax cross-entropy style
//     "weight by routing probability" — see paper Eqs. For simplicity we
//     use the average gradient flow: d_logits[b, e] is the softmax-weight
//     average of the dL/dlogit chain from the experts that selected b.
//   - The d_input chain sums dL/dx contributions from each expert that
//     received token b (same dispatch shape as the forward).
//
// We implement routing with a fixed assignment matrix M[B, E] ∈ {0, 1}
// (M[b, e] = 1 iff expert e selected token b). For each expert e, the
// dispatched tokens are at rows where M[:, e] = 1; the local batch is
// (k, d_model). The forward then computes
//     out[b] = (1/sum_e M[b, e]) * sum_e M[b, e] * expert_e(x[b]).
// We use sum_e M[b, e] as the divisor and guard for 0.
//
// Experts are 2-layer ReLU FFN with input (d_model,), hidden
// (expert_hidden, default = 4*d_model), output (d_model,).
// ============================================================================

class ExpertsChoiceMoE : public Layer {
public:
    // d_model: input/output feature dim.
    // num_experts: E (number of parallel FFNs).
    // expert_hidden: hidden dim of each expert FFN (defaults to 4*d_model).
    // capacity_factor: cf — multiplies ceil(B/E) to set per-expert cap.
    ExpertsChoiceMoE(size_t d_model, size_t num_experts,
                     size_t expert_hidden = 0,
                     double capacity_factor = 1.0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_router; }
    Tensor get_gradients() const override { return grad_W_router; }
    std::string name() const override { return "ExpertsChoiceMoE"; }

    // Public accessors (tests).
    size_t d_model() const { return d_model_; }
    size_t num_experts() const { return num_experts_; }
    size_t expert_hidden() const { return expert_hidden_; }
    double capacity_factor() const { return capacity_factor_; }
    Tensor& router_W() { return W_router; }
    Tensor& router_b() { return b_router; }
    Tensor& grad_router_W() { return grad_W_router; }
    Tensor& grad_router_b() { return grad_b_router; }
    Tensor& grad_input() { return last_d_input_; }

    // Per-expert weight accessors
    Tensor& expert_W1(size_t e) { return W1_[e]; }
    Tensor& expert_W2(size_t e) { return W2_[e]; }
    Tensor& expert_b1(size_t e) { return b1_[e]; }
    Tensor& expert_b2(size_t e) { return b2_[e]; }
    Tensor& grad_expert_W1(size_t e) { return gW1_[e]; }
    Tensor& grad_expert_W2(size_t e) { return gW2_[e]; }
    Tensor& grad_expert_b1(size_t e) { return gb1_[e]; }
    Tensor& grad_expert_b2(size_t e) { return gb2_[e]; }

    // Returns the indices of tokens assigned to expert e in the last forward.
    // Length is exactly the per-expert capacity (k = ceil(B/E)*cf).
    const std::vector<size_t>& last_expert_assignment(size_t e) const { return assign_[e]; }

private:
    size_t d_model_;
    size_t num_experts_;
    size_t expert_hidden_;
    double capacity_factor_;

    // Router: W_router (E, d_model), b_router (1, E)
    Tensor W_router, b_router;
    Tensor grad_W_router, grad_b_router;

    // Per-expert FFN (ReLU).
    // W1_[e]: (H, d_model), W2_[e]: (d_model, H)
    std::vector<Tensor> W1_, W2_, b1_, b2_;
    std::vector<Tensor> gW1_, gW2_, gb1_, gb2_;

    // Forward caches
    Tensor input_;             // (B, d_model)
    Tensor gate_logits_;       // (B, E)
    std::vector<std::vector<size_t>> assign_;   // assign_[e] = indices assigned to e (length k)
    Tensor dispatch_mask_;     // (B, E) 1 iff token b selected by expert e

    // Per-expert cached activations of dispatched tokens
    std::vector<Tensor> expert_h_pre_;   // (k, H) pre-ReLU
    std::vector<Tensor> expert_h_act_;   // (k, H) post-ReLU
    std::vector<Tensor> expert_inputs_;  // (k, d_model) dispatched inputs
    std::vector<Tensor> expert_outputs_; // (k, d_model) post-FFN cache (for backward)

    Tensor last_d_input_;       // (B, d_model)
};

// ============================================================================
// SparseMixerBlock — pre-norm + mixing/attention + residual
//                  + pre-norm + MoE/dense FFN + residual
// ============================================================================

class SparseMixerBlock : public Layer {
public:
    enum Mode { LINEAR_MIXING, SELF_ATTENTION };

    // d_model:    feature dim.
    // seq_len:    token count S (must be known statically — T_seq has shape S×S).
    // mode:       LINEAR_MIXING or SELF_ATTENTION.
    // num_experts: 0 → dense FFN, >0 → ExpertsChoiceMoE with this many experts.
    // expert_hidden: hidden dim of each expert FFN (default 4*d_model).
    // capacity_factor: cf for ExpertsChoiceMoE (default 1.0).
    // ffn_mult: hidden width of the dense FFN (default 4*d_model).
    SparseMixerBlock(Mode mode, size_t d_model, size_t seq_len,
                     size_t num_experts = 0,
                     size_t expert_hidden = 0,
                     double capacity_factor = 1.0,
                     size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "SparseMixerBlock"; }

    // Accessors for testing
    Mode mode() const { return mode_; }
    size_t d_model() const { return d_model_; }
    size_t seq_len() const { return seq_len_; }
    Tensor& grad_input() { return last_d_input_; }

private:
    Mode mode_;
    size_t d_model_;
    size_t seq_len_;
    bool has_moe_;

    // Pre-norms
    LayerNorm ln1_, ln2_;

    // First sublayer (mixing or self-attention)
    // LINEAR_MIXING: holds LinearMixingSublayer
    // SELF_ATTENTION: holds 4 Dense layers (W_q, W_k, W_v, W_o) for single-head
    //                 attention + cached Q, K, V, A, attn_output.
    std::unique_ptr<Layer> mixer_;          // LinearMixingSublayer when LINEAR_MIXING
    Tensor W_q, W_k, W_v, W_o;              // (d_model, d_model) — single-head attention
    Tensor b_q, b_k, b_v, b_o;              // (1, d_model)
    Tensor grad_W_q, grad_W_k, grad_W_v, grad_W_o;
    Tensor grad_b_q, grad_b_k, grad_b_v, grad_b_o;
    Tensor last_z1_;        // pre-norm output (B, S*D)
    Tensor last_q_, last_k_, last_v_;  // (B, S, d_model) — cached for attention backward
    Tensor last_attn_;      // (B, S, S) attention probabilities
    Tensor last_attn_out_;  // (B, S, d_model) post-attention
    Tensor last_d_z1_;      // (B, S*D) input grad to mixer / attn sublayer
    Tensor last_d_input_;   // (B, S*D) final grad_input

    // Second sublayer (MoE or dense FFN)
    std::unique_ptr<Layer> ffn_;             // Either ExpertsChoiceMoE or a small Dense-pair wrapper
    Dense ffn_dense1_ = Dense(1, 1);  // default-constructed; real init in body
    Dense ffn_dense2_ = Dense(1, 1);
    Tensor last_z2_;        // pre-norm output (B, S*D)
    Tensor last_ffn_h_;     // (B*S, ffn_dim) pre-activation of ffn_dense1_
    Tensor last_ffn_h_act_; // (B*S, ffn_dim) post-GELU
    Tensor last_ffn_out_;   // (B, S*D)
    Tensor last_d_z2_;      // (B, S*D)
};

// ============================================================================
// SparseMixerModel — full encoder stack
// ============================================================================

class SparseMixerModel : public Layer {
public:
    // input_dim: feature dim of the raw input.
    // d_model:   mixer/attention dim.
    // output_dim: classifier output dim.
    // seq_len:   token count S.
    // num_layers: total block count.
    // num_attention_layers: how many of the LAST blocks use SELF_ATTENTION.
    // num_experts: 0 → all FFNs are dense; >0 → FFN sublayer uses ExpertsChoiceMoE.
    SparseMixerModel(size_t input_dim, size_t d_model, size_t output_dim,
                     size_t seq_len,
                     size_t num_layers,
                     size_t num_attention_layers,
                     size_t num_experts,
                     size_t expert_hidden = 0,
                     double capacity_factor = 1.0,
                     size_t ffn_mult = 4);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj_.weights; }
    Tensor get_gradients() const override { return input_proj_.grad_weights; }
    std::string name() const override { return "SparseMixerModel"; }

private:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t seq_len_;
    size_t num_layers_;
    size_t num_attention_layers_;

    Dense input_proj_;          // (input_dim, d_model)
    LayerNorm head_ln_;
    Dense classifier_;          // (d_model, output_dim)
    std::vector<SparseMixerBlock> blocks_;

    Tensor last_input_;          // (B, input_dim * seq_len)
    Tensor last_z_proj_;         // (B, S * d_model) post input projection (pre-block-stack)
    Tensor last_blocks_out_;     // (B, S * d_model) post-block-stack
    Tensor last_head_ln_;        // (B, S * d_model) post head-LN
    Tensor last_pooled_;         // (B, d_model) post mean-pool
    Tensor last_logits_;         // (B, output_dim)
    Tensor last_d_input_;        // (B, input_dim * seq_len)
};

#endif