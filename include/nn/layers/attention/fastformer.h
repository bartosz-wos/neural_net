#ifndef FASTFORMER_H
#define FASTFORMER_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// ============================================================================
// FastFormer — Wu, Liu, Meng, Zheng, Wei, Que, Wang, Sui (AAAI 2021)
//   "FastFormer: Additive Attention Can Be All You Need"
//   (https://arxiv.org/abs/2108.09084)
//
// The key idea is to replace the O(n²) softmax(QK^T) attention with two
// stages of O(n) additive pooling, plus a single per-token softmax over
// the modulated K^T (which is still O(n) per token).
//
// Per-head forward (input X ∈ R^{n × d_model}, projected to Q, K, V ∈ R^{n × d}):
//
//   1. q_global = (1/n) Σ_i Q_i                                     ∈ R^d
//   2. p_i = sigmoid(W_p · [Q_i ; q_global] + b_p)                  ∈ R^d
//      K_mod_i = p_i ⊙ K_i                                          ∈ R^d
//      k_global = (1/n) Σ_i K_mod_i                                 ∈ R^d
//   3. b_i = sigmoid(W_b · [K_i ; k_global] + b_b)                  ∈ R^d
//      Q_mod_i = b_i ⊙ Q_i                                          ∈ R^d
//   4. α_i = softmax( Q_mod_i · K^T / sqrt(d) )                    ∈ R^n
//      c_i = α_i · V                                                ∈ R^d
//   5. γ_i = sigmoid( w_γ · c_i + b_γ )                             ∈ R    (scalar gate)
//      out_i = b_i ⊙ V_i + γ_i · c_i                                ∈ R^d
//
// Multi-head: split d_model = num_heads × head_dim, run per head, concat, W_o project.
//
// The paper additionally proposes the linear-attention variant
//   c_i = (Q_mod_i · K^T) · V  =  Q_mod_i · (K^T · V)
// where the d×d matrix K^T V is computed once. For v1 we keep the explicit
// softmax form (more numerically robust, cleaner gradient chain).
//
// Layout convention: (n, d_model) — rows are tokens, cols are features.
// Pre-LN block pattern matches the rest of the repo.
// ----------------------------------------------------------------------------

class FastFormerAttention : public Layer {
public:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;     // d_model_ / num_heads_
    double scale_;        // 1/sqrt(head_dim_)
    bool causal_;

    // Projections (raw Tensors, not Dense — keeps (n, d_model) layout).
    Tensor W_q_;  Tensor b_q_;  Tensor grad_W_q_;  Tensor grad_b_q_;  // (d, d), (1, d)
    Tensor W_k_;  Tensor b_k_;  Tensor grad_W_k_;  Tensor grad_b_k_;  // (d, d), (1, d)
    Tensor W_v_;  Tensor b_v_;  Tensor grad_W_v_;  Tensor grad_b_v_;  // (d, d), (1, d)
    Tensor W_p_;  Tensor b_p_;  Tensor grad_W_p_;  Tensor grad_b_p_;  // (d, 2d), (1, d)
    Tensor W_b_;  Tensor b_b_;  Tensor grad_W_b_;  Tensor grad_b_b_;  // (d, 2d), (1, d)
    Tensor W_gamma_; Tensor b_gamma_;                                  // (1, d), (1, 1)
    Tensor grad_W_gamma_; Tensor grad_b_gamma_;
    Tensor W_o_;  Tensor b_o_;  Tensor grad_W_o_;  Tensor grad_b_o_;  // (d, d), (1, d)

    // Cache for backward
    Tensor last_input_;    // (n, d_model)
    Tensor last_q_;        // (n, d)
    Tensor last_k_;        // (n, d)
    Tensor last_v_;        // (n, d)
    Tensor last_q_global_; // (1, d)
    Tensor last_p_;        // (n, d)
    Tensor last_k_mod_;    // (n, d)
    Tensor last_k_global_; // (1, d)
    Tensor last_b_;        // (n, d)   (the Q-modulating gate; called "b" in paper)
    Tensor last_q_mod_;    // (n, d)
    Tensor last_alpha_;    // (n, n)   softmax(Q_mod · K^T / sqrt(d))
    Tensor last_c_;        // (n, d)
    Tensor last_gamma_;    // (n, 1)
    Tensor last_y_pre_;    // (n, d_model)  — the per-head concatenated output before W_o
    Tensor last_output_;   // (n, d)

    FastFormerAttention(size_t d_model, size_t num_heads, bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_q_; }
    Tensor get_gradients() const override { return grad_W_q_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FastFormerAttention"; }
};


// ============================================================================
// FastFormerBlock — pre-LN → FastFormerAttention → residual → pre-LN → FFN → residual.
// FFN is optional: when ffn_hidden = 0 the FFN path is the identity.
// FFN: GELU(W1 · x + b1) · W2 + b2   with W1: (ffn_hidden, d_model), W2: (d_model, ffn_hidden).
// ============================================================================
class FastFormerBlock : public Layer {
public:
    size_t d_model_;
    size_t ffn_hidden_;
    bool has_ffn_;
    FastFormerAttention attn_;
    LayerNorm ln1_;     // pre-attn
    LayerNorm ln2_;     // pre-FFN

    // FFN weights
    Tensor W1_; Tensor b1_; Tensor W2_; Tensor b2_;
    Tensor grad_W1_; Tensor grad_b1_; Tensor grad_W2_; Tensor grad_b2_;

    // Cache
    Tensor last_input_;
    Tensor last_ln1_out_;
    Tensor last_attn_out_;
    Tensor last_r1_;        // residual after attn
    Tensor last_ln2_out_;
    Tensor last_ffn_pregelu_;
    Tensor last_ffn_gelu_;
    Tensor last_ffn_out_;
    Tensor last_output_;

    FastFormerBlock(size_t d_model, size_t num_heads,
                    size_t ffn_hidden = 0, bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W1_; }
    Tensor get_gradients() const override { return grad_W1_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FastFormerBlock"; }
};


// ============================================================================
// FastFormerModel — encoder for sequence classification.
// Forward: input proj (Dense) → stack of FastFormerBlocks → mean pool → classifier (Dense).
// Input shape (input_dim, seq_len), output shape (output_dim, 1).
// ============================================================================
class FastFormerModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t seq_len_;
    size_t num_blocks_;
    size_t num_heads_;
    size_t ffn_hidden_;
    bool causal_;

    // Input projection (kept as raw Tensors to match (d_model, seq_len) layout).
    Tensor W_in_;  Tensor b_in_;  Tensor grad_W_in_;  Tensor grad_b_in_;  // (d_model, input_dim), (1, d_model)

    // Classifier (input is mean-pooled: (1, d_model) → Dense → (1, output_dim)).
    Tensor W_cls_;  Tensor b_cls_;  Tensor grad_W_cls_;  Tensor grad_b_cls_;  // (output_dim, d_model), (1, output_dim)

    std::vector<FastFormerBlock> blocks_;
    LayerNorm ln_out_;

    // Cache
    Tensor last_input_;        // (input_dim, seq_len)
    Tensor last_proj_out_;     // (d_model, seq_len)
    std::vector<Tensor> last_block_outs_;
    Tensor last_pool_;         // (1, d_model)
    Tensor last_ln_out_;       // (1, d_model)
    Tensor last_output_;       // (output_dim, 1)

    FastFormerModel(size_t input_dim, size_t d_model, size_t output_dim,
                    size_t seq_len, size_t num_blocks = 2,
                    size_t num_heads = 4, size_t ffn_hidden = 0,
                    bool causal = false);
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return W_in_; }
    Tensor get_gradients() const override { return grad_W_in_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "FastFormerModel"; }
};

#endif
