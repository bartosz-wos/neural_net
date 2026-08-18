#ifndef MEGA_H
#define MEGA_H

#include "../../core/layer.h"
#include "../../activations/activations.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================================
// MEGA — Moving Average Equipped Gated Attention
//   Ma, Zhou, Li, Kong, Wei, Zheng (Apple), 2022.
//   "Mega: Moving Average Equipped Gated Attention"
//   https://arxiv.org/abs/2209.10655
//
// A MEGA block is a pre-norm residual sequence mixer that:
//   (1) smooths the input features with a per-channel exponential moving
//       average (EMA), parametrized by a learnable per-channel decay α ∈ (0,1);
//   (2) projects the smoothed stream to Q/K/V via Dense layers and to an
//       output gate Z = sigmoid(W_g · u);
//   (3) computes causal soft attention over Q/K/V with a learned relative
//       position bias β (a single vector of length 2T-1 covering offsets
//       -(T-1)..(T-1)); applies the output gate g = o ⊙ z;
//   (4) projects the gated attention output through W_o;
//   (5) sums into the residual stream;
//   (6) applies a position-wise GELU FFN with the canonical pre-norm residual.
//
// Per-block forward (input x ∈ R^(T × d_model), T = sequence length):
//
//   u_0 = 0
//   u_t = α ⊙ u_{t-1} + (1 - α) ⊙ x_t                            for t = 0..T-1
//   q_t = u_t · W_q^T                                          ∈ R^(d_inner)
//   k_t = u_t · W_k^T                                          ∈ R^(d_inner)
//   v_t = u_t · W_v^T                                          ∈ R^(d_inner)
//   z_t = sigmoid(u_t · W_g^T)                                 ∈ R^(d_inner)
//
//   score[t,s] = q_t · k_s / sqrt(d_inner) + β[t - s + (T-1)]    for s ≤ t  (causal)
//   attn[t,:]  = softmax(score[t, 0..t])
//   o_t        = Σ_{s=0..t} attn[t,s] · v_s
//   g_t        = o_t ⊙ z_t                                       (output gate)
//
//   h_t = g_t · W_o^T                                          ∈ R^(d_model)
//
//   y_t = x_t + h_t                                            (residual)
//   y_t = y_t + FFN(LN(y_t))                                   (FFN with pre-norm)
//
// d_inner = d_model for the single-head (num_heads = 1) v1.
// Multi-head attention is not in scope for this header — the constructor throws
// if num_heads != 1 to keep the gradient math and tests tractable. Multi-head
// extension would mirror the existing MultiHeadAttention split, but MEGA's
// output gate is applied per-feature and works cleanly with single-head Q/K/V.
// ============================================================================

class MegaBlock : public Layer {
public:
    size_t d_model_;
    size_t num_heads_;
    size_t ffn_mult_;
    size_t T_;  // sequence length seen during the most recent forward (cached for backward)

    // Q/K/V projections from the EMA-smoothed stream u.
    Dense W_q;  // Dense(d_model, d_model)  → weights (d_model, d_model),   y = u @ W_q^T
    Dense W_k;  // Dense(d_model, d_model)
    Dense W_v;  // Dense(d_model, d_model)
    Dense W_o;  // Dense(d_model, d_model)  output projection from gated attn

    // Output gate Z = sigmoid(W_g · u).
    Dense W_g;  // Dense(d_model, d_model)

    // EMA decay: α = sigmoid(W_α). W_α is a raw (1, d_model) row tensor; bias-like.
    Tensor alpha_log;       // (1, d_model) — sigmoid gives α ∈ (0, 1)
    Tensor grad_alpha_log;  // (1, d_model)

    // Relative position bias β of length 2T-1; index = (t - s) + (T - 1).
    // Stored as a (1, 2*T_max-1) row tensor; only the first (2*T_-1) entries are used
    // when T is known. We pad with zeros beyond to allow variable T without realloc.
    size_t bias_max_len_;   // = 2 * T_bias_max_ - 1 for the current allocation
    size_t T_bias_max_;     // max T this bias vector has been allocated for
    Tensor pos_bias;        // (1, 2*T_bias_max_ - 1)
    Tensor grad_pos_bias;   // (1, 2*T_bias_max_ - 1)

    // FFN sublayer: Dense(d → mult*d) → GELU → Dense(mult*d → d).
    Dense ffn_W1;  // Dense(d_model, mult*d_model)  weights (mult*d_model, d_model)
    Dense ffn_W2;  // Dense(mult*d_model, d_model)  weights (d_model, mult*d_model)
    LayerNorm ffn_ln;

    // Forward caches (T_, d_model) shape — exposed for tests.
    Tensor last_input;       // (T_, d_model) original input x
    Tensor last_u;           // (T_, d_model) EMA-smoothed stream
    Tensor last_u_prev;      // (T_, d_model) u shifted by 1 (last_u_prev[0] = 0)
    Tensor last_alpha;       // (1, d_model)  sigmoid(alpha_log)
    Tensor last_q;           // (T_, d_model)
    Tensor last_k;           // (T_, d_model)
    Tensor last_v;           // (T_, d_model)
    Tensor last_z;           // (T_, d_model) sigmoid(W_g · u) — output gate
    Tensor last_z_pre;       // (T_, d_model) pre-sigmoid (for sigmoid derivative)
    Tensor last_score;       // (T_, T_)      pre-softmax scores (causal: row t has s≤t finite)
    Tensor last_attn;        // (T_, T_)      softmax of last_score (causal: row t has s≤t nonzero)
    Tensor last_o;           // (T_, d_model) attention output pre-gate
    Tensor last_g;           // (T_, d_model) o ⊙ z
    Tensor last_h;           // (T_, d_model) g @ W_o^T (pre-residual contribution)
    Tensor last_residual;    // (T_, d_model) x + h  (input to FFN sublayer)
    Tensor last_ffn_ln_in;   // (T_, d_model) = last_residual (cache for FFN LN backward)
    Tensor last_ffn_ln_out;  // (T_, d_model) LN output
    Tensor last_ffn_hidden;  // (T_, mult*d_model) pre-GELU (W1 · LN)
    Tensor last_ffn_act;     // (T_, mult*d_model) post-GELU
    Tensor last_ffn_out;     // (T_, d_model) W2 · act

    MegaBlock(size_t d_model, size_t num_heads = 1, size_t ffn_mult = 4);
    ~MegaBlock() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "MegaBlock"; }

    // Copy all learnable parameters (W_q, W_k, W_v, W_o, W_g, alpha_log, pos_bias,
    // ffn_W1, ffn_W2, ffn_ln gamma/beta) from `other` into this block.
    // Both blocks must have identical config.
    void copy_params_from(const MegaBlock& other);

    // Accessors.
    size_t d_model()    const { return d_model_; }
    size_t num_heads()  const { return num_heads_; }
    size_t ffn_mult()   const { return ffn_mult_; }
};

// Stack of `num_layers` MegaBlock + input projection + final LayerNorm + classifier.
class MegaModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t num_heads_;
    size_t ffn_mult_;

    Dense input_proj;  // Dense(input_dim, d_model)
    std::vector<std::unique_ptr<MegaBlock>> blocks;
    LayerNorm final_ln;
    Dense output_proj;  // Dense(d_model, output_dim)

    MegaModel(size_t input_dim, size_t d_model, size_t output_dim,
              size_t num_layers, size_t num_heads = 1, size_t ffn_mult = 4);
    ~MegaModel() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return input_proj.weights; }
    Tensor get_gradients() const override { return input_proj.grad_weights; }
    std::string name() const override { return "MegaModel"; }
};

#endif // MEGA_H