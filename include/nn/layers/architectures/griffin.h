#ifndef GRIFFIN_H
#define GRIFFIN_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include "../recurrent/hawk.h"
#include <vector>
#include <memory>

// ============================================================================
// Griffin Hybrid Block (De et al. 2024, https://arxiv.org/abs/2402.19427,
// "Griffin: Mixing Gated Linear Recurrences with Local Attention for
// Efficient Language Models").
//
// A Griffin block is the parallel composition of three sublayers fed by a
// shared LayerNorm, summed into a single residual stream:
//
//   h = Hawk(LN(x)) + Attn_local(LN(x)) + MLP(LN(x))
//   out = x + h
//
// Key architectural distinction vs Jamba (which is sequential Mamba → Attn
// → FFN): Griffin composes its sublayers in parallel. Each sublayer sees
// the same pre-norm input.
//
// Sublayer definitions:
//   (1) HawkBlock (RG-LRU) — gated linear recurrence. See
//       include/nn/layers/recurrent/hawk.h for the full math.
//   (2) LocalSlidingWindowAttention — causal local attention. Query at
//       time t attends to keys in [max(0, t - window_size + 1), ..., t]
//       via softmax. window_size=1 reduces to identity (only the query
//       itself attends to itself).
//   (3) GriffinMLP — 2-layer GELU FFN. Linear(d → mult*d) → GELU →
//       Linear(mult*d → d).
//
// Forward shape: (T, d_model) → (T, d_model). Layout is (T, d_model)
// throughout (matches HawkBlock and LayerNorm convention).
// ============================================================================

// Local sliding-window causal attention (single head, d_model in/out).
class LocalSlidingWindowAttention : public Layer {
public:
    size_t d_;
    size_t window_size_;
    Dense W_q;  // (d, d)
    Dense W_k;  // (d, d)
    Dense W_v;  // (d, d)
    Dense W_o;  // (d, d)

    // Forward caches (public for tests)
    Tensor last_q;       // (T, d)
    Tensor last_k;       // (T, d)
    Tensor last_v;       // (T, d)
    Tensor last_attn;    // (T, T_w_max) per-row attention weights (T rows, padded with 0)

    LocalSlidingWindowAttention(size_t d, size_t window_size);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "LocalSlidingWindowAttention"; }
};

// 2-layer GELU MLP for the Griffin FFN branch.
class GriffinMLP : public Layer {
public:
    size_t d_;
    size_t mult_;
    size_t hidden_;
    Dense W1;  // Dense(d, hidden) → weights shape (hidden, d)  so y = x @ W1^T → (T, hidden)
    Dense W2;  // Dense(hidden, d) → weights shape (d, hidden)  so y = x @ W2^T → (T, d)

    // Forward caches
    Tensor last_input;     // (T, d)
    Tensor last_hidden;    // (T, hidden)
    Tensor last_act;       // (T, hidden) — gelu(hidden)

    GriffinMLP(size_t d, size_t mult);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W1.weights; }
    Tensor get_gradients() const override { return W1.grad_weights; }
    std::string name() const override { return "GriffinMLP"; }
};

// Full Griffin block: shared LN feeding 3 parallel branches.
class GriffinBlock : public Layer {
public:
    size_t d_;
    size_t num_heads_;
    size_t window_size_;
    size_t ffn_mult_;

    // Sublayers (public for test introspection)
    LayerNorm ln;                              // shared pre-norm
    HawkBlock hawk;                            // gated linear recurrence
    LocalSlidingWindowAttention attn;          // local sliding-window attention
    GriffinMLP mlp;                            // GELU FFN

    // Forward caches
    Tensor last_ln_input;   // (T, d) — input to LN (= x)
    Tensor last_ln_out;     // (T, d) — LN(x), shared by all 3 branches
    Tensor last_hawk_out;   // (T, d) — output of hawk branch
    Tensor last_attn_out;   // (T, d) — output of attn branch
    Tensor last_mlp_out;    // (T, d) — output of mlp branch

    GriffinBlock(size_t d_model, size_t num_heads,
                  size_t window_size = 4, size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return ln.gamma; }
    Tensor get_gradients() const override { return ln.grad_gamma_; }
    std::string name() const override { return "GriffinBlock"; }

    // Copy all learnable parameters from `other` into this block. Useful
    // for determinism / reproducibility tests. Both blocks must have
    // identical shape configuration.
    void copy_params_from(const GriffinBlock& other);

    // Accessors
    size_t d() const { return d_; }
    size_t num_heads() const { return num_heads_; }
    size_t window_size() const { return window_size_; }
    size_t ffn_mult() const { return ffn_mult_; }
    size_t num_blocks() const { return 1; }  // for stack contract parity
};

// Stack of GriffinBlocks with a final classifier.
class GriffinModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t num_heads_;
    size_t window_size_;
    size_t ffn_mult_;

    Dense embed;                              // input_dim → d_model
    std::vector<std::unique_ptr<GriffinBlock>> blocks;
    LayerNorm final_ln;
    Dense classifier;                         // d_model → output_dim

    GriffinModel(size_t input_dim, size_t d_model, size_t output_dim,
                size_t num_layers, size_t num_heads,
                size_t window_size = 4, size_t ffn_mult = 2);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return embed.weights; }
    Tensor get_gradients() const override { return embed.grad_weights; }
    std::string name() const override { return "GriffinModel"; }
};

#endif // GRIFFIN_H