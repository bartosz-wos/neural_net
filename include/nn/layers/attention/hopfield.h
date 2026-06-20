#ifndef HOPFIELD_H
#define HOPFIELD_H

#include "../../core/layer.h"
#include "../normalization/layer_norm.h"
#include <vector>
#include <cmath>

// softplus(x) = log(1 + exp(x)).  Numerically stable: use log1p(exp(-|x|)) + max(x, 0).
inline double softplus(double x) {
    if (x > 0.0) {
        return x + std::log1p(std::exp(-x));
    } else {
        return std::log1p(std::exp(x));
    }
}

// ============================================================================
// Modern Hopfield Attention — Ramsauer et al. 2020
//   "Hopfield Networks is All You Need" (https://arxiv.org/abs/2008.02217)
//
// Innovation: reinterpret the transformer attention mechanism as a *Hopfield
// network energy-minimization retrieval* over a stored set of "memory patterns".
//
// The classical discrete Hopfield network retrieves a stored pattern by
// repeatedly applying x <- sign(W x) until convergence. The *modern* Hopfield
// network (Ramsauer 2020) generalizes the energy function so that the *exact*
// fixed point is given by a softmax-weighted average over the stored patterns:
//
//   x* = P^T softmax(beta * P x)        (one query, m patterns of dim d)
//
// In matrix form (batched over n queries, m patterns):
//
//   Q = X @ W_q                          (n, d)
//   scores = beta * Q @ P^T              (n, m)        — beta = inverse temperature
//   attn = row_softmax(scores)           (n, m)
//   out_pre = attn @ P                   (n, d)        — Hopfield retrieval
//   out = out_pre @ W_o^T + b_o          (n, d)        — output projection
//
// Compare to standard transformer attention (Vaswani 2017):
//   attn = softmax(Q K^T / sqrt(d)) V
//
// The mapping is exact when we set K = V = P (shared projection) and
// use a learnable inverse temperature beta instead of 1/sqrt(d). This means
// the Hopfield layer is *strictly more general* than self-attention: it can
// (a) decouple the pattern count m from the sequence length n, and (b) learn
// the right temperature for the data.
//
// Implementation choices (matching repo conventions):
//   * Input X: (n, d_model), Output: (n, d_model).
//   * Patterns P: (m, d_model), with m a hyperparameter (default = d_model).
//   * W_q, W_o: (d_model, d_model) — Dense convention (y = X @ W^T + b).
//   * Inverse temperature beta: a single learnable scalar, stored as
//     beta_log_ and reparameterized as beta = softplus(beta_log_)
//     (so beta > 0 always; this matches Ramsauer's recommendation to learn
//     beta with the positive reparam).
//   * Pattern bias b_p ∈ R^m (one bias per pattern) — optional, but it gives
//     the patterns a "default" and helps when queries are degenerate.
//   * Default scale: 1/sqrt(d_model). The Hopfield literature uses beta =
//     1/sqrt(d) at init, which is the most numerically stable starting point.
//
// We expose:
//   * HopfieldAttention        — single modern Hopfield retrieval block.
//   * HopfieldBlock            — pre-LN → HopfieldAttention → residual →
//                                pre-LN → GELU FFN → residual.
//   * HopfieldModel            — stack of HopfieldBlocks + per-token
//                                classifier head.
// ============================================================================

class HopfieldAttention : public Layer {
public:
    using Layer::forward;  // bring base forward into scope (1-arg override)

    // d_model:    input/output feature dim
    // num_patterns: number of stored "memory patterns" m
    //                (default d_model; the most common convention).
    HopfieldAttention(size_t d_model, size_t num_patterns = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return W_q.weights; }
    Tensor get_gradients() const override { return W_q.grad_weights; }
    std::string name() const override { return "HopfieldAttention"; }

    // Accessors
    size_t d_model()      const { return d_model_; }
    size_t num_patterns() const { return num_patterns_; }
    double beta()         const { return softplus(beta_log_(0, 0)); }
    const Tensor& patterns() const { return P_; }

    // Public parameters (for tests / introspection)
    size_t d_model_;
    size_t num_patterns_;
    Dense W_q;             // (d_model, d_model)
    Tensor P_;             // (m, d_model) — stored patterns
    Tensor b_p_;           // (1, m)      — pattern bias
    Dense W_o;             // (d_model, d_model)
    // Inverse temperature scalar stored as a 1x1 tensor so the test
    // harness can perturb it via the standard parameters() / gradients()
    // interface. softplus(x) = log(1 + exp(x)) > 0 always, so beta > 0.
    Tensor beta_log_;      // (1, 1) — scalar; beta_pos = softplus(beta_log_)

private:

    // Gradient accumulators for raw tensors
    Tensor grad_P_;        // (m, d_model)
    Tensor grad_b_p_;      // (1, m)
    Tensor grad_beta_log_; // (1, 1) — scalar stored as 1x1 tensor for uniformity

    // Caches for backward
    Tensor last_input_;    // (n, d_model)
    Tensor last_Q_;        // (n, d_model)
    Tensor last_attn_;     // (n, m)
    Tensor last_out_pre_;  // (n, d_model)
    double last_beta_pos_; // cached beta at the time of forward
};

// ============================================================================
// HopfieldBlock — pre-LN → HopfieldAttention → residual →
//                  pre-LN → GELU FFN → residual
// ============================================================================

class HopfieldBlock : public Layer {
public:
    HopfieldBlock(size_t d_model, size_t num_patterns = 0, size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "HopfieldBlock"; }

private:
    size_t d_model_;
    size_t ffn_dim_;

    LayerNorm ln1_;              // pre-attn
    HopfieldAttention attn_;
    LayerNorm ln2_;              // pre-FFN
    Dense ffn_fc1_;              // (ffn_dim, d_model)
    Dense ffn_fc2_;              // (d_model, ffn_dim)

    // Cache
    Tensor last_input_;
    Tensor last_z1_;
    Tensor last_attn_out_;
    Tensor last_res1_;
    Tensor last_z2_;
    Tensor last_h_pre_;          // (n, ffn_dim)
    Tensor last_h_act_;          // (n, ffn_dim)  GELU(last_h_pre_)
    Tensor last_ffn_out_;
};

// ============================================================================
// HopfieldModel — stack of blocks + per-token classifier
// ============================================================================

class HopfieldModel : public Layer {
public:
    HopfieldModel(size_t d_model, size_t out_features,
                  size_t num_blocks = 1, size_t num_patterns = 0,
                  size_t ffn_dim = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return Tensor(); }
    Tensor get_gradients() const override { return Tensor(); }
    std::string name() const override { return "HopfieldModel"; }

private:
    size_t d_model_;
    size_t out_features_;
    size_t num_blocks_;
    std::vector<std::unique_ptr<HopfieldBlock>> blocks_;
    Dense classifier_;

    Tensor last_input_;
    Tensor last_block_output_;
};

#endif
