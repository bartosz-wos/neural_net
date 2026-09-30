#ifndef QK_NORM_H
#define QK_NORM_H

#include "../../core/layer.h"
#include <vector>
#include <cstddef>
#include <string>
#include <utility>

// ============================================================================
// QKNorm — Dehghani et al. 2023, "Scaling Vision Transformers to 22 Billion
// Parameters" (https://arxiv.org/abs/2302.05442), §3.2.
//
// Per-head L2 normalization applied to the Q and K tensors inside the
// attention score computation. Replaces the raw QK^T/√d_k with the normalized
// variant:
//
//   q̂[t, j] = q[t, j] * (γ / (||q[t, :]||_2 + eps))      per head, per row
//   k̂[s, j] = k[s, j] * (γ / (||k[s, :]||_2 + eps))      per head, per row
//   S[t, s]  = (q̂[t, :] · k̂[s, :]) / sqrt(head_dim)      per head
//
// γ is a single per-head scalar (broadcast to all head_dim channels), so
// num_heads learnable parameters in total. Initialized to γ = 1.0 so the
// forward starts as plain L2 normalization. During training γ moves and the
// effective temperature inside the softmax changes — this is the training
// stabilizer that ViT-22B credits with making 22B-param ViTs trainable.
//
// Conventions:
//   * Input/output:  (N, d_model) row-major, matching every other attention
//     variant in the repo.
//   * Cache: stores Q, K, the normalized Q̂, K̂, the per-row norms (for
//     backward), and the per-head scalar γ used in the last forward.
//   * QKNorm is a STANDALONE primitive — it does not own a softmax or
//     attention map; callers (the test, or a hypothetical attention wrapper)
//     compute S = Q̂·K̂^T / √d_k and apply softmax themselves.
//     QKNorm just transforms (Q, K) → (Q̂, K̂). This makes the gradient check
//     independent of attention math, which is the standard pedagogical
//     decomposition used in the paper and in timm.
// ============================================================================

class QKNorm : public Layer {
public:
    // d_model:       total feature dim
    // num_heads:     number of attention heads (default 1 — single-head is the
    //                canonical FD-check config)
    // eps:           numerical stabilizer for the L2 norm denominator (default 1e-6)
    QKNorm(size_t d_model, size_t num_heads = 1, double eps = 1e-6);

    // forward takes two tensors (Q, K) and returns two normalized tensors.
    // Both inputs must have shape (N, d_model). Returns a pair: {Q̂, K̂}.
    // The repo's Layer interface takes a single Tensor input, so we expose a
    // dedicated `forward_pair(q, k)` API. The base `forward(input)` is kept as
    // a thin convenience that treats `input` as Q and returns Q̂ only — used
    // by tests that want to FD-check the Q path in isolation.
    Tensor forward(const Tensor& input) override;
    std::pair<Tensor, Tensor> forward_pair(const Tensor& Q, const Tensor& K);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma_; }
    Tensor get_gradients() const override { return grad_gamma_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

    // Accessors for tests
    size_t d_model() const { return d_model_; }
    size_t num_heads() const { return num_heads_; }
    size_t head_dim() const { return head_dim_; }
    double eps() const { return eps_; }
    double gamma_value(size_t h) const { return gamma_(0, h); }

    std::string name() const override { return "QKNorm"; }

    // Public parameters (for tests to mutate)
    Tensor gamma_;          // shape (1, num_heads), init 1.0
    Tensor grad_gamma_;     // shape (1, num_heads)

    // BPTT caches (public for tests)
    Tensor last_q_;         // (N, d_model)  — Q after projection (clone)
    Tensor last_k_;         // (N, d_model)  — K after projection (clone)
    Tensor last_q_norm_;    // (N, num_heads)  — per-row L2 norms of Q (per head)
    Tensor last_k_norm_;    // (N, num_heads)  — per-row L2 norms of K (per head)

    // Backward caches (which gradient path was last taken)
    enum class LastPath { QOnly, QAndK };
    LastPath last_path() const { return last_path_; }

private:
    size_t d_model_;
    size_t num_heads_;
    size_t head_dim_;       // d_model / num_heads
    double eps_;
    LastPath last_path_ = LastPath::QOnly;
};

#endif // QK_NORM_H
