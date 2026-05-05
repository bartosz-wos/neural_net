#ifndef ROPE_H
#define ROPE_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// RoPE (Rotary Position Embedding) from "RoFormer: Enhanced Transformer with Rotary Position Embedding"
// Applied to query and key tensors to encode relative position through rotation.
//
// Core math: for dimension i in pair (i, i+d/2), the rotation at position `pos` is:
//   RoPE(x_i, pos)       = x_i * cos(pos * θ_i) - x_{i+d/2} * sin(pos * θ_i)
//   RoPE(x_{i+d/2}, pos) = x_{i+d/2} * cos(pos * θ_i) + x_i * sin(pos * θ_i)
// where θ_i = base^(-2i/d) for i in [0, d/2)
//
// Cache layout: cos_cache and sin_cache are [max_seq_len, dim] row-major,
// where row `pos` contains the rotation angles for that position across all dimensions.
class RoPE : public Layer {
public:
    int dim_;           // embedding dimension (must be even)
    int max_seq_len_;   // maximum sequence length
    float base_;        // theta base (default 10000)
    int current_seq_len_; // actual sequence length for current forward

    Tensor cos_cache;   // [max_seq_len, dim] — precomputed cos(pos * θ_i)
    Tensor sin_cache;   // [max_seq_len, dim] — precomputed sin(pos * θ_i)

    // Gradient accumulation (in-place addition for efficiency)
    Tensor grad_cos_cache;
    Tensor grad_sin_cache;

    // Cached inputs for backward (q and k before rotation)
    Tensor last_q_;
    Tensor last_k_;
    Tensor last_cos_;  // cos values actually used per position
    Tensor last_sin_;  // sin values actually used per position

    RoPE(int dim, int max_seq_len = 2048, float base = 10000.0f);
    virtual ~RoPE() = default;

    // Precompute θ_i = base^(-2i/d) and build cos/sin cache for positions 0..seq_len-1
    void precompute_theta_freqs(int seq_len);

    // Apply RoPE to Q and K tensors. Both expected in [batch, heads, seq, head_dim].
    // Returns {q_rotated, k_rotated} pair with same shape as input.
    // Internal representation: treat as (batch*heads, seq, head_dim) = (B, seq, dim)
    std::pair<Tensor, Tensor> forward(const Tensor& q, const Tensor& k);

    // Layer interface (forward takes single input — compatibility stub)
    // For RoPE, use forward(q, k) directly. This stub returns q rotated alone.
    Tensor forward(const Tensor& input) override {
        return forward(input, input).first;
    }

    Tensor backward(const Tensor& grad_output, double) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return cos_cache; }      // cos_cache as "weights"
    Tensor get_gradients() const override { return grad_cos_cache; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "RoPE"; }

    // Serialization
    void serialize(std::ostream& os) const;
    static RoPE* deserialize(std::istream& is, Layer* base = nullptr);

    // Accessors
    int dim() const { return dim_; }
    int max_seq_len() const { return max_seq_len_; }
    float base() const { return base_; }

private:
    // Apply rotation to a single row (token) at position `pos`
    // Operates on the flat row [0 .. dim_-1], rotating pairs (i, i+dim/2)
    static void rotate_row(double* row, const double* cos_row, const double* sin_row, int dim);

    // Backward helper: compute gradients for cos/sin cache given per-token gradients
    void compute_cache_gradients(int seq_len, int dim_per_head);
};

#endif