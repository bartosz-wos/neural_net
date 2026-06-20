#ifndef ROPE_V_H
#define ROPE_V_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// RoPE-on-K-and-V — extension of RoPE that also rotates the V tensor.
//
// Reference: same rotation math as RoPE, applied to a third tensor (V).
//   For pair (i, i+d/2) at position `pos`:
//     V_i'        = V_i        * cos(pos * θ_i) - V_{i+d/2} * sin(pos * θ_i)
//     V_{i+d/2}'  = V_{i+d/2}  * cos(pos * θ_i) + V_i        * sin(pos * θ_i)
//
// The rotation is the *same* per position as Q/K, so the Q·K^T score depends
// only on relative position (the original RoPE property), and V is rotated
// the same way so all three tensors share a positional encoding.
//
// Cache layout: cos_cache and sin_cache are [max_seq_len, dim] row-major,
// shared with the standard RoPE convention.
//
// API mirrors RoPE: Layer interface (forward(input) rotates a single
// tensor — included for completeness) plus a free `forward(q, k, v)`
// returning all three rotated tensors. Backward takes the grad with
// respect to one of the outputs (Q for the Layer interface) and stores
// the dL/dQ, dL/dK, dL/dV gradients separately — call `backward_q()`,
// `backward_k()`, `backward_v()` to fetch each.
class RoPEWithV : public Layer {
public:
    int dim_;           // embedding dimension (must be even)
    int max_seq_len_;   // maximum sequence length
    float base_;        // theta base (default 10000)
    int current_seq_len_; // actual sequence length for current forward

    Tensor cos_cache;   // [max_seq_len, dim]
    Tensor sin_cache;   // [max_seq_len, dim]

    // Gradient accumulators
    Tensor grad_cos_cache;
    Tensor grad_sin_cache;

    // Cached inputs for backward
    Tensor last_q_;
    Tensor last_k_;
    Tensor last_v_;

    // Gradients produced by the last backward
    Tensor last_grad_q_;
    Tensor last_grad_k_;
    Tensor last_grad_v_;

    RoPEWithV(int dim, int max_seq_len = 2048, float base = 10000.0f);
    virtual ~RoPEWithV() = default;

    void precompute_theta_freqs(int seq_len);

    // Apply RoPE to Q, K, and V. All inputs are (batch, seq*dim) tensors
    // with the same layout as RoPE. Returns {q_rot, k_rot, v_rot}.
    std::tuple<Tensor, Tensor, Tensor> forward(const Tensor& q, const Tensor& k, const Tensor& v);

    // Layer interface: rotate a single input (treated as Q).
    Tensor forward(const Tensor& input) override {
        auto out = forward(input, input, input);
        return std::get<0>(out);
    }

    // Layer backward (uses dL/dV to compute dV; ignores Q/K gradient flow here
    // — call forward(q,k,v) then fetch each gradient via backward_q/k/v()).
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return cos_cache; }
    Tensor get_gradients() const override { return grad_cos_cache; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "RoPEWithV"; }

    // Fetch per-tensor gradients produced by the last backward.
    // These are populated by `backward` (which uses the V path of the Layer
    // interface — see implementation note). For full q/k/v gradients, call
    // `backward_qkv(q_grad, k_grad, v_grad)` directly.
    const Tensor& backward_q() const { return last_grad_q_; }
    const Tensor& backward_k() const { return last_grad_k_; }
    const Tensor& backward_v() const { return last_grad_v_; }

    // Full backward: takes per-output gradients and updates internal cache
    // gradients. Returns the gradient with respect to Q (matching the
    // existing RoPE convention of returning one tensor). Callers needing
    // dL/dK and dL/dV should read backward_k() and backward_v() afterwards.
    Tensor backward_qkv(const Tensor& grad_q, const Tensor& grad_k, const Tensor& grad_v);

private:
    // Apply rotation to a single row (token) at position `pos`
    static void rotate_row(double* row, const double* cos_row, const double* sin_row, int dim);
};

#endif