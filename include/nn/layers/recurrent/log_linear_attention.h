#ifndef LOG_LINEAR_ATTENTION_H
#define LOG_LINEAR_ATTENTION_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Log-Linear Attention — Guo, Yang, Goel, Xing, Dao, Kim, ICLR 2026
//   "Log-Linear Attention"
//   https://arxiv.org/abs/2506.04761
//
// Log-linear attention generalizes linear attention / state-space models to
// use a logarithmically-growing set of Fenwick-tree partitioned hidden states.
// The masking matrix M_H is a hierarchical (H) matrix with L = ⌈log₂ T⌉ + 1
// levels; memory at decoding is O(log T) instead of O(1) (linear attention) or
// O(T) (softmax attention). Compute is O(T log T) instead of O(T) or O(T²).
//
// This file implements the **Log-Linear Mamba-2** variant — the simpler of
// the two case studies in the paper, §3.1 (recurrent form, Eq. 3) + §3.4
// (composing with the Mamba-2 A transition / scalar per-head decay). The
// recurrence per token (t) per head (h) is:
//
//   Let lssb(t+1) = index of least significant set bit of (t+1), 0-indexed.
//   (e.g. lssb(1)=0, lssb(2)=1, lssb(3)=0, lssb(4)=2, lssb(5)=0, lssb(6)=1,
//         lssb(7)=0, lssb(8)=3, ...).
//
//   For each level ℓ in 0..L-1:
//     S^(ℓ)_t[h] = {
//       b_t[h] ⊗ k_t[h]                            if ℓ == 0
//       0                                          if 0 < ℓ ≤ lssb(t+1)
//       Σ_{ℓ'=0}^{ℓ-1} S^(ℓ')_{t-1}[h] · a_t[h]   if ℓ == lssb(t+1) + 1 and ℓ < L
//       a_t[h] · S^(ℓ)_{t-1}[h]                    if ℓ > lssb(t+1) + 1
//     }
//
//   Output:
//     o_t[h, dh] = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · <q_t[h, dh, :], S^(ℓ)_t[h, dh, :]>
//                = Σ_{ℓ=0}^{L-1} λ^(ℓ)_t[h] · Σ_{ds} q_t[h, dh, ds] · S^(ℓ)_t[h, dh, ds]
//
//   where λ^(ℓ)_t[h] is a per-token per-head per-level learnable coefficient.
//
// Then the standard Mamba-2 tail:
//   y_t = o_t + D_skip ⊙ x_ssm_t
//   gated_t = silu(g_t) ⊙ y_t
//   out_t = out_proj(gated_t)
//
// ----------------------------------------------------------------------------
// Shapes:
//   x:               (T, d_model)
//   in_proj:         (T, 2*d_inner)         weights (2*d_inner, d_model)
//   x_ssm_t:         (T, d_inner)           first half of in_proj output
//   gate_t:          (T, d_inner)           second half
//   a_proj:          (T, n_heads)           pre-sigmoid log-decay
//   a_t:             (T, n_heads)           per-head scalar decay ∈ (0,1)
//   b_proj:          (T, d_inner)           flattened per-(head) value matrix
//   k_proj:          (T, d_inner)           flattened per-(head) key matrix
//   q_proj:          (T, d_inner)           flattened per-(head) query matrix
//   λ_proj:          (T, n_heads * L)       per-(token, head, level) mix coef
//   o_t:             (T, d_inner)           SSM output per token
//   D_skip:          (1, d_inner)           per-channel skip connection
//   dt_bias:         (1, n_heads)           bias added before sigmoid(a_proj)
//   out_proj:        (d_model, d_inner)
//
//   For head h, the slice [h*head_dim, (h+1)*head_dim) of b_proj, k_proj,
//   q_proj output is a (head_dim, d_state) matrix, but flattened to d_inner
//   since head_dim × d_state = d_inner for our layout (head_dim = d_state
//   by convention here so that S^(ℓ)_t[h] is head_dim × d_state).
//
//   Wait — for Mamba-2 compatibility we use head_dim = d_inner / n_heads.
//   The state shape for head h is (head_dim, d_state). b_proj output for
//   head h is a (head_dim, d_state) matrix flattened to head_dim * d_state,
//   so the per-head slice is d_inner (since d_inner = n_heads * head_dim,
//   and we set head_dim = d_state so head_dim * d_state = head_dim^2 ==
//   d_inner only if d_state = head_dim). To match Mamba-2 exactly, we use
//   d_inner = n_heads * head_dim, and the per-head state is (head_dim, d_state)
//   flattened. b_proj, k_proj, q_proj all output (T, d_inner) where for head h
//   the slice [h*head_dim, (h+1)*head_dim) is the head_dim key-vector of
//   length head_dim — that's only the ROW of the per-head state. The state
//   is (head_dim, d_state), but with d_state = head_dim, that becomes
//   (head_dim, head_dim), flattened to head_dim * head_dim = d_inner. So
//   for head h: rows are dh in 0..head_dim, cols are ds in 0..d_state.
//   Flattened as d_inner tensor: index [h*head_dim + dh*d_state + ds] for
//   element (h, dh, ds) of the per-head state.
// ============================================================================

class LogLinearAttention : public Layer {
public:
    // d_model:    input/output feature dim
    // n_heads:    number of heads (each maintains its own L-level Fenwick tree)
    // d_state:    per-head state width (d_state = head_dim, so the per-head
    //              state is head_dim × head_dim)
    // d_inner:    inner feature dim (default = 2 * d_model, matching Mamba-1/2)
    // max_levels: L (number of Fenwick-tree levels). Default = 0 means use
    //              ceil(log2(T_max)) + 1 where T_max is the sequence length
    //              at first forward pass.
    LogLinearAttention(size_t d_model, size_t n_heads, size_t d_state,
                      size_t d_inner = 0, size_t max_levels = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return in_proj.weights; }
    Tensor get_gradients() const override { return in_proj.grad_weights; }
    std::string name() const override { return "LogLinearAttention"; }

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t n_heads() const { return n_heads_; }
    size_t d_state() const { return d_state_; }
    size_t d_inner() const { return d_inner_; }
    size_t head_dim() const { return head_dim_; }
    size_t L() const { return L_; }
    size_t T() const { return last_T_; }

    // Read-only access to last lambda tensor (T, n_heads * L) for tests
    const Tensor& last_lambda() const { return last_lambda_; }

    // Parameter tensors (public for gradient tests)
    Dense in_proj;          // (d_model -> 2*d_inner)
    Dense out_proj;         // (d_inner -> d_model)
    Dense a_proj;           // (d_model -> n_heads)        — pre-sigmoid log-decay
    Dense b_proj;           // (d_model -> d_inner)         — value (flattened across heads)
    Dense k_proj;           // (d_model -> d_inner)         — key (flattened across heads)
    Dense q_proj;           // (d_model -> d_inner)         — query (flattened across heads)
    Dense lambda_proj;      // (d_model -> n_heads * L)     — per-(token, head, level) mix coef
    Tensor D_skip;          // (1, d_inner)                 — learnable skip connection
    Tensor dt_bias;         // (1, n_heads)                 — bias added before sigmoid(a_proj)

    // Hidden gradient buffers
    Tensor grad_D_skip_;
    Tensor grad_dt_bias_;

private:
    size_t d_model_;
    size_t n_heads_;
    size_t d_state_;
    size_t d_inner_;
    size_t head_dim_;       // = d_inner / n_heads (must be == d_state at runtime)
    size_t L_;              // number of Fenwick-tree levels
    size_t last_T_ = 0;     // sequence length at last forward

    // Caches for forward (filled in forward, used in backward)
        // All initialized with non-empty (1, 1) shape so .operator() never reads past end.
        Tensor last_input_;         // (T, d_model) — sized in forward
        Tensor last_p_;             // (T, 2*d_inner)
        Tensor last_x_ssm_;         // (T, d_inner)
        Tensor last_gate_;          // (T, d_inner)
        Tensor last_a_pre_;         // (T, n_heads)
        Tensor last_a_;             // (T, n_heads)
        Tensor last_b_;             // (T, d_inner)
        Tensor last_k_;             // (T, d_inner)
        Tensor last_q_;             // (T, d_inner)
        Tensor last_lambda_;        // (T, n_heads * L)

        // State cache: S at every (t, ℓ) per head.
        Tensor last_S_;
        Tensor last_o_;
        Tensor last_gated_;

    // Case cache: which case was active at each (t, ℓ). 0=immediate, 1=cleared,
    // 2=merge-promoted, 3=carry-forward. Stored as a uint8_t matrix of shape (T, L).
    std::vector<uint8_t> last_case_;

    // Helpers
    static double sigmoid(double x) {
        if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
        double ez = std::exp(x);
        return ez / (1.0 + ez);
    }
    static double silu(double x) {
        return x * sigmoid(x);
    }
};

// ============================================================================
// LogLinearAttentionModel — stack of LogLinearAttention + input proj + classifier
// ============================================================================
class LogLinearAttentionModel : public Layer {
public:
    size_t input_dim_;
    size_t d_model_;
    size_t output_dim_;
    size_t num_layers_;
    size_t n_heads_;
    size_t d_state_;
    size_t d_inner_;
    size_t L_;

    Dense input_proj_;
    std::vector<std::unique_ptr<LogLinearAttention>> blocks_;
    Dense classifier_;
    // (initialized in constructor initializer list to handle Dense's lack of default ctor)

    Tensor last_input_;
    std::vector<Tensor> block_outputs_;  // (num_layers + 2) cached tensors

    LogLinearAttentionModel(size_t input_dim, size_t d_model, size_t output_dim,
                            size_t num_layers, size_t n_heads, size_t d_state,
                            size_t d_inner = 0, size_t max_levels = 0);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return classifier_.get_weights(); }
    Tensor get_gradients() const override { return classifier_.get_gradients(); }
    std::string name() const override { return "LogLinearAttentionModel"; }
};

#endif // LOG_LINEAR_ATTENTION_H