#ifndef TITANS_MAC_H
#define TITANS_MAC_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Titans MAC (Memory as a Context) — Behrouz et al. 2025
//   "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663)
//
// Implements the simplest of the three Titans variants (MAC < MAG < MAL):
// a recurrent layer with a learnable neural long-term memory
// M ∈ R^{d_model × d_model} that is *updated at test time* (i.e. at every
// forward call, not just during training) using a surprise-weighted momentum
// rule.
//
// ----------------------------------------------------------------------------
// Per-token per-segment recurrence:
//   1. q_t, k_t, v_t = W_qkv · x_t   (joint projection; we split into 3 views)
//   2. surprise η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
//        — how poorly the *current* memory predicts x_t via k_t
//   3. gate α_t = sigmoid(W_α · [x_t ; ||v_t||] + b_α) · η_t
//        — per-token momentum coefficient, surprise-weighted so that inputs
//          that surprise memory more aggressively overwrite it
//   4. memory update:
//        M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
//   5. output:
//        y_t = M_t · q_t
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   * x:               (T, d_model)
//   * W_qkv.weights:   (3*d_model, d_model)  — concatenated [W_q; W_k; W_v]
//   * W_qkv.bias:      (1, 3*d_model)
//   * W_alpha.weights: (1, d_model + 1)      — input is concat [x_t, ||v_t||]
//   * W_alpha.bias:    (1, 1)
//   * M (persistent):  (d_model, d_model)    — the test-time memory state
//
// ----------------------------------------------------------------------------
// Caching for backward:
//   For T tokens, we cache:
//     last_x_t   (T, d_model)        — input tokens
//     last_q_t   (T, d_model)        — query projection
//     last_k_t   (T, d_model)        — key projection
//     last_v_t   (T, d_model)        — value projection
//     last_v_norm (T, 1)             — ||v_t||_2 (scalar per token)
//     last_eta   (T, 1)              — surprise at each token
//     last_alpha (T, 1)              — gate (sigmoid(W_α·[x;||v||]) · η)
//     last_M_t   (T+1, d, d)         — memory after each token's update
//                                     — last_M_t[0] = initial M, last_M_t[t] = M_t
//     last_y_t   (T, d_model)        — output (same as the returned Tensor)
//
// Backward chain (per-token, last to first):
//   dL/dy_t → via M_t → q_t         (so dL/dq_t = M_t^T · dL/dy_t)
//            → via M_t → k_t, v_t   (via the v_t⊗k_t term in M_t)
//            → via M_t ← α_t        (gate chain: dL/dα_t carries through (1-α) and the
//                                     update term, including from later tokens' M updates)
//            → via α_t ← η_t        (surprise chain: dL/dη_t via the ·η_t factor in α_t)
//            → via η_t ← M_{t-1}, x_t, k_t  (the ||x_t - M_{t-1}·k_t|| term)
//            → via M_{t-1} carrier  (accumulate dL/dM_{t-1} for the NEXT iteration)
//            → via M_{t-1} → k_t, q_t for the previous token
//            → finally, M_0 gradient → the persistent memory M
// ============================================================================

class TitansMAC : public Layer {
public:
    // d_model: input/output feature dim
    // d_inner: (v1) must equal d_model — the memory is square (d_model × d_model).
    //          The constructor validates this; future versions may add an inner projection.
    // seg_len: segment length (tokens per "test-time chunk" before optional forget reset)
    //          In v1, we process one segment of T tokens per forward call; the segment
    //          length is implicit (= T). seg_len is reserved for future use.
    size_t d_model_;
    size_t d_inner_;
    size_t seg_len_;

    // Learnable parameters (exposed for gradient tests)
    Dense W_qkv_;          // (d_model -> 3*d_model)
    Dense W_alpha_;        // (d_model+1 -> 1)
    Tensor M_;             // (d_model, d_model) — persistent test-time memory

    // Hidden gradient buffers (for parameters that aren't Denses)
    Tensor grad_W_qkv_w_;    // alias for W_qkv_.grad_weights
    Tensor grad_W_qkv_b_;    // alias for W_qkv_.grad_bias
    Tensor grad_W_alpha_w_;  // alias for W_alpha_.grad_weights
    Tensor grad_W_alpha_b_;  // alias for W_alpha_.grad_bias
    Tensor grad_M_;          // (d_model, d_model)

    TitansMAC(size_t d_model, size_t d_inner = 0, size_t seg_len = 0);
    ~TitansMAC() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return M_; }
    Tensor get_gradients() const override { return grad_M_; }
    std::string name() const override { return "TitansMAC"; }

    // Copy all learnable params (W_qkv, W_alpha, M) from another layer.
    void copy_params_from(const TitansMAC& other);

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_inner() const { return d_inner_; }
    size_t seg_len() const { return seg_len_; }

    // Cached forward state (used by backward).
    Tensor last_input_;
    Tensor last_q_t_;
    Tensor last_k_t_;
    Tensor last_v_t_;
    Tensor last_v_norm_;
    Tensor last_eta_;
    Tensor last_alpha_;
    Tensor last_M_t_;        // (T+1)*d_model rows, d_model cols
    Tensor last_y_t_;
    Tensor last_qkv_;
};

#endif // TITANS_MAC_H
