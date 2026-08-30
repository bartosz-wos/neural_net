#ifndef TITANS_MAG_H
#define TITANS_MAG_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Titans MAG (Memory as a Gate) — Behrouz et al. 2025
//   "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663)
//
// Implements the middle of the three Titans variants (MAC < MAG < MAL): a
// recurrent layer with a learnable neural long-term memory M ∈ R^{d_model × d_model}
// that is *updated at test time* via a surprise-weighted momentum rule.
//
// The MAG variant uses M as a *gate* for the input:
//   y_t = (M_{t-1} · x_t) ⊙ x_t            (element-wise product)
//
// where M is updated by the same surprise-weighted momentum rule as MAC.
//
// ----------------------------------------------------------------------------
// Per-token per-segment recurrence:
//   1. q_t, k_t, v_t = W_qkv · x_t   (joint projection; we split into 3 views)
//   2. surprise η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
//        — how poorly the *current* memory predicts x_t via k_t
//   3. gate α_t = sigmoid(W_α · [x_t ; ||v_t||]) · η_t
//        — per-token momentum coefficient, surprise-weighted
//   4. memory update:
//        M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
//   5. output (MAG):
//        y_t = (M_t · x_t) ⊙ x_t
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
//     last_input_   (T, d_model)        — input tokens
//     last_q_t_     (T, d_model)        — query projection (used for q-side chain)
//     last_k_t_     (T, d_model)        — key projection
//     last_v_t_     (T, d_model)        — value projection
//     last_v_norm_  (T, 1)             — ||v_t||_2 (scalar per token)
//     last_eta_     (T, 1)              — surprise at each token
//     last_alpha_   (T, 1)              — gate (sigmoid(W_α·[x;||v||]) · η)
//     last_M_t_     (T+1, d, d)         — memory after each token's update
//                                         — last_M_t[0] = initial M
//     last_mx_      (T, d_model)        — (M_t · x_t) gate values BEFORE ⊙ x_t
//     last_y_t_     (T, d_model)        — output (same as returned Tensor)
//
// Backward chain (per-token, last to first):
//   dL/dy_t → split into two paths due to y_t = (M·x) ⊙ x:
//              (a) direct: y[j] = mx[j] · x[j] → dL/dmx[j] = dy[j] · x[j]
//                                                        dL/dx[j] += dy[j] · mx[j]
//              (b) indirect via mx: mx = M·x → dL/dM = (dL/dmx) · x^T
//                                              dL/dx += M^T · dL/dmx
//            Both paths feed back into M_t (path b) and x_t (both paths).
//   The chain continues through the M-update recurrence → α_t → η_t → M_{t-1}
//   and the projection W_qkv for q/k/v — exactly as in MAC, plus the MAG-
//   specific direct path (a) into dL/dx_t.
// ============================================================================

class TitansMAG : public Layer {
public:
    // d_model: input/output feature dim
    // d_inner: must equal d_model in v1 (memory is square d_model × d_model).
    // seg_len: segment length; reserved for future segment-forget.
    size_t d_model_;
    size_t d_inner_;
    size_t seg_len_;

    // Learnable parameters
    Dense W_qkv_;          // (d_model -> 3*d_model)
    Dense W_alpha_;        // (d_model+1 -> 1)
    Tensor M_;             // (d_model, d_model) — persistent test-time memory

    // Hidden gradient buffers
    Tensor grad_W_qkv_w_;
    Tensor grad_W_qkv_b_;
    Tensor grad_W_alpha_w_;
    Tensor grad_W_alpha_b_;
    Tensor grad_M_;

    TitansMAG(size_t d_model, size_t d_inner = 0, size_t seg_len = 0);
    ~TitansMAG() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return M_; }
    Tensor get_gradients() const override { return grad_M_; }
    std::string name() const override { return "TitansMAG"; }

    // Copy all learnable params (W_qkv, W_alpha, M) from another layer.
    void copy_params_from(const TitansMAG& other);

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
    Tensor last_mx_;         // (T, d_model) — (M_t · x_t) gate values before ⊙ x_t
    Tensor last_y_t_;
    Tensor last_qkv_;
};

#endif // TITANS_MAG_H
