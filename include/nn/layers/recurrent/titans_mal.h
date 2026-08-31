#ifndef TITANS_MAL_H
#define TITANS_MAL_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>

// ============================================================================
// Titans MAL (Memory as a Layer) — Behrouz et al. 2025 §4.3
//   "Titans: Learning to Memorize at Test Time" (https://arxiv.org/abs/2501.00663)
//
// Third of the three Titans variants (MAC < MAG < MAL): a recurrent layer with
// a learnable test-time neural long-term memory M ∈ R^{d_model × d_model} that
// is *updated at test time* via a surprise-weighted momentum rule, used as a
// *layer* that processes the input after a per-token-learned gating:
//
//   p_t  = sigmoid(W_p · x_t + b_p)             — input gate (per channel)
//   x̃_t  = p_t ⊙ x_t                           — gated input
//   y_t  = M_t · x̃_t                            — clean output (no ⊙)
//
// M-update (shared with MAC/MAG, §3.2 Eqs. 9-10):
//   q_t, k_t, v_t = W_qkv · x_t    (k,v used; q-slice is unused in MAL output)
//   η_t = ||x_t - M_{t-1} · k_t||_2 / (||x_t||_2 + eps)
//   α_t = sigmoid(W_α · [x_t ; ||v_t||]) · η_t
//   M_t = (1 - α_t) · M_{t-1} + α_t · outer(v_t, k_t)
//
// The MAL-specific contribution over MAC is the W_p input-gate projection
// (one Dense + bias) and the fact that the per-token "query" to M is x̃_t
// rather than q_t or x_t. The paper's downstream sliding-window attention
// (Eq. 31) composes cleanly with the existing SlidingWindowAttention layer
// and is intentionally NOT bundled here — this file ships the MAL memory cell.
//
// ----------------------------------------------------------------------------
// Shape conventions:
//   * x:               (T, d_model)
//   * W_qkv.weights:   (3*d_model, d_model)
//   * W_qkv.bias:      (1, 3*d_model)
//   * W_alpha.weights: (1, d_model + 1)
//   * W_alpha.bias:    (1, 1)
//   * W_p.weights:     (d_model, d_model)        — MAL-specific input gate
//   * W_p.bias:        (1, d_model)
//   * M (persistent):  (d_model, d_model)
//
// Caching for backward (T tokens):
//   last_input_       (T, d_model)
//   last_k_t_         (T, d_model)
//   last_v_t_         (T, d_model)
//   last_v_norm_      (T, 1)
//   last_eta_         (T, 1)
//   last_alpha_       (T, 1)
//   last_M_t_         (T+1)*d_model rows × d_model cols
//   last_p_t_         (T, d_model)               — input gate (post-sigmoid)
//   last_x_tilde_     (T, d_model)               — p_t ⊙ x_t (the q-equivalent)
//   last_y_t_         (T, d_model)
//
// Backward chain (per-token, last to first):
//   dL/dy_t → split:
//     (a) M-chain: dx_tilde_t[i] += dL/dy_t[j] · M_t[j,i] (T×d²),
//                  dM_post[j,i] += dL/dy_t[j] · x̃_t[i] (then propagates through
//                  the M-update recurrence identical to MAC).
//     (b) Gate chain: dx_t[k] += dx_tilde_t[k] · p_t[k]   (the ⊙ path)
//                     dp_t[k]  = dx_tilde_t[k] · x_t[k]
//                     dW_p.weights[k,m] += dp_t[k] · p_t(1-p_t) · x_t[m]
//                     dW_p.bias[k]      += dp_t[k] · p_t(1-p_t)
//                     dx_t[m]            += dp_t[k] · p_t(1-p_t) · W_p.weights[k,m]
//   The M-chain then continues:
//     dα ← dM_post·(v⊗k - M_prev); dα → dz → W_alpha + W_qkv(k,v slices)
//     + the surprise chain dα → dx, dk, dM_carrier (identical to MAC)
//   Finally: dM_carrier (after the per-token loop) = dL/dM_ (persistent).
//
// Note on conventions: the q-slice of W_qkv is unused in MAL's output
// (q_t doesn't enter y_t), so its parameter gradients are zero — same as MAG.
// ============================================================================

class TitansMAL : public Layer {
public:
    // d_model: input/output feature dim
    // d_inner: must equal d_model in v1 (memory is square d_model × d_model).
    // seg_len: reserved for future segment-forget implementation.
    size_t d_model_;
    size_t d_inner_;
    size_t seg_len_;

    // Learnable parameters
    Dense W_qkv_;            // (d_model -> 3*d_model)
    Dense W_alpha_;          // (d_model+1 -> 1)
    Dense W_p_;              // (d_model -> d_model)   — MAL-specific input gate
    Tensor M_;               // (d_model, d_model)     — persistent memory

    // Hidden gradient buffers
    Tensor grad_W_qkv_w_;
    Tensor grad_W_qkv_b_;
    Tensor grad_W_alpha_w_;
    Tensor grad_W_alpha_b_;
    Tensor grad_W_p_w_;
    Tensor grad_W_p_b_;
    Tensor grad_M_;

    TitansMAL(size_t d_model, size_t d_inner = 0, size_t seg_len = 0);
    ~TitansMAL() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override { return M_; }
    Tensor get_gradients() const override { return grad_M_; }
    std::string name() const override { return "TitansMAL"; }

    // Copy all learnable params (W_qkv, W_alpha, W_p, M) from another layer.
    void copy_params_from(const TitansMAL& other);

    // Accessors
    size_t d_model() const { return d_model_; }
    size_t d_inner() const { return d_inner_; }
    size_t seg_len() const { return seg_len_; }

    // Cached forward state (used by backward).
    Tensor last_input_;
    Tensor last_k_t_;
    Tensor last_v_t_;
    Tensor last_v_norm_;
    Tensor last_eta_;
    Tensor last_alpha_;
    Tensor last_M_t_;        // (T+1)*d_model rows, d_model cols
    Tensor last_p_t_;
    Tensor last_x_tilde_;
    Tensor last_y_t_;
    Tensor last_qkv_;
};

#endif // TITANS_MAL_H
