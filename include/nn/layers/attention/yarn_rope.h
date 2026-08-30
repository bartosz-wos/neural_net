#ifndef YARN_ROPE_H
#define YARN_ROPE_H

#include "../../core/layer.h"
#include <vector>
#include <cmath>
#include <tuple>

// ============================================================================
// YaRN — "YaRN: Yet another RoPE extensioN" (Peng, Quesnelle, Fan, Shippole, 2023,
// https://arxiv.org/abs/2309.00071)
//
// Long-context extension of Rotary Position Embeddings (RoPE) via:
//   1. NTK-aware per-dim frequency scaling: for each rotation pair index i in
//      [0, d/2), the original RoPE frequency theta_i = base^(-2i/d) is divided
//      by a per-dim factor:
//          freq_scale_by_dim[i] = 1 / (alpha · i / ((d/2) - 1) + 1)
//      so high-frequency dims (small i) are barely affected while low-
//      frequency dims (large i) are stretched by a factor up to (alpha+1).
//      This avoids the extrapolation failure mode of vanilla RoPE when the
//      sequence length exceeds the trained max.
//   2. Attention temperature scaling: sqrt(1 / t) where
//          t = 1 - ramp_factor · (1 - current_step / total_steps)
//      At ramp_factor = 0, t = 1 (no temperature restoration). The paper
//      ramps `t` from `1 - ramp` to `1` over `ramp` tokens; in this minimal
//      reference implementation, `ramp_factor` is a fixed scalar between 0
//      and 1 set by the user. `current_step/total_steps` is not tracked
//      here; the caller (training loop) computes and passes the desired
//      `t` directly via the `attention_temperature()` accessor.
//
// Per pair (i, i+d/2) at position `pos`:
//   angle = pos · (theta_i / freq_scale_by_dim[i])
//   r_i   = x_i       · cos(angle) - x_{i+d/2} · sin(angle)
//   r_{i+d/2} = x_{i+d/2} · cos(angle) + x_i · sin(angle)
//
// Properties:
//   * At `scale = 1`, `alpha = 0` (or any scale but alpha = 0 forces
//     freq_scale_by_dim[i] = 1), YaRN degenerates to vanilla RoPE — the
//     rotation is bit-exact.
//   * At `scale > 1` and `alpha = 0.1`, low-frequency dims are scaled down
//     by up to ~10x, which stretches their effective wavelength to fit
//     the longer sequence.
//
// Cache layout: cos_cache and sin_cache are [max_seq_len, dim] row-major,
// where row `pos` contains the YaRN-scaled rotation angles for that position.
// Mirrors `RoPE` / `RoPEWithV` convention: cache[*, i] == cache[*, i+d/2]
// for each i.
//
// API mirrors `RoPEWithV`: forward(q, k, v) returns all three rotated
// tensors; backward_qkv(grad_q, grad_k, grad_v) returns the gradient w.r.t.
// Q and populates backward_k() / backward_v() for the K and V gradients.
// ============================================================================

class YaRNRoPE : public Layer {
public:
    int dim_;           // embedding dimension (must be even)
    int max_seq_len_;   // maximum sequence length
    float base_;        // theta base (default 10000.0)
    float scale_;       // context extension factor (1.0 = vanilla RoPE)
    float alpha_;       // NTK ramp factor (default 0.1, paper §3.2)
    float ramp_factor_; // attention temperature ramp (0..1; 0 = full cold)
    int current_seq_len_;

    Tensor cos_cache;   // [max_seq_len, dim]
    Tensor sin_cache;   // [max_seq_len, dim]

    // Gradient accumulators (cached cos/sin have no learnable parameters;
    // we still expose them for symmetry with the RoPE API).
    Tensor grad_cos_cache;
    Tensor grad_sin_cache;

    // Per-pair-index NTK scale table [dim/2] — precomputed once.
    Tensor freq_scale_by_dim_;

    // Cached inputs for backward
    Tensor last_q_;
    Tensor last_k_;
    Tensor last_v_;

    // Gradients produced by the last backward
    Tensor last_grad_q_;
    Tensor last_grad_k_;
    Tensor last_grad_v_;

    YaRNRoPE(int dim,
             int max_seq_len = 2048,
             float base      = 10000.0f,
             float scale     = 1.0f,
             float alpha     = 0.1f,
             float ramp_factor = 0.0f);
    virtual ~YaRNRoPE() = default;

    // Precompute theta_i = base^(-2i/d), then divide each by
    // freq_scale_by_dim[i] to get YaRN-scaled angles. Builds cos/sin cache.
    void precompute_theta_freqs(int seq_len);

    // Apply YaRN-RoPE to Q, K, V. All inputs are (batch, seq*dim) tensors
    // with the same layout as RoPE. Returns {q_rot, k_rot, v_rot}.
    std::tuple<Tensor, Tensor, Tensor>
    forward(const Tensor& q, const Tensor& k, const Tensor& v);

    // Layer interface: rotate a single input (treated as Q).
    Tensor forward(const Tensor& input) override {
        auto out = forward(input, input, input);
        return std::get<0>(out);
    }

    // Layer backward — single-input path, returns dL/dQ treating input as V.
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return cos_cache; }
    Tensor get_gradients() const override { return grad_cos_cache; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    std::string name() const override { return "YaRNRoPE"; }

    // Fetch per-tensor gradients produced by the last backward_qkv call.
    const Tensor& backward_q() const { return last_grad_q_; }
    const Tensor& backward_k() const { return last_grad_k_; }
    const Tensor& backward_v() const { return last_grad_v_; }

    // Full backward: takes per-output gradients and updates internal cache
    // gradients. Returns the gradient with respect to Q (matching the
    // RoPEWithV convention of returning one tensor). Callers needing
    // dL/dK and dL/dV should read backward_k() and backward_v() afterwards.
    Tensor backward_qkv(const Tensor& grad_q,
                        const Tensor& grad_k,
                        const Tensor& grad_v);

    // Attention temperature multiplier (paper §3.3). With ramp_factor r in
    // [0, 1] and t = 1 - r, the multiplier is sqrt(1/t). Callers should
    // multiply attention scores by this value when computing Q·K^T to
    // apply YaRN's temperature correction.
    //
    // Special cases:
    //   ramp_factor = 0  -> t = 1       -> temperature = 1 (no scaling)
    //   ramp_factor in (0, 1) -> t in (0, 1) -> temperature > 1 (sharper)
    //   ramp_factor >= 1  -> t = 0 (clamped) -> temperature = inf
    //
    // We expose a temperature_for_step(step, total) helper that takes the
    // training progress and returns sqrt(1 / t).
    double attention_temperature() const;
    double temperature_for_step(int step, int total) const;

    // Accessors
    int dim() const { return dim_; }
    int max_seq_len() const { return max_seq_len_; }
    float base() const { return base_; }
    float scale() const { return scale_; }
    float alpha() const { return alpha_; }
    float ramp_factor() const { return ramp_factor_; }
    const Tensor& freq_scale_table() const { return freq_scale_by_dim_; }
};

#endif
