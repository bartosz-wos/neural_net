#ifndef MARS_H
#define MARS_H

#include "optimizer.h"
#include <map>
#include <vector>

// =========================================================================
// MARS: "MARS: Unleashing the Power of Variance Reduction for Training Large
//        Language Models"
// Yuan, Zhang, Chen, Yu, Liu, Sun, Wang, 2024
// (https://arxiv.org/abs/2401.11615)
//
// MARS is a variance-reduced Adam variant. The key insight: in Adam the
// first moment m_t is an EMA of raw gradients. MARS instead applies a
// "shift correction" g̃_t = g_t + γ · (g_t − g_{t−1}) BEFORE the m/v EMA —
// the (g_t − g_{t−1}) term is approximately the gradient noise direction,
// and the small `γ` weight (paper default 0.025) subtracts some of that
// noise from g_t before the EMA, reducing variance.
//
// Full MARS / MARSM update per parameter `θ`:
//   1. g̃_t = g_t + γ · (g_t − g_{t−1})              (MARSM shift correction)
//   2. m_t = β1 · m_{t−1} + (1 − β1) · g̃_t           (first moment on corrected grad)
//   3. v_t = β2 · v_{t−1} + (1 − β2) · g̃_t²          (second moment on corrected grad)
//   4. m̂_t = m_t / (1 − β1^t),  v̂_t = v_t / (1 − β2^t)   (bias correction)
//   5. θ_{t+1} = θ_t − lr · m̂_t / (√v̂_t + ε) + lr · wd · θ_t   (AdamW-style decoupled wd)
//
// MARSE (epsilon-clip) variant: when clip=true, g̃ is clipped per element
// before the m/v update:
//   g̃_clip = sign(g̃) · clamp(|g̃| / max(|g_prev|, eps_grad), eps_max, 1.0)
// This keeps the shift ratio bounded — useful when g_prev can be very small.
//
// Hyperparameters (paper defaults):
//   lr          = 1e-3   (paper used 3e-4 to 5e-4 for LLM training)
//   beta1       = 0.9
//   beta2       = 0.999
//   gamma       = 0.025  (MARS-specific; γ ∈ [0, 1]). γ=0 -> reduces to Adam.
//   epsilon     = 1e-8
//   weight_decay= 0      (AdamW-style decoupled)
//   clip        = false  (MARSE mode)
//   eps_max     = 1.0    (only used when clip=true; clamps shift ratio)
//   eps_grad    = 1e-8   (only used when clip=true; min |g_prev| for ratio)
//
// State per parameter: (m, v, g_prev) — three tensors, same shape as θ.
// One more than Adam's two because we need g_prev for the shift.
//
// Reference: https://arxiv.org/abs/2401.11615
class MARS : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double gamma;       // MARSM shift coefficient (γ ∈ [0, 1])
    double epsilon;
    double weight_decay;
    bool   clip;        // MARSE variant (epsilon-clip the shift ratio)
    double eps_max;     // clipping floor for |g̃|/|g_prev|, only used if clip=true
    double eps_grad;    // min |g_prev| for ratio (avoid div-by-zero)
    int    t;           // step counter, starts at 1

    explicit MARS(double lr           = 1e-3,
                  double b1           = 0.9,
                  double b2           = 0.999,
                  double gamma_       = 0.025,
                  double eps          = 1e-8,
                  double wd           = 0.0,
                  bool   clip_        = false,
                  double eps_max_     = 1.0,
                  double eps_grad_    = 1e-8);

    void step(Model& model) override;

    // MARS applies weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // --- Test / introspection accessors ---
    // Returns true if state has been initialized for `layer_ptr`.
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    // Read the stored (m, v, g_prev) tensor for a (layer, param_index) pair.
    bool get_m(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_v(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_g_prev(void* layer_ptr, size_t param_idx, Tensor& out) const;

private:
    // Per-layer state: maps Layer* -> vector of (m, v, g_prev) triplets
    // indexed as [3*i], [3*i+1], [3*i+2] for the i-th parameter of the layer.
    std::map<void*, std::vector<Tensor>> state_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, size_t base_idx,
                      std::vector<Tensor>& st);
};

#endif
