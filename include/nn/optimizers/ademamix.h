#ifndef ADEMAMIX_H
#define ADEMAMIX_H

#include "optimizer.h"
#include <map>
#include <vector>

// AdEMAMix: "AdEMAMix: Theoretical Insights and a New Algorithm for
// Adam-style Optimization"
// Pagliardini, Manunza, Jaggi, Chavdarova (2024, https://arxiv.org/abs/2409.03137)
//
// Key idea: replace Adam's per-parameter second-moment estimator (v_t) with a
// *very slow* exponential moving average of the gradient (m_slow). The slow
// EMA tracks the gradient over a far longer window (β3 ≈ 0.9999) than the
// traditional fast EMA (β1 ≈ 0.9), capturing fine-grained curvature information
// without the variance-noise that v_t brings. The actual update direction is a
// linear mix of the bias-corrected fast EMA and the bias-corrected slow EMA:
//
//   m_fast_t = β1 * m_fast_{t-1} + (1 - β1) * g_t          (fast EMA)
//   m_slow_t = β3 * m_slow_{t-1} + (1 - β3) * g_t          (slow EMA)
//
//   m_fast_hat_t = m_fast_t / (1 - β1^t)
//   m_slow_hat_t = m_slow_t / (1 - β3^t)
//
//   combined_t  = m_fast_hat_t + α * m_slow_hat_t          (linear mix)
//   param_t     -= lr * (combined_t + wd * param_t)        (decoupled weight decay)
//
// Hyperparameters (paper §5 "Implementation details"):
//   - β1 ∈ [0.9, 0.99]   : fast-EMA coefficient (sensitivity to recent grad)
//   - β3 ∈ [0.999, 0.9999]: slow-EMA coefficient (sensitivity to long-term grad)
//   - α  ∈ [2, 5]        : mix coefficient for the slow EMA contribution
//   - lr ∈ [1e-4, 3e-4]  : same order as Adam's lr (the slow EMA amplifies the
//                            effective step size, hence often Adam's lr works)
//   - wd ≥ 0            : decoupled weight decay (AdamW-style). 0 disables.
//
// Default reparam (paper): β1=0.9, β3=0.9999, α=2.0, wd=0.0, lr=1e-4.
//
// State: per-parameter TWO Tensors (m_fast, m_slow) + an int timestep t for
// the bias-correction denominators. No second-moment memory overhead.
//
// Recommended lr from the paper for LLM pre-training: ~5e-5 with α=5; for
// smaller models, lr≈1e-4 with α=2–4 is a good starting point. AdEMAMix has
// been shown to converge in ~3x fewer tokens than AdamW for transformer
// pre-training at scale.
class AdEMAMix : public Optimizer {
public:
    double lr;
    double beta1;          // fast-EMA coefficient
    double beta3;          // slow-EMA coefficient (paper uses 0.999 or 0.9999)
    double alpha;          // mix coefficient on m_slow_hat
    double weight_decay;   // AdamW-style decoupled weight decay (default 0)

    explicit AdEMAMix(double lr = 1e-4,
                      double b1 = 0.9,
                      double b3 = 0.9999,
                      double a = 2.0,
                      double wd = 0.0);

    void step(Model& model) override;

    // AdEMAMix already applies weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // Testing accessors — read out the stored fast/slow EMAs and step counter
    // for invariant checks.
    int timestep() const { return t_; }
    double last_m_fast_value(void* layer_ptr, size_t param_idx, size_t r, size_t c) const;
    double last_m_slow_value(void* layer_ptr, size_t param_idx, size_t r, size_t c) const;
    const std::map<void*, std::vector<Tensor>>& m_fast_state() const { return m_fast_state_; }
    const std::map<void*, std::vector<Tensor>>& m_slow_state() const { return m_slow_state_; }

    // Was the state map for this layer populated yet? (Tests inspect this.)
    bool has_state(void* layer_ptr) const {
        return m_fast_state_.find(layer_ptr) != m_fast_state_.end();
    }

private:
    // Per-layer state: Layer* -> vector of (m_fast or m_slow) tensors, one per parameter.
    std::map<void*, std::vector<Tensor>> m_fast_state_;
    std::map<void*, std::vector<Tensor>> m_slow_state_;

    int t_;  // step counter (1-indexed; t-th update uses (1 - β1^t) bias correction)

    // Initialize state for a layer if not already done.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update (one element at a time).
    void update_param(Tensor* param, Tensor* grad, Tensor& m_fast, Tensor& m_slow,
                      double b1_corr, double b3_corr);
};

#endif
