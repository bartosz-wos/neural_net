#ifndef STABLEADAMW_H
#define STABLEADAMW_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

class Model;

// ============================================================================
// StableAdamW: "Stable and Low-Precision Training for Large-Scale Vision-
// Language Models"
// Wortsman et al. 2024 (Google DeepMind, https://arxiv.org/abs/2404.00441,
// also documented in https://optimi.benjaminwarner.dev/optimizers/stableadamw/).
//
// Key idea: standard AdamW practice is to clip the *gradient* (or its global
// L2 norm) BEFORE the optimizer step. This is unstable at low precision
// (bf16) because the gradient's norm itself is imprecise — dividing by it
// propagates large error into every parameter's update.
//
// StableAdamW instead clips the *update direction* AFTER computing it. Per
// parameter, per step:
//
//   m_t       = β1 * m_{t-1} + (1-β1) * g_t
//   v_t       = β2 * v_{t-1} + (1-β2) * g_t²
//   m_hat_t   = m_t / (1 - β1^t)                            (bias correction)
//   v_hat_t   = v_t / (1 - β2^t)                            (bias correction)
//   update_t  = m_hat_t / (sqrt(v_hat_t) + ε)               (Adam direction)
//   update_t  = clip(update_t, -1, +1)                      ← KEY: update clipping
//   if (wd > 0):  θ *= (1 - lr * wd)                        (decoupled WD, AdamW-style)
//   θ        -= lr * update_t
//
// The per-coordinate update is bounded by lr * (1 + |wd*θ|), regardless of
// how large the gradient gets. This means:
//   (1) Training is stable even at very large gradients (e.g. exploration
//       phase of RL or post-warmup spikes).
//   (2) Low-precision (bf16) casting of the *update* (not the *gradient*)
//       is well-conditioned — values are in [-1, 1] which bf16 represents
//       without loss.
//   (3) The standard `clip_grad_norm_(g, max_norm)` step is no longer
//       needed — the optimizer is self-stabilizing.
//
// The paper shows that StableAdamW allows dropping gradient clipping
// entirely on a 2B-parameter vision-language model (PaLI-3) without any
// degradation in training stability or final loss.
//
// === Algorithm (per Wortsman 2024 + optimi reference) ===
//
// PER PARAMETER (per step):
//   m ← β1 * m + (1-β1) * g
//   v ← β2 * v + (1-β2) * g²
//   m_hat = m / (1 - β1^t)
//   v_hat = v / (1 - β2^t)
//   update = m_hat / (sqrt(v_hat) + ε)
//   update = clamp(update, -1.0, +1.0)                  ← update clipping
//   if weight_decay > 0:  θ *= (1 - lr * weight_decay)   (decoupled WD)
//   θ -= lr * update
//
// === Defaults (paper / optimi reference) ===
//   lr            = 1e-3          (same as Adam; optimi default)
//   beta1         = 0.9
//   beta2         = 0.999
//   epsilon       = 1e-8
//   weight_decay  = 0.01          (paper: AdamW-equivalent; optimi default)
//
// === State per parameter (lazy on first step) ===
//   m, v — same shape as param
//   t    — global step counter (shared across all params; starts at 1)
//
// === Validation (mirrors optimi reference) ===
//   lr > 0
//   0 ≤ beta1 < 1
//   0 ≤ beta2 < 1
//   epsilon > 0
//   weight_decay ≥ 0
//
// === Public API ===
//   StableAdamW(lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.01)
//   set_lr / set_beta1 / set_beta2 / set_epsilon / set_weight_decay
//   get_lr / get_beta1 / get_beta2 / get_epsilon / get_weight_decay / get_t
//   has_state(layer) / get_m / get_v          — state accessors
//   step(model)
//   handles_weight_decay() → true
//
// === Reference ===
//   Wortsman, T. et al. "Stable and Low-Precision Training for Large-Scale
//   Vision-Language Models." 2024. https://arxiv.org/abs/2404.00441
//   optimi (Benjamin Warner) reference: https://github.com/warner-benjamin/optimi
// ============================================================================
class StableAdamW : public Optimizer {
public:
    // --- Hyperparameters (settable via constructor or mutators) ---
    double lr;            // base learning rate (default 1e-3)
    double beta1;         // first-moment EMA coefficient (default 0.9)
    double beta2;         // second-moment EMA coefficient (default 0.999)
    double epsilon;       // numerical stabilizer in denom (default 1e-8)
    double weight_decay;  // decoupled WD, AdamW-style (default 0.01)

    // --- Step counter (paper convention: starts at 1) ---
    int t;

    // --- Constructor with paper/optimi defaults ---
    explicit StableAdamW(double lr_ = 1e-3,
                         double b1 = 0.9,
                         double b2 = 0.999,
                         double eps = 1e-8,
                         double wd = 0.01);

    // --- Validated setters (throw std::invalid_argument on invalid input) ---
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);

    // --- Accessors ---
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_epsilon() const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t() const { return t; }

    // --- Optimizer interface ---
    void step(Model& model) override;

    // StableAdamW applies decoupled weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // --- State introspection (for tests + debugging) ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    size_t num_params_with_state(void* layer_ptr) const;

    // Retrieve the m or v tensor for (layer, param_idx).
    // Returns an empty (0,0) Tensor if (layer, param_idx) has not been seen.
    const Tensor& get_m(void* layer_ptr, size_t param_idx) const;
    const Tensor& get_v(void* layer_ptr, size_t param_idx) const;

private:
    // --- Per-parameter state ---
    struct ParamState {
        Tensor m;   // first moment, shape (rows, cols)
        Tensor v;   // second moment, shape (rows, cols)
    };
    std::map<void*, std::vector<ParamState>> state_;

    // --- Helpers ---
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
};

#endif
