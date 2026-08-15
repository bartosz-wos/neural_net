#ifndef APOLLO_H
#define APOLLO_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

class Model;

// ============================================================================
// APOLLO: Approximated Gradient Scaling for Memory-Efficient LLM Optimization
// Zhu et al. 2024, "APOLLO: SGD-like Memory, AdamW-level Performance"
// https://arxiv.org/abs/2412.05270   (MLSys 2025, Outstanding Paper Honorable Mention)
//
// ----------------------------------------------------------------------------
// Core insight
//
// Adam's per-element learning rate is redundant for LLM training. APOLLO
// coarsens it to a STRUCTURED (per-channel or tensor-wide) scaling factor
// computed from a low-rank Adam update, then applies the scaling factor to
// the raw gradient. The result is an optimizer with SGD-like memory that
// still matches AdamW-level convergence on LLM training.
//
// ----------------------------------------------------------------------------
// Algorithm (per parameter W of shape (m, n), rank r ≥ 1)
//
//   norm_dim = (m < n) ? 0 : 1     // dim with the FEWER elements = "low-rank"
//   min_dim  = std::min(m, n)
//
//   Step 1. Refresh projection R ∈ R^{max_dim × r} every `update_proj_gap`
//           steps. R is filled with randn(rng) / sqrt(r) so that E[R R^T] = I.
//
//   Step 2. Project the gradient to low rank:
//             g_low = (m < n) ? G @ R : R^T @ G          ∈ R^{min_dim × r}
//
//   Step 3. Standard Adam EMA on the low-rank gradient:
//             m_t  = β1 · m_{t-1} + (1 − β1) · g_low
//             v_t  = β2 · v_{t-1} + (1 − β2) · g_low²
//
//   Step 4. Bias correction (optional):
//             m̂ = m_t / (1 − β1^t)
//             v̂ = v_t / (1 − β2^t)
//
//   Step 5. Low-rank "Adam update":
//             u_low = m̂ / (sqrt(v̂) + ε)                 ∈ R^{min_dim × r}
//
//   Step 6. Compute scaling factor s ∈ R^r:
//             if CHANNEL:
//                 s[k] = ||u_low[:, k]||_2 / (||g_low[:, k]||_2 + 1e-8)
//             else (TENSOR):
//                 s[0] = ||u_low||_F / (||g_low||_F + 1e-8)  (scalar)
//
//   Step 7. Scale the low-rank update:
//             u_low_scaled = u_low ⊙ s_broadcast         ∈ R^{min_dim × r}
//
//   Step 8. Project back to FULL space:
//             u_full = (m < n) ? u_low_scaled @ R^T
//                               : R @ u_low_scaled       ∈ R^{m × n}
//
//   Step 9. (Optional) Norm-Growth Limiter (Fira, arXiv:2410.01623):
//             limiter = max(||u_full||_F / (cached_norm + 1e-8), 1.01) / 1.01
//             u_full /= limiter
//
//  Step 10. Apply the (scalar) `scale` factor (default 128.0):
//             u_full *= sqrt(scale)
//
//  Step 11. Decoupled weight decay (if weight_decay > 0):
//             W *= (1 − lr · weight_decay)
//
//  Step 12. Update parameters:
//             W -= lr · u_full
//
// ----------------------------------------------------------------------------
// State per parameter (lazy on first step)
//
//   exp_avg        ∈ R^{min_dim × r}  — first  moment in low-rank space
//   exp_avg_sq     ∈ R^{min_dim × r}  — second moment in low-rank space
//   R              ∈ R^{max_dim × r}  — current random projection
//   projection_initialized          — whether R has been seeded
//   cached_scaled_grad_norm          — for the Norm-Growth Limiter
//   step_pt                          — last step when projection was refreshed
//
// For 1-D parameters (shape (1, C)) we treat them as a degenerate 2-D case:
// "min_dim = 1, max_dim = C, g_low shape (1, r)". The Tensor API in this repo
// stores biases as (1, C), so this naturally covers biases/norms without a
// special path.
//
// ----------------------------------------------------------------------------
// Defaults (Zhu et al. 2024 §5.1; APOLLO-Mini paper-recommended for LLM
// pre-training at scales 60M-7B)
//
//   lr              = 1e-3
//   beta1           = 0.9
//   beta2           = 0.999
//   epsilon         = 1e-6
//   weight_decay    = 0.0
//   rank            = 1              (APOLLO-Mini; APOLLO uses 256 for 7B)
//   scale_type      = TENSOR
//   scale           = 128.0          (paper-recommended for rank=1)
//   update_proj_gap = 200
//   scale_front     = false          (apply scale AFTER NL)
//   use_nl          = true
//   bias_correction = true
//
// ----------------------------------------------------------------------------
// Memory cost
//
// For a (m, n) weight with rank r:
//   Adam:    2·m·n floats
//   APOLLO:  (min(m,n)·r + min(m,n)·r + max(m,n)·r) ≈ 3·min(m,n)·r
// When r = 1, APOLLO memory ≈ 3·min(m,n) ≈ SGD memory (one momentum, one
// variance, one projection).
//
// ----------------------------------------------------------------------------
// Reference
//
//   Paper:  https://arxiv.org/abs/2412.05270
//   Code:   https://github.com/zhuhanqing/APOLLO
// ============================================================================

class APOLLO : public Optimizer {
public:
    enum class ScaleType { CHANNEL, TENSOR };

    APOLLO(double lr = 1e-3,
           double beta1 = 0.9,
           double beta2 = 0.999,
           double epsilon = 1e-6,
           double weight_decay = 0.0,
           size_t rank = 1,
           ScaleType scale_type = ScaleType::TENSOR,
           double scale = 128.0,
           size_t update_proj_gap = 200,
           bool scale_front = false,
           bool use_nl = true);

    void step(Model& model) override;
    bool handles_weight_decay() const override { return true; }

    // Validated setters
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_weight_decay(double v);
    void set_rank(size_t v);
    void set_scale(double v);
    void set_update_proj_gap(size_t v);
    void set_scale_type(ScaleType v);
    void set_scale_front(bool v);
    void set_use_nl(bool v);

    // Hyperparameter accessors
    double get_lr() const { return Optimizer::lr; }
    double get_beta1() const { return beta1_; }
    double get_beta2() const { return beta2_; }
    double get_epsilon() const { return epsilon_; }
    double get_weight_decay() const { return weight_decay_; }
    size_t get_rank() const { return rank_; }
    ScaleType get_scale_type() const { return scale_type_; }
    double get_scale() const { return scale_; }
    size_t get_update_proj_gap() const { return update_proj_gap_; }
    bool get_scale_front() const { return scale_front_; }
    bool get_use_nl() const { return use_nl_; }
    size_t num_steps() const { return num_steps_; }

    // State introspection
    bool has_state(void* layer_ptr) const;
    size_t num_params_with_state(void* layer_ptr) const;
    bool get_exp_avg(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_sq(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_R(void* layer_ptr, size_t param_idx, Tensor& out) const;
    size_t get_step_pt(void* layer_ptr, size_t param_idx) const;

private:
    struct ParameterState {
        Tensor exp_avg;
        Tensor exp_avg_sq;
        Tensor R;
        bool projection_initialized = false;
        double cached_scaled_grad_norm = 0.0;
        size_t step_pt = 0;
    };

    double beta1_;
    double beta2_;
    double epsilon_;
    double weight_decay_;
    size_t rank_;
    ScaleType scale_type_;
    double scale_;
    size_t update_proj_gap_;
    bool scale_front_;
    bool use_nl_;
    size_t num_steps_;

    std::map<void*, std::vector<ParameterState>> state_;

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, ParameterState& state);
    void refresh_projection(ParameterState& state, size_t m, size_t n);

    static void validate(double lr,
                         double beta1,
                         double beta2,
                         double epsilon,
                         double weight_decay,
                         size_t rank,
                         double scale,
                         size_t update_proj_gap);
};

#endif
