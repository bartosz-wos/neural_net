#ifndef CAME_H
#define CAME_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cstddef>
#include <map>
#include <vector>

// =========================================================================
// CAME: Confidence-guided Adaptive Memory Efficient Optimization
//
// Luo, Ren, Zheng, Jiang, Jiang, You 2023, "CAME: Confidence-guided Adaptive
// Memory Efficient Optimization", ACL 2023 Long Papers, pp. 4442-4453
// (https://arxiv.org/abs/2307.02047)
//
// Reference implementation: https://github.com/yangluo7/CAME/blob/master/
//   came_pytorch/CAME.py
//
// CAME achieves Adam-like convergence with Adafactor-like memory footprint
// by combining (a) a factored row/column EMA of g² to approximate the Adam
// adaptive denominator, and (b) a confidence-guided residual correction that
// scales the momentum direction by the per-parameter "instability" estimate.
//
// For 2-D parameters (factored path):
//   raw            = g² + eps1
//   exp_avg_sq_row = beta2 * exp_avg_sq_row + (1-beta2) * mean_cols(raw)   # R^{d1×1}
//   exp_avg_sq_col = beta2 * exp_avg_sq_col + (1-beta2) * mean_rows(raw)   # R^{1×d2}
//   update         = g / sqrt(exp_avg_sq_row * exp_avg_sq_col / mean(exp_avg_sq_row))
//   update        /= max(1, RMS(update) / clip_threshold)
//   exp_avg        = beta1 * exp_avg + (1-beta1) * update
//   res            = (update - exp_avg)² + eps2
//   exp_avg_res_row = beta3 * exp_avg_res_row + (1-beta3) * mean_cols(res) # R^{d1×1}
//   exp_avg_res_col = beta3 * exp_avg_res_col + (1-beta3) * mean_rows(res) # R^{1×d2}
//   update         = exp_avg * sqrt(exp_avg_res_row * exp_avg_res_col / mean(exp_avg_res_row))
//
// For 1-D parameters (bias, norm) the residual correction is identity
// (the canonical reference implementation falls back to a per-element EMA):
//   raw        = g² + eps1
//   exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * raw
//   update     = g / sqrt(exp_avg_sq)
//   update    /= max(1, RMS(update) / clip_threshold)
//   exp_avg    = beta1 * exp_avg + (1-beta1) * update
//   update     = exp_avg                              # residual is identity
//
// Decoupled weight decay (AdamW form): param *= (1 − lr * weight_decay)
//
// State per 2-D parameter:
//   exp_avg          : (rows, cols) — Adam-style first moment
//   exp_avg_sq_row   : (rows, 1)    — row EMA of g²
//   exp_avg_sq_col   : (1, cols)    — column EMA of g²
//   exp_avg_res_row  : (rows, 1)    — row EMA of residual instability
//   exp_avg_res_col  : (1, cols)    — column EMA of residual instability
//   rms              : scalar       — last-update RMS (cached for inspection)
//
// State per 1-D parameter:
//   exp_avg          : (rows, cols) — same shape as parameter
//   exp_avg_sq       : (rows, cols) — per-element EMA of g²
//   rms              : scalar
//
// Defaults match the official came_pytorch reference implementation:
//   lr = 2e-3, betas = (0.9, 0.999, 0.9999),
//   eps = (1e-30, 1e-16), clip_threshold = 1.0, weight_decay = 0.0
//
// CAME deliberately uses its own `lr` field (not inherited Optimizer::lr)
// so the API is symmetric with the paper's exposed lr parameter.
// =========================================================================
class CAME : public Optimizer {
public:
    // --- Hyperparameters (public for inspection / test access) ---
    double lr;
    double beta1;
    double beta2;
    double beta3;
    double eps1;
    double eps2;
    double clip_threshold;
    double weight_decay;
    int    t;

    explicit CAME(double lr = 2e-3,
                  double beta1 = 0.9,
                  double beta2 = 0.999,
                  double beta3 = 0.9999,
                  double eps1 = 1e-30,
                  double eps2 = 1e-16,
                  double clip_threshold = 1.0,
                  double weight_decay = 0.0);

    void step(Model& model) override;

    // CAME applies weight decay internally (decoupled, AdamW form).
    bool handles_weight_decay() const override { return true; }

    // --- Validated mutators ---
    void set_lr(double new_lr);
    void set_beta1(double new_beta1);
    void set_beta2(double new_beta2);
    void set_beta3(double new_beta3);
    void set_eps1(double new_eps1);
    void set_eps2(double new_eps2);
    void set_clip_threshold(double new_clip);
    void set_weight_decay(double new_wd);

    // --- Accessors ---
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_beta3() const { return beta3; }
    double get_eps1() const { return eps1; }
    double get_eps2() const { return eps2; }
    double get_clip_threshold() const { return clip_threshold; }
    double get_weight_decay() const { return weight_decay; }
    int get_t() const { return t; }

    // --- State introspection ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    Tensor get_exp_avg(void* layer_ptr, size_t param_idx) const;
    bool get_exp_avg_sq_row(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_sq_col(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_res_row(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_res_col(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_exp_avg_sq(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_rms(void* layer_ptr, size_t param_idx, double& out) const;
    bool is_1d(void* layer_ptr, size_t param_idx) const;

private:
    struct ParameterState {
        Tensor exp_avg;             // (rows, cols)
        Tensor exp_avg_sq_row;      // (rows, 1) — 2-D only
        Tensor exp_avg_sq_col;      // (1, cols) — 2-D only
        Tensor exp_avg_res_row;     // (rows, 1) — 2-D only
        Tensor exp_avg_res_col;     // (1, cols) — 2-D only
        Tensor exp_avg_sq;          // (rows, cols) — 1-D only
        double rms = 0.0;
        bool is_1d = false;
    };

    std::map<void*, std::vector<ParameterState>> state_;

    static void validate(double lr,
                         double beta1, double beta2, double beta3,
                         double eps1, double eps2,
                         double clip_threshold, double weight_decay);

    // Compute root-mean-square of a tensor (Frobenius norm / sqrt(numel)).
    static double rms(const Tensor& t);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    void update_param_2d(Tensor* param, Tensor* grad, ParameterState& state);
    void update_param_1d(Tensor* param, Tensor* grad, ParameterState& state);
};

#endif
