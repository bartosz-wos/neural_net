#ifndef ADAFACTOR_H
#define ADAFACTOR_H

#include "optimizer.h"
#include <cmath>
#include <map>
#include <vector>

// =========================================================================
// Adafactor: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
// Shazeer & Stern, 2018 (https://arxiv.org/abs/1804.04235)
//
// Key idea: Adam stores a full second-moment matrix v_t ∈ R^{d1×d2} for
// every parameter, costing O(d1·d2) memory per parameter. For very large
// matrices (transformer hidden weights) this dominates optimizer state
// memory. Adafactor replaces v_t with a factored approximation:
//
//   v_t ≈ R_t · C_t / s_t          (outer product of row/column vectors)
//
// where:
//   R_t ∈ R^{d1×1} : per-row EMA of mean-squared-gradient
//   C_t ∈ R^{1×d2} : per-col EMA of mean-squared-gradient
//   s_t = mean(R_t) (a scalar)    : so v_t has the right per-row means
//
// Reconstruction v_ij ≈ R_i · C_j / s_0  (s_0 = mean of R at step 0, the
// paper's "outer product" version) or v_ij ≈ R_i · C_j / mean(R_t) (the
// "averaged" version). We use the paper's "averaged" version, which is
// slightly more stable.
//
// For 1-D parameters (biases, norms) the row/column factorization is
// undefined; we fall back to a per-element EMA (Adam-style) for 1-D
// parameters only.
//
// Update rule per 2-D parameter (W ∈ R^{d1×d2}):
//   g_t       = grad_t                                          (current gradient)
//   R_t       = β2_t · R_{t-1} + (1 - β2_t) · row_mean(g_t ⊙ g_t)  ∈ R^{d1×1}
//   C_t       = β2_t · C_{t-1} + (1 - β2_t) · col_mean(g_t ⊙ g_t)  ∈ R^{1×d2}
//   v_t       = R_t · C_t / max(ε2, mean(R_t))                 ∈ R^{d1×d2}  (reconstructed)
//   v̂_t      = v_t / (1 - ∏_{i=1..t} β2_i)                      (bias correction)
//
//   u_t       = g_t / sqrt(v̂_t + ε1)                           (per-coord update direction)
//
//   if (relative_step):
//       lr_t = max(ε2, RMS(W_{t-1})) / RMS(u_t)                  (paper's option 1)
//   else:
//       lr_t = lr                                               (constant)
//
//   W_t       = W_{t-1} − lr_t · (u_t + wd · W_{t-1})           (decoupled weight decay)
//
// The β2 schedule is the paper's canonical choice: β2_t = 1 − t^(−0.8).
// We allow a fixed-β2 mode too (for cases where the user has external
// hyperparameter tuning), but the schedule is the recommended default.
//
// Hyperparameters (paper §5.4 / T5 / PaLM defaults):
//   - lr: base learning rate. Paper recommends `relative_step=true` (no lr).
//   - beta2: fixed β2 if `use_beta2_schedule=false`. Else ignored.
//   - epsilon1: numerical stabilizer in the sqrt(.). Default 1e-30 (paper).
//   - epsilon2: numerical stabilizer in mean(R_t) and RMS comparisons. Default 1e-3.
//   - weight_decay: decoupled, AdamW-style. Default 0.
//   - relative_step: paper's option 1, lr_t = max(eps2, RMS(W))/RMS(u).
//                    Default true (paper recommendation).
//   - use_beta2_schedule: schedule β2_t = 1 - t^(-0.8). Default true.
//   - dmax: max(ε2, d1) clamp for the relative-step scale. Default 1.0 (paper).
//           When d1 is large (e.g. vocab-sized projection), the paper scales
//           the effective LR by 1/dmax to dampen the per-tensor effect.
//   - 1-D handling: per-element EMA (Adam-style), β1=0.9 (only used to pick
//     the update direction) and v_t is the per-element EMA with the same
//     schedule. This keeps 1-D parameter behavior sane when most of the
//     model is 2-D.
//
// State per 2-D parameter:
//   R ∈ R^{d1×1} : row accumulator
//   C ∈ R^{1×d2} : column accumulator
// State per 1-D parameter (rare):
//   v ∈ R^{d1×d2} : per-element EMA (one full tensor)
//
// Step counter t starts at 1; bias correction is `1 − ∏ β2_i` (or
// `(1 − β2^t)` for the fixed-β2 path). t increments after each `step()`.
//
// Reference: Shazeer & Stern 2018, "Adafactor: Adaptive Learning Rates
// with Sublinear Memory Cost" (https://arxiv.org/abs/1804.04235).
// Used in T5, PaLM, and many other LLM training recipes.
class Adafactor : public Optimizer {
public:
    // --- Hyperparameters (public for inspection / test access) ---
    double lr;                    // base learning rate (used when relative_step=false)
    double beta2;                 // fixed second-moment EMA coeff (when !use_beta2_schedule)
    double epsilon1;              // numerical stabilizer in sqrt(.) (default 1e-30)
    double epsilon2;              // numerical stabilizer in mean(R), RMS comparisons (1e-3)
    double weight_decay;          // AdamW-style decoupled weight decay (default 0)
    bool   relative_step;         // paper option 1: lr_t = max(eps2, RMS(W))/RMS(u) (default true)
    bool   use_beta2_schedule;    // β2_t = 1 - t^(-0.8); else constant β2 (default true)
    double dmax;                  // paper's per-tensor LR scale (default 1.0)
    int    t;                     // step counter, starts at 1

    // Running bias-correction factor B_t = 1 - ∏_{i=1..t} β2_i, used by the
    // scheduled-β2 path. Tracked incrementally because the schedule
    // β2_t = 1 - t^(-0.8) has no clean closed form. Initialized to 0 in the
    // constructor; the first call to step() uses β2_1 to set B_1.
    double B_prev_;

    // Constructor. Defaults match the paper's recommendation for LLM training
    // (relative_step=true, scheduled β2). Pass `relative_step=false` and a
    // `lr` to use the constant-LR mode (e.g. for fine-tuning or sanity checks).
    explicit Adafactor(double lr            = 1e-3,
                       double beta2_fixed   = 0.999,
                       double eps1          = 1e-30,
                       double eps2          = 1e-3,
                       double wd            = 0.0,
                       bool   rel_step      = true,
                       bool   beta2_sched   = true,
                       double dmax_         = 1.0);

    void step(Model& model) override;

    // Adafactor applies weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // --- Test / introspection accessors ---
    // Per-layer state maps. 2-D parameters live in (R_state_, C_state_);
    // 1-D parameters live in v1d_state_ with the same keyed layout.
    const std::map<void*, std::vector<Tensor>>& R_state() const { return R_state_; }
    const std::map<void*, std::vector<Tensor>>& C_state() const { return C_state_; }
    const std::map<void*, std::vector<Tensor>>& v1d_state() const { return v1d_state_; }
    // True if `layer_ptr` has had its state initialized yet.
    bool has_state(void* layer_ptr) const {
        return R_state_.find(layer_ptr) != R_state_.end()
            || v1d_state_.find(layer_ptr) != v1d_state_.end();
    }
    // Read the stored R or C tensor for a (layer, param_index) pair.
    // Returns true if the entry exists and `out` was filled; false otherwise.
    bool get_R(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_C(void* layer_ptr, size_t param_idx, Tensor& out) const;
    // Read the stored 1-D-EMA tensor for a 1-D parameter.
    bool get_v1d(void* layer_ptr, size_t param_idx, Tensor& out) const;

    // Current β2_t (per the schedule, if enabled). Useful for tests that
    // want to inspect the β2_t at the current step.
    double current_beta2() const {
        return use_beta2_schedule ? std::max(0.0, 1.0 - std::pow((double)t, -0.8)) : beta2;
    }

private:
    // --- State ---
    // 2-D parameters: row/column accumulators (R and C have shape d1×1 and 1×d2)
    std::map<void*, std::vector<Tensor>> R_state_;
    std::map<void*, std::vector<Tensor>> C_state_;
    // 1-D parameters: per-element EMA (same shape as the parameter)
    std::map<void*, std::vector<Tensor>> v1d_state_;

    // --- Helpers ---
    // Lazy-init state for a layer's parameters; populates R/C (2-D) or v1d (1-D)
    // entries as appropriate based on each parameter's row/column shape.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update. `is_1d` selects the 1-D code path (per-element
    // EMA, Adam-style update); the 2-D path uses row/col factorization.
    void update_param_2d(Tensor* param, Tensor* grad,
                         Tensor& R, Tensor& C,
                         double b2, double bias_corr);
    void update_param_1d(Tensor* param, Tensor* grad,
                         Tensor& v,
                         double b2, double bias_corr);

    // Compute the per-parameter RMS (root mean square) of a tensor.
    static double rms(const Tensor& t);
};

#endif
