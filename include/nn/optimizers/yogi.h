#ifndef YOGI_H
#define YOGI_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cmath>
#include <map>
#include <vector>

class Tensor;

// =========================================================================
// Yogi: "Adaptive Methods for Nonconvex Optimization"
// Zaheer, Keenan, et al. (2019), NeurIPS 2019
//
// Yogi is an Adam-family optimizer that self-corrects Adam's v_t update to
// keep the running estimate of the second moment from drifting on
// non-stationary objectives. The paper's Algorithm 2 uses:
//
//   m_t = β1 · m_{t-1} + (1 − β1) · g_t
//   v_t = v_{t-1} − (1 − β2) · sign(v_{t-1} − g_t²) · g_t²     ← Yogi rule
//   m̂_t = m_t / (1 − β1^t)
//   v̂_t = v_t / (1 − β2^t)
//   param_t = param_{t-1} − lr · (m̂_t / (sqrt(|v̂_t|) + ε) + wd · param_{t-1})
//
// Notice how the v_t update is:
//   - pure SUBTRACTION of a sign-controlled (1-β2)·g_t² increment
//   - sign(v_{t-1} − g_t²) is +1 when the running v is "too high" (=> v decreases)
//                              is −1 when the running v is "too low" (=> v increases)
//
// Hence v_t can never decrease below zero in *magnitude* when sign=−1, but v_t
// can transiently go negative when the sign=+1 branch is taken several times in
// a row. To keep the step well-defined we therefore use sqrt(|v̂_t|) in the
// denominator. This matches every Yogi reference implementation (incl. the
// original paper's experimental code).
//
// We implement the AdamW-style decoupled weight decay path (lr·wd·param),
// which is the de-facto convention for modern Adam-family optimizers and is
// what the paper recommends for non-convex deep learning.
//
// Default hyperparameters (paper §5, BERT-large recipe):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-3, wd = 0
//
// Validation:
//   β1 ∈ (0, 1), β2 ∈ (0, 1), ε > 0   (constructor / setter throws otherwise)
//
// State per parameter (lazy-initialized on first step()):
//   m: first-moment EMA, same shape as the parameter
//   v: second-moment estimate (Yogi rule), same shape as the parameter
//
// Reference:
//   Zaheer, Keenan, et al. 2019, "Adaptive Methods for Nonconvex Optimization."
//   NeurIPS 2019. https://papers.nips.cc/paper/2019/hash/adaedb1dfcccb1e1f4ce16e299c2e08c-Abstract.html
// =========================================================================
class Yogi : public Optimizer {
public:
    // --- Public hyperparameters (settable via constructor or mutators) ---
    double lr;            // base learning rate (default 1e-3)
    double beta1;         // first-moment EMA coeff (default 0.9)
    double beta2;         // second-moment (Yogi) EMA coeff (default 0.999)
    double epsilon;       // numerical stabilizer in the sqrt(.) denom (1e-3)
    double weight_decay;  // AdamW-style decoupled weight decay (default 0)

    // Step counter t, starts at 1 (paper convention).
    // The first call to step() uses t=1 (b1_c = 1 - β1, b2_c = 1 - β2) and
    // then increments t.
    int t;

    // --- Constructor with paper defaults ---
    explicit Yogi(double lr          = 1e-3,
                  double beta1_in    = 0.9,
                  double beta2_in    = 0.999,
                  double eps         = 1e-3,
                  double wd          = 0.0);

    // --- Mutators (validate + update + return *this for chaining) ---
    void set_lr(double new_lr);
    void set_beta1(double new_b1);
    void set_beta2(double new_b2);
    void set_epsilon(double new_eps);
    void set_weight_decay(double new_wd);

    // --- Accessors ---
    double get_lr()          const { return lr; }
    double get_beta1()       const { return beta1; }
    double get_beta2()       const { return beta2; }
    double get_epsilon()     const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    int    get_t()           const { return t; }

    // --- Optimizer interface ---
    void step(Model& model) override;

    // Yogi applies weight decay internally via its step() (matches AdamW convention).
    bool handles_weight_decay() const override { return true; }

    // --- State introspection (for tests + debugging) ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    // Retrieve the m or v tensor for (layer, param_idx).
    // Returns an empty (0, 0) Tensor if the layer/param has not been seen yet,
    // so callers can write `y.get_m(&layer, 0)(i, j)` without an out-param.
    Tensor get_m(void* layer_ptr, size_t param_idx) const;
    Tensor get_v(void* layer_ptr, size_t param_idx) const;

private:
    // --- State ---
    struct YogiState {
        Tensor m;  // first moment
        Tensor v;  // second moment (Yogi rule)
    };
    std::map<void*, std::vector<YogiState>> state_;

    // --- Helpers ---
    // Lazy-init state for a layer on first encounter; populates m and v
    // tensors of the same shape as each parameter in `params`.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update. Reads `grad`, updates `param` and the per-entry
    // m, v state. Coordinates (i, j) sweep the entire tensor.
    void update_param(Tensor* param,
                      Tensor* grad,
                      YogiState& st,
                      double lr,
                      double b1,
                      double b2,
                      double eps,
                      double wd,
                      double b1_c,
                      double b2_c) const;

    // Validate hyperparameters; throws std::invalid_argument on violation.
    static void validate(double b1, double b2, double eps);
};

#endif
