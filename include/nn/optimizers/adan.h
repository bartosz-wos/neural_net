#ifndef ADAN_H
#define ADAN_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <cmath>
#include <map>
#include <vector>

class Tensor;

// =========================================================================
// Adan: "Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models"
// Xie, Chen, Li, Chen, Chen, Wang, Tang (2022), NeurIPS 2022.
//
// Adan introduces a third EMA on the *gradient difference* (g_k - g_{k-1})
// in addition to the standard first-moment and second-moment EMAs. The
// combination of:
//   m_k = β1 * m_{k-1} + (1 - β1) * g_k                                          (first moment, EMA)
//   v_k = β2 * v_{k-1} + (1 - β2) * (g_k - g_{k-1})                              (gradient-difference EMA)
//   n_k = β3 * n_{k-1} + (1 - β3) * (g_k + β2 * (g_k - g_{k-1}))²                (third moment, squared "Nesterov-lookahead")
//   update: param = (param - lr/bc1 * m/denom - lr*β2/bc2 * v/denom) / (1 + wd*lr)
// where bc_i = 1 - β_i^t is the standard bias correction, denom = sqrt(n/bc3) + ε,
// gives an optimizer that empirically converges 2× faster than AdamW on
// ViT/CNN/Transformer training. It is the default optimizer for NVIDIA NeMo,
// timm, InternLM, DeepSeek's Stable Diffusion 3 pipeline, and Masked Diffusion
// Transformer V2.
//
// Algorithm reference: official sail-sg/Adan implementation
//   https://github.com/sail-sg/Adan/blob/main/adan.py
// (the canonical reference used by NVIDIA, Huggingface timm, MMClassification,
// InternLM, and PaddlePaddle). We follow the official implementation's
// β3-placement convention: `n = β3*n + (1-β3) * (g + β2*(g-prev_g))²`.
// This differs from the paper's Algorithm 1 pseudocode where `β3` and
// `1-β3` are swapped. The two forms differ by a constant factor absorbed
// into the bias correction and yield equivalent optimization behavior;
// we follow the official code because it is what the production frameworks
// ship and what users would expect for reproduction.
//
// We use the integrated "prox" weight-decay form (paper §3.2 default,
// no_prox=false in sail-sg):
//   param = (param - lr/bc1 * m/denom - lr*β2/bc2 * v/denom) / (1 + wd*lr)
// rather than the AdamW-style decoupled path. This is the default in the
// official sail-sg implementation; it keeps `handles_weight_decay()=true`
// so the project's WeightDecay wrapper correctly skips re-application.
//
// Default hyperparameters (paper §5.1, sail-sg/Adan defaults):
//   lr = 1e-3, β1 = 0.98, β2 = 0.92, β3 = 0.99, ε = 1e-8, wd = 0
//
// Validation:
//   β1, β2, β3 ∈ [0, 1), ε > 0
//
// State per parameter (lazy-initialized on first step()):
//   m              : first-moment EMA,   same shape as param
//   v              : gradient-difference EMA, same shape as param
//   n              : third-moment EMA,   same shape as param
//   neg_prev_grad  : -(g_{k-1}), stored negated for memory efficiency (matches sail-sg trick)
//
// Reference:
//   Xie, X., Chen, P., Li, J., Chen, Y., Chen, R., Wang, S., Tang, F.
//   "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models."
//   NeurIPS 2022. https://arxiv.org/abs/2208.06677
// =========================================================================
class Adan : public Optimizer {
public:
    // --- Public hyperparameters (settable via constructor or mutators) ---
    double lr;            // base learning rate (default 1e-3)
    double beta1;         // first-moment EMA coeff (default 0.98)
    double beta2;         // gradient-difference EMA coeff (default 0.92)
    double beta3;         // third-moment EMA coeff (default 0.99)
    double epsilon;       // numerical stabilizer in the sqrt(.) denom (1e-8)
    double weight_decay;  // integrated prox weight decay (default 0)
    bool   no_prox;       // when true, use AdamW-style decoupled wd; when false, use prox form

    // Step counter t, starts at 1 (paper convention).
    int t;

    // --- Constructor with paper defaults ---
    explicit Adan(double lr_         = 1e-3,
                  double beta1_      = 0.98,
                  double beta2_      = 0.92,
                  double beta3_      = 0.99,
                  double eps         = 1e-8,
                  double wd          = 0.0,
                  bool   no_prox_    = false);

    // --- Mutators (validate + update) ---
    void set_lr(double new_lr);
    void set_beta1(double new_b1);
    void set_beta2(double new_b2);
    void set_beta3(double new_b3);
    void set_epsilon(double new_eps);
    void set_weight_decay(double new_wd);
    void set_no_prox(bool new_no_prox);

    // --- Accessors ---
    double get_lr()           const { return lr; }
    double get_beta1()        const { return beta1; }
    double get_beta2()        const { return beta2; }
    double get_beta3()        const { return beta3; }
    double get_epsilon()      const { return epsilon; }
    double get_weight_decay() const { return weight_decay; }
    bool   get_no_prox()      const { return no_prox; }
    int    get_t()            const { return t; }

    // --- Optimizer interface ---
    void step(Model& model) override;

    // Adan applies weight decay internally (prox or decoupled form).
    bool handles_weight_decay() const override { return true; }

    // --- State introspection (for tests + debugging) ---
    bool has_state(void* layer_ptr) const {
        return state_.find(layer_ptr) != state_.end();
    }
    // Retrieve the m, v, or n tensor for (layer, param_idx).
    // Returns an empty (0, 0) Tensor if the layer/param has not been seen yet,
    // so callers can write `a.get_m(&layer, 0)(i, j)` without an out-param.
    Tensor get_m(void* layer_ptr, size_t param_idx) const;
    Tensor get_v(void* layer_ptr, size_t param_idx) const;
    Tensor get_n(void* layer_ptr, size_t param_idx) const;

private:
    // --- State ---
    struct AdanState {
        Tensor m;             // first moment
        Tensor v;             // gradient-difference EMA
        Tensor n;             // third moment
        Tensor neg_prev_grad; // -(g_{k-1}), stored negated for memory efficiency
    };
    std::map<void*, std::vector<AdanState>> state_;

    // --- Helpers ---
    // Lazy-init state for a layer on first encounter; populates m, v, n,
    // neg_prev_grad tensors of the same shape as each parameter in `params`.
    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);

    // Per-parameter update. Reads `grad`, updates `param` and the per-entry
    // m, v, n, neg_prev_grad state. Coordinates (i, j) sweep the entire tensor.
    void update_param(Tensor* param,
                      Tensor* grad,
                      AdanState& st,
                      double lr_,
                      double b1,
                      double b2,
                      double b3,
                      double eps,
                      double wd,
                      bool   no_prox_,
                      double bc1,
                      double bc2,
                      double bc3_sqrt) const;

    // Validate hyperparameters; throws std::invalid_argument on violation.
    static void validate(double b1, double b2, double b3, double eps);
};

#endif