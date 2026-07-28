#ifndef ADAMP_H
#define ADAMP_H

#include "optimizer.h"
#include "../core/tensor.h"
#include <map>
#include <vector>

// =========================================================================
// AdamP: "AdamP: Slowing Down the Slowdown for Momentum Optimizers on
//        Scale-invariant Weights"
// Heo, Kim, Yun, Han 2021 (https://arxiv.org/abs/2106.04522)
//
// AdamP adds a "projection" step on top of Adam: before applying the
// bias-corrected first-moment step, if the gradient direction is "too
// aligned" with the weight vector (cosine similarity > delta), remove the
// component along the weight direction. This specifically targets the
// scale-invariant weights (Dense layers, Conv layers) where the loss is
// invariant to w → c·w, so any step along w is wasted + destabilizing.
//
// Algorithm (per parameter):
//   m_t   = β1 · m_{t-1} + (1 − β1) · g                                  # first moment
//   v_t   = β2 · v_{t-1} + (1 − β2) · g²                                 # second moment
//   m̂_t   = m_t / (1 − β1^t)                                             # bias-correction
//   v̂_t   = v_t / (1 − β2^t)
//   cos_sim = (w · m̂_t) / (||w|| · ||m̂_t|| + ε)                        # cosine similarity
//   if cos_sim > delta:                                                   # projection gate
//     proj = (w · m̂_t) / (||w||² + ε) · w                                # gram-projection
//     m̂_t = m̂_t − proj
//   step = m̂_t / (√v̂_t + ε)
//   param -= lr · step
//   (decoupled weight decay: param *= (1 − lr · wd))
//
// Default hyperparameters (paper §4, also PyTorch reference):
//   lr = 1e-3, β1 = 0.9, β2 = 0.999, ε = 1e-8, delta = 0.1, wd = 0
//
// `handles_weight_decay()` returns true (decoupled WD).
// =========================================================================
class AdamP : public Optimizer {
public:
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    double delta;       // cosine-similarity threshold for projection
    double weight_decay;
    int    t;           // step counter (starts at 1)

    explicit AdamP(double lr = 1e-3,
                   double b1 = 0.9,
                   double b2 = 0.999,
                   double eps = 1e-8,
                   double delta = 0.1,
                   double wd = 0.0);

    void step(Model& model) override;

    // AdamP applies decoupled weight decay internally.
    bool handles_weight_decay() const override { return true; }

    // ---- Validated setters ----
    void set_lr(double v);
    void set_beta1(double v);
    void set_beta2(double v);
    void set_epsilon(double v);
    void set_delta(double v);
    void set_weight_decay(double v);

    // ---- Accessors ----
    double get_lr() const { return lr; }
    double get_beta1() const { return beta1; }
    double get_beta2() const { return beta2; }
    double get_epsilon() const { return epsilon; }
    double get_delta() const { return delta; }
    double get_weight_decay() const { return weight_decay; }
    int    get_step() const { return t; }

    // ---- State introspection ----
    bool has_state(void* layer_ptr, size_t param_idx) const;
    bool get_m(void* layer_ptr, size_t param_idx, Tensor& out) const;
    bool get_v(void* layer_ptr, size_t param_idx, Tensor& out) const;

private:
    struct State {
        Tensor m;  // first moment
        Tensor v;  // second moment
    };

    std::map<void*, std::vector<State>> state_;

    static void validate(double lr, double b1, double b2, double eps,
                         double delta, double wd);

    void ensure_state(void* layer_ptr, const std::vector<Tensor*>& params);
    void update_param(Tensor* param, Tensor* grad, State& st,
                      double lr, double b1, double b2,
                      double eps, double delta, double wd,
                      double b1_c, double b2_c);
};

#endif
