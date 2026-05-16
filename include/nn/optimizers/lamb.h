#ifndef LAMB_H
#define LAMB_H

#include "optimizer.h"
#include <map>
#include <vector>

// LAMB: Layer-wise Adaptive Moment estimation
// Large-batch training optimizer for BERT/SOTA NLP
// Paper: "Large Batch Optimization for Deep Learning: BERT Training in 76 Minutes"
// https://arxiv.org/abs/1904.00962
//
// Trust ratio: r = ||w|| / ||g||, clamped to [1/gamma, gamma]
// Update: w' = w - lr * r * (m_hat / (sqrt(v_hat) + eps))
//
// Key differences from Adam:
//   - Layerwise/parameter-wise trust ratio normalization
//   - No bias correction needed for trust ratio (uses raw moments)
//   - Uses max_norm of parameter for norm tracking
class LAMB : public Optimizer {
public:
    double lr;
    double beta1;         // first moment decay
    double beta2;         // second moment decay
    double epsilon;       // numerical stability
    double beta1_corr;    // current bias correction for first moment
    double beta2_corr;    // current bias correction for second moment
    int t;                // timestep (starts at 1)
    double trust_ratio_gamma; // clamp trust ratio to [1/gamma, gamma]

    explicit LAMB(double lr = 0.001,
                 double b1 = 0.9,
                 double b2 = 0.999,
                 double eps = 1e-6,
                 double trust_gamma = 10.0);

    void step(Model& model) override;

private:
    // Per-layer state: maps Layer* -> vector of (m, v) per parameter
    std::map<void*, std::vector<std::pair<Tensor, Tensor>>> state_;

    // Normalize parameter norm: ||w||_2
    double param_norm(const Tensor* w) const;

    // Trust ratio = ||w|| / ||g||, clamped
    double trust_ratio(double w_norm, double g_norm, double gamma) const;

    // Initialize state for a layer if not already done (keyed by void* for portability)
    void ensure_state(void* layer_ptr);

    // Compute per-parameter update with trust ratio
    void update_param(Tensor* param, Tensor* grad,
                      Tensor& m, Tensor& v,
                      double w_norm, double lr,
                      double epsilon, double trust_gamma);
};

#endif