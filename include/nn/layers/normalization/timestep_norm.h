#ifndef TIMESTEP_NORM_H
#define TIMESTEP_NORM_H

#include "../../core/layer.h"

// ============================================================================
// TimestepNorm — MEGALODON (Ma et al. 2024) §3.2 (Eq. 4) + Appendix B.2
//   "MEGALODON: Efficient LLM Pretraining and Inference with Unlimited
//   Context Length", https://arxiv.org/abs/2404.08801
//
// Group Normalization (Wu & He 2018) transposed to normalize along the
// *timestep* axis instead of the spatial axis. For an input sequence
// X ∈ R^{n×d} split into `num_groups` groups of d_g = d/num_groups features:
//
//   mu_{t,g}  = 1/(t*d_g) * sum_{i=1..t} sum_{j in g} x_{i,j}
//   var_{t,g} = 1/(t*d_g) * sum_{i=1..t} sum_{j in g} (x_{i,j} - mu_{t,g})^2
//   y_{t,j}   = (gamma_j + 1) * (x_{t,j} - mu_{t,g(j)}) / sqrt(var_{t,g(j)} + eps)
//               + beta_j
//
// The statistics are CUMULATIVE over t, so only x_1..x_t influence y_t. That
// makes the layer causal — the paper's stated motivation: plain GroupNorm
// leaks future information through its timestep-axis mean/variance and
// therefore cannot be used in autoregressive sequence modeling, whereas the
// cumulative form can.
//
// The "+1" on the scale parameter is the paper's plus-1 reparameterization
// (Appendix B.2): gamma is initialized to 0 so that the effective scale
// (gamma + 1) starts at 1 and weight decay keeps gamma centered at zero
// rather than pulling it away from its initialization.
//
// num_groups = 0 (the default) means a single group spanning all d features.
// An explicit k in [1, d] is honoured as-is; k == d gives per-feature groups
// (group size 1) and k == 1 gives one group over everything. k > d or a k
// that does not divide d throws.
// ============================================================================

class TimestepNorm : public Layer {
public:
    size_t features_;
    size_t num_groups_;  // 0 means "single group over all features"
    double eps_;
    size_t group_size_;  // d_g

    Tensor gamma;  // (1, features) — initialized to 0 (plus-1 reparam)
    Tensor beta;   // (1, features) — initialized to 0
    Tensor grad_gamma_;
    Tensor grad_beta_;

    // Forward cache
    Tensor last_x;        // (n, d)   input
    Tensor last_mu;       // (n, G)   cumulative mean per group per timestep
    Tensor last_var;      // (n, G)   cumulative variance per group per timestep
    Tensor last_inv_std;  // (n, G)   1/sqrt(var + eps), cached to avoid a divide
    Tensor grad_x;        // (n, d)

    TimestepNorm(size_t features, size_t num_groups = 0, double eps = 1e-5);
    ~TimestepNorm() override = default;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double learning_rate) override;
    void update_weights(double learning_rate) override;
    void zero_grad() override;

    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma_; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    std::string name() const override { return "TimestepNorm"; }

    size_t features() const { return features_; }
    size_t num_groups() const { return num_groups_; }
    double eps() const { return eps_; }
    size_t group_size() const { return group_size_; }

    // Cumulative per-timestep statistics, exposed for tests.
    const Tensor& last_mu_tensor() const { return last_mu; }
    const Tensor& last_var_tensor() const { return last_var; }
};

#endif  // TIMESTEP_NORM_H
