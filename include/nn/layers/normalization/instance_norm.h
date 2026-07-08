#ifndef INSTANCE_NORM_H
#define INSTANCE_NORM_H

#include "../../core/layer.h"

// ============================================================================
// InstanceNorm1D — Ulyanov et al. 2016 "Instance Normalization: The Missing
// Ingredient for Fast Stylization" (https://arxiv.org/abs/1607.08022).
//
// Per-sample normalization across the feature axis (analogous to LayerNorm but
// without averaging over the batch). For input X ∈ R^{N×C}, computes:
//
//   For each n in [0, N):
//     mu_n   = mean_c x[n, c]
//     var_n  = var_c  x[n, c]
//     y[n,c] = gamma[c] * (x[n,c] - mu_n) / sqrt(var_n + eps) + beta[c]
//
// No running statistics — InstanceNorm is per-sample and does not use a moving
// average (unlike BatchNorm).
// ============================================================================
class InstanceNorm1D : public Layer {
public:
    Tensor gamma;             // (1, C)
    Tensor beta;              // (1, C)
    double eps;

    Tensor last_x;            // (N, C) — cached input
    Tensor last_mean;         // (N, 1) row-vector of per-sample means
    Tensor last_var;          // (N, 1) row-vector of per-sample variances
    Tensor grad_gamma;
    Tensor grad_beta;
    Tensor grad_x;            // (N, C) — gradient flowing back to input

    bool training;            // InstanceNorm has the same behaviour in
                              // train and eval modes, but we honour the
                              // Layer interface by exposing set_training().

    // Constructor takes the channel count (C); input rows (N) are inferred
    // from the first forward call. We allocate gamma/beta/grad_gamma/grad_beta
    // up-front so parameters() returns correctly-shaped tensors.
    explicit InstanceNorm1D(size_t channels, double eps = 1e-5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training = t; }
    std::string name() const override { return "InstanceNorm1D"; }

    // Accessors for tests
    size_t channels() const { return gamma.cols; }
};

// ============================================================================
// InstanceNorm2D — same idea, but for 2D feature maps flattened as
// (N, C*H*W). Per-sample AND per-channel normalization across the H*W slice.
//
//   For each n in [0, N):
//     For each c in [0, C):
//       idx range = [c*S, (c+1)*S)  where S = H*W
//       mu_{n,c}   = mean_s x[n, idx]
//       var_{n,c}  = var_s  x[n, idx]
//       y[n, idx]  = gamma[c] * (x[n, idx] - mu_{n,c}) / sqrt(var_{n,c} + eps)
//                              + beta[c]
//
// Layout assumption: the input tensor (N, C*H*W) stores channel c's slice at
// indices [c*S, c*S+S), matching how the existing GroupNorm layer reads its
// flattened CNN output (see layers/normalization/group_norm.{h,cpp}).
// ============================================================================
class InstanceNorm2D : public Layer {
public:
    Tensor gamma;             // (1, C)
    Tensor beta;              // (1, C)
    double eps;

    int num_channels_;
    int spatial_;             // H * W

    Tensor last_x;            // (N, C*H*W)
    Tensor last_mean;         // (N, C)
    Tensor last_var;          // (N, C)
    Tensor grad_gamma;
    Tensor grad_beta;
    Tensor grad_x;

    bool training;

    InstanceNorm2D(int num_channels, int H, int W, double eps = 1e-5);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output, double /*learning_rate*/) override;
    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return gamma; }
    Tensor get_gradients() const override { return grad_gamma; }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;
    void set_training(bool t) { training = t; }
    std::string name() const override { return "InstanceNorm2D"; }

    int num_channels() const { return num_channels_; }
    int spatial() const { return spatial_; }
};

#endif // INSTANCE_NORM_H
