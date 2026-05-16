#ifndef COUPLING_LAYER_H
#define COUPLING_LAYER_H

#include "../../core/layer.h"
#include <vector>
#include <random>

// Affine Coupling Layer for RealNVP Normalizing Flow
// Split-based coupling: x = [x1, x2]; y1 = x1; y2 = x2 * exp(s(x1)) + t(x1)
// where s, t are MLPs computing log_scale and translation from x1.
class CouplingLayer : public Layer {
public:
    // input_dim: total number of elements (rows * cols)
    // split_dim: 0 = split rows, 1 = split cols (for row-major, use 1 for col-wise split)
    // hidden_size: hidden dimension for s/t MLPs (auto-computed if 0)
    CouplingLayer(size_t input_dim, size_t split_dim, size_t hidden_size = 0);

    // Forward pass: x -> y
    Tensor forward(const Tensor& input) override;

    // Inverse pass: y -> x (for sampling)
    Tensor inverse(const Tensor& y);

    // Log determinant of Jacobian from forward pass
    double forward_log_det_jacobian() const { return log_det_; }

    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void set_log_det_gradient(double g) { dL_d_log_det_ = g; }

    void update_weights(double learning_rate) override;
    Tensor get_weights() const override { return s_net_.get_weights(); }
    Tensor get_gradients() const override { return s_net_.get_gradients(); }
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    void zero_grad() override;

private:
    size_t input_dim_;
    size_t split_dim_;
    size_t half_dim_;
    size_t hidden_size_;

    // s_net: half_dim -> hidden_size
    // s_scale: hidden_size -> half_dim (projects to log_scale output)
    Dense s_net_;
    Dense s_scale_;
    // t_net: half_dim -> hidden_size
    // t_scale: hidden_size -> half_dim (projects to translation output)
    Dense t_net_;
    Dense t_scale_;

    Tensor x1_;
    Tensor x2_;
    Tensor log_scale_;
    Tensor translation_;
    Tensor last_input_;

    double log_det_;
    double dL_d_log_det_;

    std::mt19937 rng_;

    void compute_st_from_x1(const Tensor& x1);
};

#endif