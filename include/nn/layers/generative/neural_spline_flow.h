#ifndef NN_LAYERS_GENERATIVE_NEURAL_SPLINE_FLOW_H
#define NN_LAYERS_GENERATIVE_NEURAL_SPLINE_FLOW_H

#include "../../core/layer.h"

#include <cstddef>
#include <string>
#include <vector>

// Result of a scalar monotone spline transform.
struct SplineTransformResult {
    double value;
    double log_abs_det;
};

// Monotone rational-quadratic spline from Neural Spline Flows
// (Durkan et al., NeurIPS 2019). The transform is learned on
// [-tail_bound, tail_bound] and is the identity outside that interval.
// Boundary derivatives are fixed to one for C1-compatible linear tails.
class RationalQuadraticSpline {
public:
    RationalQuadraticSpline(size_t num_bins = 8,
                            double tail_bound = 3.0,
                            double min_bin_width = 1e-3,
                            double min_bin_height = 1e-3,
                            double min_derivative = 1e-3);

    SplineTransformResult forward(
        double input,
        const std::vector<double>& unnormalized_widths,
        const std::vector<double>& unnormalized_heights,
        const std::vector<double>& unnormalized_internal_derivatives) const;

    SplineTransformResult inverse(
        double input,
        const std::vector<double>& unnormalized_widths,
        const std::vector<double>& unnormalized_heights,
        const std::vector<double>& unnormalized_internal_derivatives) const;

    size_t num_bins() const { return num_bins_; }
    size_t num_derivative_params() const { return num_bins_ - 1; }
    double tail_bound() const { return tail_bound_; }
    double min_bin_width() const { return min_bin_width_; }
    double min_bin_height() const { return min_bin_height_; }
    double min_derivative() const { return min_derivative_; }

private:
    size_t num_bins_;
    double tail_bound_;
    double min_bin_width_;
    double min_bin_height_;
    double min_derivative_;
    double derivative_identity_shift_;
};

// Split coupling layer whose conditioner predicts a rational-quadratic
// spline for every scalar in the transformed half of each batch row.
// Conditioner: Dense(half_dim, hidden_size) -> tanh ->
//              Dense(hidden_size, half_dim * (3*num_bins - 1)).
class NSFCouplingLayer : public Layer {
public:
    NSFCouplingLayer(size_t input_dim,
                     size_t hidden_size,
                     size_t num_bins = 8,
                     double tail_bound = 3.0,
                     bool swap_halves = false,
                     double min_bin_width = 1e-3,
                     double min_bin_height = 1e-3,
                     double min_derivative = 1e-3);

    Tensor forward(const Tensor& input) override;
    Tensor inverse(const Tensor& output);
    Tensor backward(const Tensor& grad_output, double learning_rate) override;

    void set_log_det_gradient(double gradient);
    double log_det_jacobian() const { return log_det_jacobian_; }

    void update_weights(double learning_rate) override;
    void zero_grad() override;
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    Tensor get_weights() const override;
    Tensor get_gradients() const override;
    std::string name() const override { return "NSFCouplingLayer"; }

    size_t input_dim() const { return input_dim_; }
    size_t half_dim() const { return half_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    size_t num_bins() const { return num_bins_; }
    size_t params_per_scalar() const { return params_per_scalar_; }
    bool swap_halves() const { return swap_halves_; }

private:
    size_t input_dim_;
    size_t half_dim_;
    size_t hidden_size_;
    size_t num_bins_;
    size_t params_per_scalar_;
    bool swap_halves_;

    RationalQuadraticSpline spline_;
    Dense conditioner_hidden_;
    Dense conditioner_output_;

    Tensor last_input_;
    Tensor last_conditioning_;
    Tensor last_transformed_;
    Tensor last_hidden_;
    Tensor last_raw_parameters_;

    double log_det_jacobian_;
    double dL_d_log_det_;
    bool forward_cache_valid_;

    size_t conditioning_column(size_t local_column) const;
    size_t transformed_column(size_t local_column) const;
    void split_input(const Tensor& input, Tensor& conditioning,
                     Tensor& transformed) const;
    void validate_input(const Tensor& input, const char* operation) const;
};

#endif  // NN_LAYERS_GENERATIVE_NEURAL_SPLINE_FLOW_H
