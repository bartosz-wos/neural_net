#include "nn/layers/generative/neural_spline_flow.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kRootRoundoffTolerance = 1e-10;

bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

double stable_softplus(double value) {
    if (value > 20.0) return value;
    if (value < -20.0) return std::exp(value);
    return std::log1p(std::exp(value));
}

double inverse_softplus(double value) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument("inverse_softplus requires a positive finite value");
    }
    if (value > 20.0) return value;
    return std::log(std::expm1(value));
}

void validate_raw_parameters(size_t num_bins,
                             const std::vector<double>& raw_widths,
                             const std::vector<double>& raw_heights,
                             const std::vector<double>& raw_derivatives) {
    if (raw_widths.size() != num_bins) {
        throw std::invalid_argument("RationalQuadraticSpline expected K width parameters");
    }
    if (raw_heights.size() != num_bins) {
        throw std::invalid_argument("RationalQuadraticSpline expected K height parameters");
    }
    if (raw_derivatives.size() != num_bins - 1) {
        throw std::invalid_argument(
            "RationalQuadraticSpline expected K-1 internal derivative parameters");
    }
    for (double value : raw_widths) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("RationalQuadraticSpline width parameter is not finite");
        }
    }
    for (double value : raw_heights) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("RationalQuadraticSpline height parameter is not finite");
        }
    }
    for (double value : raw_derivatives) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(
                "RationalQuadraticSpline derivative parameter is not finite");
        }
    }
}

std::vector<double> stable_softmax(const std::vector<double>& raw) {
    const double maximum = *std::max_element(raw.begin(), raw.end());
    std::vector<double> probabilities(raw.size());
    double total = 0.0;
    for (size_t i = 0; i < raw.size(); ++i) {
        probabilities[i] = std::exp(raw[i] - maximum);
        total += probabilities[i];
    }
    for (double& probability : probabilities) probability /= total;
    return probabilities;
}

struct SplineParameters {
    std::vector<double> widths;
    std::vector<double> heights;
    std::vector<double> derivatives;
    std::vector<double> cumulative_widths;
    std::vector<double> cumulative_heights;
};

SplineParameters constrain_parameters(
    size_t num_bins,
    double tail_bound,
    double min_bin_width,
    double min_bin_height,
    double min_derivative,
    double derivative_identity_shift,
    const std::vector<double>& raw_widths,
    const std::vector<double>& raw_heights,
    const std::vector<double>& raw_derivatives) {
    validate_raw_parameters(num_bins, raw_widths, raw_heights, raw_derivatives);

    const std::vector<double> soft_widths = stable_softmax(raw_widths);
    const std::vector<double> soft_heights = stable_softmax(raw_heights);
    const double interval = 2.0 * tail_bound;
    const double width_scale = 1.0 - min_bin_width * static_cast<double>(num_bins);
    const double height_scale = 1.0 - min_bin_height * static_cast<double>(num_bins);

    SplineParameters parameters;
    parameters.widths.resize(num_bins);
    parameters.heights.resize(num_bins);
    parameters.derivatives.resize(num_bins + 1, 1.0);
    parameters.cumulative_widths.resize(num_bins + 1);
    parameters.cumulative_heights.resize(num_bins + 1);
    parameters.cumulative_widths[0] = -tail_bound;
    parameters.cumulative_heights[0] = -tail_bound;

    for (size_t bin = 0; bin < num_bins; ++bin) {
        parameters.widths[bin] = interval *
            (min_bin_width + width_scale * soft_widths[bin]);
        parameters.heights[bin] = interval *
            (min_bin_height + height_scale * soft_heights[bin]);
        parameters.cumulative_widths[bin + 1] =
            parameters.cumulative_widths[bin] + parameters.widths[bin];
        parameters.cumulative_heights[bin + 1] =
            parameters.cumulative_heights[bin] + parameters.heights[bin];
    }

    // Pin exact boundaries rather than carrying a softmax summation roundoff.
    parameters.cumulative_widths.back() = tail_bound;
    parameters.cumulative_heights.back() = tail_bound;
    for (size_t knot = 1; knot < num_bins; ++knot) {
        parameters.derivatives[knot] = min_derivative + stable_softplus(
            raw_derivatives[knot - 1] + derivative_identity_shift);
    }
    return parameters;
}

size_t find_bin(const std::vector<double>& cumulative, double value) {
    const size_t num_bins = cumulative.size() - 1;
    if (value >= cumulative.back()) return num_bins - 1;
    const auto upper = std::upper_bound(cumulative.begin(), cumulative.end(), value);
    if (upper == cumulative.begin()) return 0;
    const size_t index = static_cast<size_t>(upper - cumulative.begin() - 1);
    return std::min(index, num_bins - 1);
}

SplineTransformResult evaluate_forward_in_bin(
    double input, size_t bin, const SplineParameters& parameters) {
    const double x_start = parameters.cumulative_widths[bin];
    const double y_start = parameters.cumulative_heights[bin];
    const double width = parameters.widths[bin];
    const double height = parameters.heights[bin];
    const double delta = height / width;
    const double derivative_left = parameters.derivatives[bin];
    const double derivative_right = parameters.derivatives[bin + 1];
    const double theta = (input - x_start) / width;
    const double one_minus_theta = 1.0 - theta;
    const double theta_one_minus_theta = theta * one_minus_theta;
    const double denominator = delta +
        (derivative_left + derivative_right - 2.0 * delta) *
            theta_one_minus_theta;
    const double numerator = height *
        (delta * theta * theta + derivative_left * theta_one_minus_theta);
    const double output = y_start + numerator / denominator;

    const double derivative_numerator = delta * delta *
        (derivative_right * theta * theta +
         2.0 * delta * theta_one_minus_theta +
         derivative_left * one_minus_theta * one_minus_theta);
    if (!(derivative_numerator > 0.0) || !(denominator > 0.0)) {
        throw std::runtime_error("RationalQuadraticSpline lost monotonicity");
    }
    return {output,
            std::log(derivative_numerator) - 2.0 * std::log(denominator)};
}

// Small forward-mode dual number used to differentiate a scalar spline
// transform with respect to its input and local raw parameter vector.
struct Dual {
    double value = 0.0;
    std::vector<double> derivative;

    Dual() = default;
    Dual(double primal, size_t dimension)
        : value(primal), derivative(dimension, 0.0) {}
};

Dual seed(double value, size_t dimension, size_t coordinate) {
    Dual result(value, dimension);
    result.derivative[coordinate] = 1.0;
    return result;
}

Dual operator+(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value + rhs.value, lhs.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = lhs.derivative[i] + rhs.derivative[i];
    }
    return result;
}

Dual operator-(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value - rhs.value, lhs.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = lhs.derivative[i] - rhs.derivative[i];
    }
    return result;
}

Dual operator*(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value * rhs.value, lhs.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = lhs.derivative[i] * rhs.value +
                               lhs.value * rhs.derivative[i];
    }
    return result;
}

Dual operator/(const Dual& lhs, const Dual& rhs) {
    Dual result(lhs.value / rhs.value, lhs.derivative.size());
    const double inverse_square = 1.0 / (rhs.value * rhs.value);
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] =
            (lhs.derivative[i] * rhs.value - lhs.value * rhs.derivative[i]) *
            inverse_square;
    }
    return result;
}

Dual operator+(const Dual& lhs, double rhs) {
    Dual result = lhs;
    result.value += rhs;
    return result;
}

Dual operator+(double lhs, const Dual& rhs) { return rhs + lhs; }

Dual operator-(const Dual& lhs, double rhs) {
    Dual result = lhs;
    result.value -= rhs;
    return result;
}

Dual operator-(double lhs, const Dual& rhs) {
    Dual result(lhs - rhs.value, rhs.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = -rhs.derivative[i];
    }
    return result;
}

Dual operator*(const Dual& lhs, double rhs) {
    Dual result(lhs.value * rhs, lhs.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = lhs.derivative[i] * rhs;
    }
    return result;
}

Dual operator*(double lhs, const Dual& rhs) { return rhs * lhs; }

Dual dual_exp(const Dual& input) {
    const double exponential = std::exp(input.value);
    Dual result(exponential, input.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = exponential * input.derivative[i];
    }
    return result;
}

Dual dual_log(const Dual& input) {
    Dual result(std::log(input.value), input.derivative.size());
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = input.derivative[i] / input.value;
    }
    return result;
}

Dual dual_softplus(const Dual& input) {
    if (input.value > 20.0) return input;
    if (input.value < -20.0) return dual_exp(input);
    Dual result(std::log1p(std::exp(input.value)), input.derivative.size());
    const double sigmoid = 1.0 / (1.0 + std::exp(-input.value));
    for (size_t i = 0; i < result.derivative.size(); ++i) {
        result.derivative[i] = sigmoid * input.derivative[i];
    }
    return result;
}

std::vector<Dual> dual_softmax(const std::vector<Dual>& raw) {
    double maximum = raw.front().value;
    for (const Dual& value : raw) maximum = std::max(maximum, value.value);
    std::vector<Dual> exponentials;
    exponentials.reserve(raw.size());
    Dual total(0.0, raw.front().derivative.size());
    for (const Dual& value : raw) {
        exponentials.push_back(dual_exp(value - maximum));
        total = total + exponentials.back();
    }
    for (Dual& exponential : exponentials) exponential = exponential / total;
    return exponentials;
}

struct DualSplineResult {
    Dual output;
    Dual log_abs_det;
};

DualSplineResult evaluate_spline_with_jacobian(
    double input,
    const std::vector<double>& raw_parameters,
    size_t num_bins,
    double tail_bound,
    double min_bin_width,
    double min_bin_height,
    double min_derivative) {
    const size_t parameter_count = 3 * num_bins - 1;
    if (raw_parameters.size() != parameter_count) {
        throw std::invalid_argument("NSFCouplingLayer local parameter slice is invalid");
    }
    const size_t dimension = parameter_count + 1;
    if (input < -tail_bound || input > tail_bound) {
        return {seed(input, dimension, 0), Dual(0.0, dimension)};
    }

    std::vector<Dual> raw_widths;
    std::vector<Dual> raw_heights;
    std::vector<Dual> raw_derivatives;
    raw_widths.reserve(num_bins);
    raw_heights.reserve(num_bins);
    raw_derivatives.reserve(num_bins - 1);
    for (size_t i = 0; i < num_bins; ++i) {
        raw_widths.push_back(seed(raw_parameters[i], dimension, 1 + i));
        raw_heights.push_back(
            seed(raw_parameters[num_bins + i], dimension, 1 + num_bins + i));
    }
    for (size_t i = 0; i < num_bins - 1; ++i) {
        raw_derivatives.push_back(seed(
            raw_parameters[2 * num_bins + i], dimension,
            1 + 2 * num_bins + i));
    }

    const std::vector<Dual> soft_widths = dual_softmax(raw_widths);
    const std::vector<Dual> soft_heights = dual_softmax(raw_heights);
    const double interval = 2.0 * tail_bound;
    const double width_scale = 1.0 - min_bin_width * static_cast<double>(num_bins);
    const double height_scale = 1.0 - min_bin_height * static_cast<double>(num_bins);
    const double derivative_shift = inverse_softplus(1.0 - min_derivative);

    std::vector<Dual> widths(num_bins);
    std::vector<Dual> heights(num_bins);
    std::vector<Dual> derivatives(num_bins + 1, Dual(1.0, dimension));
    std::vector<Dual> cumulative_widths(num_bins + 1, Dual(0.0, dimension));
    std::vector<Dual> cumulative_heights(num_bins + 1, Dual(0.0, dimension));
    cumulative_widths[0] = Dual(-tail_bound, dimension);
    cumulative_heights[0] = Dual(-tail_bound, dimension);
    for (size_t bin = 0; bin < num_bins; ++bin) {
        widths[bin] = interval *
            (min_bin_width + width_scale * soft_widths[bin]);
        heights[bin] = interval *
            (min_bin_height + height_scale * soft_heights[bin]);
        cumulative_widths[bin + 1] = cumulative_widths[bin] + widths[bin];
        cumulative_heights[bin + 1] = cumulative_heights[bin] + heights[bin];
    }
    for (size_t knot = 1; knot < num_bins; ++knot) {
        derivatives[knot] =
            min_derivative + dual_softplus(raw_derivatives[knot - 1] +
                                           derivative_shift);
    }

    std::vector<double> cumulative_primal(num_bins + 1);
    for (size_t i = 0; i <= num_bins; ++i) {
        cumulative_primal[i] = cumulative_widths[i].value;
    }
    cumulative_primal.front() = -tail_bound;
    cumulative_primal.back() = tail_bound;
    const size_t bin = find_bin(cumulative_primal, input);

    const Dual dual_input = seed(input, dimension, 0);
    const Dual delta = heights[bin] / widths[bin];
    const Dual theta = (dual_input - cumulative_widths[bin]) / widths[bin];
    const Dual one_minus_theta = 1.0 - theta;
    const Dual theta_one_minus_theta = theta * one_minus_theta;
    const Dual denominator = delta +
        (derivatives[bin] + derivatives[bin + 1] - 2.0 * delta) *
            theta_one_minus_theta;
    const Dual numerator = heights[bin] *
        (delta * theta * theta +
         derivatives[bin] * theta_one_minus_theta);
    const Dual output = cumulative_heights[bin] + numerator / denominator;
    const Dual derivative_numerator = delta * delta *
        (derivatives[bin + 1] * theta * theta +
         2.0 * delta * theta_one_minus_theta +
         derivatives[bin] * one_minus_theta * one_minus_theta);
    const Dual log_abs_det = dual_log(derivative_numerator) -
                             2.0 * dual_log(denominator);
    return {output, log_abs_det};
}

void extract_local_parameters(const Tensor& raw_parameters,
                              size_t row, size_t transformed_dimension,
                              size_t params_per_scalar,
                              std::vector<double>& local) {
    local.resize(params_per_scalar);
    const size_t offset = transformed_dimension * params_per_scalar;
    for (size_t parameter = 0; parameter < params_per_scalar; ++parameter) {
        local[parameter] = raw_parameters(row, offset + parameter);
    }
}

void split_local_parameters(const std::vector<double>& local,
                            size_t num_bins,
                            std::vector<double>& widths,
                            std::vector<double>& heights,
                            std::vector<double>& derivatives) {
    widths.assign(local.begin(), local.begin() + static_cast<std::ptrdiff_t>(num_bins));
    heights.assign(local.begin() + static_cast<std::ptrdiff_t>(num_bins),
                   local.begin() + static_cast<std::ptrdiff_t>(2 * num_bins));
    derivatives.assign(local.begin() + static_cast<std::ptrdiff_t>(2 * num_bins),
                       local.end());
}

}  // namespace

RationalQuadraticSpline::RationalQuadraticSpline(
    size_t num_bins, double tail_bound, double min_bin_width,
    double min_bin_height, double min_derivative)
    : num_bins_(num_bins),
      tail_bound_(tail_bound),
      min_bin_width_(min_bin_width),
      min_bin_height_(min_bin_height),
      min_derivative_(min_derivative),
      derivative_identity_shift_(0.0) {
    if (num_bins_ < 2) {
        throw std::invalid_argument("RationalQuadraticSpline requires at least two bins");
    }
    if (!finite_positive(tail_bound_)) {
        throw std::invalid_argument("RationalQuadraticSpline tail_bound must be positive and finite");
    }
    if (!finite_positive(min_bin_width_) ||
        min_bin_width_ * static_cast<double>(num_bins_) >= 1.0) {
        throw std::invalid_argument("RationalQuadraticSpline min_bin_width is infeasible");
    }
    if (!finite_positive(min_bin_height_) ||
        min_bin_height_ * static_cast<double>(num_bins_) >= 1.0) {
        throw std::invalid_argument("RationalQuadraticSpline min_bin_height is infeasible");
    }
    if (!finite_positive(min_derivative_) || min_derivative_ >= 1.0) {
        throw std::invalid_argument(
            "RationalQuadraticSpline min_derivative must lie in (0, 1)");
    }
    derivative_identity_shift_ = inverse_softplus(1.0 - min_derivative_);
}

SplineTransformResult RationalQuadraticSpline::forward(
    double input, const std::vector<double>& unnormalized_widths,
    const std::vector<double>& unnormalized_heights,
    const std::vector<double>& unnormalized_internal_derivatives) const {
    validate_raw_parameters(num_bins_, unnormalized_widths, unnormalized_heights,
                            unnormalized_internal_derivatives);
    if (!std::isfinite(input)) {
        throw std::invalid_argument("RationalQuadraticSpline input must be finite");
    }
    if (input < -tail_bound_ || input > tail_bound_) return {input, 0.0};

    const SplineParameters parameters = constrain_parameters(
        num_bins_, tail_bound_, min_bin_width_, min_bin_height_, min_derivative_,
        derivative_identity_shift_, unnormalized_widths, unnormalized_heights,
        unnormalized_internal_derivatives);
    const size_t bin = find_bin(parameters.cumulative_widths, input);
    return evaluate_forward_in_bin(input, bin, parameters);
}

SplineTransformResult RationalQuadraticSpline::inverse(
    double input, const std::vector<double>& unnormalized_widths,
    const std::vector<double>& unnormalized_heights,
    const std::vector<double>& unnormalized_internal_derivatives) const {
    validate_raw_parameters(num_bins_, unnormalized_widths, unnormalized_heights,
                            unnormalized_internal_derivatives);
    if (!std::isfinite(input)) {
        throw std::invalid_argument("RationalQuadraticSpline inverse input must be finite");
    }
    if (input < -tail_bound_ || input > tail_bound_) return {input, 0.0};

    const SplineParameters parameters = constrain_parameters(
        num_bins_, tail_bound_, min_bin_width_, min_bin_height_, min_derivative_,
        derivative_identity_shift_, unnormalized_widths, unnormalized_heights,
        unnormalized_internal_derivatives);
    const size_t bin = find_bin(parameters.cumulative_heights, input);
    const double width = parameters.widths[bin];
    const double height = parameters.heights[bin];
    const double delta = height / width;
    const double derivative_left = parameters.derivatives[bin];
    const double derivative_right = parameters.derivatives[bin + 1];
    const double y_relative = input - parameters.cumulative_heights[bin];
    const double derivative_sum = derivative_left + derivative_right - 2.0 * delta;
    const double a = y_relative * derivative_sum +
                     height * (delta - derivative_left);
    const double b = height * derivative_left - y_relative * derivative_sum;
    const double c = -delta * y_relative;
    double discriminant = b * b - 4.0 * a * c;
    if (discriminant < -kRootRoundoffTolerance) {
        throw std::runtime_error("RationalQuadraticSpline inverse has negative discriminant");
    }
    discriminant = std::max(0.0, discriminant);
    const double root_denominator = -b - std::sqrt(discriminant);
    double theta = 0.0;
    if (std::abs(a) < 1e-14) {
        if (std::abs(b) < 1e-14) {
            theta = 0.0;
        } else {
            theta = -c / b;
        }
    } else if (std::abs(root_denominator) < 1e-14) {
        theta = (-b + std::sqrt(discriminant)) / (2.0 * a);
    } else {
        theta = (2.0 * c) / root_denominator;
    }
    if (theta < -kRootRoundoffTolerance || theta > 1.0 + kRootRoundoffTolerance) {
        throw std::runtime_error("RationalQuadraticSpline inverse root is outside its bin");
    }
    theta = std::min(1.0, std::max(0.0, theta));
    const double output = parameters.cumulative_widths[bin] + theta * width;
    const SplineTransformResult forward_result =
        evaluate_forward_in_bin(output, bin, parameters);
    return {output, -forward_result.log_abs_det};
}

NSFCouplingLayer::NSFCouplingLayer(
    size_t input_dim, size_t hidden_size, size_t num_bins, double tail_bound,
    bool swap_halves, double min_bin_width, double min_bin_height,
    double min_derivative)
    : input_dim_(input_dim),
      half_dim_(input_dim / 2),
      hidden_size_(hidden_size),
      num_bins_(num_bins),
      params_per_scalar_(3 * num_bins - 1),
      swap_halves_(swap_halves),
      spline_(num_bins, tail_bound, min_bin_width, min_bin_height,
              min_derivative),
      conditioner_hidden_(input_dim / 2, hidden_size),
      conditioner_output_(hidden_size, (input_dim / 2) * (3 * num_bins - 1)),
      last_input_(),
      last_conditioning_(),
      last_transformed_(),
      last_hidden_(),
      last_raw_parameters_(),
      log_det_jacobian_(0.0),
      dL_d_log_det_(0.0),
      forward_cache_valid_(false) {
    if (input_dim_ == 0 || input_dim_ % 2 != 0) {
        throw std::invalid_argument("NSFCouplingLayer requires a positive even input_dim");
    }
    if (hidden_size_ == 0) {
        throw std::invalid_argument("NSFCouplingLayer requires hidden_size > 0");
    }
    conditioner_hidden_.init_weights("xavier");
    conditioner_output_.init_weights("zeros");
}

size_t NSFCouplingLayer::conditioning_column(size_t local_column) const {
    return swap_halves_ ? half_dim_ + local_column : local_column;
}

size_t NSFCouplingLayer::transformed_column(size_t local_column) const {
    return swap_halves_ ? local_column : half_dim_ + local_column;
}

void NSFCouplingLayer::validate_input(const Tensor& input,
                                      const char* operation) const {
    if (input.rows == 0) {
        throw std::invalid_argument(std::string(operation) +
                                    " requires a non-empty batch");
    }
    if (input.cols != input_dim_) {
        throw std::invalid_argument(std::string(operation) +
                                    " input feature dimension mismatch");
    }
    for (double value : input.data) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument(std::string(operation) +
                                        " input contains a non-finite value");
        }
    }
}

void NSFCouplingLayer::split_input(const Tensor& input, Tensor& conditioning,
                                   Tensor& transformed) const {
    conditioning = Tensor(input.rows, half_dim_);
    transformed = Tensor(input.rows, half_dim_);
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t column = 0; column < half_dim_; ++column) {
            conditioning(row, column) = input(row, conditioning_column(column));
            transformed(row, column) = input(row, transformed_column(column));
        }
    }
}

Tensor NSFCouplingLayer::forward(const Tensor& input) {
    validate_input(input, "NSFCouplingLayer::forward");
    split_input(input, last_conditioning_, last_transformed_);
    last_input_ = input.clone();

    Tensor hidden_pre = conditioner_hidden_.forward(last_conditioning_);
    last_hidden_ = hidden_pre.apply([](double value) { return std::tanh(value); });
    last_raw_parameters_ = conditioner_output_.forward(last_hidden_);

    Tensor output = input.clone();
    log_det_jacobian_ = 0.0;
    std::vector<double> local, widths, heights, derivatives;
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t dimension = 0; dimension < half_dim_; ++dimension) {
            extract_local_parameters(last_raw_parameters_, row, dimension,
                                     params_per_scalar_, local);
            split_local_parameters(local, num_bins_, widths, heights, derivatives);
            const SplineTransformResult result = spline_.forward(
                last_transformed_(row, dimension), widths, heights, derivatives);
            output(row, transformed_column(dimension)) = result.value;
            log_det_jacobian_ += result.log_abs_det;
        }
    }
    forward_cache_valid_ = true;
    return output;
}

Tensor NSFCouplingLayer::inverse(const Tensor& output) {
    validate_input(output, "NSFCouplingLayer::inverse");
    Tensor conditioning;
    Tensor transformed;
    split_input(output, conditioning, transformed);

    Tensor hidden_pre = conditioner_hidden_.forward(conditioning);
    Tensor hidden = hidden_pre.apply([](double value) { return std::tanh(value); });
    Tensor raw_parameters = conditioner_output_.forward(hidden);

    Tensor input = output.clone();
    log_det_jacobian_ = 0.0;
    std::vector<double> local, widths, heights, derivatives;
    for (size_t row = 0; row < output.rows; ++row) {
        for (size_t dimension = 0; dimension < half_dim_; ++dimension) {
            extract_local_parameters(raw_parameters, row, dimension,
                                     params_per_scalar_, local);
            split_local_parameters(local, num_bins_, widths, heights, derivatives);
            const SplineTransformResult result = spline_.inverse(
                transformed(row, dimension), widths, heights, derivatives);
            input(row, transformed_column(dimension)) = result.value;
            log_det_jacobian_ += result.log_abs_det;
        }
    }
    forward_cache_valid_ = false;
    return input;
}

void NSFCouplingLayer::set_log_det_gradient(double gradient) {
    if (!std::isfinite(gradient)) {
        throw std::invalid_argument("NSFCouplingLayer log-det gradient must be finite");
    }
    dL_d_log_det_ = gradient;
}

Tensor NSFCouplingLayer::backward(const Tensor& grad_output,
                                  double /*learning_rate*/) {
    if (!forward_cache_valid_) {
        throw std::logic_error(
            "NSFCouplingLayer::backward requires a preceding forward call");
    }
    if (grad_output.rows != last_input_.rows ||
        grad_output.cols != last_input_.cols) {
        throw std::invalid_argument(
            "NSFCouplingLayer::backward gradient shape mismatch");
    }

    Tensor grad_raw(grad_output.rows, last_raw_parameters_.cols);
    grad_raw.fill(0.0);
    Tensor grad_transformed(grad_output.rows, half_dim_);
    grad_transformed.fill(0.0);

    std::vector<double> local;
    for (size_t row = 0; row < last_input_.rows; ++row) {
        for (size_t dimension = 0; dimension < half_dim_; ++dimension) {
            extract_local_parameters(last_raw_parameters_, row, dimension,
                                     params_per_scalar_, local);
            const DualSplineResult result = evaluate_spline_with_jacobian(
                last_transformed_(row, dimension), local, num_bins_,
                spline_.tail_bound(), spline_.min_bin_width(),
                spline_.min_bin_height(), spline_.min_derivative());
            const double grad_y = grad_output(row, transformed_column(dimension));
            grad_transformed(row, dimension) =
                grad_y * result.output.derivative[0] +
                dL_d_log_det_ * result.log_abs_det.derivative[0];
            const size_t offset = dimension * params_per_scalar_;
            for (size_t parameter = 0; parameter < params_per_scalar_; ++parameter) {
                grad_raw(row, offset + parameter) =
                    grad_y * result.output.derivative[1 + parameter] +
                    dL_d_log_det_ *
                        result.log_abs_det.derivative[1 + parameter];
            }
        }
    }

    // inverse() calls the same Dense objects and overwrites their caches, so
    // backward is intentionally forbidden after inverse. For the valid path,
    // these caches still correspond exactly to the latest forward call.
    Tensor grad_hidden = conditioner_output_.backward(grad_raw, 0.0);
    Tensor grad_hidden_pre(grad_hidden.rows, grad_hidden.cols);
    for (size_t i = 0; i < grad_hidden.data.size(); ++i) {
        grad_hidden_pre.data[i] =
            grad_hidden.data[i] *
            (1.0 - last_hidden_.data[i] * last_hidden_.data[i]);
    }
    Tensor grad_conditioning = conditioner_hidden_.backward(grad_hidden_pre, 0.0);

    Tensor grad_input(grad_output.rows, input_dim_);
    grad_input.fill(0.0);
    for (size_t row = 0; row < grad_output.rows; ++row) {
        for (size_t dimension = 0; dimension < half_dim_; ++dimension) {
            grad_input(row, conditioning_column(dimension)) =
                grad_output(row, conditioning_column(dimension)) +
                grad_conditioning(row, dimension);
            grad_input(row, transformed_column(dimension)) =
                grad_transformed(row, dimension);
        }
    }
    return grad_input;
}

void NSFCouplingLayer::update_weights(double learning_rate) {
    conditioner_hidden_.update_weights(learning_rate);
    conditioner_output_.update_weights(learning_rate);
}

void NSFCouplingLayer::zero_grad() {
    conditioner_hidden_.zero_grad();
    conditioner_output_.zero_grad();
}

std::vector<Tensor*> NSFCouplingLayer::parameters() {
    std::vector<Tensor*> result;
    for (Tensor* parameter : conditioner_hidden_.parameters()) {
        result.push_back(parameter);
    }
    for (Tensor* parameter : conditioner_output_.parameters()) {
        result.push_back(parameter);
    }
    return result;
}

std::vector<Tensor*> NSFCouplingLayer::gradients() {
    std::vector<Tensor*> result;
    for (Tensor* gradient : conditioner_hidden_.gradients()) {
        result.push_back(gradient);
    }
    for (Tensor* gradient : conditioner_output_.gradients()) {
        result.push_back(gradient);
    }
    return result;
}

Tensor NSFCouplingLayer::get_weights() const {
    return conditioner_output_.get_weights();
}

Tensor NSFCouplingLayer::get_gradients() const {
    return conditioner_output_.get_gradients();
}
