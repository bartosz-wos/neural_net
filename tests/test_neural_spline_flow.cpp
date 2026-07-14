#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/layers/generative/neural_spline_flow.h"

namespace {

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        ++passed;
        std::cout << "  [PASS] " << name << '\n';
    } else {
        ++failed;
        std::cout << "  [FAIL] " << name << '\n';
    }
}

void check_near(double actual, double expected, double tolerance,
                const std::string& name) {
    const bool ok = std::isfinite(actual) && std::isfinite(expected) &&
                    std::abs(actual - expected) <= tolerance;
    if (!ok) {
        std::cout << std::setprecision(12)
                  << "         actual=" << actual
                  << " expected=" << expected
                  << " tolerance=" << tolerance << '\n';
    }
    check(ok, name);
}

template <typename Fn>
void check_throws_invalid_argument(Fn&& fn, const std::string& name) {
    bool threw = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, name);
}

template <typename Fn>
void check_throws_logic_error(Fn&& fn, const std::string& name) {
    bool threw = false;
    try {
        fn();
    } catch (const std::logic_error&) {
        threw = true;
    }
    check(threw, name);
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        return std::numeric_limits<double>::infinity();
    }
    double result = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        result = std::max(result, std::abs(a.data[i] - b.data[i]));
    }
    return result;
}

bool all_finite(const Tensor& tensor) {
    for (double value : tensor.data) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

void configure_nontrivial(NSFCouplingLayer& layer) {
    auto params = layer.parameters();
    for (size_t i = 0; i < params[0]->rows; ++i) {
        for (size_t j = 0; j < params[0]->cols; ++j) {
            (*params[0])(i, j) = 0.17 * std::sin(0.8 * static_cast<double>(i + 1))
                               - 0.09 * static_cast<double>(j + 1);
        }
        (*params[1])(0, i) = 0.04 * static_cast<double>(i) - 0.07;
    }
    for (size_t i = 0; i < params[2]->rows; ++i) {
        for (size_t j = 0; j < params[2]->cols; ++j) {
            (*params[2])(i, j) = 0.08 * std::sin(0.37 * static_cast<double>(i + 1)
                                                + 0.23 * static_cast<double>(j + 1));
        }
        (*params[3])(0, i) = 0.16 * std::cos(0.31 * static_cast<double>(i + 1));
    }
}

Tensor make_coupling_input() {
    Tensor input(3, 4);
    input(0, 0) = -0.8; input(0, 1) =  0.4;
    input(0, 2) = -0.6; input(0, 3) =  1.1;
    input(1, 0) =  0.3; input(1, 1) = -1.2;
    input(1, 2) =  0.7; input(1, 3) = -0.9;
    input(2, 0) =  1.0; input(2, 1) =  0.5;
    input(2, 2) = -1.4; input(2, 3) =  0.2;
    return input;
}

double objective(NSFCouplingLayer& layer, const Tensor& input,
                 const Tensor& target, double lambda) {
    Tensor output = layer.forward(input);
    double loss = lambda * layer.log_det_jacobian();
    for (size_t i = 0; i < output.data.size(); ++i) {
        const double residual = output.data[i] - target.data[i];
        loss += 0.5 * residual * residual;
    }
    return loss;
}

struct GradientCheckStats {
    double max_abs_error = 0.0;
    double max_relative_error = 0.0;
    size_t failures = 0;
};

void record_gradient_error(GradientCheckStats& stats,
                           double analytical, double numerical) {
    const double absolute = std::abs(analytical - numerical);
    const double scale = std::max({std::abs(analytical), std::abs(numerical), 1e-12});
    const double relative = absolute / scale;
    stats.max_abs_error = std::max(stats.max_abs_error, absolute);
    if (scale > 1e-7) {
        stats.max_relative_error = std::max(stats.max_relative_error, relative);
    }
    const double tolerance = std::max(2e-7, 5e-4 * scale);
    if (absolute > tolerance) ++stats.failures;
}

void print_gradient_stats(const std::string& label,
                          const GradientCheckStats& stats) {
    std::cout << std::scientific << std::setprecision(3)
              << "         " << label
              << " max_abs=" << stats.max_abs_error
              << " max_rel=" << stats.max_relative_error
              << " failures=" << stats.failures << '\n'
              << std::defaultfloat;
}

void test_validation_and_accessors() {
    std::cout << "\n-- Validation and public contract --\n";

    RationalQuadraticSpline spline(8, 3.0);
    check(spline.num_bins() == 8, "spline stores bin count");
    check(spline.num_derivative_params() == 7,
          "linear tails learn K-1 internal derivatives");
    check_near(spline.tail_bound(), 3.0, 0.0, "spline stores tail bound");
    check_near(spline.min_bin_width(), 1e-3, 0.0, "default minimum bin width");
    check_near(spline.min_bin_height(), 1e-3, 0.0, "default minimum bin height");
    check_near(spline.min_derivative(), 1e-3, 0.0, "default minimum derivative");

    check_throws_invalid_argument(
        [] { RationalQuadraticSpline bad(1, 3.0); },
        "rejects fewer than two bins");
    check_throws_invalid_argument(
        [] { RationalQuadraticSpline bad(8, 0.0); },
        "rejects non-positive tail bound");
    check_throws_invalid_argument(
        [] { RationalQuadraticSpline bad(8, std::numeric_limits<double>::infinity()); },
        "rejects non-finite tail bound");
    check_throws_invalid_argument(
        [] { RationalQuadraticSpline bad(4, 3.0, 0.25, 1e-3, 1e-3); },
        "rejects infeasible minimum bin width");
    check_throws_invalid_argument(
        [] { RationalQuadraticSpline bad(4, 3.0, 1e-3, 0.25, 1e-3); },
        "rejects infeasible minimum bin height");

    NSFCouplingLayer layer(4, 5, 4, 2.5);
    check(layer.input_dim() == 4, "coupling stores input dimension");
    check(layer.half_dim() == 2, "coupling derives half dimension");
    check(layer.hidden_size() == 5, "coupling stores hidden size");
    check(layer.num_bins() == 4, "coupling stores bin count");
    check(layer.params_per_scalar() == 11, "coupling uses 3K-1 params per scalar");
    check(!layer.swap_halves(), "coupling stores default split orientation");

    check_throws_invalid_argument(
        [] { NSFCouplingLayer bad(0, 5); },
        "rejects zero input dimension");
    check_throws_invalid_argument(
        [] { NSFCouplingLayer bad(3, 5); },
        "rejects odd input dimension");
    check_throws_invalid_argument(
        [] { NSFCouplingLayer bad(4, 0); },
        "rejects zero hidden size");

    Tensor wrong_cols(2, 3);
    check_throws_invalid_argument(
        [&] { (void)layer.forward(wrong_cols); },
        "forward rejects wrong feature dimension");
    Tensor empty_batch(0, 4);
    check_throws_invalid_argument(
        [&] { (void)layer.forward(empty_batch); },
        "forward rejects empty batch");

    std::vector<double> w(4, 0.0), h(4, 0.0), d(3, 0.0);
    check_throws_invalid_argument(
        [&] { (void)spline.forward(0.0, std::vector<double>(3, 0.0), h, d); },
        "spline rejects wrong width parameter count");
    w[2] = std::numeric_limits<double>::quiet_NaN();
    check_throws_invalid_argument(
        [&] { (void)spline.forward(0.0, w, h, d); },
        "spline rejects non-finite raw parameters");
}

void test_spline_primitive() {
    std::cout << "\n-- Rational-quadratic spline primitive --\n";

    RationalQuadraticSpline spline(4, 2.0);
    const std::vector<double> zeros_w(4, 0.0);
    const std::vector<double> zeros_h(4, 0.0);
    const std::vector<double> zeros_d(3, 0.0);

    double identity_error = 0.0;
    for (double x : {-2.0, -1.3, -0.1, 0.0, 0.8, 1.7, 2.0}) {
        const auto result = spline.forward(x, zeros_w, zeros_h, zeros_d);
        identity_error = std::max(identity_error, std::abs(result.value - x));
        identity_error = std::max(identity_error, std::abs(result.log_abs_det));
    }
    check(identity_error < 1e-12, "zero raw parameters produce identity spline");

    for (double x : {-4.5, -2.1, 2.1, 5.0}) {
        const auto result = spline.forward(x, zeros_w, zeros_h, zeros_d);
        check(result.value == x && result.log_abs_det == 0.0,
              "linear tail is exact identity at x=" + std::to_string(x));
    }

    const std::vector<double> raw_w = {-0.7, 0.2, 1.1, -0.3};
    const std::vector<double> raw_h = {0.4, 1.0, -0.5, 0.1};
    const std::vector<double> raw_d = {-0.2, 0.8, 0.3};

    bool strictly_increasing = true;
    bool finite = true;
    double previous = -std::numeric_limits<double>::infinity();
    double max_roundtrip_error = 0.0;
    double max_inverse_logdet_error = 0.0;
    for (size_t i = 0; i <= 400; ++i) {
        const double x = -2.0 + 4.0 * static_cast<double>(i) / 400.0;
        const auto forward = spline.forward(x, raw_w, raw_h, raw_d);
        const auto inverse = spline.inverse(forward.value, raw_w, raw_h, raw_d);
        finite = finite && std::isfinite(forward.value) &&
                 std::isfinite(forward.log_abs_det) &&
                 std::isfinite(inverse.value) &&
                 std::isfinite(inverse.log_abs_det);
        if (i > 0 && !(forward.value > previous)) strictly_increasing = false;
        previous = forward.value;
        max_roundtrip_error = std::max(max_roundtrip_error,
                                       std::abs(inverse.value - x));
        max_inverse_logdet_error = std::max(
            max_inverse_logdet_error,
            std::abs(forward.log_abs_det + inverse.log_abs_det));
    }
    check(finite, "non-uniform spline stays finite across the domain");
    check(strictly_increasing, "non-uniform spline is strictly monotone");
    check(max_roundtrip_error < 1e-10,
          "forward then inverse recovers inputs to 1e-10");
    check(max_inverse_logdet_error < 1e-10,
          "forward and inverse log determinants cancel");

    const double eps = 1e-6;
    double max_derivative_rel_error = 0.0;
    for (double x : {-1.73, -1.1, -0.35, 0.23, 0.91, 1.62}) {
        const double yp = spline.forward(x + eps, raw_w, raw_h, raw_d).value;
        const double ym = spline.forward(x - eps, raw_w, raw_h, raw_d).value;
        const double numerical = (yp - ym) / (2.0 * eps);
        const auto result = spline.forward(x, raw_w, raw_h, raw_d);
        const double analytical = std::exp(result.log_abs_det);
        const double rel = std::abs(analytical - numerical) /
                           std::max({std::abs(analytical), std::abs(numerical), 1e-12});
        max_derivative_rel_error = std::max(max_derivative_rel_error, rel);
    }
    std::cout << "         max derivative rel_err=" << std::scientific
              << max_derivative_rel_error << std::defaultfloat << '\n';
    check(max_derivative_rel_error < 1e-6,
          "reported log determinant matches numerical derivative");

    const auto left = spline.forward(-2.0, raw_w, raw_h, raw_d);
    const auto right = spline.forward(2.0, raw_w, raw_h, raw_d);
    check_near(left.value, -2.0, 1e-12, "left boundary maps exactly");
    check_near(right.value, 2.0, 1e-12, "right boundary maps exactly");
    check_near(std::exp(left.log_abs_det), 1.0, 1e-10,
               "left boundary joins identity tail with unit derivative");
    check_near(std::exp(right.log_abs_det), 1.0, 1e-10,
               "right boundary joins identity tail with unit derivative");

    const std::vector<double> extreme_w = {-1000.0, 1000.0, -500.0, 500.0};
    const std::vector<double> extreme_h = {800.0, -900.0, 600.0, -700.0};
    const std::vector<double> extreme_d = {-1000.0, 1000.0, -400.0};
    bool extreme_finite = true;
    bool extreme_monotone = true;
    previous = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i <= 200; ++i) {
        const double x = -2.0 + 4.0 * static_cast<double>(i) / 200.0;
        const auto result = spline.forward(x, extreme_w, extreme_h, extreme_d);
        extreme_finite = extreme_finite && std::isfinite(result.value) &&
                         std::isfinite(result.log_abs_det);
        if (i > 0 && !(result.value > previous)) extreme_monotone = false;
        previous = result.value;
    }
    check(extreme_finite, "extreme finite logits remain numerically finite");
    check(extreme_monotone, "extreme finite logits preserve monotonicity");
}

void test_coupling_forward_inverse() {
    std::cout << "\n-- Neural spline coupling forward/inverse --\n";

    Tensor input = make_coupling_input();
    NSFCouplingLayer identity(4, 5, 4, 2.5);
    Tensor identity_output = identity.forward(input);
    check(identity_output.rows == input.rows && identity_output.cols == input.cols,
          "coupling preserves batch shape");
    check(max_abs_diff(identity_output, input) < 1e-12,
          "zero-initialized conditioner gives identity coupling");
    check(std::abs(identity.log_det_jacobian()) < 1e-12,
          "identity coupling has zero log determinant");

    auto params = identity.parameters();
    auto grads = identity.gradients();
    check(params.size() == 4, "coupling exposes two Dense parameter pairs");
    check(params.size() == grads.size(), "parameter and gradient counts match");
    bool shapes_match = true;
    for (size_t i = 0; i < params.size(); ++i) {
        shapes_match = shapes_match && params[i]->rows == grads[i]->rows &&
                       params[i]->cols == grads[i]->cols;
    }
    check(shapes_match, "every coupling gradient mirrors its parameter shape");

    NSFCouplingLayer layer(4, 5, 4, 2.5);
    configure_nontrivial(layer);
    Tensor output = layer.forward(input);
    const double analytical_logdet = layer.log_det_jacobian();
    check(max_abs_diff(output, input) > 1e-4,
          "configured conditioner produces a non-trivial transform");
    check(all_finite(output) && std::isfinite(analytical_logdet),
          "configured coupling output and log determinant are finite");
    for (size_t row = 0; row < input.rows; ++row) {
        check(output(row, 0) == input(row, 0) && output(row, 1) == input(row, 1),
              "default split leaves left half unchanged for row " +
                  std::to_string(row));
    }

    Tensor reconstructed = layer.inverse(output);
    check(max_abs_diff(reconstructed, input) < 1e-10,
          "batch forward/inverse round trip recovers input");

    NSFCouplingLayer swapped(4, 5, 4, 2.5, true);
    configure_nontrivial(swapped);
    Tensor swapped_output = swapped.forward(input);
    check(swapped.swap_halves(), "swapped coupling stores split orientation");
    for (size_t row = 0; row < input.rows; ++row) {
        check(swapped_output(row, 2) == input(row, 2) &&
                  swapped_output(row, 3) == input(row, 3),
              "swapped split leaves right half unchanged for row " +
                  std::to_string(row));
    }
    check(max_abs_diff(swapped.inverse(swapped_output), input) < 1e-10,
          "swapped coupling round trip recovers input");

    Tensor same_transformed(2, 4);
    same_transformed(0, 0) = -1.1; same_transformed(0, 1) = 0.2;
    same_transformed(1, 0) =  1.0; same_transformed(1, 1) = -0.7;
    same_transformed(0, 2) = same_transformed(1, 2) = 0.35;
    same_transformed(0, 3) = same_transformed(1, 3) = -0.45;
    Tensor independent = layer.forward(same_transformed);
    check(std::abs(independent(0, 2) - independent(1, 2)) > 1e-7 ||
              std::abs(independent(0, 3) - independent(1, 3)) > 1e-7,
          "different batch rows receive independent conditioner outputs");

    layer.forward(input);
    const double expected_logdet = layer.log_det_jacobian();
    double numerical_logdet = 0.0;
    const double eps = 1e-5;
    for (size_t row = 0; row < input.rows; ++row) {
        for (size_t col = 2; col < input.cols; ++col) {
            Tensor plus = input.clone();
            Tensor minus = input.clone();
            plus(row, col) += eps;
            minus(row, col) -= eps;
            const double yp = layer.forward(plus)(row, col);
            const double ym = layer.forward(minus)(row, col);
            const double derivative = (yp - ym) / (2.0 * eps);
            numerical_logdet += std::log(std::abs(derivative));
        }
    }
    check_near(expected_logdet, numerical_logdet, 2e-6,
               "coupling log determinant matches triangular numerical Jacobian");

    layer.forward(input);
    Tensor latest_output = layer.forward(input);
    (void)layer.inverse(latest_output);
    Tensor grad(latest_output.rows, latest_output.cols);
    grad.fill(1.0);
    check_throws_logic_error(
        [&] { (void)layer.backward(grad, 0.0); },
        "inverse invalidates forward cache for backward");
}

void test_coupling_gradients() {
    std::cout << "\n-- Coupling analytical gradients --\n";

    NSFCouplingLayer layer(4, 5, 4, 2.5);
    configure_nontrivial(layer);

    Tensor input(2, 4);
    input(0, 0) = -0.73; input(0, 1) = 0.41;
    input(0, 2) = -0.52; input(0, 3) = 0.88;
    input(1, 0) = 0.67; input(1, 1) = -1.07;
    input(1, 2) = 1.21; input(1, 3) = -0.34;

    Tensor target(2, 4);
    target(0, 0) = 0.2; target(0, 1) = -0.4;
    target(0, 2) = 0.7; target(0, 3) = -0.1;
    target(1, 0) = -0.3; target(1, 1) = 0.6;
    target(1, 2) = -0.5; target(1, 3) = 0.9;

    const double lambda = 0.37;
    layer.zero_grad();
    Tensor output = layer.forward(input);
    Tensor grad_output(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        grad_output.data[i] = output.data[i] - target.data[i];
    }
    layer.set_log_det_gradient(lambda);
    Tensor analytical_input = layer.backward(grad_output, 0.0).clone();

    std::vector<Tensor> analytical_params;
    for (Tensor* grad : layer.gradients()) analytical_params.push_back(grad->clone());

    const double eps = 1e-5;
    GradientCheckStats input_stats;
    for (size_t i = 0; i < input.data.size(); ++i) {
        Tensor plus = input.clone();
        Tensor minus = input.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        const double numerical = (objective(layer, plus, target, lambda) -
                                  objective(layer, minus, target, lambda)) /
                                 (2.0 * eps);
        record_gradient_error(input_stats, analytical_input.data[i], numerical);
    }
    print_gradient_stats("combined input", input_stats);
    check(input_stats.failures == 0,
          "input gradient matches finite difference for output + logdet loss");

    auto params = layer.parameters();
    const std::vector<std::string> names = {
        "hidden weights", "hidden bias", "output weights", "output bias"};
    for (size_t p = 0; p < params.size(); ++p) {
        GradientCheckStats stats;
        for (size_t i = 0; i < params[p]->data.size(); ++i) {
            const double original = params[p]->data[i];
            params[p]->data[i] = original + eps;
            const double plus = objective(layer, input, target, lambda);
            params[p]->data[i] = original - eps;
            const double minus = objective(layer, input, target, lambda);
            params[p]->data[i] = original;
            const double numerical = (plus - minus) / (2.0 * eps);
            record_gradient_error(stats, analytical_params[p].data[i], numerical);
        }
        print_gradient_stats(names[p], stats);
        check(stats.failures == 0,
              names[p] + " gradient matches finite difference");
    }

    layer.zero_grad();
    output = layer.forward(input);
    Tensor zero_grad(output.rows, output.cols);
    zero_grad.fill(0.0);
    layer.set_log_det_gradient(1.0);
    Tensor analytical_logdet_input = layer.backward(zero_grad, 0.0).clone();
    GradientCheckStats logdet_stats;
    for (size_t i = 0; i < input.data.size(); ++i) {
        Tensor plus = input.clone();
        Tensor minus = input.clone();
        plus.data[i] += eps;
        minus.data[i] -= eps;
        layer.forward(plus);
        const double lp = layer.log_det_jacobian();
        layer.forward(minus);
        const double lm = layer.log_det_jacobian();
        const double numerical = (lp - lm) / (2.0 * eps);
        record_gradient_error(logdet_stats,
                              analytical_logdet_input.data[i], numerical);
    }
    print_gradient_stats("logdet-only input", logdet_stats);
    check(logdet_stats.failures == 0,
          "logdet-only input gradient is independently non-vacuous");
}

void test_state_and_lifecycle() {
    std::cout << "\n-- State and lifecycle contracts --\n";

    NSFCouplingLayer layer(4, 5, 4, 2.5);
    Tensor grad(2, 4);
    grad.fill(1.0);
    check_throws_logic_error(
        [&] { (void)layer.backward(grad, 0.0); },
        "backward before forward throws");

    Tensor input(2, 4);
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = -0.8 + 0.19 * static_cast<double>(i);
    }
    layer.forward(input);
    Tensor wrong_grad(1, 4);
    check_throws_invalid_argument(
        [&] { (void)layer.backward(wrong_grad, 0.0); },
        "backward rejects gradient shape mismatch");

    configure_nontrivial(layer);
    layer.zero_grad();
    Tensor output = layer.forward(input);
    Tensor real_grad(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        real_grad.data[i] = output.data[i] + 0.1 * static_cast<double>(i + 1);
    }
    layer.set_log_det_gradient(-0.25);
    (void)layer.backward(real_grad, 0.0);

    auto params = layer.parameters();
    Tensor before = params[2]->clone();
    layer.update_weights(1e-3);
    check(max_abs_diff(before, *params[2]) > 0.0,
          "update_weights moves conditioner parameters after backward");

    layer.zero_grad();
    bool all_zero = true;
    for (Tensor* gradient : layer.gradients()) {
        for (double value : gradient->data) all_zero = all_zero && value == 0.0;
    }
    check(all_zero, "zero_grad clears all conditioner gradients exactly");
}

void test_density_estimation_training() {
    std::cout << "\n-- End-to-end density-estimation smoke test --\n";

    NSFCouplingLayer layer(4, 10, 4, 4.0);
    Tensor data(12, 4);
    for (size_t i = 0; i < data.rows; ++i) {
        const double c0 = -1.5 + 3.0 * static_cast<double>(i) /
                                   static_cast<double>(data.rows - 1);
        const double c1 = std::sin(0.7 * static_cast<double>(i));
        data(i, 0) = c0;
        data(i, 1) = c1;
        data(i, 2) = 1.35 + 0.65 * c0 + 0.15 * c1;
        data(i, 3) = -1.15 + 0.38 * c0 * c0 - 0.22 * c1;
    }

    auto density_loss = [&]() {
        Tensor z = layer.forward(data);
        double value = -layer.log_det_jacobian();
        for (double element : z.data) value += 0.5 * element * element;
        return value / static_cast<double>(data.rows);
    };

    const double initial = density_loss();
    const Tensor initial_output_weights = layer.parameters()[2]->clone();
    bool finite = std::isfinite(initial);
    const double learning_rate = 2e-3;
    for (size_t step = 0; step < 350; ++step) {
        layer.zero_grad();
        Tensor z = layer.forward(data);
        Tensor grad_z(z.rows, z.cols);
        const double inv_batch = 1.0 / static_cast<double>(data.rows);
        for (size_t i = 0; i < z.data.size(); ++i) {
            grad_z.data[i] = z.data[i] * inv_batch;
        }
        layer.set_log_det_gradient(-inv_batch);
        (void)layer.backward(grad_z, 0.0);
        layer.update_weights(learning_rate);
        if (step % 25 == 0) finite = finite && std::isfinite(density_loss());
        for (Tensor* parameter : layer.parameters()) {
            finite = finite && all_finite(*parameter);
        }
    }
    const double final = density_loss();
    std::cout << std::fixed << std::setprecision(6)
              << "         density NLL " << initial << " -> " << final << '\n'
              << std::defaultfloat;
    check(finite && std::isfinite(final),
          "density training remains finite");
    check(final < initial * 0.8,
          "density NLL falls by at least 20 percent");
    check(max_abs_diff(initial_output_weights, *layer.parameters()[2]) > 1e-6,
          "density objective updates conditioner weights");
}

}  // namespace

int main() {
    std::cout << "============================================\n"
              << "       Neural Spline Flow Tests             \n"
              << "============================================\n";

    test_validation_and_accessors();
    test_spline_primitive();
    test_coupling_forward_inverse();
    test_coupling_gradients();
    test_state_and_lifecycle();
    test_density_estimation_training();

    std::cout << "\n=== Summary: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
