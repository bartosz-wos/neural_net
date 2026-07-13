#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nn/layers/attention/conv_bert.h"

namespace {

int checks = 0;
int passed = 0;

void check(bool condition, const std::string& message) {
    ++checks;
    if (condition) {
        ++passed;
        std::cout << "[PASS] " << message << '\n';
    } else {
        std::cout << "[FAIL] " << message << '\n';
    }
}

void check_near(double actual, double expected, double tolerance,
                const std::string& message) {
    check(std::fabs(actual - expected) <= tolerance,
          message + " (actual=" + std::to_string(actual) +
          ", expected=" + std::to_string(expected) + ")");
}

bool throws_invalid_argument(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

Tensor make_input(size_t tokens, size_t d_model, double phase = 0.0) {
    Tensor x(tokens, d_model);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t c = 0; c < d_model; ++c) {
            const double z = static_cast<double>((t + 1) * (c + 2));
            x(t, c) = 0.31 * std::sin(0.47 * z + phase)
                    + 0.17 * std::cos(0.29 * (t + 2 * c + 1) - phase)
                    + 0.03 * static_cast<double>(t)
                    - 0.02 * static_cast<double>(c);
        }
    }
    return x;
}

Tensor make_target(size_t tokens, size_t d_model) {
    Tensor y(tokens, d_model);
    for (size_t t = 0; t < tokens; ++t) {
        for (size_t c = 0; c < d_model; ++c) {
            y(t, c) = 0.21 * std::cos(0.37 * static_cast<double>((t + 2) * (c + 1)))
                    - 0.08 * std::sin(0.51 * static_cast<double>(t + c + 1));
        }
    }
    return y;
}

double half_squared_loss(const Tensor& output, const Tensor& target) {
    double loss = 0.0;
    for (size_t i = 0; i < output.data.size(); ++i) {
        const double error = output.data[i] - target.data[i];
        loss += 0.5 * error * error;
    }
    return loss;
}

Tensor half_squared_loss_gradient(const Tensor& output, const Tensor& target) {
    Tensor grad(output.rows, output.cols);
    for (size_t i = 0; i < output.data.size(); ++i) {
        grad.data[i] = output.data[i] - target.data[i];
    }
    return grad;
}

bool all_finite(const Tensor& tensor) {
    for (double value : tensor.data) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

double max_abs_difference(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.rows != rhs.rows || lhs.cols != rhs.cols) return INFINITY;
    double result = 0.0;
    for (size_t i = 0; i < lhs.data.size(); ++i) {
        result = std::max(result, std::fabs(lhs.data[i] - rhs.data[i]));
    }
    return result;
}

void make_parameters_asymmetric(ConvBertLayer& layer) {
    auto params = layer.parameters();
    for (size_t p = 0; p < params.size(); ++p) {
        if (p == params.size() - 1) continue;  // alpha is set via its public API below
        Tensor& tensor = *params[p];
        for (size_t i = 0; i < tensor.data.size(); ++i) {
            const double z = static_cast<double>((p + 2) * (i + 3));
            tensor.data[i] = 0.19 * std::sin(0.31 * z)
                           + 0.07 * std::cos(0.17 * (z + p));
        }
    }
    layer.set_alpha(0.37);
}

struct GradientStats {
    bool close = true;
    double max_relative_error = 0.0;
    double max_absolute_error = 0.0;
};

void record_gradient(GradientStats& stats, double analytical, double numerical) {
    const double scale = std::max({1e-12, std::fabs(analytical), std::fabs(numerical)});
    const double absolute_error = std::fabs(analytical - numerical);
    const double relative_error = absolute_error / scale;
    stats.max_absolute_error = std::max(stats.max_absolute_error, absolute_error);
    stats.max_relative_error = std::max(stats.max_relative_error, relative_error);
    if (absolute_error > std::max(1e-7, 2e-4 * scale)) stats.close = false;
}

GradientStats input_gradient_stats(ConvBertLayer& layer, Tensor input,
                                   const Tensor& target, double epsilon) {
    Tensor output = layer.forward(input);
    layer.zero_grad();
    Tensor analytical = layer.backward(half_squared_loss_gradient(output, target), 0.0);

    GradientStats stats;
    for (size_t r = 0; r < input.rows; ++r) {
        for (size_t c = 0; c < input.cols; ++c) {
            const double original = input(r, c);
            input(r, c) = original + epsilon;
            const double loss_plus = half_squared_loss(layer.forward(input), target);
            input(r, c) = original - epsilon;
            const double loss_minus = half_squared_loss(layer.forward(input), target);
            input(r, c) = original;
            const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            record_gradient(stats, analytical(r, c), numerical);
        }
    }
    return stats;
}

GradientStats parameter_gradient_stats(ConvBertLayer& layer, const Tensor& input,
                                       const Tensor& target, size_t parameter_index,
                                       double epsilon) {
    Tensor output = layer.forward(input);
    layer.zero_grad();
    layer.backward(half_squared_loss_gradient(output, target), 0.0);

    auto params = layer.parameters();
    auto grads = layer.gradients();
    Tensor* parameter = params.at(parameter_index);
    Tensor analytical = grads.at(parameter_index)->clone();

    GradientStats stats;
    for (size_t i = 0; i < parameter->data.size(); ++i) {
        const double original = parameter->data[i];
        parameter->data[i] = original + epsilon;
        const double loss_plus = half_squared_loss(layer.forward(input), target);
        parameter->data[i] = original - epsilon;
        const double loss_minus = half_squared_loss(layer.forward(input), target);
        parameter->data[i] = original;
        const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        record_gradient(stats, analytical.data[i], numerical);
    }
    return stats;
}

void report_gradient_check(const std::string& name, const GradientStats& stats) {
    std::cout << "  " << name << ": max_rel=" << stats.max_relative_error
              << " max_abs=" << stats.max_absolute_error << '\n';
    check(stats.close, name + " analytical gradient matches centered finite difference");
}

void test_constructor_and_validation() {
    std::cout << "\n--- constructor and validation ---\n";
    ConvBertLayer layer(4, 3, 2);
    check(layer.d_model() == 4, "stores d_model");
    check(layer.kernel_size() == 3, "stores kernel_size");
    check(layer.num_heads() == 2, "stores num_heads");
    check(layer.head_dim() == 2, "derives head_dim");
    check_near(layer.alpha(), 0.5, 0.0, "default alpha balances branches");
    check(layer.name() == "ConvBertLayer", "reports layer name");

    layer.set_alpha(0.25);
    check_near(layer.alpha(), 0.25, 0.0, "set_alpha updates mixing coefficient");

    check(throws_invalid_argument([] { ConvBertLayer invalid(0, 3, 1); }),
          "rejects d_model=0");
    check(throws_invalid_argument([] { ConvBertLayer invalid(4, 0, 1); }),
          "rejects kernel_size=0");
    check(throws_invalid_argument([] { ConvBertLayer invalid(4, 2, 1); }),
          "rejects even kernel sizes");
    check(throws_invalid_argument([] { ConvBertLayer invalid(4, 3, 0); }),
          "rejects num_heads=0");
    check(throws_invalid_argument([] { ConvBertLayer invalid(5, 3, 2); }),
          "rejects non-divisible head layout");
    check(throws_invalid_argument([] { ConvBertLayer invalid(4, 3, 2, -0.1); }),
          "rejects alpha below zero");
    check(throws_invalid_argument([] { ConvBertLayer invalid(4, 3, 2, 1.1); }),
          "rejects alpha above one");
    check(throws_invalid_argument([&layer] { layer.set_alpha(-0.01); }),
          "set_alpha rejects alpha below zero");
    check(throws_invalid_argument([&layer] { layer.set_alpha(1.01); }),
          "set_alpha rejects alpha above one");

    auto params = layer.parameters();
    (*params.back())(0, 0) = 100.0;  // simulate an adaptive optimizer's direct update
    check(layer.alpha() >= 0.0 && layer.alpha() <= 1.0,
          "external optimizer updates cannot move effective alpha outside [0,1]");
}

void test_forward_and_branch_contract() {
    std::cout << "\n--- forward and branch contract ---\n";
    ConvBertLayer layer(4, 3, 2);
    make_parameters_asymmetric(layer);
    Tensor input = make_input(5, 4);

    Tensor mixed = layer.forward(input);
    check(mixed.rows == 5 && mixed.cols == 4, "forward preserves (tokens, d_model)");
    check(all_finite(mixed), "forward output is finite");

    const Tensor attention = layer.last_attention_output().clone();
    const Tensor convolution = layer.last_convolution_output().clone();
    check(max_abs_difference(attention, convolution) > 1e-6,
          "local convolution branch differs from global attention branch");

    Tensor expected(mixed.rows, mixed.cols);
    for (size_t i = 0; i < expected.data.size(); ++i) {
        expected.data[i] = layer.alpha() * attention.data[i]
                         + (1.0 - layer.alpha()) * convolution.data[i];
    }
    check(max_abs_difference(mixed, expected) < 1e-12,
          "forward is the documented alpha-weighted branch mixture");

    layer.set_alpha(1.0);
    Tensor pure_attention = layer.forward(input);
    check(max_abs_difference(pure_attention, layer.last_attention_output()) < 1e-12,
          "alpha=1 selects pure attention");

    layer.set_alpha(0.0);
    Tensor pure_convolution = layer.forward(input);
    check(max_abs_difference(pure_convolution, layer.last_convolution_output()) < 1e-12,
          "alpha=0 selects pure convolution");

    Tensor short_input = make_input(3, 4, 0.2);
    Tensor long_input = make_input(7, 4, -0.3);
    check(layer.forward(short_input).rows == 3 && layer.forward(long_input).rows == 7,
          "one layer supports dynamic sequence lengths");

    check(throws_invalid_argument([&layer] { layer.forward(Tensor(3, 5)); }),
          "forward rejects input with wrong feature dimension");
    check(throws_invalid_argument([&layer] { layer.forward(Tensor(0, 4)); }),
          "forward rejects an empty sequence");
}

void test_locality_of_convolution_branch() {
    std::cout << "\n--- convolution locality ---\n";
    ConvBertLayer layer(4, 3, 2);
    make_parameters_asymmetric(layer);
    layer.set_alpha(0.0);

    Tensor input = make_input(7, 4);
    Tensor baseline = layer.forward(input);
    input(3, 1) += 1.0;
    Tensor perturbed = layer.forward(input);

    bool outside_unchanged = true;
    bool inside_changed = false;
    for (size_t t = 0; t < input.rows; ++t) {
        for (size_t c = 0; c < input.cols; ++c) {
            const double delta = std::fabs(perturbed(t, c) - baseline(t, c));
            if (t < 2 || t > 4) outside_unchanged = outside_unchanged && delta < 1e-12;
            if (t >= 2 && t <= 4) inside_changed = inside_changed || delta > 1e-8;
        }
    }
    check(outside_unchanged, "kernel=3 convolution does not affect positions outside one-hop window");
    check(inside_changed, "kernel=3 convolution responds inside the local window");
}

void test_parameter_contract_and_zero_grad() {
    std::cout << "\n--- parameter contract and zero_grad ---\n";
    ConvBertLayer layer(4, 3, 2);
    make_parameters_asymmetric(layer);
    Tensor input = make_input(4, 4);
    Tensor target = make_target(4, 4);

    Tensor output = layer.forward(input);
    layer.zero_grad();
    layer.backward(half_squared_loss_gradient(output, target), 0.0);

    auto params = layer.parameters();
    auto grads = layer.gradients();
    check(params.size() == 13, "exposes 13 parameter tensors");
    check(params.size() == grads.size(), "parameter and gradient counts match");

    bool shapes_match = true;
    bool any_nonzero = false;
    for (size_t i = 0; i < params.size(); ++i) {
        shapes_match = shapes_match && params[i]->rows == grads[i]->rows
                                     && params[i]->cols == grads[i]->cols;
        for (double value : grads[i]->data) any_nonzero = any_nonzero || std::fabs(value) > 1e-14;
    }
    check(shapes_match, "all parameter and gradient shapes match");
    check(any_nonzero, "backward produces nonzero parameter gradients");

    layer.zero_grad();
    bool all_zero = true;
    for (Tensor* grad : layer.gradients()) {
        for (double value : grad->data) all_zero = all_zero && value == 0.0;
    }
    check(all_zero, "zero_grad clears every gradient exactly");
}

void test_gradients() {
    std::cout << "\n--- analytical gradient checks ---\n";
    constexpr size_t tokens = 4;
    constexpr size_t d_model = 4;
    constexpr double epsilon = 1e-5;
    Tensor input = make_input(tokens, d_model, 0.13);
    Tensor target = make_target(tokens, d_model);

    ConvBertLayer input_layer(d_model, 3, 2);
    make_parameters_asymmetric(input_layer);
    report_gradient_check("input", input_gradient_stats(input_layer, input, target, epsilon));

    const std::vector<std::pair<size_t, std::string>> parameter_groups = {
        {0, "query projection weights"},
        {2, "key projection weights"},
        {4, "value projection weights"},
        {6, "attention output projection weights"},
        {8, "GLU projection weights (linear and gate halves)"},
        {10, "depthwise convolution weights"},
        {11, "depthwise convolution bias"},
        {12, "alpha mixing scalar"},
    };
    for (const auto& group : parameter_groups) {
        ConvBertLayer layer(d_model, 3, 2);
        make_parameters_asymmetric(layer);
        report_gradient_check(group.second,
                              parameter_gradient_stats(layer, input, target,
                                                       group.first, epsilon));
    }
}

void test_training_and_updates() {
    std::cout << "\n--- training and updates ---\n";
    ConvBertLayer layer(4, 3, 2);
    make_parameters_asymmetric(layer);
    Tensor input = make_input(5, 4, -0.17);
    Tensor target = make_target(5, 4);

    const double initial_loss = half_squared_loss(layer.forward(input), target);
    for (int step = 0; step < 160; ++step) {
        layer.zero_grad();
        Tensor output = layer.forward(input);
        layer.backward(half_squared_loss_gradient(output, target), 0.0);
        layer.update_weights(0.01);
    }
    const double final_loss = half_squared_loss(layer.forward(input), target);
    std::cout << "  loss: " << initial_loss << " -> " << final_loss << '\n';
    check(final_loss < initial_loss * 0.7, "training reduces regression loss by at least 30%");

    bool finite = std::isfinite(layer.alpha());
    for (Tensor* parameter : layer.parameters()) finite = finite && all_finite(*parameter);
    check(finite, "training keeps all parameters finite");
    check(layer.alpha() >= 0.0 && layer.alpha() <= 1.0,
          "weight updates keep alpha in the mixing interval");

    ConvBertLayer default_kernel_layer(8, 7, 2);
    Tensor smoke_input = make_input(8, 8, 0.23);
    Tensor smoke_target = make_target(8, 8);
    Tensor smoke_output = default_kernel_layer.forward(smoke_input);
    default_kernel_layer.zero_grad();
    Tensor smoke_grad = default_kernel_layer.backward(
        half_squared_loss_gradient(smoke_output, smoke_target), 0.0);
    check(smoke_output.rows == 8 && smoke_output.cols == 8 && all_finite(smoke_output),
          "default kernel=7 multi-head forward is finite");
    check(smoke_grad.rows == 8 && smoke_grad.cols == 8 && all_finite(smoke_grad),
          "default kernel=7 multi-head backward is finite");
}

}  // namespace

int main() {
    std::cout << "=== ConvBERT-Style Mixed Attention Tests ===\n";
    std::cout.setf(std::ios::unitbuf);

    test_constructor_and_validation();
    test_forward_and_branch_contract();
    test_locality_of_convolution_branch();
    test_parameter_contract_and_zero_grad();
    test_gradients();
    test_training_and_updates();

    std::cout << "\n=== ConvBERT: " << passed << "/" << checks << " checks passed ===\n";
    return passed == checks ? 0 : 1;
}
