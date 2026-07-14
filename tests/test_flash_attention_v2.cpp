#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/layers/attention/flash_attention.h"
#include "nn/layers/attention/flash_attention_v2.h"

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

Tensor make_feature_major(size_t d_model, size_t seq_len, double phase = 0.0) {
    Tensor x(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f) {
        for (size_t t = 0; t < seq_len; ++t) {
            const double z = static_cast<double>((f + 2) * (t + 3));
            x(f, t) = 0.37 * std::sin(0.31 * z + phase)
                    + 0.19 * std::cos(0.23 * (2 * f + t + 1) - phase)
                    + 0.025 * static_cast<double>(t)
                    - 0.018 * static_cast<double>(f);
        }
    }
    return x;
}

Tensor make_target(size_t d_model, size_t seq_len) {
    Tensor y(d_model, seq_len);
    for (size_t f = 0; f < d_model; ++f) {
        for (size_t t = 0; t < seq_len; ++t) {
            y(f, t) = 0.24 * std::cos(0.41 * static_cast<double>((f + 1) * (t + 2)))
                    - 0.11 * std::sin(0.29 * static_cast<double>(f + 2 * t + 1));
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

void make_parameters_asymmetric(FlashAttentionV2Layer& layer) {
    auto params = layer.parameters();
    for (size_t p = 0; p < params.size(); ++p) {
        Tensor& tensor = *params[p];
        for (size_t i = 0; i < tensor.data.size(); ++i) {
            const double z = static_cast<double>((p + 2) * (i + 3));
            tensor.data[i] = 0.21 * std::sin(0.27 * z)
                           + 0.09 * std::cos(0.19 * (z + 2 * p + 1));
        }
    }
}

void copy_parameters(const FlashAttentionV2Layer& source, FlashAttentionV2Layer& target) {
    target.W_q = source.W_q.clone();
    target.W_k = source.W_k.clone();
    target.W_v = source.W_v.clone();
    target.W_o = source.W_o.clone();
}

void copy_parameters_to_legacy(const FlashAttentionV2Layer& source,
                               FlashAttentionLayer& target) {
    target.W_q = source.W_q.clone();
    target.W_k = source.W_k.clone();
    target.W_v = source.W_v.clone();
    target.W_o = source.W_o.clone();
}

Tensor dense_reference(const FlashAttentionV2Layer& layer, const Tensor& input,
                       bool causal) {
    const size_t d_model = layer.d_model();
    const size_t num_heads = layer.num_heads();
    const size_t head_dim = layer.head_dim();
    const size_t seq_len = input.cols;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    Tensor x(seq_len, d_model);
    for (size_t t = 0; t < seq_len; ++t)
        for (size_t f = 0; f < d_model; ++f)
            x(t, f) = input(f, t);

    Tensor q(seq_len, d_model), k(seq_len, d_model), v(seq_len, d_model);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t j = 0; j < d_model; ++j) {
            double q_value = 0.0, k_value = 0.0, v_value = 0.0;
            for (size_t f = 0; f < d_model; ++f) {
                q_value += x(t, f) * layer.W_q(f, j);
                k_value += x(t, f) * layer.W_k(f, j);
                v_value += x(t, f) * layer.W_v(f, j);
            }
            q(t, j) = q_value;
            k(t, j) = k_value;
            v(t, j) = v_value;
        }
    }

    Tensor context(seq_len, d_model);
    context.fill(0.0);
    for (size_t h = 0; h < num_heads; ++h) {
        const size_t offset = h * head_dim;
        for (size_t query = 0; query < seq_len; ++query) {
            const size_t key_count = causal ? query + 1 : seq_len;
            std::vector<double> scores(key_count, 0.0);
            double row_max = -INFINITY;
            for (size_t key = 0; key < key_count; ++key) {
                for (size_t d = 0; d < head_dim; ++d)
                    scores[key] += q(query, offset + d) * k(key, offset + d);
                scores[key] *= scale;
                row_max = std::max(row_max, scores[key]);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - row_max);
                denominator += score;
            }
            for (size_t d = 0; d < head_dim; ++d) {
                for (size_t key = 0; key < key_count; ++key)
                    context(query, offset + d) +=
                        (scores[key] / denominator) * v(key, offset + d);
            }
        }
    }

    Tensor output(d_model, seq_len);
    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t j = 0; j < d_model; ++j) {
            double value = 0.0;
            for (size_t f = 0; f < d_model; ++f)
                value += context(t, f) * layer.W_o(f, j);
            output(j, t) = value;
        }
    }
    return output;
}

struct GradientStats {
    bool close = true;
    double max_relative_error = 0.0;
    double max_absolute_error = 0.0;
};

void record_gradient(GradientStats& stats, double analytical, double numerical) {
    const double scale = std::max({1e-12, std::fabs(analytical), std::fabs(numerical)});
    const double absolute_error = std::fabs(analytical - numerical);
    stats.max_absolute_error = std::max(stats.max_absolute_error, absolute_error);
    stats.max_relative_error = std::max(stats.max_relative_error, absolute_error / scale);
    if (absolute_error > std::max(1e-7, 2e-4 * scale)) stats.close = false;
}

GradientStats input_gradient_stats(FlashAttentionV2Layer& layer, Tensor input,
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
            record_gradient(stats, analytical(r, c),
                            (loss_plus - loss_minus) / (2.0 * epsilon));
        }
    }
    return stats;
}

GradientStats parameter_gradient_stats(FlashAttentionV2Layer& layer,
                                       const Tensor& input,
                                       const Tensor& target,
                                       size_t parameter_index,
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
        record_gradient(stats, analytical.data[i],
                        (loss_plus - loss_minus) / (2.0 * epsilon));
    }
    return stats;
}

void report_gradient_check(const std::string& name, const GradientStats& stats) {
    std::cout << "  " << name << ": max_rel=" << stats.max_relative_error
              << " max_abs=" << stats.max_absolute_error << '\n';
    check(stats.close, name + " matches centered finite differences");
}

void test_constructor_and_validation() {
    std::cout << "\n--- constructor and validation ---\n";
    FlashAttentionV2Layer layer(4, 2, 2, 3, true);
    check(layer.d_model() == 4, "stores d_model");
    check(layer.num_heads() == 2, "stores num_heads");
    check(layer.head_dim() == 2, "derives head_dim");
    check(layer.query_block_size() == 2, "stores query block size");
    check(layer.key_block_size() == 3, "stores key/value block size");
    check(layer.causal(), "stores causal mode");
    check(layer.name() == "FlashAttentionV2Layer", "reports layer name");
    check(throws_invalid_argument([] { FlashAttentionV2Layer x(0, 1); }), "rejects d_model=0");
    check(throws_invalid_argument([] { FlashAttentionV2Layer x(4, 0); }), "rejects num_heads=0");
    check(throws_invalid_argument([] { FlashAttentionV2Layer x(5, 2); }), "rejects non-divisible head layout");
    check(throws_invalid_argument([] { FlashAttentionV2Layer x(4, 2, 0, 2); }), "rejects query_block_size=0");
    check(throws_invalid_argument([] { FlashAttentionV2Layer x(4, 2, 2, 0); }), "rejects key_block_size=0");
    check(throws_invalid_argument([&layer] { layer.forward(Tensor(5, 3)); }), "forward rejects input rows different from d_model");
    check(throws_invalid_argument([&layer] { layer.forward(Tensor(4, 0)); }), "forward rejects an empty sequence");
    check(throws_invalid_argument([&layer] { layer.backward(Tensor(4, 3), 0.0); }), "backward rejects calls before forward");
}

void test_forward_reference_and_legacy_parity() {
    std::cout << "\n--- forward reference and FA-1 parity ---\n";
    const std::vector<std::pair<size_t, size_t>> cases = {{4, 1}, {8, 2}};
    const std::vector<size_t> seq_lengths = {5, 7};
    for (size_t index = 0; index < cases.size(); ++index) {
        const size_t d_model = cases[index].first;
        const size_t num_heads = cases[index].second;
        const size_t seq_len = seq_lengths[index];
        FlashAttentionV2Layer v2(d_model, num_heads, 3, 2, true);
        make_parameters_asymmetric(v2);
        Tensor input = make_feature_major(d_model, seq_len, 0.11 * index);
        Tensor output = v2.forward(input);
        check(output.rows == d_model && output.cols == seq_len,
              "forward preserves feature-major shape for case " + std::to_string(index));
        check(all_finite(output), "forward is finite for case " + std::to_string(index));
        check(max_abs_difference(output, dense_reference(v2, input, true)) < 1e-10,
              "forward matches independent dense causal reference for case " + std::to_string(index));
        FlashAttentionLayer legacy(d_model, num_heads);
        copy_parameters_to_legacy(v2, legacy);
        check(max_abs_difference(output, legacy.forward(input)) < 1e-9,
              "forward matches legacy FA-1 for case " + std::to_string(index));
        check(v2.last_logsumexp().rows == num_heads && v2.last_logsumexp().cols == seq_len,
              "forward caches one log-sum-exp scalar per head and query");
    }
    FlashAttentionV2Layer long_v2(4, 2, 17, 13, true);
    make_parameters_asymmetric(long_v2);
    Tensor input = make_feature_major(4, 65, 0.27);
    check(max_abs_difference(long_v2.forward(input), dense_reference(long_v2, input, true)) < 1e-9,
          "forward handles sequence longer than one legacy tile");
}

void test_causal_prefix_and_noncausal_mode() {
    std::cout << "\n--- causal and non-causal behavior ---\n";
    Tensor input = make_feature_major(4, 6, 0.13);
    Tensor perturbed = input.clone();
    for (size_t f = 0; f < 4; ++f) perturbed(f, 5) += 1.1 + 0.2 * f;
    FlashAttentionV2Layer causal(4, 2, 2, 3, true);
    make_parameters_asymmetric(causal);
    Tensor before = causal.forward(input);
    Tensor after = causal.forward(perturbed);
    bool prefix_unchanged = true;
    for (size_t f = 0; f < 4; ++f)
        for (size_t t = 0; t < 5; ++t)
            prefix_unchanged = prefix_unchanged && std::fabs(before(f, t) - after(f, t)) < 1e-12;
    check(prefix_unchanged, "future-token perturbation cannot change causal prefix outputs");
    FlashAttentionV2Layer noncausal(4, 2, 2, 3, false);
    copy_parameters(causal, noncausal);
    Tensor nc_before = noncausal.forward(input);
    Tensor nc_after = noncausal.forward(perturbed);
    bool prefix_changed = false;
    for (size_t f = 0; f < 4; ++f)
        for (size_t t = 0; t < 5; ++t)
            prefix_changed = prefix_changed || std::fabs(nc_before(f, t) - nc_after(f, t)) > 1e-8;
    check(prefix_changed, "non-causal mode lets later tokens influence earlier queries");
    check(max_abs_difference(nc_before, dense_reference(noncausal, input, false)) < 1e-10,
          "non-causal forward matches independent dense reference");
}

void test_parameter_contract_and_errors() {
    std::cout << "\n--- parameter contract and error handling ---\n";
    FlashAttentionV2Layer layer(4, 2, 2, 3, true);
    make_parameters_asymmetric(layer);
    Tensor input = make_feature_major(4, 5);
    Tensor target = make_target(4, 5);
    Tensor output = layer.forward(input);
    check(throws_invalid_argument([&layer] { layer.backward(Tensor(4, 4), 0.0); }),
          "backward rejects a gradient with the wrong cached shape");
    layer.zero_grad();
    layer.backward(half_squared_loss_gradient(output, target), 0.0);
    auto params = layer.parameters();
    auto grads = layer.gradients();
    check(params.size() == 4, "exposes four projection parameter tensors");
    check(params.size() == grads.size(), "parameter and gradient counts match");
    bool shapes_match = true, all_nonzero = true;
    for (size_t i = 0; i < params.size(); ++i) {
        shapes_match = shapes_match && params[i]->rows == grads[i]->rows && params[i]->cols == grads[i]->cols;
        double norm = 0.0;
        for (double value : grads[i]->data) norm += value * value;
        all_nonzero = all_nonzero && norm > 1e-16;
    }
    check(shapes_match, "all parameter and gradient shapes match");
    check(all_nonzero, "backward exercises Q, K, V, and O parameter paths");
    Tensor old_wq = layer.W_q.clone();
    layer.update_weights(0.01);
    check(max_abs_difference(old_wq, layer.W_q) > 1e-12, "update_weights moves a parameter after backward");
    layer.zero_grad();
    bool all_zero = true;
    for (Tensor* grad : layer.gradients())
        for (double value : grad->data) all_zero = all_zero && value == 0.0;
    check(all_zero, "zero_grad clears every gradient exactly");
}

void test_gradients() {
    std::cout << "\n--- analytical gradient checks ---\n";
    constexpr double epsilon = 1e-5;
    Tensor input = make_feature_major(4, 5, 0.19);
    Tensor target = make_target(4, 5);
    FlashAttentionV2Layer input_layer(4, 2, 2, 3, true);
    make_parameters_asymmetric(input_layer);
    report_gradient_check("input (Q + K + V paths)", input_gradient_stats(input_layer, input, target, epsilon));
    const std::vector<std::string> names = {"W_q", "W_k", "W_v", "W_o"};
    for (size_t i = 0; i < names.size(); ++i) {
        FlashAttentionV2Layer layer(4, 2, 2, 3, true);
        make_parameters_asymmetric(layer);
        report_gradient_check(names[i], parameter_gradient_stats(layer, input, target, i, epsilon));
    }
}

void test_block_size_invariance() {
    std::cout << "\n--- block-size invariance ---\n";
    Tensor input = make_feature_major(4, 7, -0.17);
    Tensor target = make_target(4, 7);
    FlashAttentionV2Layer baseline(4, 2, 1, 1, true), uneven(4, 2, 2, 3, true), oversized(4, 2, 32, 32, true);
    make_parameters_asymmetric(baseline);
    copy_parameters(baseline, uneven);
    copy_parameters(baseline, oversized);
    Tensor out_baseline = baseline.forward(input);
    Tensor out_uneven = uneven.forward(input);
    Tensor out_oversized = oversized.forward(input);
    check(max_abs_difference(out_baseline, out_uneven) < 1e-10,
          "Q=1/K=1 and uneven blocks produce identical output");
    check(max_abs_difference(out_baseline, out_oversized) < 1e-10,
          "single-block and tiled kernels produce identical output");
    Tensor grad_output = half_squared_loss_gradient(out_baseline, target);
    baseline.zero_grad(); uneven.zero_grad(); oversized.zero_grad();
    Tensor dx0 = baseline.backward(grad_output, 0.0);
    Tensor dx1 = uneven.backward(grad_output, 0.0);
    Tensor dx2 = oversized.backward(grad_output, 0.0);
    bool equal = max_abs_difference(dx0, dx1) < 1e-10 && max_abs_difference(dx0, dx2) < 1e-10;
    auto g0 = baseline.gradients(); auto g1 = uneven.gradients(); auto g2 = oversized.gradients();
    for (size_t i = 0; i < g0.size(); ++i)
        equal = equal && max_abs_difference(*g0[i], *g1[i]) < 1e-10
                      && max_abs_difference(*g0[i], *g2[i]) < 1e-10;
    check(equal, "block partition changes scheduling but not gradients");
}

void test_training() {
    std::cout << "\n--- training smoke test ---\n";
    FlashAttentionV2Layer layer(4, 2, 2, 3, true);
    make_parameters_asymmetric(layer);
    Tensor input = make_feature_major(4, 5, 0.07);
    Tensor target = make_target(4, 5);
    const double initial = half_squared_loss(layer.forward(input), target);
    for (int step = 0; step < 240; ++step) {
        layer.zero_grad();
        Tensor output = layer.forward(input);
        layer.backward(half_squared_loss_gradient(output, target), 0.0);
        layer.update_weights(0.02);
    }
    const double final = half_squared_loss(layer.forward(input), target);
    std::cout << "  loss: " << initial << " -> " << final << '\n';
    check(final < initial * 0.7, "training reduces fixed regression loss by at least 30%");
    bool finite = std::isfinite(final);
    for (Tensor* parameter : layer.parameters()) finite = finite && all_finite(*parameter);
    check(finite, "training keeps outputs and parameters finite");
}

}  // namespace

int main() {
    std::cout << "=== FlashAttention-2 Work-Partitioning Tests ===\n";
    std::cout.setf(std::ios::unitbuf);
    test_constructor_and_validation();
    test_forward_reference_and_legacy_parity();
    test_causal_prefix_and_noncausal_mode();
    test_parameter_contract_and_errors();
    test_gradients();
    test_block_size_invariance();
    test_training();
    std::cout << "\n=== FlashAttention-2: " << passed << "/" << checks << " checks passed ===\n";
    return passed == checks ? 0 : 1;
}
