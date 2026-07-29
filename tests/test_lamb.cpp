// test_lamb.cpp — behavioral tests for canonical LAMB.
//
// Primary reference: You et al. (2019), Algorithm 2,
// "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
// (https://arxiv.org/abs/1904.00962).
//
// Per-parameter-tensor update tested here:
//   m_t = beta1*m_{t-1} + (1-beta1)*g_t
//   v_t = beta2*v_{t-1} + (1-beta2)*g_t^2
//   r_t = m_hat/(sqrt(v_hat)+epsilon)
//   u_t = r_t + weight_decay*w_t
//   q_t = ||w_t||/||u_t||, with q_t=1 on either zero norm
//   q_t = clamp(q_t, 1/gamma, gamma)  // repository extension
//   w_{t+1} = w_t - lr*q_t*u_t

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/lamb.h"
#include "nn/optimizers/scheduler.h"

using namespace std;

static int passed = 0;
static int failed = 0;

static bool check(const string& name, bool condition) {
    if (condition) {
        cout << "  [PASS] " << name << '\n';
        ++passed;
    } else {
        cout << "  [FAIL] " << name << '\n';
        ++failed;
    }
    return condition;
}

static bool near(double actual, double expected, double tol = 1e-12) {
    return std::abs(actual - expected) <= tol;
}

static double tensor_l1(const Tensor& tensor) {
    double total = 0.0;
    for (size_t row = 0; row < tensor.rows; ++row)
        for (size_t col = 0; col < tensor.cols; ++col)
            total += std::abs(tensor[row][col]);
    return total;
}

class ParameterLayer : public Layer {
public:
    Tensor parameter;
    Tensor gradient;

    ParameterLayer(size_t rows, size_t cols)
        : parameter(rows, cols), gradient(rows, cols) {
        parameter.fill(0.0);
        gradient.fill(0.0);
    }

    Tensor forward(const Tensor& input) override { return input.clone(); }
    Tensor backward(const Tensor& grad_output, double) override {
        return grad_output.clone();
    }
    void update_weights(double) override {}
    Tensor get_weights() const override { return parameter.clone(); }
    Tensor get_gradients() const override { return gradient.clone(); }
    vector<Tensor*> parameters() override { return {&parameter}; }
    vector<Tensor*> gradients() override { return {&gradient}; }
    void zero_grad() override { gradient.fill(0.0); }
};

class MalformedLayer : public Layer {
public:
    enum class Mode { COUNT_MISMATCH, SHAPE_MISMATCH };

    explicit MalformedLayer(Mode mode)
        : mode_(mode), parameter_(2, 2), gradient_(2, 2), wrong_gradient_(1, 2) {
        parameter_.fill(0.0);
        gradient_.fill(0.0);
        wrong_gradient_.fill(0.0);
    }

    Tensor forward(const Tensor& input) override { return input.clone(); }
    Tensor backward(const Tensor& grad_output, double) override {
        return grad_output.clone();
    }
    void update_weights(double) override {}
    Tensor get_weights() const override { return parameter_.clone(); }
    Tensor get_gradients() const override { return gradient_.clone(); }
    vector<Tensor*> parameters() override { return {&parameter_}; }
    vector<Tensor*> gradients() override {
        if (mode_ == Mode::COUNT_MISMATCH) return {};
        return {&wrong_gradient_};
    }
    void zero_grad() override { gradient_.fill(0.0); }

private:
    Mode mode_;
    Tensor parameter_;
    Tensor gradient_;
    Tensor wrong_gradient_;
};

static pair<double, double> run_deterministic_trajectory() {
    Model model;
    auto* layer = new ParameterLayer(2, 3);
    for (size_t row = 0; row < 2; ++row)
        for (size_t col = 0; col < 3; ++col)
            layer->parameter[row][col] = 0.2 + 0.1 * row - 0.03 * col;
    model.add_layer(layer);

    LAMB optimizer(0.02, 0.8, 0.9, 1e-6, 5.0, 0.01);
    for (int step = 0; step < 8; ++step) {
        for (size_t row = 0; row < 2; ++row) {
            for (size_t col = 0; col < 3; ++col) {
                layer->gradient[row][col] =
                    std::sin(0.3 * step + 0.7 * row - 0.2 * col);
            }
        }
        optimizer.step(model);
    }

    const double first = layer->parameter[0][0] +
                         2.0 * layer->parameter[0][1] +
                         3.0 * layer->parameter[1][2];
    const double second = layer->parameter[0][2] +
                          5.0 * layer->parameter[1][0];
    return {first, second};
}

int main() {
    cout << setprecision(15);
    cout << "=== LAMB Optimizer Tests ===\n\n";

    cout << "T1: defaults and public contract\n";
    {
        LAMB optimizer;
        check("default lr = 1e-3", near(optimizer.get_lr(), 1e-3));
        check("default beta1 = 0.9", near(optimizer.get_beta1(), 0.9));
        check("default beta2 = 0.999", near(optimizer.get_beta2(), 0.999));
        check("default epsilon = 1e-6", near(optimizer.get_epsilon(), 1e-6));
        check("default trust gamma = 10", near(optimizer.get_trust_ratio_gamma(), 10.0));
        check("default weight decay = 0", near(optimizer.get_weight_decay(), 0.0));
        check("step counter starts at 1", optimizer.get_step() == 1);
        check("handles_weight_decay is true", optimizer.handles_weight_decay());

        LAMB custom(0.2, 0.7, 0.8, 1e-5, 4.0, 0.03);
        check("custom lr stored in Optimizer base",
              near(custom.get_lr(), 0.2) && near(custom.Optimizer::lr, 0.2));
        check("custom beta1 stored", near(custom.get_beta1(), 0.7));
        check("custom beta2 stored", near(custom.get_beta2(), 0.8));
        check("custom epsilon stored", near(custom.get_epsilon(), 1e-5));
        check("custom trust gamma stored", near(custom.get_trust_ratio_gamma(), 4.0));
        check("custom weight decay stored", near(custom.get_weight_decay(), 0.03));
    }
    cout << '\n';

    cout << "T2: constructor and setter validation\n";
    {
        auto constructor_throws = [](double lr, double b1, double b2,
                                     double eps, double gamma, double wd) {
            try {
                LAMB bad(lr, b1, b2, eps, gamma, wd);
                (void)bad;
                return false;
            } catch (const invalid_argument&) {
                return true;
            }
        };
        check("negative lr throws", constructor_throws(-0.1, 0.9, 0.999, 1e-6, 10, 0));
        check("negative beta1 throws", constructor_throws(0.1, -0.1, 0.999, 1e-6, 10, 0));
        check("beta1 >= 1 throws", constructor_throws(0.1, 1.0, 0.999, 1e-6, 10, 0));
        check("negative beta2 throws", constructor_throws(0.1, 0.9, -0.1, 1e-6, 10, 0));
        check("beta2 >= 1 throws", constructor_throws(0.1, 0.9, 1.0, 1e-6, 10, 0));
        check("zero epsilon throws", constructor_throws(0.1, 0.9, 0.999, 0, 10, 0));
        check("trust gamma below one throws", constructor_throws(0.1, 0.9, 0.999, 1e-6, 0.5, 0));
        check("negative weight decay throws", constructor_throws(0.1, 0.9, 0.999, 1e-6, 10, -0.1));

        LAMB optimizer;
        bool threw = false;
        try { optimizer.set_beta1(1.0); }
        catch (const invalid_argument&) { threw = true; }
        check("set_beta1 validates", threw);
        threw = false;
        try { optimizer.set_epsilon(0.0); }
        catch (const invalid_argument&) { threw = true; }
        check("set_epsilon validates", threw);
        threw = false;
        try { optimizer.set_trust_ratio_gamma(0.9); }
        catch (const invalid_argument&) { threw = true; }
        check("set_trust_ratio_gamma validates", threw);
        threw = false;
        try { optimizer.set_weight_decay(-1e-3); }
        catch (const invalid_argument&) { threw = true; }
        check("set_weight_decay validates", threw);

        optimizer.set_lr(0.0);
        optimizer.set_beta1(0.0);
        optimizer.set_beta2(0.0);
        optimizer.set_epsilon(1e-8);
        optimizer.set_trust_ratio_gamma(1.0);
        optimizer.set_weight_decay(0.2);
        check("valid setter boundaries are accepted",
              near(optimizer.get_lr(), 0.0) && near(optimizer.get_beta1(), 0.0) &&
              near(optimizer.get_beta2(), 0.0) && near(optimizer.get_epsilon(), 1e-8) &&
              near(optimizer.get_trust_ratio_gamma(), 1.0) &&
              near(optimizer.get_weight_decay(), 0.2));
    }
    cout << '\n';

    cout << "T3: lazy state mirrors parameter shapes\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);  // weights (2,3), bias (1,2)
        layer->weights.fill(0.0);
        layer->bias.fill(0.0);
        layer->zero_grad();
        model.add_layer(layer);
        LAMB optimizer;

        check("state absent before step", !optimizer.has_state(layer));
        Tensor state;
        check("m accessor misses before step", !optimizer.get_m(layer, 0, state));
        check("v accessor misses before step", !optimizer.get_v(layer, 0, state));
        optimizer.step(model);
        check("state present after step", optimizer.has_state(layer));
        check("weight m accessor succeeds", optimizer.get_m(layer, 0, state));
        check("weight m shape is (2,3)", state.rows == 2 && state.cols == 3);
        check("weight m starts at zero", near(tensor_l1(state), 0.0));
        check("bias v accessor succeeds", optimizer.get_v(layer, 1, state));
        check("bias v shape is (1,2)", state.rows == 1 && state.cols == 2);
        check("out-of-range state accessor fails", !optimizer.get_m(layer, 9, state));
        check("step counter advances to 2", optimizer.get_step() == 2);
    }
    cout << '\n';

    cout << "T4: canonical trust ratio uses the Adam direction\n";
    {
        Model model;
        auto* layer = new ParameterLayer(2, 2);
        layer->parameter[0][0] = 3.0;
        layer->parameter[0][1] = 4.0;  // ||w|| = 5
        layer->gradient[0][0] = 6.0;
        layer->gradient[0][1] = 8.0;
        model.add_layer(layer);

        const double eps = 1.0;
        const double u0 = 6.0 / 7.0;
        const double u1 = 8.0 / 9.0;
        const double expected_q = 5.0 / std::sqrt(u0 * u0 + u1 * u1);
        LAMB optimizer(0.1, 0.0, 0.0, eps, 100.0, 0.0);
        optimizer.step(model);

        double actual_q = 0.0;
        check("trust-ratio accessor succeeds",
              optimizer.get_last_trust_ratio(layer, 0, actual_q));
        check("q = ||w||/||m_hat/(sqrt(v_hat)+eps)||",
              near(actual_q, expected_q, 1e-12));
        check("first coordinate uses canonical q-scaled Adam direction",
              near(layer->parameter[0][0], 3.0 - 0.1 * expected_q * u0, 1e-12));
        check("second coordinate uses canonical q-scaled Adam direction",
              near(layer->parameter[0][1], 4.0 - 0.1 * expected_q * u1, 1e-12));
        check("signature differs from raw-gradient ratio q=0.5",
              std::abs(actual_q - 0.5) > 1.0);
    }
    cout << '\n';

    cout << "T5: weight decay participates in the adapted update\n";
    {
        Model model;
        auto* layer = new ParameterLayer(1, 2);
        layer->parameter[0][0] = 3.0;
        layer->parameter[0][1] = 4.0;
        model.add_layer(layer);

        // g=0 -> r=0; u=0.5*w; ||u||=2.5; q=5/2.5=2.
        LAMB optimizer(0.1, 0.0, 0.0, 1.0, 100.0, 0.5);
        optimizer.step(model);
        double q = 0.0;
        optimizer.get_last_trust_ratio(layer, 0, q);
        check("decay-only trust ratio is two", near(q, 2.0));
        check("decay-only first coordinate becomes 0.9*w", near(layer->parameter[0][0], 2.7));
        check("decay-only second coordinate becomes 0.9*w", near(layer->parameter[0][1], 3.6));
    }
    cout << '\n';

    cout << "T6: trust-ratio clamp boundaries\n";
    {
        Model upper_model;
        auto* upper = new ParameterLayer(1, 2);
        upper->parameter[0][0] = 10.0;
        upper->gradient[0][0] = 1.0;
        upper_model.add_layer(upper);
        LAMB upper_optimizer(0.1, 0.0, 0.0, 1e-12, 2.0, 0.0);
        upper_optimizer.step(upper_model);
        double q = 0.0;
        upper_optimizer.get_last_trust_ratio(upper, 0, q);
        check("upper trust ratio clamps to gamma", near(q, 2.0));
        check("upper-clamped update uses q=2", near(upper->parameter[0][0], 9.8, 1e-11));

        Model lower_model;
        auto* lower = new ParameterLayer(1, 2);
        lower->parameter[0][0] = 1.0;
        lower->gradient[0][0] = 1.0;
        lower_model.add_layer(lower);
        // r ~= 1 and wd*w = 9, so raw q ~= 0.1 and clamps to 1/gamma=0.5.
        LAMB lower_optimizer(0.1, 0.0, 0.0, 1e-12, 2.0, 9.0);
        lower_optimizer.step(lower_model);
        lower_optimizer.get_last_trust_ratio(lower, 0, q);
        check("lower trust ratio clamps to reciprocal gamma", near(q, 0.5));
        check("lower-clamped update uses q=0.5", near(lower->parameter[0][0], 0.5, 1e-11));
    }
    cout << '\n';

    cout << "T7: zero-norm fallbacks remain finite\n";
    {
        Model zero_weight_model;
        auto* zero_weight = new ParameterLayer(1, 1);
        zero_weight->gradient[0][0] = 2.0;
        zero_weight_model.add_layer(zero_weight);
        LAMB optimizer(0.1, 0.0, 0.0, 1.0, 10.0, 0.0);
        optimizer.step(zero_weight_model);
        double q = 0.0;
        optimizer.get_last_trust_ratio(zero_weight, 0, q);
        check("zero weight norm falls back to q=1", near(q, 1.0));
        check("zero weight still takes finite step", near(zero_weight->parameter[0][0], -0.1 * (2.0 / 3.0)));

        Model zero_update_model;
        auto* zero_update = new ParameterLayer(1, 1);
        zero_update->parameter[0][0] = 2.0;
        zero_update_model.add_layer(zero_update);
        LAMB optimizer2(0.1, 0.0, 0.0, 1.0, 10.0, 0.0);
        optimizer2.step(zero_update_model);
        optimizer2.get_last_trust_ratio(zero_update, 0, q);
        check("zero update norm falls back to q=1", near(q, 1.0));
        check("zero update leaves parameter unchanged", near(zero_update->parameter[0][0], 2.0));
        check("zero-update result is finite", std::isfinite(zero_update->parameter[0][0]));
    }
    cout << '\n';

    cout << "T8: two-step moments and bias correction\n";
    {
        Model model;
        auto* layer = new ParameterLayer(1, 1);
        model.add_layer(layer);
        LAMB optimizer(0.1, 0.5, 0.5, 1.0, 1.0, 0.0);  // gamma=1 forces q=1

        layer->gradient[0][0] = 1.0;
        optimizer.step(model);
        check("step-one parameter uses bias-corrected direction", near(layer->parameter[0][0], -0.05));
        Tensor m, v;
        optimizer.get_m(layer, 0, m);
        optimizer.get_v(layer, 0, v);
        check("step-one m = 0.5", near(m[0][0], 0.5));
        check("step-one v = 0.5", near(v[0][0], 0.5));

        layer->gradient[0][0] = 3.0;
        optimizer.step(model);
        optimizer.get_m(layer, 0, m);
        optimizer.get_v(layer, 0, v);
        const double expected_m = 1.75;
        const double expected_v = 4.75;
        const double expected_direction =
            (expected_m / 0.75) / (std::sqrt(expected_v / 0.75) + 1.0);
        check("step-two m recurrence", near(m[0][0], expected_m));
        check("step-two v recurrence", near(v[0][0], expected_v));
        check("step-two parameter uses t=2 bias correction",
              near(layer->parameter[0][0], -0.05 - 0.1 * expected_direction));
        check("beta1 correction is 0.75", near(optimizer.beta1_corr, 0.75));
        check("beta2 correction is 0.75", near(optimizer.beta2_corr, 0.75));
        check("step counter advances to 3", optimizer.get_step() == 3);
    }
    cout << '\n';

    cout << "T9: LR scheduler compatibility\n";
    {
        Model model;
        auto* layer = new ParameterLayer(1, 1);
        layer->gradient[0][0] = 1.0;
        model.add_layer(layer);
        LAMB optimizer(0.1, 0.0, 0.0, 1.0, 1.0, 0.0);
        PolynomialLR scheduler(0.1, 2, 1.0, 0.0);
        scheduler.set_optimizer(&optimizer);
        scheduler.step(model);  // epoch 1 -> lr=0.05
        check("scheduler updates LAMB get_lr", near(optimizer.get_lr(), 0.05));
        optimizer.step(model);
        check("step consumes scheduled lr", near(layer->parameter[0][0], -0.025));
    }
    cout << '\n';

    cout << "T10: multi-layer state isolation\n";
    {
        Model model;
        auto* first = new ParameterLayer(1, 2);
        auto* second = new ParameterLayer(2, 1);
        first->gradient[0][0] = 1.0;
        second->gradient[0][0] = 3.0;
        model.add_layer(first);
        model.add_layer(second);
        LAMB optimizer(0.1, 0.5, 0.5, 1.0, 10.0, 0.0);
        optimizer.step(model);
        check("first layer has state", optimizer.has_state(first));
        check("second layer has state", optimizer.has_state(second));
        Tensor first_m, second_m;
        optimizer.get_m(first, 0, first_m);
        optimizer.get_m(second, 0, second_m);
        check("first state shape is independent", first_m.rows == 1 && first_m.cols == 2);
        check("second state shape is independent", second_m.rows == 2 && second_m.cols == 1);
        check("layer moments contain distinct values", !near(first_m[0][0], second_m[0][0]));
    }
    cout << '\n';

    cout << "T11: malformed layers fail fast\n";
    {
        bool threw = false;
        try {
            Model model;
            model.add_layer(new MalformedLayer(MalformedLayer::Mode::COUNT_MISMATCH));
            LAMB optimizer;
            optimizer.step(model);
        } catch (const runtime_error&) {
            threw = true;
        }
        check("parameter-gradient count mismatch throws", threw);

        threw = false;
        try {
            Model model;
            model.add_layer(new MalformedLayer(MalformedLayer::Mode::SHAPE_MISMATCH));
            LAMB optimizer;
            optimizer.step(model);
        } catch (const runtime_error&) {
            threw = true;
        }
        check("parameter-gradient shape mismatch throws", threw);
    }
    cout << '\n';

    cout << "T12: deterministic trajectory\n";
    {
        const auto first = run_deterministic_trajectory();
        const auto second = run_deterministic_trajectory();
        check("first trajectory signature is bit-identical", near(first.first, second.first, 0.0));
        check("second trajectory signature is bit-identical", near(first.second, second.second, 0.0));
    }
    cout << '\n';

    cout << "T13: gradients are cleared after step\n";
    {
        Model model;
        auto* layer = new ParameterLayer(2, 2);
        layer->gradient[0][0] = 1.0;
        layer->gradient[1][1] = -2.0;
        model.add_layer(layer);
        LAMB optimizer;
        optimizer.step(model);
        check("all consumed gradients are zero", near(tensor_l1(layer->gradient), 0.0));
    }
    cout << '\n';

    cout << "T14: end-to-end training reduces asymmetric regression loss\n";
    {
        std::srand(1234);
        Model model;
        Dense* layer = new Dense(2, 2);
        model.add_layer(layer);
        LAMB optimizer(0.03, 0.9, 0.999, 1e-6, 10.0, 0.0);

        Tensor input(6, 2);
        const double values[12] = {
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            -1.0, 0.5, 0.5, -1.0, 2.0, -0.5
        };
        for (size_t row = 0; row < 6; ++row)
            for (size_t col = 0; col < 2; ++col)
                input[row][col] = values[2 * row + col];

        Tensor target(6, 2);
        for (size_t row = 0; row < 6; ++row) {
            target[row][0] = 2.0 * input[row][0] - input[row][1] + 0.5;
            target[row][1] = -input[row][0] + 3.0 * input[row][1] - 0.25;
        }

        auto loss = [&]() {
            Tensor prediction = layer->forward(input);
            double total = 0.0;
            for (size_t row = 0; row < prediction.rows; ++row) {
                for (size_t col = 0; col < prediction.cols; ++col) {
                    const double error = prediction[row][col] - target[row][col];
                    total += error * error;
                }
            }
            return total / static_cast<double>(prediction.rows * prediction.cols);
        };

        const double initial_loss = loss();
        for (int step = 0; step < 300; ++step) {
            Tensor prediction = layer->forward(input);
            Tensor grad(prediction.rows, prediction.cols);
            const double scale = 2.0 / static_cast<double>(prediction.rows * prediction.cols);
            for (size_t row = 0; row < prediction.rows; ++row)
                for (size_t col = 0; col < prediction.cols; ++col)
                    grad[row][col] = scale * (prediction[row][col] - target[row][col]);
            layer->backward(grad, 0.0);
            optimizer.step(model);
        }
        const double final_loss = loss();
        cout << "  loss: " << initial_loss << " -> " << final_loss << '\n';
        check("training loss stays finite", std::isfinite(final_loss));
        check("LAMB reduces loss by at least 95 percent", final_loss < initial_loss * 0.05);
    }
    cout << '\n';

    cout << "=== LAMB: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
