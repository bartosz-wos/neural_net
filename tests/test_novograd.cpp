// test_novograd.cpp — behavioral tests for NovoGrad.
//
// Reference: Ginsburg et al. (2020),
// "Training Deep Networks with Stochastic Gradient Normalized by Layerwise
// Adaptive Second Moments" (https://arxiv.org/abs/1905.11286).
//
// Per-parameter-tensor update tested here:
//   s_t = ||g_t||_2^2
//   v_1 = s_1; v_t = beta2*v_{t-1} + (1-beta2)*s_t for t > 1
//   u_t = g_t/(sqrt(v_t)+epsilon) + weight_decay*w_t
//   m_t = beta1*m_{t-1} + (grad_averaging ? 1-beta1 : 1)*u_t
//   w_{t+1} = w_t - lr*m_t

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/novograd.h"
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

static void zero_dense(Dense* layer) {
    layer->weights.fill(0.0);
    layer->bias.fill(0.0);
    layer->zero_grad();
}

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
    Tensor backward(const Tensor& grad_output, double) override { return grad_output.clone(); }
    void update_weights(double) override {}
    Tensor get_weights() const override { return parameter_.clone(); }
    Tensor get_gradients() const override { return gradient_.clone(); }
    vector<Tensor*> parameters() override { return {&parameter_}; }
    vector<Tensor*> gradients() override {
        if (mode_ == Mode::COUNT_MISMATCH) return {};
        return {&wrong_gradient_};
    }
    void zero_grad() override {
        gradient_.fill(0.0);
        wrong_gradient_.fill(0.0);
    }

private:
    Mode mode_;
    Tensor parameter_;
    Tensor gradient_;
    Tensor wrong_gradient_;
};

static pair<double, double> run_deterministic_trajectory() {
    Model model;
    Dense* layer = new Dense(2, 2);
    zero_dense(layer);
    layer->weights[0][0] = 0.4;
    layer->weights[0][1] = -0.2;
    layer->weights[1][0] = 0.7;
    layer->weights[1][1] = 0.1;
    layer->bias[0][0] = 0.05;
    layer->bias[0][1] = -0.03;
    model.add_layer(layer);

    NovoGrad optimizer(0.03, 0.8, 0.6, 1e-9, 0.01, false, true);
    for (int step = 0; step < 8; ++step) {
        layer->grad_weights[0][0] = 0.2 + 0.03 * step;
        layer->grad_weights[0][1] = -0.1 + 0.02 * step;
        layer->grad_weights[1][0] = 0.05 - 0.01 * step;
        layer->grad_weights[1][1] = -0.3 + 0.04 * step;
        layer->grad_bias[0][0] = 0.1 * (step + 1);
        layer->grad_bias[0][1] = -0.05 * (step + 2);
        optimizer.step(model);
    }

    const double weight_signature = layer->weights[0][0]
                                  + 2.0 * layer->weights[0][1]
                                  + 3.0 * layer->weights[1][0]
                                  + 5.0 * layer->weights[1][1];
    const double bias_signature = layer->bias[0][0] + 7.0 * layer->bias[0][1];
    return {weight_signature, bias_signature};
}

int main() {
    cout << setprecision(12);
    cout << "=== NovoGrad Optimizer Tests ===\n\n";

    // ---------------------------------------------------------------------
    // T1: defaults, custom values, setters, and public contract.
    // ---------------------------------------------------------------------
    cout << "T1: defaults and accessors\n";
    {
        NovoGrad optimizer;
        check("default lr = 1e-3", near(optimizer.get_lr(), 1e-3));
        check("default beta1 = 0.95", near(optimizer.get_beta1(), 0.95));
        check("default beta2 = 0.98", near(optimizer.get_beta2(), 0.98));
        check("default epsilon = 1e-8", near(optimizer.get_epsilon(), 1e-8));
        check("default weight decay = 0", near(optimizer.get_weight_decay(), 0.0));
        check("default grad averaging is false", !optimizer.get_grad_averaging());
        check("default AMSGrad is false", !optimizer.get_amsgrad());
        check("handles_weight_decay is true", optimizer.handles_weight_decay());
        check("num_steps starts at zero", optimizer.num_steps() == 0);

        NovoGrad custom(0.2, 0.7, 0.6, 1e-5, 0.03, true, true);
        check("custom lr uses Optimizer base",
              near(custom.get_lr(), 0.2) && near(custom.Optimizer::lr, 0.2));
        check("custom beta1", near(custom.get_beta1(), 0.7));
        check("custom beta2", near(custom.get_beta2(), 0.6));
        check("custom epsilon", near(custom.get_epsilon(), 1e-5));
        check("custom weight decay", near(custom.get_weight_decay(), 0.03));
        check("custom grad averaging", custom.get_grad_averaging());
        check("custom AMSGrad", custom.get_amsgrad());

        custom.set_lr(0.15);
        custom.set_beta1(0.4);
        custom.set_beta2(0.3);
        custom.set_epsilon(1e-6);
        custom.set_weight_decay(0.02);
        custom.set_grad_averaging(false);
        custom.set_amsgrad(false);
        check("valid setters update every value",
              near(custom.get_lr(), 0.15) && near(custom.Optimizer::lr, 0.15) &&
              near(custom.get_beta1(), 0.4) && near(custom.get_beta2(), 0.3) &&
              near(custom.get_epsilon(), 1e-6) &&
              near(custom.get_weight_decay(), 0.02) &&
              !custom.get_grad_averaging() && !custom.get_amsgrad());
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T2: invalid hyperparameters fail immediately and setters validate.
    // ---------------------------------------------------------------------
    cout << "T2: validation\n";
    {
        auto throws = [](double lr, double beta1, double beta2, double eps, double wd) {
            try {
                NovoGrad bad(lr, beta1, beta2, eps, wd);
                (void)bad;
                return false;
            } catch (const invalid_argument&) {
                return true;
            }
        };
        check("negative lr throws", throws(-0.1, 0.95, 0.98, 1e-8, 0.0));
        check("negative beta1 throws", throws(0.1, -0.1, 0.98, 1e-8, 0.0));
        check("beta1 >= 1 throws", throws(0.1, 1.0, 0.98, 1e-8, 0.0));
        check("negative beta2 throws", throws(0.1, 0.95, -0.1, 1e-8, 0.0));
        check("beta2 >= 1 throws", throws(0.1, 0.95, 1.0, 1e-8, 0.0));
        check("zero epsilon throws", throws(0.1, 0.95, 0.98, 0.0, 0.0));
        check("negative weight decay throws", throws(0.1, 0.95, 0.98, 1e-8, -0.1));
        check("beta1 = 0 is allowed", !throws(0.1, 0.0, 0.98, 1e-8, 0.0));
        check("beta2 = 0 is allowed", !throws(0.1, 0.95, 0.0, 1e-8, 0.0));

        NovoGrad optimizer;
        bool setter_threw = false;
        try { optimizer.set_lr(-1.0); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_lr validates", setter_threw);
        setter_threw = false;
        try { optimizer.set_beta1(1.0); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_beta1 validates", setter_threw);
        setter_threw = false;
        try { optimizer.set_beta2(-0.1); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_beta2 validates", setter_threw);
        setter_threw = false;
        try { optimizer.set_epsilon(0.0); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_epsilon validates", setter_threw);
        setter_threw = false;
        try { optimizer.set_weight_decay(-0.1); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_weight_decay validates", setter_threw);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T3: state is lazy; momentum mirrors parameter shape; norm state scalar.
    // ---------------------------------------------------------------------
    cout << "T3: lazy state and shape\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);  // weights (2,3), bias (1,2)
        zero_dense(layer);
        model.add_layer(layer);
        NovoGrad optimizer;

        check("state absent before step", !optimizer.has_state(layer));
        double scalar = -1.0;
        check("second moment unavailable before step",
              !optimizer.get_second_moment(layer, 0, scalar));
        optimizer.step(model);
        check("state present after step", optimizer.has_state(layer));
        check("step counter increments", optimizer.num_steps() == 1);

        Tensor weight_momentum = optimizer.get_momentum(layer, 0);
        Tensor bias_momentum = optimizer.get_momentum(layer, 1);
        check("weight momentum shape is (2,3)",
              weight_momentum.rows == 2 && weight_momentum.cols == 3);
        check("bias momentum shape is (1,2)",
              bias_momentum.rows == 1 && bias_momentum.cols == 2);
        check("zero-gradient momentum is zero",
              near(tensor_l1(weight_momentum), 0.0) &&
              near(tensor_l1(bias_momentum), 0.0));
        check("weight second moment is one scalar equal to zero",
              optimizer.get_second_moment(layer, 0, scalar) && near(scalar, 0.0));
        check("out-of-range momentum query is empty",
              optimizer.get_momentum(layer, 99).rows == 0);
        check("out-of-range scalar query fails",
              !optimizer.get_second_moment(layer, 99, scalar));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T4: soul test — first v equals the full tensor's squared L2 norm.
    // ---------------------------------------------------------------------
    cout << "T4: first-step tensor-wise normalization\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->weights[0][0] = 1.0;
        layer->weights[0][1] = -2.0;
        layer->grad_weights[0][0] = 3.0;
        layer->grad_weights[0][1] = 4.0;
        model.add_layer(layer);

        const double eps = 1e-12;
        NovoGrad optimizer(0.2, 0.0, 0.98, eps, 0.0, false, false);
        optimizer.step(model);

        double v = 0.0;
        check("v1 = ||g||^2 = 25",
              optimizer.get_second_moment(layer, 0, v) && near(v, 25.0));
        Tensor momentum = optimizer.get_momentum(layer, 0);
        const double denom = 5.0 + eps;
        check("m[0][0] = 3/(5+eps)", near(momentum[0][0], 3.0 / denom));
        check("m[0][1] = 4/(5+eps)", near(momentum[0][1], 4.0 / denom));
        check("both coordinates share one tensor denominator",
              near(momentum[0][0] / momentum[0][1], 0.75));
        check("weight 0 uses normalized gradient",
              near(layer->weights[0][0], 1.0 - 0.2 * 3.0 / denom));
        check("weight 1 uses normalized gradient",
              near(layer->weights[0][1], -2.0 - 0.2 * 4.0 / denom));
        check("zero coordinates remain zero",
              near(layer->weights[1][0], 0.0) && near(layer->weights[1][1], 0.0));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T5: second-step scalar EMA uses the entire new gradient tensor norm.
    // ---------------------------------------------------------------------
    cout << "T5: second-moment EMA recurrence\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        NovoGrad optimizer(0.0, 0.0, 0.5, 1e-12, 0.0, false, false);

        layer->grad_bias[0][0] = 4.0;
        optimizer.step(model);  // v1 = 16
        double v = 0.0;
        optimizer.get_second_moment(layer, 1, v);
        check("step 1 v = 16", near(v, 16.0));

        layer->grad_bias[0][0] = 2.0;
        optimizer.step(model);  // v2 = 0.5*16 + 0.5*4 = 10
        optimizer.get_second_moment(layer, 1, v);
        Tensor momentum = optimizer.get_momentum(layer, 1);
        check("v2 = beta2*v1 + (1-beta2)*g2", near(v, 10.0));
        check("step 2 normalizes by sqrt(v2)",
              near(momentum[0][0], 2.0 / (std::sqrt(10.0) + 1e-12)));
        check("num_steps = 2", optimizer.num_steps() == 2);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T6: default momentum is Polyak-style (no 1-beta1 multiplier).
    // ---------------------------------------------------------------------
    cout << "T6: momentum recurrence without gradient averaging\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        NovoGrad optimizer(0.0, 0.5, 0.0, 1e-12, 0.0, false, false);

        layer->grad_bias[0][0] = 2.0;
        optimizer.step(model);  // normalized = 1, m1 = 1
        Tensor momentum = optimizer.get_momentum(layer, 1);
        check("m1 = normalized gradient", near(momentum[0][0], 2.0 / (2.0 + 1e-12)));

        layer->grad_bias[0][0] = 2.0;
        optimizer.step(model);  // m2 = 0.5*m1 + 1
        momentum = optimizer.get_momentum(layer, 1);
        const double normalized = 2.0 / (2.0 + 1e-12);
        check("m2 = beta1*m1 + normalized gradient",
              near(momentum[0][0], 1.5 * normalized));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T7: grad averaging scales normalized gradient and decay together.
    // ---------------------------------------------------------------------
    cout << "T7: optional Adam-style gradient averaging\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->bias[0][0] = 2.0;
        layer->grad_bias[0][0] = 2.0;
        model.add_layer(layer);

        NovoGrad optimizer(0.0, 0.5, 0.0, 1e-12, 0.25, true, false);
        optimizer.step(model);
        Tensor momentum = optimizer.get_momentum(layer, 1);
        const double direction = 2.0 / (2.0 + 1e-12) + 0.25 * 2.0;
        check("grad averaging multiplies full normalized-plus-decay direction",
              near(momentum[0][0], 0.5 * direction));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T8: paper weight decay is added after normalization, inside momentum.
    // ---------------------------------------------------------------------
    cout << "T8: weight decay placement\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->bias[0][0] = 2.0;
        model.add_layer(layer);
        NovoGrad optimizer(0.1, 0.0, 0.98, 1e-8, 0.25, false, false);
        optimizer.step(model);

        Tensor momentum = optimizer.get_momentum(layer, 1);
        double v = -1.0;
        optimizer.get_second_moment(layer, 1, v);
        check("zero gradient gives v = 0", near(v, 0.0));
        check("decay direction is wd*w = 0.5", near(momentum[0][0], 0.5));
        check("decay inside momentum shrinks 2.0 to 1.95", near(layer->bias[0][0], 1.95));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T9: AMSGrad max statistic is monotone and controls normalization.
    // ---------------------------------------------------------------------
    cout << "T9: AMSGrad scalar maximum\n";
    {
        Model model_ams;
        Dense* layer_ams = new Dense(1, 1);
        zero_dense(layer_ams);
        model_ams.add_layer(layer_ams);
        NovoGrad ams(0.0, 0.0, 0.5, 1e-12, 0.0, false, true);

        Model model_plain;
        Dense* layer_plain = new Dense(1, 1);
        zero_dense(layer_plain);
        model_plain.add_layer(layer_plain);
        NovoGrad plain(0.0, 0.0, 0.5, 1e-12, 0.0, false, false);

        layer_ams->grad_bias[0][0] = 4.0;
        layer_plain->grad_bias[0][0] = 4.0;
        ams.step(model_ams);
        plain.step(model_plain);  // v1 = 16

        layer_ams->grad_bias[0][0] = 2.0;
        layer_plain->grad_bias[0][0] = 2.0;
        ams.step(model_ams);
        plain.step(model_plain);  // v2 = 10, max remains 16

        double v = 0.0, vmax = 0.0;
        ams.get_second_moment(layer_ams, 1, v);
        ams.get_max_second_moment(layer_ams, 1, vmax);
        check("current v falls to 10", near(v, 10.0));
        check("AMSGrad v_max remains 16", near(vmax, 16.0));
        Tensor ams_m = ams.get_momentum(layer_ams, 1);
        Tensor plain_m = plain.get_momentum(layer_plain, 1);
        check("AMSGrad step uses sqrt(v_max)",
              near(ams_m[0][0], 2.0 / (4.0 + 1e-12)));
        check("plain step uses smaller current v",
              near(plain_m[0][0], 2.0 / (std::sqrt(10.0) + 1e-12)));
        check("AMSGrad and plain normalization differ", ams_m[0][0] < plain_m[0][0]);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T10: zero norm stays finite and does not move without decay/momentum.
    // ---------------------------------------------------------------------
    cout << "T10: zero-gradient stability\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->weights[0][0] = 3.0;
        model.add_layer(layer);
        NovoGrad optimizer(0.1, 0.9, 0.98, 1e-8, 0.0, false, true);
        optimizer.step(model);
        double v = -1.0;
        optimizer.get_second_moment(layer, 0, v);
        check("zero norm stores finite v=0", near(v, 0.0) && std::isfinite(v));
        check("zero gradient leaves parameter unchanged", near(layer->weights[0][0], 3.0));
        Tensor momentum = optimizer.get_momentum(layer, 0);
        check("zero-gradient momentum is finite and zero",
              near(tensor_l1(momentum), 0.0) && std::isfinite(momentum[0][0]));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T11: every parameter and every layer gets independent scalar state.
    // ---------------------------------------------------------------------
    cout << "T11: parameter and layer state isolation\n";
    {
        Model model;
        Dense* layer1 = new Dense(2, 2);
        Dense* layer2 = new Dense(2, 2);
        zero_dense(layer1);
        zero_dense(layer2);
        layer1->grad_weights[0][0] = 3.0;  // v = 9
        layer1->grad_bias[0][0] = 4.0;     // v = 16
        layer2->grad_weights[0][0] = 5.0;  // v = 25
        model.add_layer(layer1);
        model.add_layer(layer2);
        NovoGrad optimizer(0.0, 0.0, 0.5);
        optimizer.step(model);

        double l1w = 0.0, l1b = 0.0, l2w = 0.0;
        optimizer.get_second_moment(layer1, 0, l1w);
        optimizer.get_second_moment(layer1, 1, l1b);
        optimizer.get_second_moment(layer2, 0, l2w);
        check("layer 1 weight v = 9", near(l1w, 9.0));
        check("layer 1 bias v = 16", near(l1b, 16.0));
        check("layer 2 weight v = 25", near(l2w, 25.0));
        check("both layer states exist",
              optimizer.has_state(layer1) && optimizer.has_state(layer2));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T12: scheduler writes inherited lr and step consumes it.
    // ---------------------------------------------------------------------
    cout << "T12: scheduler compatibility\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->grad_bias[0][0] = 1.0;
        model.add_layer(layer);

        NovoGrad optimizer(0.1, 0.0, 0.98, 1e-12);
        PolynomialLR scheduler(0.1, 2, 1.0, 0.0);
        scheduler.set_optimizer(&optimizer);
        scheduler.step(model);  // inherited lr becomes 0.05
        check("scheduler updates NovoGrad get_lr", near(optimizer.get_lr(), 0.05));
        optimizer.step(model);
        check("step consumes scheduled lr", near(layer->bias[0][0], -0.05, 1e-12));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T13: deterministic state/update trajectory.
    // ---------------------------------------------------------------------
    cout << "T13: determinism\n";
    {
        const auto first = run_deterministic_trajectory();
        const auto second = run_deterministic_trajectory();
        check("weight trajectory is bit-identical", near(first.first, second.first, 0.0));
        check("bias trajectory is bit-identical", near(first.second, second.second, 0.0));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T14: end-to-end optimization on asymmetric two-output regression.
    // ---------------------------------------------------------------------
    cout << "T14: end-to-end training reduces loss\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        model.add_layer(layer);
        NovoGrad optimizer(0.08, 0.9, 0.98, 1e-8, 0.0, false, false);

        Tensor input(6, 2);
        const double raw_input[12] = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                      -1.0, 0.5, 0.5, -1.0, 2.0, -0.5};
        for (size_t row = 0; row < 6; ++row)
            for (size_t col = 0; col < 2; ++col)
                input[row][col] = raw_input[2 * row + col];

        Tensor target(6, 2);
        for (size_t row = 0; row < 6; ++row) {
            target[row][0] = 2.0 * input[row][0] - input[row][1] + 0.5;
            target[row][1] = -input[row][0] + 3.0 * input[row][1] - 0.25;
        }

        auto loss = [&]() {
            Tensor prediction = layer->forward(input);
            double total = 0.0;
            for (size_t row = 0; row < prediction.rows; ++row)
                for (size_t col = 0; col < prediction.cols; ++col) {
                    const double error = prediction[row][col] - target[row][col];
                    total += error * error;
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
        check("training loss is finite", std::isfinite(final_loss));
        check("NovoGrad reduces loss by at least 95%", final_loss < initial_loss * 0.05);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T15: gradients are consumed and malformed Layer contracts throw.
    // ---------------------------------------------------------------------
    cout << "T15: gradient clearing and contract guards\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->grad_weights[0][0] = 1.0;
        layer->grad_bias[0][0] = -2.0;
        model.add_layer(layer);
        NovoGrad optimizer;
        optimizer.step(model);
        check("weight gradients cleared", near(tensor_l1(layer->grad_weights), 0.0));
        check("bias gradients cleared", near(tensor_l1(layer->grad_bias), 0.0));

        bool threw = false;
        try {
            Model malformed;
            malformed.add_layer(new MalformedLayer(MalformedLayer::Mode::COUNT_MISMATCH));
            NovoGrad bad;
            bad.step(malformed);
        } catch (const runtime_error&) {
            threw = true;
        }
        check("parameter/gradient count mismatch throws", threw);

        threw = false;
        try {
            Model malformed;
            malformed.add_layer(new MalformedLayer(MalformedLayer::Mode::SHAPE_MISMATCH));
            NovoGrad bad;
            bad.step(malformed);
        } catch (const runtime_error&) {
            threw = true;
        }
        check("parameter/gradient shape mismatch throws", threw);
    }

    cout << "\n=== NovoGrad: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
