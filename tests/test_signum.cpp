// test_signum.cpp — behavioral tests for Signum (Bernstein et al., ICML 2018).
// Paper algorithm: m_t = beta*m_{t-1} + (1-beta)*g_t;
//                  param -= lr*(sign(m_t) + weight_decay*param).

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/scheduler.h"
#include "nn/optimizers/signum.h"

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
    return std::isfinite(actual) && std::isfinite(expected) &&
           std::abs(actual - expected) <= tol;
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
    Tensor backward(const Tensor& grad_output, double) override {
        return grad_output.clone();
    }
    void update_weights(double) override {}
    Tensor get_weights() const override { return parameter_.clone(); }
    Tensor get_gradients() const override { return gradient_.clone(); }
    vector<Tensor*> parameters() override { return {&parameter_}; }
    vector<Tensor*> gradients() override {
        return mode_ == Mode::COUNT_MISMATCH
                   ? vector<Tensor*>{}
                   : vector<Tensor*>{&wrong_gradient_};
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

static pair<double, double> deterministic_trajectory() {
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

    Signum optimizer(0.03, 0.8, 0.01);
    for (int step = 0; step < 8; ++step) {
        layer->grad_weights[0][0] = 0.2 + 0.03 * step;
        layer->grad_weights[0][1] = -0.1 + 0.02 * step;
        layer->grad_weights[1][0] = 0.05 - 0.01 * step;
        layer->grad_weights[1][1] = -0.3 + 0.04 * step;
        layer->grad_bias[0][0] = 0.1 * (step + 1);
        layer->grad_bias[0][1] = -0.05 * (step + 2);
        optimizer.step(model);
    }

    return {
        layer->weights[0][0] + 2.0 * layer->weights[0][1] +
            3.0 * layer->weights[1][0] + 5.0 * layer->weights[1][1],
        layer->bias[0][0] + 7.0 * layer->bias[0][1]
    };
}

int main() {
    cout << setprecision(15);
    cout << "=== Signum Optimizer Tests ===\n\n";

    cout << "T1: defaults, custom values, and accessors\n";
    {
        Signum optimizer;
        check("default lr = 0.01", near(optimizer.get_lr(), 0.01));
        check("default beta = 0.9", near(optimizer.get_beta(), 0.9));
        check("default weight decay = 0", near(optimizer.get_weight_decay(), 0.0));
        check("step count starts at zero", optimizer.num_steps() == 0);
        check("handles weight decay internally", optimizer.handles_weight_decay());

        Signum custom(0.2, 0.7, 0.03);
        check("custom constructor values",
              near(custom.get_lr(), 0.2) && near(custom.Optimizer::lr, 0.2) &&
              near(custom.get_beta(), 0.7) && near(custom.get_weight_decay(), 0.03));
        custom.set_lr(0.15);
        custom.set_beta(0.4);
        custom.set_weight_decay(0.02);
        check("validated setters update values",
              near(custom.get_lr(), 0.15) && near(custom.Optimizer::lr, 0.15) &&
              near(custom.get_beta(), 0.4) && near(custom.get_weight_decay(), 0.02));
    }
    cout << '\n';

    cout << "T2: invalid hyperparameters\n";
    {
        auto throws = [](double lr, double beta, double wd) {
            try {
                Signum bad(lr, beta, wd);
                (void)bad;
                return false;
            } catch (const invalid_argument&) {
                return true;
            }
        };
        check("negative lr throws", throws(-0.1, 0.9, 0.0));
        check("negative beta throws", throws(0.1, -0.1, 0.0));
        check("beta >= 1 throws", throws(0.1, 1.0, 0.0));
        check("negative weight decay throws", throws(0.1, 0.9, -0.1));
        check("lr = 0 is allowed", !throws(0.0, 0.9, 0.0));
        check("beta = 0 is allowed", !throws(0.1, 0.0, 0.0));

        Signum optimizer;
        bool threw = false;
        try { optimizer.set_lr(-1.0); } catch (const invalid_argument&) { threw = true; }
        check("set_lr validates", threw);
        threw = false;
        try { optimizer.set_beta(1.0); } catch (const invalid_argument&) { threw = true; }
        check("set_beta validates", threw);
        threw = false;
        try { optimizer.set_weight_decay(-0.1); } catch (const invalid_argument&) { threw = true; }
        check("set_weight_decay validates", threw);
    }
    cout << '\n';

    cout << "T3: lazy one-tensor state\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer;
        check("state absent before step", !optimizer.has_state(layer));
        check("missing state query returns empty", optimizer.get_momentum(layer, 0).rows == 0);
        optimizer.step(model);
        check("state present after step", optimizer.has_state(layer));
        check("step count increments", optimizer.num_steps() == 1);
        Tensor weight_state = optimizer.get_momentum(layer, 0);
        Tensor bias_state = optimizer.get_momentum(layer, 1);
        check("weight momentum mirrors (2,3)",
              weight_state.rows == 2 && weight_state.cols == 3);
        check("bias momentum mirrors (1,2)",
              bias_state.rows == 1 && bias_state.cols == 2);
        check("zero-gradient momentum remains zero", near(tensor_l1(weight_state), 0.0));
        check("out-of-range state query is empty",
              optimizer.get_momentum(layer, 99).rows == 0);
    }
    cout << '\n';

    cout << "T4: first-step closed form\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->grad_weights[0][0] = 2.0;
        model.add_layer(layer);
        Signum optimizer(0.1, 0.8, 0.0);
        optimizer.step(model);
        check("m1 = (1-beta)*g", near(optimizer.get_momentum(layer, 0)[0][0], 0.4));
        check("positive momentum moves parameter by exactly -lr",
              near(layer->weights[0][0], -0.1));
    }
    cout << '\n';

    cout << "T5: beta=0 reduces exactly to signSGD\n";
    {
        Model model;
        Dense* layer = new Dense(1, 2);
        zero_dense(layer);
        layer->grad_weights[0][0] = 1000.0;
        layer->grad_weights[1][0] = -1e-9;
        model.add_layer(layer);
        Signum optimizer(0.07, 0.0, 0.0);
        optimizer.step(model);
        check("large positive gradient gives -lr", near(layer->weights[0][0], -0.07));
        check("tiny negative gradient gives +lr", near(layer->weights[1][0], 0.07));
        check("beta-zero state equals gradient",
              near(optimizer.get_momentum(layer, 0)[0][0], 1000.0) &&
              near(optimizer.get_momentum(layer, 0)[1][0], -1e-9));
    }
    cout << '\n';

    cout << "T6: persistent gradients have constant-magnitude updates\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer(0.05, 0.9, 0.0);
        double previous = 0.0;
        bool exact = true;
        for (int step = 0; step < 5; ++step) {
            layer->grad_weights[0][0] = 1e-3 * (step + 1);
            optimizer.step(model);
            const double current = layer->weights[0][0];
            exact = exact && near(current - previous, -0.05);
            previous = current;
        }
        check("five positive-gradient updates are all exactly -lr", exact);
        check("final parameter is -5*lr", near(layer->weights[0][0], -0.25));
    }
    cout << '\n';

    cout << "T7: Signum sequence differs from Lion interpolation\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer(0.1, 0.9, 0.0);
        layer->grad_weights[0][0] = 10.0;
        optimizer.step(model);
        layer->grad_weights[0][0] = -1.0;
        optimizer.step(model);
        // Signum m: 1.0 -> 0.8, so both steps are negative. Lion with
        // beta1=.9,beta2=.99 has c2=.09-.1<0 and reverses on step two.
        check("Signum momentum remains positive after [10,-1]",
              near(optimizer.get_momentum(layer, 0)[0][0], 0.8));
        check("Signum takes two negative steps on [10,-1]",
              near(layer->weights[0][0], -0.2));
    }
    cout << '\n';

    cout << "T8: zero gradient and momentum carry\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer(0.1, 0.5, 0.0);
        optimizer.step(model);
        check("fresh zero gradient leaves parameter unchanged",
              near(layer->weights[0][0], 0.0));
        layer->grad_weights[0][0] = 1.0;
        optimizer.step(model);
        layer->grad_weights[0][0] = 0.0;
        optimizer.step(model);
        check("zero current gradient still follows nonzero momentum",
              near(layer->weights[0][0], -0.2));
        check("momentum decays but remains positive",
              near(optimizer.get_momentum(layer, 0)[0][0], 0.25));
    }
    cout << '\n';

    cout << "T9: decoupled weight decay\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->weights[0][0] = 1.0;
        model.add_layer(layer);
        Signum optimizer(0.1, 0.9, 0.1);
        optimizer.step(model);
        check("zero-gradient weight decay is exact 1 -> 0.99",
              near(layer->weights[0][0], 0.99));
        check("weight decay does not pollute momentum",
              near(optimizer.get_momentum(layer, 0)[0][0], 0.0));
    }
    cout << '\n';

    cout << "T10: LR scheduler compatibility\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer(0.1, 0.0, 0.0);
        StepLR scheduler(0.1, 1, 0.5);
        scheduler.set_optimizer(&optimizer);
        scheduler.step(model);
        check("scheduler writes effective Signum lr", near(optimizer.get_lr(), 0.05));
        layer->grad_weights[0][0] = 1.0;
        optimizer.step(model);
        check("scheduled lr controls actual update",
              near(layer->weights[0][0], -0.05));
    }
    cout << '\n';

    cout << "T11: independent state across layers and parameters\n";
    {
        Model model;
        Dense* first = new Dense(1, 1);
        Dense* second = new Dense(1, 1);
        zero_dense(first);
        zero_dense(second);
        first->grad_weights[0][0] = 1.0;
        first->grad_bias[0][0] = 2.0;
        second->grad_weights[0][0] = -3.0;
        second->grad_bias[0][0] = 4.0;
        model.add_layer(first);
        model.add_layer(second);
        Signum optimizer(0.01, 0.5, 0.0);
        optimizer.step(model);
        check("first layer weight/bias state independent",
              near(optimizer.get_momentum(first, 0)[0][0], 0.5) &&
              near(optimizer.get_momentum(first, 1)[0][0], 1.0));
        check("second layer owns different state",
              near(optimizer.get_momentum(second, 0)[0][0], -1.5) &&
              near(optimizer.get_momentum(second, 1)[0][0], 2.0));
        check("both layers registered", optimizer.has_state(first) && optimizer.has_state(second));
    }
    cout << '\n';

    cout << "T12: gradients clear after step\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->grad_weights.fill(1.0);
        layer->grad_bias.fill(-2.0);
        model.add_layer(layer);
        Signum optimizer;
        optimizer.step(model);
        check("weight gradients cleared", near(tensor_l1(layer->grad_weights), 0.0));
        check("bias gradients cleared", near(tensor_l1(layer->grad_bias), 0.0));
    }
    cout << '\n';

    cout << "T13: malformed layers throw safely\n";
    {
        bool count_threw = false;
        try {
            Model model;
            model.add_layer(new MalformedLayer(MalformedLayer::Mode::COUNT_MISMATCH));
            Signum optimizer;
            optimizer.step(model);
        } catch (const runtime_error&) {
            count_threw = true;
        }
        check("parameter/gradient count mismatch throws", count_threw);

        bool shape_threw = false;
        try {
            Model model;
            model.add_layer(new MalformedLayer(MalformedLayer::Mode::SHAPE_MISMATCH));
            Signum optimizer;
            optimizer.step(model);
        } catch (const runtime_error&) {
            shape_threw = true;
        }
        check("parameter/gradient shape mismatch throws", shape_threw);
    }
    cout << '\n';

    cout << "T14: deterministic trajectory\n";
    {
        const auto first = deterministic_trajectory();
        const auto second = deterministic_trajectory();
        check("weight trajectory is bit-exact", first.first == second.first);
        check("bias trajectory is bit-exact", first.second == second.second);
        check("trajectory is non-trivial", std::abs(first.first) + std::abs(first.second) > 1e-6);
    }
    cout << '\n';

    cout << "T15: end-to-end linear regression\n";
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        zero_dense(layer);
        model.add_layer(layer);
        Signum optimizer(0.01, 0.9, 0.0);

        Tensor input(6, 2);
        Tensor target(6, 1);
        const double samples[6][2] = {
            {-2.0, -1.0}, {-1.0, 2.0}, {0.0, -2.0},
            {1.0, 0.5}, {2.0, 1.5}, {3.0, -1.0}
        };
        for (size_t row = 0; row < 6; ++row) {
            input[row][0] = samples[row][0];
            input[row][1] = samples[row][1];
            target[row][0] = 1.5 * samples[row][0] - 0.7 * samples[row][1] + 0.2;
        }

        auto loss_and_grad = [&](bool backward) {
            Tensor prediction = layer->forward(input);
            Tensor grad(6, 1);
            double loss = 0.0;
            for (size_t row = 0; row < 6; ++row) {
                const double error = prediction[row][0] - target[row][0];
                loss += error * error / 6.0;
                grad[row][0] = 2.0 * error / 6.0;
            }
            if (backward) layer->backward(grad, 0.0);
            return loss;
        };

        const double initial_loss = loss_and_grad(false);
        for (int step = 0; step < 300; ++step) {
            (void)loss_and_grad(true);
            optimizer.step(model);
        }
        const double final_loss = loss_and_grad(false);
        check("training loss is finite", std::isfinite(final_loss));
        check("training reduces loss by at least 95%", final_loss < 0.05 * initial_loss);
        check("trained weights approach target slope",
              std::abs(layer->weights[0][0] - 1.5) < 0.2 &&
              std::abs(layer->weights[0][1] + 0.7) < 0.2);
        check("trained bias approaches target intercept",
              std::abs(layer->bias[0][0] - 0.2) < 0.2);
        cout << "  loss: " << initial_loss << " -> " << final_loss << '\n';
    }
    cout << '\n';

    cout << "=== Summary: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
