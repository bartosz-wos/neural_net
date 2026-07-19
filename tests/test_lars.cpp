// test_lars.cpp — behavioral tests for Layer-wise Adaptive Rate Scaling.
//
// Reference: You, Gitman, Ginsburg (2017),
// "Large Batch Training of Convolutional Networks".
//
// Canonical update tested here:
//   u = grad + weight_decay * param
//   q = trust_coefficient * ||param|| / (||u|| + epsilon)
//   momentum_buffer = momentum * momentum_buffer + q * u
//   param -= lr * momentum_buffer
// Bias/norm-shaped tensors are excluded from adaptation and weight decay by
// default, matching modern Barlow Twins / SimCLR LARS recipes.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nn/core/layer.h"
#include "nn/core/model.h"
#include "nn/optimizers/lars.h"
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

static void zero_dense(Dense* layer) {
    layer->weights.fill(0.0);
    layer->bias.fill(0.0);
    layer->zero_grad();
}

static double tensor_l1(const Tensor& tensor) {
    double total = 0.0;
    for (size_t i = 0; i < tensor.rows; ++i)
        for (size_t j = 0; j < tensor.cols; ++j)
            total += std::abs(tensor[i][j]);
    return total;
}

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

    LARS optimizer(0.08, 0.8, 0.01, 0.02, 1e-9, false, false);
    for (int step = 0; step < 8; ++step) {
        layer->grad_weights[0][0] = 0.2 + 0.03 * step;
        layer->grad_weights[0][1] = -0.1 + 0.02 * step;
        layer->grad_weights[1][0] = 0.05 - 0.01 * step;
        layer->grad_weights[1][1] = -0.3 + 0.04 * step;
        layer->grad_bias[0][0] = 0.1 * (step + 1);
        layer->grad_bias[0][1] = -0.05 * (step + 2);
        optimizer.step(model);
    }

    double weight_signature = layer->weights[0][0] + 2.0 * layer->weights[0][1]
                            + 3.0 * layer->weights[1][0] + 5.0 * layer->weights[1][1];
    double bias_signature = layer->bias[0][0] + 7.0 * layer->bias[0][1];
    return {weight_signature, bias_signature};
}

int main() {
    cout << setprecision(12);
    cout << "=== LARS Optimizer Tests ===\n\n";

    // ---------------------------------------------------------------------
    // T1: constructor defaults, non-default values, and public contract.
    // ---------------------------------------------------------------------
    cout << "T1: constructor defaults and accessors\n";
    {
        LARS optimizer;
        check("default lr = 0.1", near(optimizer.get_lr(), 0.1));
        check("default momentum = 0.9", near(optimizer.get_momentum(), 0.9));
        check("default weight_decay = 1e-4", near(optimizer.get_weight_decay(), 1e-4));
        check("default trust_coefficient = 1e-3", near(optimizer.get_trust_coefficient(), 1e-3));
        check("default epsilon = 1e-8", near(optimizer.get_epsilon(), 1e-8));
        check("default excludes vector-shaped adaptation", optimizer.get_exclude_1d_from_adaptation());
        check("default excludes vector-shaped weight decay", optimizer.get_exclude_1d_from_weight_decay());
        check("handles_weight_decay() is true", optimizer.handles_weight_decay());
        check("num_steps starts at 0", optimizer.num_steps() == 0);

        LARS custom(0.3, 0.7, 0.02, 0.04, 1e-6, false, false);
        check("custom lr stored in Optimizer base", near(custom.get_lr(), 0.3) && near(custom.Optimizer::lr, 0.3));
        check("custom momentum", near(custom.get_momentum(), 0.7));
        check("custom weight decay", near(custom.get_weight_decay(), 0.02));
        check("custom trust coefficient", near(custom.get_trust_coefficient(), 0.04));
        check("custom epsilon", near(custom.get_epsilon(), 1e-6));
        check("custom adaptation filter", !custom.get_exclude_1d_from_adaptation());
        check("custom decay filter", !custom.get_exclude_1d_from_weight_decay());
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T2: invalid hyperparameters must fail immediately; setters validate too.
    // ---------------------------------------------------------------------
    cout << "T2: hyperparameter validation\n";
    {
        auto throws = [](double lr, double momentum, double wd, double eta, double eps) {
            try {
                LARS bad(lr, momentum, wd, eta, eps);
                (void)bad;
                return false;
            } catch (const invalid_argument&) {
                return true;
            }
        };
        check("negative lr throws", throws(-0.1, 0.9, 0.0, 1e-3, 1e-8));
        check("negative momentum throws", throws(0.1, -0.1, 0.0, 1e-3, 1e-8));
        check("momentum >= 1 throws", throws(0.1, 1.0, 0.0, 1e-3, 1e-8));
        check("negative weight decay throws", throws(0.1, 0.9, -0.1, 1e-3, 1e-8));
        check("zero trust coefficient throws", throws(0.1, 0.9, 0.0, 0.0, 1e-8));
        check("negative trust coefficient throws", throws(0.1, 0.9, 0.0, -1e-3, 1e-8));
        check("zero epsilon throws", throws(0.1, 0.9, 0.0, 1e-3, 0.0));

        LARS optimizer;
        bool setter_threw = false;
        try { optimizer.set_momentum(1.0); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_momentum validates", setter_threw);
        setter_threw = false;
        try { optimizer.set_trust_coefficient(0.0); }
        catch (const invalid_argument&) { setter_threw = true; }
        check("set_trust_coefficient validates", setter_threw);
        optimizer.set_lr(0.25);
        optimizer.set_weight_decay(0.03);
        optimizer.set_epsilon(1e-5);
        optimizer.set_exclude_1d_from_adaptation(false);
        optimizer.set_exclude_1d_from_weight_decay(false);
        check("valid setters update values",
              near(optimizer.get_lr(), 0.25) && near(optimizer.Optimizer::lr, 0.25) &&
              near(optimizer.get_weight_decay(), 0.03) && near(optimizer.get_epsilon(), 1e-5) &&
              !optimizer.get_exclude_1d_from_adaptation() &&
              !optimizer.get_exclude_1d_from_weight_decay());
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T3: state is lazy and mirrors every parameter shape exactly.
    // ---------------------------------------------------------------------
    cout << "T3: lazy momentum state and shape\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);  // weights (2,3), bias (1,2)
        zero_dense(layer);
        model.add_layer(layer);
        LARS optimizer;

        check("state absent before step", !optimizer.has_state(layer));
        optimizer.step(model);
        check("state present after step", optimizer.has_state(layer));
        check("step counter increments", optimizer.num_steps() == 1);

        Tensor weight_momentum = optimizer.get_momentum_buffer(layer, 0);
        Tensor bias_momentum = optimizer.get_momentum_buffer(layer, 1);
        check("weight momentum shape is (2,3)",
              weight_momentum.rows == 2 && weight_momentum.cols == 3);
        check("bias momentum shape is (1,2)",
              bias_momentum.rows == 1 && bias_momentum.cols == 2);
        check("fresh zero-gradient state remains zero",
              near(tensor_l1(weight_momentum), 0.0) && near(tensor_l1(bias_momentum), 0.0));
        check("out-of-range state query is empty",
              optimizer.get_momentum_buffer(layer, 99).rows == 0);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T4: LARS soul test — exact trust ratio and scaled update.
    // ---------------------------------------------------------------------
    cout << "T4: closed-form trust ratio without weight decay\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->weights[0][0] = 3.0;
        layer->weights[0][1] = 4.0;  // ||w|| = 5
        layer->grad_weights[0][0] = 6.0;
        layer->grad_weights[0][1] = 8.0;  // ||g|| = 10
        model.add_layer(layer);

        const double eps = 1e-12;
        const double expected_q = 0.1 * 5.0 / (10.0 + eps);
        LARS optimizer(0.2, 0.0, 0.0, 0.1, eps, false, false);
        optimizer.step(model);

        double actual_q = 0.0;
        check("trust-ratio accessor succeeds", optimizer.get_last_trust_ratio(layer, 0, actual_q));
        check("q = eta*||w||/(||g||+eps)", near(actual_q, expected_q, 1e-13));
        check("w[0][0] uses q-scaled gradient",
              near(layer->weights[0][0], 3.0 - 0.2 * expected_q * 6.0, 1e-12));
        check("w[0][1] uses q-scaled gradient",
              near(layer->weights[0][1], 4.0 - 0.2 * expected_q * 8.0, 1e-12));
        check("unrelated zero entries stay zero", near(layer->weights[1][0], 0.0));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T5: weight decay belongs inside the adapted update norm.
    // Non-parallel g and w prevents a globally scaled wrong formula from passing.
    // ---------------------------------------------------------------------
    cout << "T5: coupled weight decay is inside layer adaptation\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->weights[0][0] = 3.0;
        layer->weights[0][1] = 4.0;
        layer->grad_weights[0][1] = 3.0;
        layer->grad_weights[1][0] = 4.0;
        model.add_layer(layer);

        // u = g + 0.5*w = [[1.5, 5.0], [4.0, 0.0]]
        const double update_norm = std::sqrt(1.5 * 1.5 + 5.0 * 5.0 + 4.0 * 4.0);
        const double expected_q = 0.2 * 5.0 / (update_norm + 1e-12);
        LARS optimizer(0.1, 0.0, 0.5, 0.2, 1e-12, false, false);
        optimizer.step(model);

        double actual_q = 0.0;
        optimizer.get_last_trust_ratio(layer, 0, actual_q);
        check("trust ratio uses ||g + wd*w||", near(actual_q, expected_q, 1e-13));
        check("decayed first coordinate uses q*(g+wd*w)",
              near(layer->weights[0][0], 3.0 - 0.1 * expected_q * 1.5, 1e-12));
        check("non-parallel gradient coordinate uses q*(g+wd*w)",
              near(layer->weights[1][0], 0.0 - 0.1 * expected_q * 4.0, 1e-12));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T6: zero norms use q=1 rather than producing NaN or freezing updates.
    // ---------------------------------------------------------------------
    cout << "T6: zero-norm fallbacks\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->grad_weights[0][0] = 2.0;
        model.add_layer(layer);
        LARS optimizer(0.1, 0.0, 0.0, 0.001, 1e-8, false, false);
        optimizer.step(model);
        double q = 0.0;
        optimizer.get_last_trust_ratio(layer, 0, q);
        check("zero parameter norm falls back to q=1", near(q, 1.0));
        check("zero parameter norm still takes finite SGD step", near(layer->weights[0][0], -0.2));

        // Make u = g + wd*w exactly zero on the next independent optimizer.
        Model model2;
        Dense* layer2 = new Dense(2, 2);
        zero_dense(layer2);
        layer2->weights[0][0] = 2.0;
        layer2->grad_weights[0][0] = -1.0;
        model2.add_layer(layer2);
        LARS optimizer2(0.1, 0.0, 0.5, 0.001, 1e-8, false, false);
        optimizer2.step(model2);
        optimizer2.get_last_trust_ratio(layer2, 0, q);
        check("zero update norm falls back to q=1", near(q, 1.0));
        check("zero adapted update leaves parameter unchanged", near(layer2->weights[0][0], 2.0));
        check("zero-update result is finite", std::isfinite(layer2->weights[0][0]));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T7: momentum recurrence is applied after local-rate scaling.
    // Bias is excluded, so q=1 makes the recurrence hand-verifiable.
    // ---------------------------------------------------------------------
    cout << "T7: two-step momentum recurrence\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        model.add_layer(layer);
        LARS optimizer(0.1, 0.5, 0.0, 0.001, 1e-8);

        layer->grad_bias[0][0] = 2.0;
        optimizer.step(model);  // m1 = 2; b1 = -0.2
        check("step 1 bias = -0.2", near(layer->bias[0][0], -0.2));

        layer->grad_bias[0][0] = 4.0;
        optimizer.step(model);  // m2 = 0.5*2 + 4 = 5; b2 = -0.7
        check("step 2 bias = -0.7", near(layer->bias[0][0], -0.7));
        Tensor m = optimizer.get_momentum_buffer(layer, 1);
        check("stored momentum follows m2 = momentum*m1 + update", near(m[0][0], 5.0));
        check("num_steps = 2", optimizer.num_steps() == 2);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T8: vector-shaped parameters are excluded from adaptation and decay.
    // ---------------------------------------------------------------------
    cout << "T8: default bias/norm exclusion\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->bias[0][0] = 2.0;
        layer->grad_bias[0][0] = 1.0;
        model.add_layer(layer);
        LARS optimizer(0.1, 0.0, 0.5, 0.001, 1e-8);
        optimizer.step(model);

        double q = 0.0;
        optimizer.get_last_trust_ratio(layer, 1, q);
        check("excluded bias uses q=1", near(q, 1.0));
        check("excluded bias skips weight decay (2.0 -> 1.9)", near(layer->bias[0][0], 1.9));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T9: turning filters off adapts and decays vector-shaped parameters.
    // ---------------------------------------------------------------------
    cout << "T9: configurable bias/norm filters\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->bias[0][0] = 2.0;
        layer->grad_bias[0][0] = 1.0;
        model.add_layer(layer);
        const double eps = 1e-12;
        LARS optimizer(0.1, 0.0, 0.5, 0.1, eps, false, false);
        optimizer.step(model);

        // u = 1 + 0.5*2 = 2; q = 0.1*2/(2+eps), b' = 2 - 0.1*q*2
        const double expected_q = 0.2 / (2.0 + eps);
        double q = 0.0;
        optimizer.get_last_trust_ratio(layer, 1, q);
        check("unfiltered bias gets adapted trust ratio", near(q, expected_q, 1e-13));
        check("unfiltered bias gets adapted weight decay",
              near(layer->bias[0][0], 2.0 - 0.1 * expected_q * 2.0, 1e-12));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T10: two layers keep independent state and trust-ratio diagnostics.
    // ---------------------------------------------------------------------
    cout << "T10: multi-layer state isolation\n";
    {
        Model model;
        Dense* layer1 = new Dense(2, 2);
        Dense* layer2 = new Dense(2, 2);
        zero_dense(layer1);
        zero_dense(layer2);
        layer1->weights[0][0] = 1.0;
        layer2->weights[0][0] = 4.0;
        layer1->grad_weights[0][0] = 2.0;
        layer2->grad_weights[0][0] = 1.0;
        model.add_layer(layer1);
        model.add_layer(layer2);
        LARS optimizer(0.1, 0.5, 0.0, 0.1, 1e-12, false, false);
        optimizer.step(model);

        check("state exists for layer 1", optimizer.has_state(layer1));
        check("state exists for layer 2", optimizer.has_state(layer2));
        Tensor m1 = optimizer.get_momentum_buffer(layer1, 0);
        Tensor m2 = optimizer.get_momentum_buffer(layer2, 0);
        check("layer states contain distinct scaled updates", !near(m1[0][0], m2[0][0]));
        double q1 = 0.0, q2 = 0.0;
        optimizer.get_last_trust_ratio(layer1, 0, q1);
        optimizer.get_last_trust_ratio(layer2, 0, q2);
        check("layer trust ratios are independently computed", !near(q1, q2));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T11: scheduler writes the inherited lr field consumed by step().
    // ---------------------------------------------------------------------
    cout << "T11: LR scheduler compatibility\n";
    {
        Model model;
        Dense* layer = new Dense(1, 1);
        zero_dense(layer);
        layer->grad_bias[0][0] = 1.0;
        model.add_layer(layer);

        LARS optimizer(0.1, 0.0, 0.0, 0.001, 1e-8);
        PolynomialLR scheduler(0.1, 2, 1.0, 0.0);
        scheduler.set_optimizer(&optimizer);
        scheduler.step(model);  // inherited lr becomes 0.05
        check("scheduler updates LARS get_lr()", near(optimizer.get_lr(), 0.05));
        optimizer.step(model);
        check("step consumes scheduled lr (bias shift = 0.05)", near(layer->bias[0][0], -0.05));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T12: deterministic state/update trajectory.
    // ---------------------------------------------------------------------
    cout << "T12: determinism\n";
    {
        auto first = run_deterministic_trajectory();
        auto second = run_deterministic_trajectory();
        check("weight trajectory is bit-identical", near(first.first, second.first, 0.0));
        check("bias trajectory is bit-identical", near(first.second, second.second, 0.0));
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T13: end-to-end optimization on an asymmetric two-output regression.
    // ---------------------------------------------------------------------
    cout << "T13: end-to-end training reduces loss\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        model.add_layer(layer);
        LARS optimizer(0.15, 0.9, 0.0, 0.2, 1e-8);

        Tensor x(6, 2);
        const double raw_x[12] = {1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                                  -1.0, 0.5, 0.5, -1.0, 2.0, -0.5};
        for (size_t i = 0; i < 6; ++i)
            for (size_t j = 0; j < 2; ++j)
                x[i][j] = raw_x[2 * i + j];

        Tensor target(6, 2);
        for (size_t i = 0; i < 6; ++i) {
            target[i][0] = 2.0 * x[i][0] - x[i][1] + 0.5;
            target[i][1] = -x[i][0] + 3.0 * x[i][1] - 0.25;
        }

        auto loss = [&]() {
            Tensor prediction = layer->forward(x);
            double total = 0.0;
            for (size_t i = 0; i < prediction.rows; ++i)
                for (size_t j = 0; j < prediction.cols; ++j) {
                    const double error = prediction[i][j] - target[i][j];
                    total += error * error;
                }
            return total / static_cast<double>(prediction.rows * prediction.cols);
        };

        const double initial_loss = loss();
        for (int step = 0; step < 250; ++step) {
            Tensor prediction = layer->forward(x);
            Tensor grad(prediction.rows, prediction.cols);
            const double scale = 2.0 / static_cast<double>(prediction.rows * prediction.cols);
            for (size_t i = 0; i < prediction.rows; ++i)
                for (size_t j = 0; j < prediction.cols; ++j)
                    grad[i][j] = scale * (prediction[i][j] - target[i][j]);
            layer->backward(grad, 0.0);
            optimizer.step(model);
        }
        const double final_loss = loss();
        cout << "  loss: " << initial_loss << " -> " << final_loss << '\n';
        check("training loss is finite", std::isfinite(final_loss));
        check("LARS reduces loss by at least 95%", final_loss < initial_loss * 0.05);
    }
    cout << '\n';

    // ---------------------------------------------------------------------
    // T14: optimizer consumes and clears accumulated gradients.
    // ---------------------------------------------------------------------
    cout << "T14: gradients are cleared after step\n";
    {
        Model model;
        Dense* layer = new Dense(2, 2);
        zero_dense(layer);
        layer->grad_weights[0][0] = 1.0;
        layer->grad_bias[0][0] = -2.0;
        model.add_layer(layer);
        LARS optimizer;
        optimizer.step(model);
        check("weight gradients cleared", near(tensor_l1(layer->grad_weights), 0.0));
        check("bias gradients cleared", near(tensor_l1(layer->grad_bias), 0.0));
    }

    cout << "\n=== LARS: " << passed << " passed, " << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
