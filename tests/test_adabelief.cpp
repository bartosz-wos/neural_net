#include <iostream>
#include <cmath>
#include <iomanip>
#include "nn/optimizers/adabelief.h"
#include "nn/core/model.h"
#include "nn/core/layer.h"

// Minimal gradient check - call layer's backward to populate gradients,
// then run optimizer step.
static double tensor_l2norm(const Tensor& t) {
    double s = 0.0;
    for (size_t i = 0; i < t.rows; ++i)
        for (size_t j = 0; j < t.cols; ++j)
            s += t[i][j] * t[i][j];
    return std::sqrt(s);
}

int main() {
    std::cout << std::setprecision(10);
    std::cout << "=== AdaBelief Optimizer Test ===\n\n";

    // Test 1: Zero gradient -> no change
    std::cout << "Test 1: Zero gradient should not change parameters\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        AdaBelief opt(0.1, 0.9, 0.999, 1e-8, 0.0);

        double init_w00 = layer->weights[0][0];
        double init_w10 = layer->weights[1][0];

        std::cout << "  Initial w[0][0]=" << init_w00 << " w[1][0]=" << init_w10 << "\n";
        opt.step(model);
        double new_w00 = layer->weights[0][0];
        double new_w10 = layer->weights[1][0];
        std::cout << "  After zero-grad step: w[0][0]=" << new_w00 << " w[1][0]=" << new_w10 << "\n";

        bool pass = (std::abs(new_w00 - init_w00) < 1e-10 &&
                     std::abs(new_w10 - init_w10) < 1e-10);
        std::cout << "  Result: " << (pass ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 2: Non-zero gradient should change parameters
    // Use forward+backward to properly populate gradients
    std::cout << "Test 2: Forward+backward populates gradients, optimizer updates params\n";
    {
        Model model;
        Dense* layer = new Dense(3, 2);
        model.add_layer(layer);

        // Initialize weights to deterministic values
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                layer->weights[r][c] = 0.1;
        layer->bias.fill(0.0);

        // Input and upstream gradient
        Tensor input(1, 3);
        input[0][0] = 1.0; input[0][1] = 1.0; input[0][2] = 1.0;
        Tensor grad_output(1, 2);
        grad_output[0][0] = 1.0; grad_output[0][1] = 1.0;

        // Forward pass
        Tensor output = layer->forward(input);
        std::cout << "  Forward output: " << output[0][0] << " " << output[0][1] << "\n";

        // Backward pass - populates grad_weights
        Tensor grad = layer->backward(grad_output, 0.0);
        double grad_norm = tensor_l2norm(layer->get_gradients());
        std::cout << "  Gradient L2 norm: " << grad_norm << "\n";

        double init_w00 = layer->weights[0][0];

        AdaBelief opt(0.5, 0.9, 0.999, 1e-8, 0.0);
        opt.step(model);

        double new_w00 = layer->weights[0][0];
        std::cout << "  Before step: w[0][0]=" << init_w00 << "\n";
        std::cout << "  After step:  w[0][0]=" << new_w00 << "\n";
        std::cout << "  Change: " << (new_w00 - init_w00) << "\n";

        bool pass = (std::abs(new_w00 - init_w00) > 1e-10);
        std::cout << "  Result: " << (pass ? "PASS (params changed)" : "FAIL") << "\n\n";
    }

    // Test 3: Multiple steps with constant gradient = momentum buildup
    std::cout << "Test 3: Multiple steps with constant gradient = momentum buildup\n";
    {
        Model model;
        Dense* layer = new Dense(3, 1);
        model.add_layer(layer);

        // Initialize weights
        for (size_t r = 0; r < layer->weights.rows; ++r)
            for (size_t c = 0; c < layer->weights.cols; ++c)
                layer->weights[r][c] = 0.0;
        layer->bias.fill(0.0);

        // Input: all ones, gradient: only affects first weight
        Tensor input(1, 3);
        input[0][0] = 1.0; input[0][1] = 0.0; input[0][2] = 0.0;
        Tensor grad_output(1, 1);
        grad_output[0][0] = 1.0;

        AdaBelief opt(0.1, 0.9, 0.999, 1e-8, 0.0);

        // Step 1
        layer->forward(input);
        layer->backward(grad_output, 0.0);
        opt.step(model);
        double after1 = layer->weights[0][0];
        std::cout << "  After step 1: w[0][0]=" << after1 << "\n";

        // Step 2 - same gradient
        layer->forward(input);
        layer->backward(grad_output, 0.0);
        opt.step(model);
        double after2 = layer->weights[0][0];
        std::cout << "  After step 2: w[0][0]=" << after2 << "\n";

        bool pass = (after2 < after1);  // should decrease further due to momentum
        std::cout << "  Result: " << (pass ? "PASS (momentum accumulating)" : "FAIL") << "\n\n";
    }

    // Test 4: Weight decay (AdamW style)
    std::cout << "Test 4: Weight decay check\n";
    std::cout << "  With weight_decay=0.1, parameters should shrink even with zero gradient\n";
    {
        Model model;
        Dense* layer = new Dense(2, 1);
        layer->weights[0][0] = 1.0;
        layer->weights[0][1] = 0.5;
        model.add_layer(layer);

        AdaBelief opt(0.1, 0.9, 0.999, 1e-8, 0.1);

        double w_before = layer->weights[0][0];
        std::cout << "  Before weight decay: w[0][0]=" << w_before << "\n";
        opt.step(model);  // zero gradient, but weight decay applies
        double w_after = layer->weights[0][0];
        std::cout << "  After weight decay:  w[0][0]=" << w_after << "\n";

        bool pass = (w_after < w_before);
        std::cout << "  Result: " << (pass ? "PASS (weight decreased)" : "FAIL") << "\n\n";
    }

    std::cout << "=== All tests complete ===\n";
    return 0;
}