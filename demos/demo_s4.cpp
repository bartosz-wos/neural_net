#include "nn/nn.h"
#include <iostream>
#include <cmath>

int main() {
    std::cout << "=== S4 Layer Demo ===\n\n";

    // --- Parameters ---
    const int d_model = 8;
    const int d_state = 4;
    const int seq_len = 16;

    // --- Create S4 layer ---
    S4Layer s4(d_model, d_state);
    std::cout << "S4Layer created: d_model=" << d_model << ", d_state=" << d_state << "\n";

    // --- Create random input tensor (d_model, seq_len) ---
    // S4 expects input.rows == d_model_ and input.cols == seq_len
    Tensor input(d_model, seq_len);
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = (double(rand()) / RAND_MAX) * 0.5 - 0.25;  // uniform in [-0.25, 0.25]
    }
    std::cout << "Input shape: (" << d_model << ", " << seq_len << ")  [d_model x seq_len]\n";

    // --- Forward pass ---
    try {
        Tensor output = s4.forward(input);
        std::cout << "Forward pass OK. Output shape: (" << output.rows << ", " << output.cols << ")\n";
        if (output.rows != d_model || output.cols != seq_len) {
            std::cerr << "ERROR: Unexpected output shape!\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Forward failed: " << e.what() << "\n";
        return 1;
    }

    // --- Backward pass ---
    // Create upstream gradient tensor (same shape as output)
    Tensor grad_output(d_model, seq_len);
    for (size_t i = 0; i < grad_output.data.size(); ++i) {
        grad_output.data[i] = (double(rand()) / RAND_MAX) * 0.1 - 0.05;
    }

    try {
        Tensor grad_input = s4.backward(grad_output, 0.01);
        std::cout << "Backward pass OK. Grad input shape: (" << grad_input.rows << ", " << grad_input.cols << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Backward failed: " << e.what() << "\n";
        return 1;
    }

    // --- Training loop with simple MSE loss ---
    std::cout << "\n--- Training (d_model=" << d_model << ", seq_len=" << seq_len << ") ---\n";

    // Target: shifted version of input (simple sequence prediction task)
    // y[t] = input[t-1] (wrap around at t=0)
    Tensor target(d_model, seq_len);
    for (int d = 0; d < d_model; ++d) {
        for (int t = 0; t < seq_len; ++t) {
            int t_prev = (t == 0) ? seq_len - 1 : t - 1;
            target(d, t) = input(d, t_prev);
        }
    }

    double prev_loss = 0.0;
    const int epochs = 50;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward
        Tensor pred = s4.forward(input);

        // MSE loss: mean over all elements of (pred - target)^2
        double loss = 0.0;
        for (size_t i = 0; i < pred.data.size(); ++i) {
            double diff = pred.data[i] - target.data[i];
            loss += diff * diff;
        }
        loss /= pred.data.size();

        // Backward (grad of MSE is 2*(pred - target) / N)
        Tensor grad_loss(pred.rows, pred.cols);
        for (size_t i = 0; i < pred.data.size(); ++i) {
            grad_loss.data[i] = 2.0 * (pred.data[i] - target.data[i]) / pred.data.size();
        }

        s4.backward(grad_loss, 0.01);
        s4.update_weights(0.01);

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " | MSE loss: " << loss
                      << " | |Δ|=" << std::abs(loss - prev_loss) << "\n";
        }
        prev_loss = loss;
    }

    std::cout << "\nFinal MSE loss: " << prev_loss << "\n";

    // --- Summary ---
    std::cout << "\n--- Summary ---\n";
    std::cout << "S4Layer: d_model=" << d_model << ", d_state=" << d_state << "\n";
    std::cout << "Input/Output shape: (" << d_model << ", " << seq_len << ")\n";
    std::cout << "Parameter shapes:\n";
    std::cout << "  x_proj: (" << s4.x_proj.rows << ", " << s4.x_proj.cols << ")\n";
    std::cout << "  W_out:  (" << s4.W_out.rows << ", " << s4.W_out.cols << ")\n";
    std::cout << "  b_out:  (" << s4.b_out.rows << ", " << s4.b_out.cols << ")\n";
    std::cout << "\nDemo complete.\n";
    return 0;
}
