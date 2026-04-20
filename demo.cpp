#include "nn/nn.h"
// removed (via nn.h)
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Neural Network Demo ===\n\n";

    // XOR dataset
    std::vector<std::vector<double>> X_data = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    std::vector<std::vector<double>> y_data = {
        {0},
        {1},
        {1},
        {0}
    };

    Tensor X(X_data);
    Tensor y(y_data);

    std::cout << "Training XOR problem (2-4-4-1 MLP with ReLU hidden, sigmoid output)\n";
    std::cout << "Samples: 4, Input dim: 2, Output dim: 1\n\n";

    // Create model: 2 -> 4 -> 1 (single hidden layer) with tanh hidden, sigmoid output
    Model model = create_mlp({2, 4, 1}, "tanh");
    // Sigmoid output already added by create_mlp for last layer? Wait create_mlp does not add activation after last Dense, so we need to add it.
    model.add_layer(new Activation<Sigmoid>(Sigmoid{}));

    // Quick forward test
    Tensor sample(1, X_data[0].size(), X_data[0].data());
    try {
        Tensor out = model.forward(sample);
        std::cout << "Forward test passed. Output shape: " << out.rows << "x" << out.cols << "\n\n";
    } catch (const std::exception& e) {
        std::cerr << "Forward failed: " << e.what() << std::endl;
        return 1;
    }

    try {
        model.train(X, y, 0.5, 2000);
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Predictions after training ---\n";
    for (size_t i = 0; i < X.rows; ++i) {
        Tensor in(1, X.cols, &X.data[i * X.cols]);
        Tensor pred = model.forward(in);
        double prob = pred[0][0];
        std::cout << "Input: [" << X[i][0] << ", " << X[i][1] << "] => ";
        std::cout << "output: " << prob << " (≈ " << (prob > 0.5 ? 1 : 0) << ")\n";
    }

    double final_loss = model.evaluate(X, y);
    std::cout << "\nFinal MSE loss: " << final_loss << "\n";

    // Save model
    model.save("xor_model.nn");
    std::cout << "Model saved to xor_model.nn\n";

    return 0;
}
