#include "nn/nn.h"
// removed (via nn.h)
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Neural Network Demo (XOR, bigger net) ===\n\n";

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

    // Deeper + wider: 2 -> 16 -> 8 -> 1
    std::cout << "Architecture: 2 -> 16 -> 8 -> 1, tanh hidden, sigmoid output\n";
    std::cout << "Training for 5000 epochs, LR=0.1\n\n";

    Model model = create_mlp({2, 16, 8, 1}, "tanh");
    model.add_layer(new Activation<Sigmoid>(Sigmoid{}));

    try {
        model.train(X, y, 0.1, 5000);
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n--- Predictions ---\n";
    for (size_t i = 0; i < X.rows; ++i) {
        Tensor in = Tensor(std::vector<std::vector<double>>{X[i]});
        Tensor pred = model.forward(in);
        double p = pred[0][0];
        std::cout << "[" << X[i][0] << ", " << X[i][1] << "] => " << p
                  << " (≈ " << (p > 0.5 ? 1 : 0) << ")\n";
    }

    double loss = model.evaluate(X, y);
    std::cout << "\nFinal MSE loss: " << loss << "\n";

    return 0;
}
