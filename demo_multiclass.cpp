#include "nn/nn.h"
// removed (via nn.h)
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== Multi-class Classification Demo ===\n\n";

    // Simple 3-class dataset in 2D
    std::vector<std::vector<double>> X_data = {
        {-2, 0}, {-1, 0},   // class 0
        {0, 1},  {0, 2},    // class 1
        {2, 0},  {3, 0}     // class 2
    };
    std::vector<std::vector<double>> y_data = {
        {1, 0, 0}, {1, 0, 0},
        {0, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {0, 0, 1}
    };

    Tensor X(X_data);
    Tensor y(y_data);

    std::cout << "Samples: 6, Input dim: 2, Classes: 3\n\n";

    Model model = create_mlp({2, 8, 3}, "tanh"); // logits output (no activation on last layer)
    std::cout << "Architecture: 2 -> 8 -> 3 (logits), tanh hidden\n";
    std::cout << "Optimizer: SGD, LR=0.1, epochs=3000\n\n";

    model.train_cross_entropy(X, y, 0.1, 3000);

    std::cout << "\n--- Predictions ---\n";
    int correct = 0;
    for (size_t i = 0; i < X.rows; ++i) {
        Tensor in(1, X.cols, &X.data[i * X.cols]);
        Tensor logits = model.forward(in);
        Tensor probs = Softmax()(logits);

        // Find predicted class
        int pred = 0;
        double maxp = probs[0][0];
        for (size_t c = 1; c < probs.cols; ++c) {
            if (probs[0][c] > maxp) {
                maxp = probs[0][c];
                pred = c;
            }
        }

        // True class
        int true_label = 0;
        for (size_t c = 0; c < y.cols; ++c) {
            if (y[i][c] > 0.5) {
                true_label = c;
                break;
            }
        }

        if (pred == true_label) correct++;

        std::cout << "(" << X[i][0] << ", " << X[i][1] << ") => prob: [";
        for (size_t c = 0; c < probs.cols; ++c) {
            std::cout << probs[0][c];
            if (c + 1 < probs.cols) std::cout << ", ";
        }
        std::cout << "]  pred=" << pred << "  true=" << true_label << "\n";
    }

    double acc = 100.0 * correct / static_cast<double>(X.rows);
    std::cout << "\nAccuracy: " << correct << "/" << X.rows << " (" << acc << "%)\n";

    return 0;
}
