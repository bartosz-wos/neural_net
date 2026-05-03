#include "nn/nn.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>

// Simple MLP that we can train twice: once with Adam, once with Lookahead(Adam)
// Returns vector of loss per epoch
std::vector<double> train_model(Model& model, const Tensor& X, const Tensor& y,
                                 Optimizer& opt, int epochs) {
    std::vector<double> losses;
    losses.reserve(epochs);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train(X, y, opt, 1);
        double loss = model.evaluate(X, y);
        losses.push_back(loss);

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "  Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::scientific << std::setprecision(6) << loss
                      << std::defaultfloat << "\n";
        }
    }
    return losses;
}

int main() {
    std::cout << "=== Lookahead Optimizer Demo ===\n\n";

    // ------------------------------------------------------------
    // Synthetic regression dataset: y = sin(x1) * cos(x2) + noise
    // ------------------------------------------------------------
    const int N = 200;
    std::vector<std::vector<double>> X_data;
    std::vector<std::vector<double>> y_data;

    for (int i = 0; i < N; ++i) {
        double x1 = (static_cast<double>(i % 20) / 19.0) * 4.0 - 2.0; // [-2, 2]
        double x2 = (static_cast<double>(i / 20) / 19.0) * 4.0 - 2.0; // [-2, 2]
        double noise = 0.05 * (static_cast<double>(rand()) / RAND_MAX - 0.5);
        double y = std::sin(x1) * std::cos(x2) + noise;

        X_data.push_back({x1, x2});
        y_data.push_back({y});
    }

    Tensor X(X_data);
    Tensor y(y_data);

    std::cout << "Dataset: " << N << " samples, input dim 2, output dim 1\n";
    std::cout << "Target: y = sin(x1) * cos(x2) + small noise\n\n";

    const int INPUT_DIM  = 2;
    const int HIDDEN     = 32;
    const int OUTPUT_DIM = 1;
    const int EPOCHS     = 50;
    const double LR      = 0.005;

    std::cout << "MLP: " << INPUT_DIM << " -> " << HIDDEN << " -> " << OUTPUT_DIM << "\n";
    std::cout << "Training for " << EPOCHS << " epochs, lr=" << LR << "\n\n";

    // ------------------------------------------------------------
    // Run 1: Adam only
    // ------------------------------------------------------------
    std::cout << "--- Training with Adam (baseline) ---\n";
    Model model_adam = create_mlp({INPUT_DIM, HIDDEN, OUTPUT_DIM}, "relu");
    model_adam.add_layer(new Activation<Sigmoid>(Sigmoid{})); // output activation for bounded targets
    Adam adam_opt(LR);
    auto losses_adam = train_model(model_adam, X, y, adam_opt, EPOCHS);

    // ------------------------------------------------------------
    // Run 2: Lookahead(Adam) with k=6, alpha=0.5
    // ------------------------------------------------------------
    std::cout << "\n--- Training with Lookahead(Adam, k=6, alpha=0.5) ---\n";
    Model model_la = create_mlp({INPUT_DIM, HIDDEN, OUTPUT_DIM}, "relu");
    model_la.add_layer(new Activation<Sigmoid>(Sigmoid{}));
    Adam inner_adam(LR);
    Lookahead look_adam(&inner_adam, 6, 0.5);
    auto losses_la = train_model(model_la, X, y, look_adam, EPOCHS);

    // ------------------------------------------------------------
    // Summary comparison
    // ------------------------------------------------------------
    std::cout << "\n=== Loss Curve Comparison ===\n";
    std::cout << std::left;
    std::cout << std::setw(8)  << "Epoch"
              << std::setw(16) << "Adam"
              << std::setw(16) << "Lookahead(Adam)"
              << "\n";
    std::cout << std::string(40, '-') << "\n";

    // Print every 5 epochs
    for (size_t e = 0; e < losses_adam.size(); ++e) {
        if (e % 5 == 0 || e == losses_adam.size() - 1) {
            std::cout << std::setw(8)  << e
                      << std::setw(16) << std::scientific << std::setprecision(4) << losses_adam[e]
                      << std::setw(16) << std::scientific << std::setprecision(4) << losses_la[e]
                      << std::defaultfloat << "\n";
        }
    }

    double final_adam = losses_adam.back();
    double final_la  = losses_la.back();
    std::cout << "\n--- Final Loss ---\n";
    std::cout << "Adam only:       " << std::scientific << std::setprecision(6) << final_adam << "\n";
    std::cout << "Lookahead(Adam): " << std::scientific << std::setprecision(6) << final_la  << "\n";

    double improvement = (final_adam - final_la) / final_adam * 100.0;
    if (final_la < final_adam) {
        std::cout << "Lookahead improved final loss by " << std::fixed << std::setprecision(2)
                  << improvement << "%\n";
    } else {
        std::cout << "Lookahead is comparable (delta: " << std::fixed << std::setprecision(2)
                  << (final_adam - final_la) / final_adam * 100.0 << "%)\n";
    }

    // ------------------------------------------------------------
    // Predictions sample
    // ------------------------------------------------------------
    std::cout << "\n--- Sample Predictions (Lookahead model) ---\n";
    std::vector<std::vector<double>> test_samples = {
        { 0.0,  0.0},
        { 1.0,  1.0},
        {-1.0,  1.0},
        { 1.5, -1.5},
    };
    for (const auto& s : test_samples) {
        Tensor in(1, 2, const_cast<double*>(s.data()));
        Tensor pred = model_la.forward(in);
        double expected = std::sin(s[0]) * std::cos(s[1]);
        std::cout << "Input: [" << s[0] << ", " << s[1] << "]"
                  << " | Pred: " << std::scientific << std::setprecision(4) << pred[0][0]
                  << " | Expected≈" << expected << "\n";
    }

    std::cout << "\nDemo complete.\n";
    return 0;
}
