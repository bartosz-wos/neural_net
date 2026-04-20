#include "nn/nn.h"
#include "nn/layers/conv_layer.h"
#include "nn/layers/pool_layer.h"
// removed (via nn.h)
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// Generate 6x6 pattern images for 3 classes
// Class 0: horizontal middle line (row 2)
// Class 1: vertical middle line (col 2)
// Class 2: crossed diagonals (both diagonals)
std::vector<double> generate_sample(int cls, double noise=0.1) {
    const int H = 6, W = 6;
    std::vector<double> img(H * W, 0.0);
    std::mt19937 gen(42 + cls);
    std::uniform_real_distribution<> flip(0.0, 1.0);

    // Base pattern
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double val = 0.0;
            if (cls == 0) {
                // horizontal line at row 2 (0-indexed: row 2)
                if (i == 2) val = 1.0;
            } else if (cls == 1) {
                // vertical line at col 2
                if (j == 2) val = 1.0;
            } else if (cls == 2) {
                // both diagonals: i == j or i + j == 5
                if (i == j || i + j == 5) val = 1.0;
            }
            // Add noise
            if (flip(gen) < noise) val = 1.0 - val;
            img[i * W + j] = val;
        }
    }
    return img;
}

int main() {
    std::cout << "=== Multi-Conv CNN on 3-Class 6x6 Dataset ===\n\n";

    const int N_per_class = 50;
    const int total_samples = 3 * N_per_class;

    // Build dataset
    std::vector<std::vector<double>> X_data;
    std::vector<std::vector<double>> y_data; // one-hot for 3 classes

    std::mt19937 gen(123);
    std::vector<int> indices(total_samples);
    for (int i = 0; i < total_samples; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), gen);

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < N_per_class; ++i) {
            auto img = generate_sample(c, 0.1);
            X_data.push_back(img);
            std::vector<double> y(3, 0.0);
            y[c] = 1.0;
            y_data.push_back(y);
        }
    }

    // Shuffle globally to mix classes
    std::vector<std::vector<double>> X_shuffled, y_shuffled;
    for (int idx : indices) {
        X_shuffled.push_back(X_data[idx]);
        y_shuffled.push_back(y_data[idx]);
    }

    Tensor X(X_shuffled);
    Tensor y(y_shuffled);

    const int img_h = 6, img_w = 6, img_c = 1;
    std::cout << "Dataset: " << X.rows << " samples, " << X.cols << " features (6x6)\n";
    std::cout << "Classes: 3 (horizontal line, vertical line, cross-diagonals)\n\n";

    // Architecture
    // Input (N, 36) -> reshape internally as (N, 1, 6, 6)
    // Conv2D(1->8, 3x3, stride=1, pad=1)  -> out: 8 x 6 x 6 = 288
    // ReLU
    // MaxPool2D(2x2, stride=2)             -> out: 8 x 3 x 3 = 72
    // Conv2D(8->16, 2x2, stride=1, pad=0) -> out: 16 x 2 x 2 = 64
    // ReLU
    // Dense(64->16)
    // Dense(16->3) logits

    Model model;
    model.add_layer(new Conv2D(1, 8, 3, 3, img_h, img_w, 1, 1, 1, 1));  // preserve 6x6
    model.add_layer(new Activation<ReLU>(ReLU{}));
    model.add_layer(new MaxPool2D(2, 2, 6, 6, 2, 2));  // -> 8 x 3 x 3
    model.add_layer(new Conv2D(8, 16, 2, 2, 3, 3, 1, 1, 0, 0)); // -> 16 x 2 x 2 = 64
    model.add_layer(new Activation<ReLU>(ReLU{}));
    model.add_layer(new Dense(16 * 2 * 2, 16)); // flatten conv output: 64 -> 16
    model.add_layer(new Dense(16, 3));          // logits

    std::cout << "Architecture:\n";
    std::cout << "  Conv2D(1->8, 3x3, pad=1) + ReLU  -> 8x6x6\n";
    std::cout << "  MaxPool2D(2x2, stride=2)          -> 8x3x3\n";
    std::cout << "  Conv2D(8->16, 2x2) + ReLU        -> 16x2x2\n";
    std::cout << "  Flatten -> Dense(64->16)\n";
    std::cout << "  Dense(16->3) logits\n\n";
    std::cout << "Training: cross-entropy, SGD LR=0.001, epochs=2000\n\n";

    try {
        model.train_cross_entropy(X, y, 0.001, 2000);
    } catch (const std::exception& e) {
        std::cerr << "Training error: " << e.what() << std::endl;
        return 1;
    }

    // Evaluate accuracy
    int correct = 0;
    std::cout << "\n--- Predictions (first 10 samples) ---\n";
    for (int i = 0; i < X.rows; ++i) {
        Tensor in = Tensor(std::vector<std::vector<double>>{X_shuffled[i]});
        Tensor logits = model.forward(in);
        Tensor probs = Softmax()(logits);
        int pred = 0;
        double maxp = probs[0][0];
        for (int c = 1; c < 3; ++c) {
            if (probs[0][c] > maxp) {
                maxp = probs[0][c];
                pred = c;
            }
        }
        int true_label = 0;
        for (int c = 0; c < 3; ++c) if (y_shuffled[i][c] > 0.5) true_label = c;
        if (pred == true_label) correct++;
        if (i < 10) {
            std::cout << "Sample " << i << ": pred=" << pred << " (p=" << maxp << ")  true=" << true_label << "\n";
        }
    }
    double acc = 100.0 * correct / static_cast<double>(X.rows);
    std::cout << "\nTrain accuracy: " << correct << "/" << X.rows << " (" << acc << "%)\n";

    return 0;
}
