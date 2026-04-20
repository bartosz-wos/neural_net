#include "nn/nn.h"
#include "nn/layers/convolutions/conv_layer.h"
// removed (via nn.h)
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== CNN on XOR Parity (2x2 images) ===\n\n";

    // Each sample is a 2x2 single-channel image flattened row-major:
    // Bits are placed in top row; bottom row zeros.
    // Label = parity (odd number of 1s => 1, even => 0)
    std::vector<std::vector<double>> X_data = {
        {0, 0, 0, 0}, // 00 -> 0
        {0, 1, 0, 0}, // 01 -> 1
        {1, 0, 0, 0}, // 10 -> 1
        {1, 1, 0, 0}  // 11 -> 0
    };
    std::vector<std::vector<double>> y_data = {
        {0}, {1}, {1}, {0}
    };

    const int img_h = 2, img_w = 2, img_c = 1;
    Tensor X(X_data);
    Tensor y(y_data);
    int N = X.rows;

    // Model: Conv2D(1, 2, 2x2) -> Tanh -> Dense(2 -> 1) -> Sigmoid
    Model model;
    model.add_layer(new Conv2D(img_c, 2, 2, 2, img_h, img_w, 1, 1, 0, 0));
    model.add_layer(new Activation<Tanh>(Tanh{}));
    model.add_layer(new Dense(2, 1));
    model.add_layer(new Activation<Sigmoid>(Sigmoid{}));

    std::cout << "Architecture:\n";
    std::cout << "  Input: 2x2 single-channel image (flattened to 4 features)\n";
    std::cout << "  Conv2D: 1 -> 2 channels, 2x2 kernel, stride 1, no padding\n";
    std::cout << "    Output spatial: 1x1 per channel -> 2 features\n";
    std::cout << "  Tanh activation\n";
    std::cout << "  Dense: 2 -> 1\n";
    std::cout << "  Sigmoid output\n\n";
    std::cout << "Training: SGD, LR=0.1, epochs=5000\n\n";

    model.train(X, y, 0.1, 5000);

    std::cout << "\n--- Predictions ---\n";
    int correct = 0;
    for (int i = 0; i < N; ++i) {
        Tensor in(1, X.cols, &X.data[i * X.cols]);
        Tensor pred = model.forward(in);
        double p = pred[0][0];
        int pred_label = (p > 0.5) ? 1 : 0;
        if (pred_label == static_cast<int>(y[i][0])) correct++;
        std::cout << "Sample " << i << ": output=" << p
                  << " (≈" << pred_label << ")  true=" << y[i][0] << "\n";
    }
    double acc = 100.0 * correct / N;
    std::cout << "\nAccuracy: " << correct << "/" << N << " (" << acc << "%)\n";

    return 0;
}
