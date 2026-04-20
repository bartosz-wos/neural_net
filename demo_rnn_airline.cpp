#include "nn/nn.h"
#include "nn/layers/rnn.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cmath>

// Read CSV: returns vector of passenger counts (int)
std::vector<double> load_airline(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open dataset: " + path);
    std::string line;
    std::vector<double> series;
    bool first = true;
    while (std::getline(in, line)) {
        if (first) { first = false; continue; } // skip header
        if (line.empty()) continue;
        // Split by comma, remove quotes
        std::stringstream ss(line);
        std::string month, count_str;
        std::getline(ss, month, ',');
        std::getline(ss, count_str, ',');
        // strip quotes
        if (!count_str.empty() && count_str.front() == '"') count_str.erase(0,1);
        if (!count_str.empty() && count_str.back() == '"') count_str.pop_back();
        double val = std::stod(count_str);
        series.push_back(val);
    }
    return series;
}

int main() {
    std::cout << "=== RNN Time Series: Air Passengers Forecasting ===\n\n";

    auto series = load_airline("data_airline_passengers.csv");
    int N = series.size();
    std::cout << "Loaded " << N << " monthly observations (1949-01 to 1960-12).\n";

    // Hyperparameters
    const int seq_len = 12; // use 12 months to predict next month
    const int hidden_size = 32;
    const double lr = 0.005;
    const int epochs = 500;
    const int test_months = 24; // hold out last 2 years (24 months) for test

    // Total sequences = N - seq_len
    int total_seq = N - seq_len;
    int train_seq = total_seq - test_months;
    int test_seq = total_seq - train_seq;

    std::cout << "Sequence length: " << seq_len << "\n";
    std::cout << "Training sequences: " << train_seq << "\n";
    std::cout << "Test sequences: " << test_seq << "\n\n";

    // Compute min/max from training portion only (target values)
    // Training targets correspond to series[seq_len .. seq_len+train_seq-1]
    double y_min = series[seq_len], y_max = series[seq_len];
    for (int i = 1; i < train_seq; ++i) {
        double v = series[seq_len + i];
        if (v < y_min) y_min = v;
        if (v > y_max) y_max = v;
    }
    double y_range = y_max - y_min;
    std::cout << "Target range (train): [" << y_min << ", " << y_max << "]\n\n";

    // Build dataset
    auto build_sequence = [&](int start_idx) -> std::pair<std::vector<double>, double> {
        std::vector<double> seq;
        for (int t = 0; t < seq_len; ++t) {
            double val = series[start_idx + t];
            seq.push_back((val - y_min) / y_range);
        }
        double target = series[start_idx + seq_len];
        double target_norm = (target - y_min) / y_range;
        return {seq, target_norm};
    };

    std::vector<std::vector<double>> X_train_vec, X_test_vec;
    std::vector<double> y_train_vec, y_test_vec;

    for (int i = 0; i < train_seq; ++i) {
        auto [seq, target] = build_sequence(i);
        X_train_vec.push_back(seq);
        y_train_vec.push_back(target);
    }
    for (int i = train_seq; i < total_seq; ++i) {
        auto [seq, target] = build_sequence(i);
        X_test_vec.push_back(seq);
        y_test_vec.push_back(target);
    }

    Tensor X_train = Tensor(X_train_vec);
    Tensor X_test = Tensor(X_test_vec);
    Tensor y_train(train_seq, 1);
    for (int i = 0; i < train_seq; ++i) y_train[i][0] = y_train_vec[i];
    Tensor y_test(test_seq, 1);
    for (int i = 0; i < test_seq; ++i) y_test[i][0] = y_test_vec[i];

    std::cout << "X_train: " << X_train.rows << " x " << X_train.cols << "\n";
    std::cout << "X_test:  " << X_test.rows << " x " << X_test.cols << "\n\n";

    // Model: RNN + Dense
    Model model;
    model.add_layer(new SimpleRNN(1, hidden_size, seq_len)); // input_dim=1 (per time step)
    model.add_layer(new Dense(hidden_size, 1)); // linear output

    std::cout << "Training RNN (hidden=" << hidden_size << ") with SGD, LR=" << lr << "\n";
    std::cout << "Epochs: " << epochs << "\n\n";

    model.train(X_train, y_train, lr, epochs);

    // Evaluate on test set
    double test_loss = model.evaluate(X_test, y_test);
    std::cout << "\nTest MSE (normalized): " << test_loss << "\n";

    // Predictions on test
    std::cout << "\n--- Sample Predictions (first 12 test points) ---\n";
    int to_show = std::min(12, (int)X_test.rows);
    for (int i = 0; i < to_show; ++i) {
        Tensor in = Tensor(std::vector<std::vector<double>>{X_test_vec[i]});
        Tensor pred_norm = model.forward(in);
        double pred_denorm = pred_norm[0][0] * y_range + y_min;
        double true_denorm = y_test_vec[i] * y_range + y_min;
        std::cout << "t+" << seq_len + i << "  pred=" << pred_denorm
                  << "  actual=" << true_denorm
                  << "  err=" << std::abs(pred_denorm - true_denorm) << "\n";
    }

    // Forecast future beyond dataset? (optional)
    // Not now.

    return 0;
}
