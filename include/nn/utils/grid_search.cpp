#include "grid_search.h"
#include "../activations/activations.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <cassert>

std::vector<std::map<std::string, std::string>> GridSearchCV::cartesian_product(
    const std::vector<std::map<std::string, std::string>>& grids
) {
    if (grids.empty()) return {{}};

    std::vector<std::map<std::string, std::string>> result = {{}};
    for (const auto& grid : grids) {
        std::vector<std::map<std::string, std::string>> new_result;
        for (const auto& combo : result) {
            for (const auto& kv : grid) {
                auto new_combo = combo;
                new_combo[kv.first] = kv.second;
                new_result.push_back(std::move(new_combo));
            }
        }
        result = std::move(new_result);
    }
    return result;
}

GridSearchResult GridSearchCV::fit(
    std::function<Model(const std::map<std::string, std::string>&)> model_fn,
    const Tensor& X, const Tensor& y,
    const Tensor& X_val, const Tensor& y_val
) {
    GridSearchResult best;
    best.best_score = -1e9;

    auto combinations = cartesian_product(param_grid);

    std::cout << "[GridSearchCV] Testing " << combinations.size() << " parameter combinations...\n";

    int combo_idx = 0;
    for (const auto& params : combinations) {
        combo_idx++;
        std::ostringstream oss;
        for (auto it = params.begin(); it != params.end(); ++it) {
            if (it != params.begin()) oss << ", ";
            oss << it->first << "=" << it->second;
        }
        std::string params_str = oss.str();
        std::cout << "  [" << combo_idx << "/" << combinations.size() << "] " << params_str << "\n";

        Model model = model_fn(params);
        double lr = 0.01;
        int epochs = 50;

        auto it_lr = params.find("lr");
        if (it_lr != params.end()) lr = std::stod(it_lr->second);

        auto it_epochs = params.find("epochs");
        if (it_epochs != params.end()) epochs = std::stoi(it_epochs->second);

        // Train
        model.train(X, y, lr, epochs);

        // Evaluate
        double score = 0.0;
        // Simple accuracy on validation set
        Tensor pred = model.forward(X_val);
        assert(pred.rows == y_val.rows && "GridSearch: pred and y_val must have same row count");
        assert(pred.cols == y_val.cols && "GridSearch: pred and y_val must have same col count");
        int correct = 0;
        size_t n = X_val.rows;
        for (size_t i = 0; i < n; ++i) {
            // Find argmax
            size_t pred_class = 0, true_class = 0;
            double pred_max = pred[i][0];
            double true_max = y_val[i][0];
            for (size_t j = 1; j < pred.cols; ++j) {
                if (pred[i][j] > pred_max) { pred_max = pred[i][j]; pred_class = j; }
                if (y_val[i][j] > true_max) { true_max = y_val[i][j]; true_class = j; }
            }
            if (pred_class == true_class) correct++;
        }
        score = (double)correct / n;

        std::cout << "    -> score: " << std::fixed << std::setprecision(4) << score << "\n";

        if (score > best.best_score) {
            best.best_score = score;
            best.params = params;
            best.best_params_str = params_str;
        }
    }

    std::cout << "[GridSearchCV] Best: " << best.best_params_str
              << " -> " << std::fixed << std::setprecision(4) << best.best_score << "\n";

    return best;
}