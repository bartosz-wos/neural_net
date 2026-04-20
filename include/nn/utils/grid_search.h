#ifndef GRID_SEARCH_H
#define GRID_SEARCH_H

#include "../core/model.h"
#include <vector>
#include <map>
#include <string>
#include <functional>

struct GridSearchResult {
    std::map<std::string, std::string> params;
    double best_score;
    std::string best_params_str;
};

class GridSearchCV {
public:
    std::vector<std::map<std::string, std::string>> param_grid;
    int cv_folds;
    std::function<double(Model&, const Tensor&, const Tensor&, const Tensor&, const Tensor&)> scorer;

    GridSearchCV(int cv_folds = 3) : cv_folds(cv_folds) {}

    void add_param(const std::string& name, const std::vector<std::string>& values) {
        param_grid.push_back({{name, values[0]}});
    }

    GridSearchResult fit(
        std::function<Model(const std::map<std::string, std::string>&)> model_fn,
        const Tensor& X, const Tensor& y,
        const Tensor& X_val, const Tensor& y_val
    );

    static std::vector<std::map<std::string, std::string>> cartesian_product(
        const std::vector<std::map<std::string, std::string>>& grids
    );
};

#endif