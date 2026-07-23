#include "early_stopping.h"
#include <cmath>
#include <stdexcept>

InMemoryEarlyStopping::InMemoryEarlyStopping(size_t patience,
                             double min_delta,
                             InMemoryEarlyStoppingMode mode,
                             bool restore_best_weights)
    : patience_(patience),
      min_delta_(min_delta),
      mode_(mode),
      restore_best_weights_(restore_best_weights) {
    if (patience_ == 0) {
        throw std::invalid_argument("InMemoryEarlyStopping: patience must be > 0");
    }
    if (!std::isfinite(min_delta_) || min_delta_ < 0.0) {
        throw std::invalid_argument(
            "InMemoryEarlyStopping: min_delta must be finite and >= 0");
    }
}

std::vector<Tensor*> InMemoryEarlyStopping::collect_parameters(Model& model) {
    std::vector<Tensor*> parameters;
    for (auto& layer : model.layers) {
        std::vector<Tensor*> layer_parameters = layer->parameters();
        parameters.insert(parameters.end(),
                          layer_parameters.begin(), layer_parameters.end());
    }
    return parameters;
}

void InMemoryEarlyStopping::snapshot(const Model& model) {
    best_parameters_.clear();
    for (const auto& layer : model.layers) {
        std::vector<Tensor*> layer_parameters = layer->parameters();
        for (Tensor* parameter : layer_parameters) {
            best_parameters_.push_back(parameter->clone());
        }
    }
}

bool InMemoryEarlyStopping::is_improvement(double metric) const {
    if (!has_best_) return true;
    if (mode_ == InMemoryEarlyStoppingMode::MINIMIZE) {
        return metric < best_metric_ - min_delta_;
    }
    return metric > best_metric_ + min_delta_;
}

bool InMemoryEarlyStopping::step(double metric, Model& model) {
    if (stopped_) return true;
    if (!std::isfinite(metric)) {
        throw std::invalid_argument("InMemoryEarlyStopping: metric must be finite");
    }

    ++num_steps_;
    if (is_improvement(metric)) {
        has_best_ = true;
        best_metric_ = metric;
        best_step_ = num_steps_;
        num_bad_epochs_ = 0;
        snapshot(model);
        return false;
    }

    ++num_bad_epochs_;
    if (num_bad_epochs_ >= patience_) {
        stopped_ = true;
        if (restore_best_weights_) {
            restore_best(model);
        }
    }
    return stopped_;
}

void InMemoryEarlyStopping::restore_best(Model& model) const {
    if (!has_best_) {
        throw std::runtime_error(
            "InMemoryEarlyStopping: cannot restore before observing a metric");
    }

    std::vector<Tensor*> parameters = collect_parameters(model);
    if (parameters.size() != best_parameters_.size()) {
        throw std::runtime_error(
            "InMemoryEarlyStopping: model parameter count changed since best snapshot");
    }

    for (size_t index = 0; index < parameters.size(); ++index) {
        Tensor* parameter = parameters[index];
        const Tensor& best = best_parameters_[index];
        if (parameter->rows != best.rows || parameter->cols != best.cols) {
            throw std::runtime_error(
                "InMemoryEarlyStopping: model parameter shape changed since best snapshot");
        }
    }

    for (size_t index = 0; index < parameters.size(); ++index) {
        parameters[index]->data = best_parameters_[index].data;
    }
}

void InMemoryEarlyStopping::reset() {
    has_best_ = false;
    stopped_ = false;
    best_metric_ = 0.0;
    best_step_ = 0;
    num_steps_ = 0;
    num_bad_epochs_ = 0;
    best_parameters_.clear();
}
