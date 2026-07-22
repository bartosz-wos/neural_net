#include "../optimizers/lookahead.h"
#include <stdexcept>
#include <utility>

Lookahead::Lookahead(Optimizer* inner, int k, double alpha)
    : inner_(inner), k_(k), alpha_(alpha), steps_(0) {
    if (!inner_) {
        throw std::invalid_argument("Lookahead: inner optimizer must not be null");
    }
    if (k_ <= 0) {
        throw std::invalid_argument("Lookahead: k must be > 0");
    }
    if (alpha_ < 0.0 || alpha_ > 1.0) {
        throw std::invalid_argument("Lookahead: alpha must be in [0, 1]");
    }
    Optimizer::lr = inner_->lr;
}

void Lookahead::initialize_slow_weights(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        const std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor> copies;
        copies.reserve(params.size());
        for (Tensor* param : params) {
            if (!param) {
                throw std::runtime_error("Lookahead: null parameter pointer");
            }
            copies.push_back(param->clone());
        }
        slow_weights_[layer_ptr] = std::move(copies);
    }
}

void Lookahead::synchronize(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        const std::vector<Tensor*> params = layer->parameters();
        const auto state_it = slow_weights_.find(layer_ptr);
        if (state_it == slow_weights_.end()) {
            throw std::runtime_error("Lookahead: layer added after state initialization");
        }
        std::vector<Tensor>& slow_params = state_it->second;
        if (params.size() != slow_params.size()) {
            throw std::runtime_error("Lookahead: parameter count changed after initialization");
        }

        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* fast = params[param_idx];
            Tensor& slow = slow_params[param_idx];
            if (!fast || fast->rows != slow.rows || fast->cols != slow.cols) {
                throw std::runtime_error("Lookahead: parameter shape changed after initialization");
            }
            for (size_t row = 0; row < fast->rows; ++row) {
                for (size_t col = 0; col < fast->cols; ++col) {
                    slow[row][col] += alpha_ * ((*fast)[row][col] - slow[row][col]);
                    (*fast)[row][col] = slow[row][col];
                }
            }
        }
    }
}

Tensor Lookahead::get_slow_weight(void* layer_ptr, size_t param_idx) const {
    const auto layer_it = slow_weights_.find(layer_ptr);
    if (layer_it == slow_weights_.end() || param_idx >= layer_it->second.size()) {
        return Tensor(0, 0);
    }
    return layer_it->second[param_idx].clone();
}

void Lookahead::step(Model& model) {
    if (steps_ == 0) {
        initialize_slow_weights(model);
    }

    inner_->step(model);
    ++steps_;
    Optimizer::lr = inner_->lr;

    if (steps_ % static_cast<size_t>(k_) == 0) {
        synchronize(model);
    }
}
