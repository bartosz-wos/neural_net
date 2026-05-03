#include "../optimizers/lookahead.h"

void Lookahead::sync_slow_weights(Model& model) {
    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer_ptr->parameters();
        auto it = slow_weights_.find((void*)layer_ptr);
        if (it == slow_weights_.end()) continue;
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor& slow = it->second[i];
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c = 0; c < p->cols; ++c) {
                    (*p)[r][c] = (*p)[r][c] + alpha_ * (slow[r][c] - (*p)[r][c]);
                }
            }
        }
    }
}

void Lookahead::update_slow_weights(Model& model) {
    for (auto& layer : model.layers) {
        auto* layer_ptr = layer.get();
        auto params = layer_ptr->parameters();
        auto it = slow_weights_.find((void*)layer_ptr);
        if (it == slow_weights_.end()) continue;
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor& slow = it->second[i];
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c = 0; c < p->cols; ++c) {
                    slow[r][c] = (*p)[r][c];
                }
            }
        }
    }
}

Lookahead::Lookahead(Optimizer* inner, int k, double alpha)
    : inner_(inner), k_(k), alpha_(alpha), steps_(0) {}

void Lookahead::step(Model& model) {
    // Every k steps, snapshot current weights into slow_weights_ before inner_->step
    if (steps_ > 0 && steps_ % k_ == 0) {
        for (auto& layer : model.layers) {
            auto* layer_ptr = layer.get();
            auto params = layer_ptr->parameters();
            auto it = slow_weights_.find((void*)layer_ptr);
            if (it == slow_weights_.end()) continue;
            for (size_t i = 0; i < params.size(); ++i) {
                Tensor* p = params[i];
                Tensor& slow = it->second[i];
                for (size_t r = 0; r < p->rows; ++r) {
                    for (size_t c = 0; c < p->cols; ++c) {
                        slow[r][c] = (*p)[r][c];
                    }
                }
            }
        }
    }

    // Initialize slow_weights_ on first step (before any updates)
    if (steps_ == 0) {
        for (auto& layer : model.layers) {
            auto* layer_ptr = layer.get();
            auto params = layer_ptr->parameters();
            std::vector<Tensor> copies;
            for (Tensor* p : params) {
                Tensor copy(p->rows, p->cols);
                for (size_t r = 0; r < p->rows; ++r) {
                    for (size_t c = 0; c < p->cols; ++c) {
                        copy[r][c] = (*p)[r][c];
                    }
                }
                copies.push_back(std::move(copy));
            }
            slow_weights_[(void*)layer_ptr] = std::move(copies);
        }
    }

    // Run inner optimizer step (updates weights using inner's own LR)
    inner_->step(model);
    ++steps_;

    // Every k steps: interpolate model weights toward slow_weights_, then update slow_weights_
    if (steps_ % k_ == 0) {
        sync_slow_weights(model);
        update_slow_weights(model);
    }
}

