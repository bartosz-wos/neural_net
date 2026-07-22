#include "signum.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <stdexcept>
#include <utility>

void Signum::validate(double lr, double beta, double weight_decay) {
    if (lr < 0.0) {
        throw std::invalid_argument("Signum: learning rate must be >= 0");
    }
    if (beta < 0.0 || beta >= 1.0) {
        throw std::invalid_argument("Signum: beta must be in [0, 1)");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("Signum: weight decay must be >= 0");
    }
}

Signum::Signum(double lr, double beta, double weight_decay)
    : beta_(beta), weight_decay_(weight_decay), num_steps_(0) {
    validate(lr, beta_, weight_decay_);
    Optimizer::lr = lr;
}

void Signum::set_lr(double new_lr) {
    validate(new_lr, beta_, weight_decay_);
    Optimizer::lr = new_lr;
}

void Signum::set_beta(double new_beta) {
    validate(Optimizer::lr, new_beta, weight_decay_);
    beta_ = new_beta;
}

void Signum::set_weight_decay(double new_weight_decay) {
    validate(Optimizer::lr, beta_, new_weight_decay);
    weight_decay_ = new_weight_decay;
}

void Signum::ensure_state(void* layer_ptr,
                          const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) {
        return;
    }

    std::vector<Tensor> layer_state;
    layer_state.reserve(params.size());
    for (Tensor* parameter : params) {
        Tensor momentum(parameter->rows, parameter->cols);
        momentum.fill(0.0);
        layer_state.push_back(std::move(momentum));
    }
    state_[layer_ptr] = std::move(layer_state);
}

Tensor Signum::get_momentum(void* layer_ptr, size_t param_idx) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return Tensor(0, 0);
    }
    return layer_it->second[param_idx].clone();
}

void Signum::update_param(Tensor* param, Tensor* grad, Tensor& momentum) {
    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            const double gradient = (*grad)[row][col];
            momentum[row][col] =
                beta_ * momentum[row][col] + (1.0 - beta_) * gradient;
            const double signed_momentum =
                (momentum[row][col] > 0.0)
                    ? 1.0
                    : ((momentum[row][col] < 0.0) ? -1.0 : 0.0);
            const double update =
                signed_momentum + weight_decay_ * (*param)[row][col];
            (*param)[row][col] -= Optimizer::lr * update;
        }
    }
}

void Signum::step(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) {
            continue;
        }
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "Signum: parameter and gradient counts must match");
        }

        ensure_state(layer_ptr, params);
        std::vector<Tensor>& layer_state = state_[layer_ptr];
        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* param = params[param_idx];
            Tensor* grad = grads[param_idx];
            if (param->rows != grad->rows || param->cols != grad->cols) {
                throw std::runtime_error(
                    "Signum: parameter and gradient shapes must match");
            }
            update_param(param, grad, layer_state[param_idx]);
        }
        layer->zero_grad();
    }
    ++num_steps_;
}
