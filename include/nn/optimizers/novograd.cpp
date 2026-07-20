#include "novograd.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

void NovoGrad::validate(double lr,
                        double beta1,
                        double beta2,
                        double epsilon,
                        double weight_decay) {
    if (lr < 0.0) {
        throw std::invalid_argument("NovoGrad: learning rate must be >= 0");
    }
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("NovoGrad: beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("NovoGrad: beta2 must be in [0, 1)");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("NovoGrad: epsilon must be > 0");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("NovoGrad: weight decay must be >= 0");
    }
}

NovoGrad::NovoGrad(double lr,
                   double beta1,
                   double beta2,
                   double epsilon,
                   double weight_decay,
                   bool grad_averaging,
                   bool amsgrad)
    : beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon),
      weight_decay_(weight_decay),
      grad_averaging_(grad_averaging),
      amsgrad_(amsgrad),
      num_steps_(0) {
    validate(lr, beta1_, beta2_, epsilon_, weight_decay_);
    this->Optimizer::lr = lr;
}

void NovoGrad::set_lr(double new_lr) {
    validate(new_lr, beta1_, beta2_, epsilon_, weight_decay_);
    this->Optimizer::lr = new_lr;
}

void NovoGrad::set_beta1(double new_beta1) {
    validate(Optimizer::lr, new_beta1, beta2_, epsilon_, weight_decay_);
    beta1_ = new_beta1;
}

void NovoGrad::set_beta2(double new_beta2) {
    validate(Optimizer::lr, beta1_, new_beta2, epsilon_, weight_decay_);
    beta2_ = new_beta2;
}

void NovoGrad::set_epsilon(double new_epsilon) {
    validate(Optimizer::lr, beta1_, beta2_, new_epsilon, weight_decay_);
    epsilon_ = new_epsilon;
}

void NovoGrad::set_weight_decay(double new_weight_decay) {
    validate(Optimizer::lr, beta1_, beta2_, epsilon_, new_weight_decay);
    weight_decay_ = new_weight_decay;
}

double NovoGrad::squared_l2_norm(const Tensor& tensor) {
    double squared_norm = 0.0;
    for (size_t row = 0; row < tensor.rows; ++row) {
        for (size_t col = 0; col < tensor.cols; ++col) {
            const double value = tensor[row][col];
            squared_norm += value * value;
        }
    }
    return squared_norm;
}

void NovoGrad::ensure_state(void* layer_ptr,
                            const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) {
        return;
    }

    std::vector<ParameterState> layer_state;
    layer_state.reserve(params.size());
    for (Tensor* parameter : params) {
        ParameterState parameter_state;
        parameter_state.momentum = Tensor(parameter->rows, parameter->cols);
        parameter_state.momentum.fill(0.0);
        layer_state.push_back(std::move(parameter_state));
    }
    state_[layer_ptr] = std::move(layer_state);
}

Tensor NovoGrad::get_momentum(void* layer_ptr, size_t param_idx) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return Tensor(0, 0);
    }
    return layer_it->second[param_idx].momentum.clone();
}

bool NovoGrad::get_second_moment(void* layer_ptr,
                                 size_t param_idx,
                                 double& out) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return false;
    }
    out = layer_it->second[param_idx].second_moment;
    return true;
}

bool NovoGrad::get_max_second_moment(void* layer_ptr,
                                     size_t param_idx,
                                     double& out) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return false;
    }
    out = layer_it->second[param_idx].max_second_moment;
    return true;
}

void NovoGrad::update_param(Tensor* param,
                            Tensor* grad,
                            ParameterState& state) {
    const double gradient_squared_norm = squared_l2_norm(*grad);
    if (!state.initialized) {
        state.second_moment = gradient_squared_norm;
        state.max_second_moment = gradient_squared_norm;
        state.initialized = true;
    } else {
        state.second_moment =
            beta2_ * state.second_moment +
            (1.0 - beta2_) * gradient_squared_norm;
        state.max_second_moment =
            std::max(state.max_second_moment, state.second_moment);
    }

    const double denominator_moment =
        amsgrad_ ? state.max_second_moment : state.second_moment;
    const double denominator = std::sqrt(denominator_moment) + epsilon_;
    const double direction_scale = grad_averaging_ ? (1.0 - beta1_) : 1.0;

    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            const double direction =
                (*grad)[row][col] / denominator +
                weight_decay_ * (*param)[row][col];
            state.momentum[row][col] =
                beta1_ * state.momentum[row][col] +
                direction_scale * direction;
            (*param)[row][col] -=
                Optimizer::lr * state.momentum[row][col];
        }
    }
}

void NovoGrad::step(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) {
            continue;
        }
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "NovoGrad: parameter and gradient counts must match");
        }

        ensure_state(layer_ptr, params);
        std::vector<ParameterState>& layer_state = state_[layer_ptr];
        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* param = params[param_idx];
            Tensor* grad = grads[param_idx];
            if (param->rows != grad->rows || param->cols != grad->cols) {
                throw std::runtime_error(
                    "NovoGrad: parameter and gradient shapes must match");
            }
            update_param(param, grad, layer_state[param_idx]);
        }
        layer->zero_grad();
    }
    ++num_steps_;
}
