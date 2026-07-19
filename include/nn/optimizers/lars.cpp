#include "lars.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <cmath>
#include <stdexcept>
#include <utility>

void LARS::validate(double lr,
                    double momentum,
                    double weight_decay,
                    double trust_coefficient,
                    double epsilon) {
    if (lr < 0.0) {
        throw std::invalid_argument("LARS: learning rate must be >= 0");
    }
    if (momentum < 0.0 || momentum >= 1.0) {
        throw std::invalid_argument("LARS: momentum must be in [0, 1)");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("LARS: weight decay must be >= 0");
    }
    if (trust_coefficient <= 0.0) {
        throw std::invalid_argument("LARS: trust coefficient must be > 0");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("LARS: epsilon must be > 0");
    }
}

LARS::LARS(double lr,
           double momentum,
           double weight_decay,
           double trust_coefficient,
           double epsilon,
           bool exclude_1d_from_adaptation,
           bool exclude_1d_from_weight_decay)
    : momentum_(momentum),
      weight_decay_(weight_decay),
      trust_coefficient_(trust_coefficient),
      epsilon_(epsilon),
      exclude_1d_from_adaptation_(exclude_1d_from_adaptation),
      exclude_1d_from_weight_decay_(exclude_1d_from_weight_decay),
      num_steps_(0) {
    validate(lr, momentum_, weight_decay_, trust_coefficient_, epsilon_);
    this->Optimizer::lr = lr;
}

void LARS::set_lr(double new_lr) {
    validate(new_lr, momentum_, weight_decay_, trust_coefficient_, epsilon_);
    this->Optimizer::lr = new_lr;
}

void LARS::set_momentum(double new_momentum) {
    validate(Optimizer::lr,
             new_momentum,
             weight_decay_,
             trust_coefficient_,
             epsilon_);
    momentum_ = new_momentum;
}

void LARS::set_weight_decay(double new_weight_decay) {
    validate(Optimizer::lr,
             momentum_,
             new_weight_decay,
             trust_coefficient_,
             epsilon_);
    weight_decay_ = new_weight_decay;
}

void LARS::set_trust_coefficient(double new_trust_coefficient) {
    validate(Optimizer::lr,
             momentum_,
             weight_decay_,
             new_trust_coefficient,
             epsilon_);
    trust_coefficient_ = new_trust_coefficient;
}

void LARS::set_epsilon(double new_epsilon) {
    validate(Optimizer::lr,
             momentum_,
             weight_decay_,
             trust_coefficient_,
             new_epsilon);
    epsilon_ = new_epsilon;
}

void LARS::set_exclude_1d_from_adaptation(bool exclude) {
    exclude_1d_from_adaptation_ = exclude;
}

void LARS::set_exclude_1d_from_weight_decay(bool exclude) {
    exclude_1d_from_weight_decay_ = exclude;
}

double LARS::l2_norm(const Tensor& tensor) {
    double squared_norm = 0.0;
    for (size_t row = 0; row < tensor.rows; ++row) {
        for (size_t col = 0; col < tensor.cols; ++col) {
            const double value = tensor[row][col];
            squared_norm += value * value;
        }
    }
    return std::sqrt(squared_norm);
}

bool LARS::is_vector_shaped(const Tensor& tensor) {
    return tensor.rows == 1 || tensor.cols == 1;
}

void LARS::ensure_state(void* layer_ptr,
                        const std::vector<Tensor*>& params) {
    if (momentum_state_.find(layer_ptr) != momentum_state_.end()) {
        return;
    }

    std::vector<Tensor> buffers;
    buffers.reserve(params.size());
    for (Tensor* parameter : params) {
        Tensor buffer(parameter->rows, parameter->cols);
        buffer.fill(0.0);
        buffers.push_back(std::move(buffer));
    }
    momentum_state_[layer_ptr] = std::move(buffers);
    last_trust_ratios_[layer_ptr] = std::vector<double>(params.size(), 1.0);
}

Tensor LARS::get_momentum_buffer(void* layer_ptr, size_t param_idx) const {
    const auto layer_it = momentum_state_.find(layer_ptr);
    if (layer_it == momentum_state_.end() ||
        param_idx >= layer_it->second.size()) {
        return Tensor(0, 0);
    }
    return layer_it->second[param_idx].clone();
}

bool LARS::get_last_trust_ratio(void* layer_ptr,
                                size_t param_idx,
                                double& out) const {
    const auto layer_it = last_trust_ratios_.find(layer_ptr);
    if (layer_it == last_trust_ratios_.end() ||
        param_idx >= layer_it->second.size()) {
        return false;
    }
    out = layer_it->second[param_idx];
    return true;
}

double LARS::update_param(Tensor* param,
                          Tensor* grad,
                          Tensor& momentum_buffer) {
    const bool vector_shaped = is_vector_shaped(*param);
    const bool use_weight_decay =
        !(exclude_1d_from_weight_decay_ && vector_shaped);
    const bool use_adaptation =
        !(exclude_1d_from_adaptation_ && vector_shaped);

    Tensor update(param->rows, param->cols);
    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            const double decay = use_weight_decay
                               ? weight_decay_ * (*param)[row][col]
                               : 0.0;
            update[row][col] = (*grad)[row][col] + decay;
        }
    }

    const double param_norm = l2_norm(*param);
    const double update_norm = l2_norm(update);
    double trust_ratio = 1.0;
    if (use_adaptation && param_norm > 0.0 && update_norm > 0.0) {
        trust_ratio = trust_coefficient_ * param_norm /
                      (update_norm + epsilon_);
    }

    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            momentum_buffer[row][col] =
                momentum_ * momentum_buffer[row][col] +
                trust_ratio * update[row][col];
            (*param)[row][col] -=
                Optimizer::lr * momentum_buffer[row][col];
        }
    }

    return trust_ratio;
}

void LARS::step(Model& model) {
    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) {
            continue;
        }
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "LARS: parameter and gradient counts must match");
        }

        ensure_state(layer_ptr, params);
        std::vector<Tensor>& buffers = momentum_state_[layer_ptr];
        std::vector<double>& ratios = last_trust_ratios_[layer_ptr];

        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            Tensor* param = params[param_idx];
            Tensor* grad = grads[param_idx];
            if (param->rows != grad->rows || param->cols != grad->cols) {
                throw std::runtime_error(
                    "LARS: parameter and gradient shapes must match");
            }
            ratios[param_idx] =
                update_param(param, grad, buffers[param_idx]);
        }

        layer->zero_grad();
    }
    ++num_steps_;
}
