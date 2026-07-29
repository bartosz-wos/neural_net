#include "lamb.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

void LAMB::validate(double lr,
                    double beta1,
                    double beta2,
                    double epsilon,
                    double trust_ratio_gamma,
                    double weight_decay) {
    if (lr < 0.0) {
        throw std::invalid_argument("LAMB: learning rate must be >= 0");
    }
    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("LAMB: beta1 must be in [0, 1)");
    }
    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("LAMB: beta2 must be in [0, 1)");
    }
    if (epsilon <= 0.0) {
        throw std::invalid_argument("LAMB: epsilon must be > 0");
    }
    if (trust_ratio_gamma < 1.0) {
        throw std::invalid_argument(
            "LAMB: trust ratio gamma must be >= 1");
    }
    if (weight_decay < 0.0) {
        throw std::invalid_argument("LAMB: weight decay must be >= 0");
    }
}

LAMB::LAMB(double lr,
           double beta1,
           double beta2,
           double epsilon,
           double trust_ratio_gamma,
           double weight_decay)
    : beta1(beta1),
      beta2(beta2),
      epsilon(epsilon),
      beta1_corr(1.0),
      beta2_corr(1.0),
      t(1),
      trust_ratio_gamma(trust_ratio_gamma),
      weight_decay(weight_decay) {
    validate(lr,
             this->beta1,
             this->beta2,
             this->epsilon,
             this->trust_ratio_gamma,
             this->weight_decay);
    this->Optimizer::lr = lr;
}

void LAMB::set_lr(double new_lr) {
    validate(new_lr,
             beta1,
             beta2,
             epsilon,
             trust_ratio_gamma,
             weight_decay);
    Optimizer::lr = new_lr;
}

void LAMB::set_beta1(double new_beta1) {
    validate(Optimizer::lr,
             new_beta1,
             beta2,
             epsilon,
             trust_ratio_gamma,
             weight_decay);
    beta1 = new_beta1;
}

void LAMB::set_beta2(double new_beta2) {
    validate(Optimizer::lr,
             beta1,
             new_beta2,
             epsilon,
             trust_ratio_gamma,
             weight_decay);
    beta2 = new_beta2;
}

void LAMB::set_epsilon(double new_epsilon) {
    validate(Optimizer::lr,
             beta1,
             beta2,
             new_epsilon,
             trust_ratio_gamma,
             weight_decay);
    epsilon = new_epsilon;
}

void LAMB::set_trust_ratio_gamma(double new_gamma) {
    validate(Optimizer::lr,
             beta1,
             beta2,
             epsilon,
             new_gamma,
             weight_decay);
    trust_ratio_gamma = new_gamma;
}

void LAMB::set_weight_decay(double new_weight_decay) {
    validate(Optimizer::lr,
             beta1,
             beta2,
             epsilon,
             trust_ratio_gamma,
             new_weight_decay);
    weight_decay = new_weight_decay;
}

double LAMB::l2_norm(const Tensor& tensor) {
    double squared_norm = 0.0;
    for (size_t row = 0; row < tensor.rows; ++row) {
        for (size_t col = 0; col < tensor.cols; ++col) {
            const double value = tensor[row][col];
            squared_norm += value * value;
        }
    }
    return std::sqrt(squared_norm);
}

void LAMB::ensure_state(void* layer_ptr,
                        const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) {
        return;
    }

    std::vector<ParameterState> layer_state;
    layer_state.reserve(params.size());
    for (const Tensor* parameter : params) {
        ParameterState parameter_state;
        parameter_state.m = Tensor(parameter->rows, parameter->cols);
        parameter_state.v = Tensor(parameter->rows, parameter->cols);
        parameter_state.m.fill(0.0);
        parameter_state.v.fill(0.0);
        layer_state.push_back(std::move(parameter_state));
    }
    state_[layer_ptr] = std::move(layer_state);
    last_trust_ratios_[layer_ptr] = std::vector<double>(params.size(), 1.0);
}

bool LAMB::get_m(void* layer_ptr, size_t param_idx, Tensor& out) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return false;
    }
    out = layer_it->second[param_idx].m.clone();
    return true;
}

bool LAMB::get_v(void* layer_ptr, size_t param_idx, Tensor& out) const {
    const auto layer_it = state_.find(layer_ptr);
    if (layer_it == state_.end() || param_idx >= layer_it->second.size()) {
        return false;
    }
    out = layer_it->second[param_idx].v.clone();
    return true;
}

bool LAMB::get_last_trust_ratio(void* layer_ptr,
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

double LAMB::update_param(Tensor* param,
                          Tensor* grad,
                          ParameterState& state,
                          double beta1_correction,
                          double beta2_correction) {
    Tensor update(param->rows, param->cols);

    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            const double gradient = (*grad)[row][col];
            state.m[row][col] =
                beta1 * state.m[row][col] + (1.0 - beta1) * gradient;
            state.v[row][col] =
                beta2 * state.v[row][col] +
                (1.0 - beta2) * gradient * gradient;

            const double m_hat = state.m[row][col] / beta1_correction;
            const double v_hat = state.v[row][col] / beta2_correction;
            update[row][col] =
                m_hat / (std::sqrt(v_hat) + epsilon) +
                weight_decay * (*param)[row][col];
        }
    }

    const double parameter_norm = l2_norm(*param);
    const double update_norm = l2_norm(update);
    double trust_ratio = 1.0;
    if (parameter_norm > 0.0 && update_norm > 0.0) {
        trust_ratio = parameter_norm / update_norm;
        const double lower = 1.0 / trust_ratio_gamma;
        trust_ratio = std::max(lower,
                               std::min(trust_ratio_gamma, trust_ratio));
    }

    for (size_t row = 0; row < param->rows; ++row) {
        for (size_t col = 0; col < param->cols; ++col) {
            (*param)[row][col] -=
                Optimizer::lr * trust_ratio * update[row][col];
        }
    }
    return trust_ratio;
}

void LAMB::step(Model& model) {
    beta1_corr = 1.0 - std::pow(beta1, t);
    beta2_corr = 1.0 - std::pow(beta2, t);

    for (auto& layer : model.layers) {
        void* layer_ptr = layer.get();
        std::vector<Tensor*> params = layer->parameters();
        std::vector<Tensor*> grads = layer->gradients();
        if (params.empty()) {
            continue;
        }
        if (params.size() != grads.size()) {
            throw std::runtime_error(
                "LAMB: parameter and gradient counts must match");
        }

        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            if (params[param_idx]->rows != grads[param_idx]->rows ||
                params[param_idx]->cols != grads[param_idx]->cols) {
                throw std::runtime_error(
                    "LAMB: parameter and gradient shapes must match");
            }
        }

        ensure_state(layer_ptr, params);
        std::vector<ParameterState>& layer_state = state_[layer_ptr];
        std::vector<double>& ratios = last_trust_ratios_[layer_ptr];
        for (size_t param_idx = 0; param_idx < params.size(); ++param_idx) {
            ratios[param_idx] = update_param(params[param_idx],
                                             grads[param_idx],
                                             layer_state[param_idx],
                                             beta1_corr,
                                             beta2_corr);
        }
        layer->zero_grad();
    }

    ++t;
}
