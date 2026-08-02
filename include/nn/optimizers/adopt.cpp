#include "adopt.h"
#include "../core/layer.h"
#include "../core/model.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

ADOPT::ADOPT(double lr, double beta1, double beta2, double epsilon,
             double clip_exp, double weight_decay, bool decoupled)
    : beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      clip_exp_(clip_exp), weight_decay_(weight_decay),
      decoupled_(decoupled), t_(1) {
    set_lr(lr);
    set_beta1(beta1);
    set_beta2(beta2);
    set_epsilon(epsilon);
    set_clip_exp(clip_exp);
    set_weight_decay(weight_decay);
}

void ADOPT::set_lr(double value) {
    if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument("ADOPT: lr must be finite and >= 0");
    Optimizer::lr = value;
}

void ADOPT::set_beta1(double value) {
    if (!(value >= 0.0 && value < 1.0))
        throw std::invalid_argument("ADOPT: beta1 must be in [0,1)");
    beta1_ = value;
}

void ADOPT::set_beta2(double value) {
    if (!(value >= 0.0 && value < 1.0))
        throw std::invalid_argument("ADOPT: beta2 must be in [0,1)");
    beta2_ = value;
}

void ADOPT::set_epsilon(double value) {
    if (!(value > 0.0) || !std::isfinite(value))
        throw std::invalid_argument("ADOPT: epsilon must be finite and > 0");
    epsilon_ = value;
}

void ADOPT::set_clip_exp(double value) {
    if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument("ADOPT: clip_exp must be finite and >= 0");
    clip_exp_ = value;
}

void ADOPT::set_weight_decay(double value) {
    if (!(value >= 0.0) || !std::isfinite(value))
        throw std::invalid_argument("ADOPT: weight_decay must be finite and >= 0");
    weight_decay_ = value;
}

void ADOPT::ensure_state(void* layer_ptr, const std::vector<Tensor*>& params) {
    auto existing = state_.find(layer_ptr);
    if (existing != state_.end()) {
        if (existing->second.size() != params.size())
            throw std::logic_error("ADOPT: parameter count changed after state initialization");
        for (size_t i = 0; i < params.size(); ++i) {
            if (existing->second[i].m.rows != params[i]->rows ||
                existing->second[i].m.cols != params[i]->cols)
                throw std::logic_error("ADOPT: parameter shape changed after state initialization");
        }
        return;
    }

    std::vector<ParamState> states;
    states.reserve(params.size());
    for (Tensor* param : params) {
        if (!param) throw std::logic_error("ADOPT: null parameter");
        ParamState state;
        state.m = Tensor::zeros(param->rows, param->cols);
        state.v = Tensor::zeros(param->rows, param->cols);
        states.push_back(std::move(state));
    }
    state_[layer_ptr] = std::move(states);
}

void ADOPT::step(Model& model) {
    const bool initialization_step = (t_ == 1);
    const double clip_value = initialization_step
        ? 0.0
        : std::pow(static_cast<double>(t_ - 1), clip_exp_);

    for (auto& layer : model.layers) {
        if (!layer) continue;
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size())
            throw std::logic_error("ADOPT: parameter/gradient count mismatch");

        void* layer_ptr = layer.get();
        ensure_state(layer_ptr, params);
        auto& states = state_[layer_ptr];

        for (size_t p = 0; p < params.size(); ++p) {
            Tensor* param = params[p];
            Tensor* grad = grads[p];
            if (!param || !grad)
                throw std::logic_error("ADOPT: null parameter or gradient");
            if (param->rows != grad->rows || param->cols != grad->cols)
                throw std::logic_error("ADOPT: parameter/gradient shape mismatch");

            ParamState& state = states[p];
            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g = (*grad)[r][c];
                    if (weight_decay_ != 0.0 && !decoupled_)
                        g += weight_decay_ * (*param)[r][c];

                    if (initialization_step) {
                        state.v[r][c] += g * g;
                        continue;
                    }

                    if (weight_decay_ != 0.0 && decoupled_)
                        (*param)[r][c] *= (1.0 - Optimizer::lr * weight_decay_);

                    const double denominator = std::max(std::sqrt(state.v[r][c]), epsilon_);
                    double normalized = g / denominator;
                    normalized = std::max(-clip_value, std::min(clip_value, normalized));
                    state.m[r][c] = beta1_ * state.m[r][c] + (1.0 - beta1_) * normalized;
                    (*param)[r][c] -= Optimizer::lr * state.m[r][c];
                    state.v[r][c] = beta2_ * state.v[r][c] + (1.0 - beta2_) * g * g;
                }
            }
        }
        layer->zero_grad();
    }
    ++t_;
}

bool ADOPT::has_state(void* layer_ptr) const {
    return state_.find(layer_ptr) != state_.end();
}

size_t ADOPT::num_params_with_state(void* layer_ptr) const {
    auto it = state_.find(layer_ptr);
    return it == state_.end() ? 0 : it->second.size();
}

const Tensor& ADOPT::get_m(void* layer_ptr, size_t param_idx) const {
    static const Tensor empty;
    auto it = state_.find(layer_ptr);
    return (it == state_.end() || param_idx >= it->second.size()) ? empty : it->second[param_idx].m;
}

const Tensor& ADOPT::get_v(void* layer_ptr, size_t param_idx) const {
    static const Tensor empty;
    auto it = state_.find(layer_ptr);
    return (it == state_.end() || param_idx >= it->second.size()) ? empty : it->second[param_idx].v;
}
