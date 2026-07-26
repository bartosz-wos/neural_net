#include "diffgrad.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

DiffGrad::DiffGrad(double lr_, double b1, double b2, double eps, double wd)
    : lr(lr_), beta1(b1), beta2(b2), epsilon(eps),
      t(1), weight_decay(wd) {
    // Validate at construction time so callers don't get a surprise later.
    if (!(lr >= 0.0))
        throw std::invalid_argument("DiffGrad: lr must be >= 0");
    if (!(b1 >= 0.0 && b1 < 1.0))
        throw std::invalid_argument("DiffGrad: beta1 must be in [0, 1)");
    if (!(b2 >= 0.0 && b2 < 1.0))
        throw std::invalid_argument("DiffGrad: beta2 must be in [0, 1)");
    if (!(eps > 0.0))
        throw std::invalid_argument("DiffGrad: epsilon must be > 0");
    if (!(wd >= 0.0))
        throw std::invalid_argument("DiffGrad: weight_decay must be >= 0");
}

void DiffGrad::set_lr(double v) {
    if (!(v >= 0.0)) throw std::invalid_argument("DiffGrad::set_lr: must be >= 0");
    lr = v;
}

void DiffGrad::set_beta1(double v) {
    if (!(v >= 0.0 && v < 1.0))
        throw std::invalid_argument("DiffGrad::set_beta1: must be in [0, 1)");
    beta1 = v;
}

void DiffGrad::set_beta2(double v) {
    if (!(v >= 0.0 && v < 1.0))
        throw std::invalid_argument("DiffGrad::set_beta2: must be in [0, 1)");
    beta2 = v;
}

void DiffGrad::set_epsilon(double v) {
    if (!(v > 0.0))
        throw std::invalid_argument("DiffGrad::set_epsilon: must be > 0");
    epsilon = v;
}

void DiffGrad::set_weight_decay(double v) {
    if (!(v >= 0.0))
        throw std::invalid_argument("DiffGrad::set_weight_decay: must be >= 0");
    weight_decay = v;
}

void DiffGrad::ensure_state(void* layer_ptr,
                             const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;
    std::vector<State> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        State st;
        st.m      = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.v      = Tensor(p.rows, p.cols);
        st.v.fill(0.0);
        st.g_prev = Tensor(p.rows, p.cols);
        st.g_prev.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

bool DiffGrad::has_state(void* layer_ptr, size_t param_idx) const {
    auto it = state_.find(layer_ptr);
    if (it == state_.end()) return false;
    return param_idx < it->second.size();
}

void DiffGrad::update_param(Tensor* param, Tensor* grad, State& st,
                              double lr, double b1, double b2,
                              double eps, double wd,
                              double b1_c, double b2_c) {
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // First moment: m_t = β1 * m_{t-1} + (1−β1) * g_t
            st.m[i][j] = b1 * st.m[i][j] + (1.0 - b1) * g;

            // Second moment: v_t = β2 * v_{t-1} + (1−β2) * g_t²
            st.v[i][j] = b2 * st.v[i][j] + (1.0 - b2) * g * g;

            // DiffGrad Friction Coefficient: DFC = sigmoid(|g_{t-1} − g_t|)
            //   ∈ (0, 1):   ≈ 0.5 when gradient is unchanging (small change)
            //               → 1.0 when gradient is changing rapidly.
            double diff = std::abs(st.g_prev[i][j] - g);
            double dfc = 1.0 / (1.0 + std::exp(-diff));

            // Apply DFC to the first moment.
            double m_eff = dfc * st.m[i][j];

            // Adam-style bias correction packed into step_size and denom.
            double denom   = std::sqrt(st.v[i][j]) + eps;
            double step_size = lr * std::sqrt(b2_c) / b1_c;

            double update = step_size * m_eff / denom;

            // Optional decoupled weight decay: param *= (1 − lr * wd)
            if (wd > 0.0) {
                (*param)[i][j] -= lr * wd * (*param)[i][j];
            }

            (*param)[i][j] -= update;

            // Cache gradient for the next step's DFC.
            st.g_prev[i][j] = g;
        }
    }
}

void DiffGrad::step(Model& model) {
    // Bias correction factors for this timestep
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);
    // Guard against the very narrow degenerate case where β1 or β2 == 1.0
    // (already validated to be < 1 in the constructor / setters, so this
    // strictly positive denominator is always true here).
    if (b1_c <= 0.0 || b2_c <= 0.0) {
        throw std::logic_error("DiffGrad: bias-correction degenerated; "
                               "check that β1, β2 are strictly < 1");
    }

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads  = layer->gradients();
        if (params.empty()) continue;
        if (params.size() != grads.size()) {
            throw std::logic_error("DiffGrad: parameter/gradient count mismatch");
        }

        ensure_state(layer_ptr, params);
        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], state_vec[i],
                         lr, beta1, beta2,
                         epsilon, weight_decay,
                         b1_c, b2_c);
        }

        layer->zero_grad();
    }

    t++;
}
