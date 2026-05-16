#include "adabelief.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

AdaBelief::AdaBelief(double lr, double b1, double b2, double eps, double wd)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps),
      t(1), weight_decay(wd) {}

void AdaBelief::ensure_state(void* layer_ptr,
                              const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<BeliefState> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        BeliefState st;
        st.m = Tensor(p.rows, p.cols);
        st.m.fill(0.0);
        st.s = Tensor(p.rows, p.cols);
        st.s.fill(0.0);
        vec.push_back(std::move(st));
    }
    state_[layer_ptr] = std::move(vec);
}

void AdaBelief::update_param(Tensor* param, Tensor* grad,
                              BeliefState& st,
                              double lr, double b1, double b2,
                              double eps, double wd,
                              double b1_c, double b2_c) {
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // First moment update: m_t = β1 * m_{t-1} + (1-β1) * g_t
            double m_prev = st.m[i][j];
            st.m[i][j] = b1 * m_prev + (1.0 - b1) * g;

            // Residual: r_t = g_t - m_{t-1} (surprise relative to previous belief)
            double residual = g - m_prev;

            // Belief variance: s_t = β2 * s_{t-1} + (1-β2) * r_t^2
            st.s[i][j] = b2 * st.s[i][j] + (1.0 - b2) * residual * residual;

            // Bias-corrected first moment
            double m_hat = st.m[i][j] / b1_c;

            // Bias-corrected belief variance (used for step size)
            double s_hat = st.s[i][j] / b2_c;

            // AdaBelief update: param -= lr * m_hat / (sqrt(s_hat) + eps)
            double denom = std::sqrt(s_hat) + eps;
            double update = m_hat / denom;

            // Optional weight decay (AdamW style)
            if (wd > 0.0) {
                (*param)[i][j] -= lr * wd * (*param)[i][j];
            }

            (*param)[i][j] -= lr * update;
        }
    }
}

void AdaBelief::step(Model& model) {
    // Bias corrections for this timestep
    double b1_c = 1.0 - std::pow(beta1, t);
    double b2_c = 1.0 - std::pow(beta2, t);

    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i],
                         state_vec[i],
                         lr, beta1, beta2,
                         epsilon, weight_decay,
                         b1_c, b2_c);
        }

        layer->zero_grad();
    }

    t++;
}