#include "lion.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>

Lion::Lion(double lr, double b1, double b2, double wd)
    : lr(lr), beta1(b1), beta2(b2), weight_decay(wd) {}

void Lion::ensure_state(void* layer_ptr,
                         const std::vector<Tensor*>& params) {
    if (state_.find(layer_ptr) != state_.end()) return;

    std::vector<Tensor> vec;
    vec.reserve(params.size());
    for (size_t i = 0; i < params.size(); ++i) {
        const Tensor& p = *params[i];
        Tensor m(p.rows, p.cols);
        m.fill(0.0);
        vec.push_back(std::move(m));
    }
    state_[layer_ptr] = std::move(vec);
}

void Lion::update_param(Tensor* param, Tensor* grad, Tensor& m) {
    for (size_t i = 0; i < grad->rows; ++i) {
        for (size_t j = 0; j < grad->cols; ++j) {
            double g = (*grad)[i][j];

            // c_t = beta1 * m_{t-1} + (1 - beta1) * g
            double c = beta1 * m[i][j] + (1.0 - beta1) * g;

            // sign(c): +1 if c > 0, -1 if c < 0, 0 if c == 0
            double s = (c > 0.0) ? 1.0 : (c < 0.0 ? -1.0 : 0.0);

            // m_t = beta2 * m_{t-1} + (1 - beta2) * g
            m[i][j] = beta2 * m[i][j] + (1.0 - beta2) * g;

            // Apply weight decay (decoupled, AdamW-style)
            // param -= lr * (sign(c) + wd * param)
            double update = s + weight_decay * (*param)[i][j];
            (*param)[i][j] -= lr * update;
        }
    }
}

void Lion::step(Model& model) {
    for (size_t li = 0; li < model.layers.size(); ++li) {
        auto& layer = model.layers[li];
        auto* layer_ptr = layer.get();
        auto params = layer->parameters();
        auto grads = layer->gradients();
        if (params.empty()) continue;

        ensure_state(layer_ptr, params);
        auto& state_vec = state_[layer_ptr];

        for (size_t i = 0; i < params.size(); ++i) {
            update_param(params[i], grads[i], state_vec[i]);
        }

        layer->zero_grad();
    }
}
