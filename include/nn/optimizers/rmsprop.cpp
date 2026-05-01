#include "rmsprop.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>

RMSProp::RMSProp(double lr, double alpha, double eps, double weight_decay)
    : lr(lr), alpha(alpha), epsilon(eps), weight_decay(weight_decay), t(1) {}

void RMSProp::step(Model& model) {
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        // Initialize Eg² cache on first encounter of this layer
        if (eg_state.find(ptr) == eg_state.end()) {
            eg_state[ptr] = std::vector<Tensor>(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                eg_state[ptr][i] = Tensor(params[i]->rows, params[i]->cols);
                eg_state[ptr][i].fill(0.0);
            }
        }

        auto& eg_vec = eg_state[ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            Tensor& eg = eg_vec[i];

            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g = (*grad)[r][c];

                    // L2 regularization (Tikhonov / weight decay)
                    if (weight_decay > 0) {
                        g += weight_decay * (*param)[r][c];
                    }

                    // E[g²] = α * E[g²] + (1-α) * g²
                    eg[r][c] = alpha * eg[r][c] + (1 - alpha) * g * g;

                    // parameter update: θ ← θ - lr * g / √(E[g²] + ε)
                    (*param)[r][c] -= lr * g / (std::sqrt(eg[r][c]) + epsilon);
                }
            }
        }

        ptr->zero_grad();
    }
    ++t;
}
