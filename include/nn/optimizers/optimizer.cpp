#include "optimizer.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <map>

void SGD::step(Model& model) {
    for (auto& layer : model.layers) {
        layer->update_weights(lr);
        layer->zero_grad();
    }
}

Adam::Adam(double lr, double b1, double b2, double eps)
    : lr(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::step(Model& model) {
    t++;
    // First, ensure state exists for each layer
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        // Initialize state vectors if first time seeing this layer
        if (m_state.find(ptr) == m_state.end()) {
            m_state[ptr] = std::vector<Tensor>(params.size());
            v_state[ptr] = std::vector<Tensor>(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                Tensor& p = *params[i];
                m_state[ptr][i] = Tensor(p.rows, p.cols);
                m_state[ptr][i].fill(0.0);
                v_state[ptr][i] = Tensor(p.rows, p.cols);
                v_state[ptr][i].fill(0.0);
            }
        }

        // Update moments and parameters
        auto& m_vec = m_state[ptr];
        auto& v_vec = v_state[ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            Tensor* grad = grads[i];
            Tensor& m = m_vec[i];
            Tensor& v = v_vec[i];

            // elementwise: m = beta1*m + (1-beta1)*grad; v = beta2*v + (1-beta2)*grad*grad
            for (size_t r = 0; r < param->rows; ++r) {
                for (size_t c = 0; c < param->cols; ++c) {
                    double g = (*grad)[r][c];
                    m[r][c] = beta1 * m[r][c] + (1 - beta1) * g;
                    v[r][c] = beta2 * v[r][c] + (1 - beta2) * g * g;
                    // bias-corrected
                    double m_hat = m[r][c] / (1 - std::pow(beta1, t));
                    double v_hat = v[r][c] / (1 - std::pow(beta2, t));
                    // update
                    (*param)[r][c] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
                }
            }
        }

        // zero gradients
        ptr->zero_grad();
    }
}

double Optimizer::clip_grad_norm_(Model& model, double max_norm) {
    // Compute total L2 norm of all gradients
    double total_norm_sq = 0.0;
    for (auto& layer : model.layers) {
        for (Tensor* grad : layer->gradients()) {
            if (grad->rows == 0) continue;
            for (size_t i = 0; i < grad->rows; ++i)
                for (size_t j = 0; j < grad->cols; ++j)
                    total_norm_sq += (*grad)[i][j] * (*grad)[i][j];
        }
    }
    double total_norm = std::sqrt(total_norm_sq);
    if (total_norm > max_norm && total_norm > 0.0) {
        double scale = max_norm / total_norm;
        for (auto& layer : model.layers) {
            for (Tensor* grad : layer->gradients()) {
                for (size_t i = 0; i < grad->rows; ++i)
                    for (size_t j = 0; j < grad->cols; ++j)
                        (*grad)[i][j] *= scale;
            }
        }
    }
    return total_norm;
}
