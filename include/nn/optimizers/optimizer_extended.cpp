#include "optimizer_extended.h"
#include "../core/model.h"
#include "../core/layer.h"
#include "../core/tensor.h"
#include <cmath>
#include <algorithm>
#include <map>

void RMSprop::step(Model& model) {
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        if (cache_.find(ptr) == cache_.end()) {
            cache_[ptr] = std::vector<Tensor>(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                cache_[ptr][i] = Tensor(params[i]->rows, params[i]->cols);
                cache_[ptr][i].fill(0.0);
            }
        }

        auto& cache = cache_[ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            Tensor& c = cache[i];
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c_col = 0; c_col < p->cols; ++c_col) {
                    c[r][c_col] = alpha * c[r][c_col] + (1 - alpha) * (*g)[r][c_col] * (*g)[r][c_col];
                    (*p)[r][c_col] -= lr * (*g)[r][c_col] / (std::sqrt(c[r][c_col]) + eps);
                }
            }
        }
        ptr->zero_grad();
    }
}

void AdamW::step(Model& model) {
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        if (m_state.find(ptr) == m_state.end()) {
            m_state[ptr] = std::vector<Tensor>(params.size());
            v_state[ptr] = std::vector<Tensor>(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                m_state[ptr][i] = Tensor(params[i]->rows, params[i]->cols);
                m_state[ptr][i].fill(0.0);
                v_state[ptr][i] = Tensor(params[i]->rows, params[i]->cols);
                v_state[ptr][i].fill(0.0);
            }
        }

        auto& m_vec = m_state[ptr];
        auto& v_vec = v_state[ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            Tensor& m = m_vec[i];
            Tensor& v = v_vec[i];

            // Bias correction: 1 - beta^t, t starts at 1 so first step gets 1-beta1
            double b1_corr = 1.0 - std::pow(beta1, t);
            double b2_corr = 1.0 - std::pow(beta2, t);

            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c_col = 0; c_col < p->cols; ++c_col) {
                    double gg = (*g)[r][c_col];
                    m[r][c_col] = beta1 * m[r][c_col] + (1 - beta1) * gg;
                    v[r][c_col] = beta2 * v[r][c_col] + (1 - beta2) * gg * gg;
                    double m_hat = m[r][c_col] / b1_corr;
                    double v_hat = v[r][c_col] / b2_corr;
                    // AdamW: decoupled weight decay (Loshchilov & Hutter 2019)
                    (*p)[r][c_col] -= lr * (m_hat / (std::sqrt(v_hat) + epsilon)
                                           + weight_decay * (*p)[r][c_col]);
                }
            }
        }
        ptr->zero_grad();
    }
    t++;
}

void SGDNesterov::step(Model& model) {
    for (auto& layer : model.layers) {
        auto* ptr = layer.get();
        auto params = ptr->parameters();
        auto grads = ptr->gradients();
        if (params.empty()) continue;

        if (velocity_.find(ptr) == velocity_.end()) {
            velocity_[ptr] = std::vector<Tensor>(params.size());
            for (size_t i = 0; i < params.size(); ++i) {
                velocity_[ptr][i] = Tensor(params[i]->rows, params[i]->cols);
                velocity_[ptr][i].fill(0.0);
            }
        }

        auto& vel = velocity_[ptr];
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* p = params[i];
            Tensor* g = grads[i];
            Tensor& v = vel[i];
            for (size_t r = 0; r < p->rows; ++r) {
                for (size_t c_col = 0; c_col < p->cols; ++c_col) {
                    // Standard momentum SGD; Nesterov lookahead would require
                    // gradient re-evaluation at (p + momentum*v) which needs
                    // an extra forward pass and is not implemented here.
                    v[r][c_col] = momentum * v[r][c_col] + (*g)[r][c_col];
                    (*p)[r][c_col] -= lr * v[r][c_col];
                }
            }
        }
        ptr->zero_grad();
    }
}
void WeightDecay::step(Model& model) {
    // Only apply L2 if inner optimizer does NOT handle weight decay internally.
    // This prevents double weight decay when wrapping AdamW.
    if (weight_decay_ > 0 && !inner_->handles_weight_decay()) {
        for (auto& layer : model.layers) {
            for (Tensor* p : layer->parameters()) {
                if (p->rows == 0) continue;
                for (size_t i = 0; i < p->rows; ++i)
                    for (size_t j = 0; j < p->cols; ++j)
                        (*p)[i][j] -= weight_decay_ * (*p)[i][j];
            }
        }
    }
    // Delegate to inner optimizer
    inner_->step(model);
}
